"""MiniMax Music 3 engine worker. Runs inside the Python of a virtual environment with mlx-audio,
never inside ace-step-api's environment: it imports only the standard library, numpy and
``mlx_audio``.

Protocol: one JSON object per line. Requests arrive on stdin; replies go to the file descriptor
given by ``--proto-fd`` (libraries print progress bars and logs to stdout, so stdout is pointed
at stderr and never carries the protocol).

    -> {"event": "ready", "load_s": 2.7, "device": "mlx"}
    <- {"cmd": "generate", "id": "...", "caption": "...", "lyrics": "...", "instrumental": false,
        "duration": 60, "seed": 7, "expected_s": 580.0, "out": "/abs/raw-0.wav"}
    -> {"event": "progress", "id": "...", "value": 0.31}
    -> {"event": "done", "id": "...", "seconds": 512.4, "sample_rate": 44100, "frames": 2646000,
        "channels": 2, "gain_db": -0.4}
    -> {"event": "error", "id": "...", "message": "...", "input": true}
    <- {"cmd": "quit"}             (or stdin closed: the worker exits)

A failed load sends ``{"event": "fatal", "message": ...}`` and exits with status 1. An ``error``
with ``"input": true`` is input the engine refuses (mlx-audio raises ``ValueError``: more than 5000
prompt tokens, zero audio frames); the worker is fine and stays up. After every track MLX's
buffer cache is released, so an idle worker holds only the model.

mlx-audio's ``generate`` reports nothing until the whole track is made, so ``progress`` is an
estimate: the time spent so far over ``expected_s``, at most 0.95. All MLX work stays on the
main thread (MLX GPU streams are per thread); the progress thread only writes to the channel.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
import wave
from pathlib import Path
from typing import Any, TextIO

# the default and the maximum mlx-audio accepts for MiniMax Music 3
STEPS = 30
# the model needs lyrics; this is its marker for none
INSTRUMENTAL_LYRICS = "[instrumental]"
# the model's float output is clipped to [-1, 1]: a track that peaks above this is turned down
PEAK_CEILING_DB = -1.0
TICK_SECONDS = 1.0
MAX_ESTIMATED_PROGRESS = 0.95


class InputError(ValueError):
    """The engine refused the request's input; trying again with the same input cannot help."""


class Channel:
    """The protocol channel; the progress thread and the main thread both write to it."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._lock = threading.Lock()

    def send(self, **message: Any) -> None:
        line = json.dumps(message, ensure_ascii=False) + "\n"
        with self._lock:
            self._stream.write(line)
            self._stream.flush()


def open_channel(fd: int) -> Channel:
    """The protocol channel, then fd 1 -> stderr so nothing else can write to the channel."""
    stream = os.fdopen(os.dup(fd), "w", buffering=1, encoding="utf-8")
    if fd > 2:
        os.close(fd)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return Channel(stream)


class Progress:
    """Reports ``elapsed / expected_s`` (at most 0.95) every second until stopped."""

    def __init__(self, channel: Channel, job: Any, expected_s: float) -> None:
        self._channel = channel
        self._job = job
        self._expected = max(float(expected_s), 1.0)
        self._stop = threading.Event()
        self._started = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="progress", daemon=True)

    def __enter__(self) -> Progress:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        # stopped before "done" or "error" is sent: no progress line may follow them
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        last = 0.0
        while not self._stop.wait(TICK_SECONDS):
            value = min((time.monotonic() - self._started) / self._expected, 1.0)
            value = round(min(value, MAX_ESTIMATED_PROGRESS), 3)
            if value - last >= 0.01:
                last = value
                self._channel.send(event="progress", id=self._job, value=value)


def as_frames(data: Any) -> Any:
    """``(frames, channels)`` float32 audio from whatever layout the engine returned."""
    import numpy as np  # pyright: ignore[reportMissingImports]

    audio = np.asarray(data, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]
    elif audio.ndim == 2 and audio.shape[0] <= 2 < audio.shape[1]:
        audio = audio.T  # (channels, frames)
    if audio.ndim != 2 or audio.shape[0] == 0:
        raise RuntimeError(f"the engine returned audio of shape {tuple(audio.shape)}")
    return np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)


def fit_peak(audio: Any, ceiling_db: float = PEAK_CEILING_DB) -> tuple[Any, float]:
    """Turn the whole track down (never up) so its peak stays under ``ceiling_db``; no limiter.
    Returns the audio and the gain in dB (0.0 when nothing was done)."""
    import numpy as np  # pyright: ignore[reportMissingImports]

    ceiling = 10 ** (ceiling_db / 20)
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    if peak <= ceiling:
        return audio, 0.0
    gain = ceiling / peak
    return audio * np.float32(gain), round(float(20 * np.log10(gain)), 2)


def write_wav(path: Path, audio: Any, rate: int) -> None:
    """``(frames, channels)`` float audio in [-1, 1] as 16-bit PCM."""
    import numpy as np  # pyright: ignore[reportMissingImports]

    pcm = np.clip(np.round(np.asarray(audio) * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(int(pcm.shape[1]))
        wav.setsampwidth(2)
        wav.setframerate(int(rate))
        wav.writeframes(pcm.tobytes())


def _load(model: str) -> Any:
    from mlx_audio.music import load  # pyright: ignore[reportMissingImports]

    return load(model)


def _generate(model: Any, request: dict[str, Any], channel: Channel) -> dict[str, Any]:
    lyrics = str(request.get("lyrics") or "")
    # the engine refuses empty lyrics: no lyrics is an instrumental
    if bool(request["instrumental"]) or not lyrics.strip():
        lyrics = INSTRUMENTAL_LYRICS
    started = time.monotonic()
    with Progress(channel, request["id"], float(request.get("expected_s") or 0.0)):
        produced = model.generate(
            text=str(request["caption"]),
            lyrics=lyrics,
            duration=float(request["duration"]),
            steps=STEPS,
            seed=int(request["seed"]),
        )
        try:
            result = next(iter(produced), None)
        except ValueError as exc:  # mlx-audio's checks of the input (it raises nothing else)
            raise InputError(f"{type(exc).__name__}: {exc}") from exc
        if result is None:
            raise RuntimeError("the engine produced no audio")
        audio = as_frames(result.audio)
        rate = int(result.sample_rate)
    audio, gain_db = fit_peak(audio)
    write_wav(Path(request["out"]), audio, rate)
    return {
        "seconds": round(time.monotonic() - started, 2),
        "sample_rate": rate,
        "frames": int(audio.shape[0]),
        "channels": int(audio.shape[1]),
        "gain_db": gain_db,
    }


def release_cache() -> None:
    """Give MLX's cached GPU buffers back (``mx.clear_cache``, ``mx.metal.clear_cache`` before
    MLX 0.29); nothing when neither exists."""
    try:
        import mlx.core as mx  # pyright: ignore[reportMissingImports]
    except ImportError:
        return
    clear = getattr(mx, "clear_cache", None) or getattr(
        getattr(mx, "metal", None), "clear_cache", None
    )
    if callable(clear):
        try:
            clear()
        except Exception:
            traceback.print_exc()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proto-fd", type=int, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    channel = open_channel(args.proto_fd)
    started = time.monotonic()
    try:
        model = _load(args.model)
    except Exception as exc:
        traceback.print_exc()
        channel.send(event="fatal", message=f"{type(exc).__name__}: {exc}")
        return 1
    channel.send(event="ready", load_s=round(time.monotonic() - started, 2), device="mlx")
    for line in sys.stdin:
        try:
            request = json.loads(line)
        except ValueError:
            continue
        if not isinstance(request, dict) or request.get("cmd") == "quit":
            break
        if request.get("cmd") != "generate":
            continue
        try:
            reply = _generate(model, request, channel)
        except InputError as exc:
            channel.send(event="error", id=request.get("id"), message=str(exc), input=True)
            continue
        except Exception as exc:
            traceback.print_exc()
            channel.send(
                event="error", id=request.get("id"), message=f"{type(exc).__name__}: {exc}"
            )
            continue
        finally:
            release_cache()
        channel.send(event="done", id=request["id"], **reply)
    return 0


if __name__ == "__main__":
    sys.exit(main())
