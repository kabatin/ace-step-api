"""ACE-Step 1.5 engine worker. Runs inside the ACE-Step checkout's own Python (3.11 + torch),
never inside ace-step-api's environment: it imports only the standard library, ``acestep`` and
``soundfile`` (an ACE-Step dependency).

Protocol: one JSON object per line. Requests arrive on stdin; replies go to the file descriptor
given by ``--proto-fd`` (ACE-Step prints progress bars and logs to stdout, so stdout is pointed
at stderr and never carries the protocol).

    -> {"event": "ready", "load_s": 41.2, "device": "mps", "lm_backend": "mlx"}
    <- {"cmd": "generate", "id": "...", "caption": "...", "lyrics": "...", "instrumental": false,
        "vocal_language": "ja", "duration": 60, "seed": 7, "out": "/abs/raw-0.wav"}
    -> {"event": "progress", "id": "...", "value": 0.52}
    -> {"event": "done", "id": "...", "seconds": 21.0, "sample_rate": 48000, "frames": 2880000,
        "channels": 2, "gain_db": -3.1, "limiter_db": -1.2}
    -> {"event": "error", "id": "...", "message": "..."}
    <- {"cmd": "quit"}             (or stdin closed: the worker exits)

A failed load sends ``{"event": "fatal", "message": ...}`` and exits with status 1.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, TextIO

# ACE-Step's own recommendation for the turbo model
INFERENCE_STEPS = 8
SHIFT = 3.0
INSTRUMENTAL_LYRICS = "[Instrumental]"
# ACE-Step's float output can exceed full scale; 16-bit PCM would clip it for good
PEAK_CEILING_DB = -1.0
LIMITER_BLOCK = 256  # ~5 ms at 48 kHz
LIMITER_RELEASE_BLOCKS = 40  # back to unity gain over ~0.2 s
# the share of blocks the limiter may handle; louder tracks are first turned down as a whole
LIMITED_SHARE = 0.001


def limit_peaks(data: Any, ceiling_db: float = PEAK_CEILING_DB) -> tuple[Any, dict[str, float]]:
    """Keep ``(frames, channels)`` float audio under ``ceiling_db`` without squashing it.

    1. Gain: when more than ``LIMITED_SHARE`` of the 5 ms blocks would exceed the ceiling, the
       whole track is turned down until only that share does (the port's output often runs
       hot all the way through; limiting all of it would pump and distort).
    2. Limiter for what is left: the gain drops one block before a peak (look-ahead), holds
       one block after it and recovers linearly within ``LIMITER_RELEASE_BLOCKS`` blocks.

    Audio that never reaches the ceiling is returned unchanged. Returns the audio and
    ``{"gain_db": ..., "limiter_db": ...}`` (both <= 0)."""
    import numpy as np  # pyright: ignore[reportMissingImports]

    audio = np.asarray(data, dtype=np.float32)
    frames = audio.shape[0]
    ceiling = 10 ** (ceiling_db / 20)
    if frames == 0 or float(np.abs(audio).max()) <= ceiling:
        return audio, {"gain_db": 0.0, "limiter_db": 0.0}
    blocks = -(-frames // LIMITER_BLOCK)
    padded = np.zeros((blocks * LIMITER_BLOCK, audio.shape[1]), dtype=np.float32)
    padded[:frames] = np.abs(audio)
    peaks = padded.reshape(blocks, -1).max(axis=1)
    loud = float(np.quantile(peaks, 1.0 - LIMITED_SHARE, method="lower"))
    gain = min(1.0, ceiling / max(loud, 1e-9))
    audio = audio * np.float32(gain)
    peaks = peaks * gain
    need = np.minimum(1.0, ceiling / np.maximum(peaks, 1e-9))
    need = np.minimum(need, np.minimum(np.r_[need[1:], 1.0], np.r_[1.0, need[:-1]]))
    gains = np.empty_like(need)
    level = 1.0
    for index, wanted in enumerate(need):
        level = min(float(wanted), level + 1.0 / LIMITER_RELEASE_BLOCKS, 1.0)
        gains[index] = level
    centres = (np.arange(blocks) + 0.5) * LIMITER_BLOCK
    per_frame = np.interp(np.arange(frames), centres, gains).astype(np.float32)
    limited = np.clip(audio * per_frame[:, None], -ceiling, ceiling)
    return limited, {
        "gain_db": round(float(20 * np.log10(gain)), 2),
        "limiter_db": round(float(20 * np.log10(gains.min())), 2),
    }


def _audio_of(produced: dict[str, Any], sf: Any) -> tuple[Any, int]:
    """``(frames, channels)`` float audio of one result: the engine's own float tensor when it
    hands one over (before any file format could clip it), else the file it wrote."""
    tensor = produced.get("tensor")
    if tensor is not None:
        for step in ("detach", "cpu", "float"):
            if hasattr(tensor, step):
                tensor = getattr(tensor, step)()
        return tensor.numpy().T, int(produced.get("sample_rate") or 48000)
    data, rate = sf.read(str(produced["path"]), always_2d=True, dtype="float32")
    return data, int(rate)


def _protocol(fd: int) -> TextIO:
    """The protocol channel, then fd 1 -> stderr so nothing else can write to the channel."""
    channel = os.fdopen(os.dup(fd), "w", buffering=1, encoding="utf-8")
    if fd > 2:
        os.close(fd)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return channel


def _send(channel: TextIO, **message: Any) -> None:
    channel.write(json.dumps(message, ensure_ascii=False) + "\n")
    channel.flush()


def _load(args: argparse.Namespace) -> tuple[Any, Any]:
    from acestep.handler import AceStepHandler  # pyright: ignore[reportMissingImports]
    from acestep.llm_inference import LLMHandler  # pyright: ignore[reportMissingImports]

    dit = AceStepHandler()
    options: dict[str, Any] = {}
    # only engines that have the switch (the official checkout) get it
    switchable = "use_mlx_dit" in inspect.signature(dit.initialize_service).parameters
    if args.mlx_dit is not None and switchable:
        options["use_mlx_dit"] = args.mlx_dit == "on"
    status, ok = dit.initialize_service(
        project_root=args.project_root,
        config_path=args.dit_model,
        device="auto",
        prefer_source="huggingface",
        **options,
    )
    if not ok:
        raise RuntimeError(f"DiT load failed: {status}")
    llm = LLMHandler()
    status, ok = llm.initialize(
        checkpoint_dir=str(Path(args.project_root) / "checkpoints"),
        lm_model_path=args.lm_model,
        backend=args.lm_backend,
        device="auto",
    )
    if not ok:
        raise RuntimeError(f"LM load failed: {status}")
    return dit, llm


def _generate(dit: Any, llm: Any, request: dict[str, Any], channel: TextIO) -> dict[str, Any]:
    import soundfile as sf  # pyright: ignore[reportMissingImports]
    from acestep.inference import (  # pyright: ignore[reportMissingImports]
        GenerationConfig,
        GenerationParams,
        generate_music,
    )

    job = request["id"]
    instrumental = bool(request["instrumental"])
    params = GenerationParams(
        caption=request["caption"],
        lyrics=INSTRUMENTAL_LYRICS if instrumental else request["lyrics"],
        instrumental=instrumental,
        vocal_language=request.get("vocal_language") or "unknown",
        duration=float(request["duration"]),
        inference_steps=INFERENCE_STEPS,
        shift=SHIFT,
        seed=int(request["seed"]),
    )
    config = GenerationConfig(
        batch_size=1, use_random_seed=False, seeds=[int(request["seed"])], audio_format="wav"
    )
    last = [-1.0]

    def progress(value: Any = None, *_args: Any, **_kwargs: Any) -> None:
        if isinstance(value, (int, float)) and 0.0 <= value <= 1.0 and value - last[0] >= 0.01:
            last[0] = float(value)
            _send(channel, event="progress", id=job, value=round(float(value), 3))

    out = Path(request["out"])
    scratch = Path(tempfile.mkdtemp(prefix="ace-step-", dir=out.parent))
    started = time.monotonic()
    try:
        result = generate_music(dit, llm, params, config, save_dir=str(scratch), progress=progress)
        if not result.success or not result.audios:
            raise RuntimeError(result.error or result.status_message or "no audio produced")
        data, rate = _audio_of(result.audios[0], sf)
        data, levels = limit_peaks(data)
        # 16-bit PCM: what every client reads
        sf.write(str(out), data, rate, subtype="PCM_16")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return {
        "seconds": round(time.monotonic() - started, 2),
        "sample_rate": rate,
        "frames": int(data.shape[0]),
        "channels": int(data.shape[1]),
        "gain_db": levels["gain_db"],
        "limiter_db": levels["limiter_db"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proto-fd", type=int, required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--dit-model", default="acestep-v15-turbo")
    parser.add_argument("--lm-model", default="acestep-5Hz-lm-1.7B")
    parser.add_argument("--lm-backend", default="vllm")  # switched to MLX on Apple Silicon
    parser.add_argument("--mlx-dit", choices=("on", "off"), default=None)  # engine default if unset
    args = parser.parse_args()
    channel = _protocol(args.proto_fd)
    started = time.monotonic()
    try:
        dit, llm = _load(args)
    except Exception as exc:
        traceback.print_exc()
        _send(channel, event="fatal", message=f"{type(exc).__name__}: {exc}")
        return 1
    _send(
        channel,
        event="ready",
        load_s=round(time.monotonic() - started, 2),
        device=str(getattr(dit, "device", "")),
        lm_backend=str(getattr(llm, "llm_backend", "")),
    )
    for line in sys.stdin:
        try:
            request = json.loads(line)
        except ValueError:
            continue
        if request.get("cmd") == "quit":
            break
        if request.get("cmd") != "generate":
            continue
        try:
            reply = _generate(dit, llm, request, channel)
        except Exception as exc:
            traceback.print_exc()
            _send(
                channel, event="error", id=request.get("id"), message=f"{type(exc).__name__}: {exc}"
            )
            continue
        _send(channel, event="done", id=request["id"], **reply)
    return 0


if __name__ == "__main__":
    sys.exit(main())
