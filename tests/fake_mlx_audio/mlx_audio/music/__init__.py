"""A stand-in for ``mlx_audio.music``: the real MiniMax worker script runs against it in tests.

It checks its inputs the way mlx-audio's MiniMax Music 3 does (lyrics required, 0 < duration
<= 360, 1 <= steps <= 30) and returns a sine at 1/100 of real time (``duration * 441`` frames at
44.1 kHz) as numpy float32, ``(frames, 2)`` like the real model.

``FAKE_MINIMAX_MODE`` picks the behaviour: ok, channels_first (audio as ``(2, frames)``), hot
(full-scale peaks, as the model clips to [-1, 1]), early (ends at a third of the requested
length), load_fail, gen_error, hang (generation never ends), crash (the process dies
mid-track), noisy (prints to stdout like the real engine), too_long (refuses the prompt as over
5000 tokens), no_frames (ends before the first frame); the last two raise ``ValueError`` like
mlx-audio does for input it cannot use.
``FAKE_MINIMAX_DELAY`` (seconds) slows each generation down. ``FAKE_MINIMAX_RECORD`` (a file)
gets one JSON line per load and per generation: what the engine was asked.
"""

import json
import os
import time
from dataclasses import dataclass

import numpy as np

MODE = os.environ.get("FAKE_MINIMAX_MODE", "ok")
SAMPLE_RATE = 44100


def _record(**entry):
    path = os.environ.get("FAKE_MINIMAX_RECORD")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


@dataclass
class GenerationResult:
    audio: object
    sample_rate: int


class Model:
    def __init__(self, path):
        self.path = path
        self.sample_rate = SAMPLE_RATE

    def generate(self, text, *, lyrics, duration=None, steps=30, seed=0, **_):
        if not lyrics.strip():
            raise ValueError("Lyrics are required; use [instrumental] explicitly")
        if duration is None or duration <= 0 or duration > 360:
            raise ValueError("duration must be between 0 and 360 seconds")
        if steps < 1 or steps > 30:
            raise ValueError("steps must be between 1 and 30")
        _record(text=text, lyrics=lyrics, duration=duration, steps=steps, seed=seed)
        if MODE == "noisy":
            print("generating frames 52%", flush=True)
        if MODE == "hang":
            time.sleep(3600)
        if MODE == "crash":
            os._exit(3)
        if MODE == "gen_error":
            raise RuntimeError("[METAL] Command buffer execution failed: out of memory")
        if MODE == "too_long":
            raise ValueError("The assembled prompt has 5123 tokens; the maximum is 5000")
        if MODE == "no_frames":
            raise ValueError("MiniMax Music 3 generated zero audio frames")
        time.sleep(float(os.environ.get("FAKE_MINIMAX_DELAY", "0")))
        seconds = duration / 3 if MODE == "early" else duration
        frames = int(seconds * 441)
        level = 1.0 if MODE == "hot" else 0.5
        wave = level * np.sin(2 * np.pi * 441 * np.arange(frames) / SAMPLE_RATE)
        audio = np.stack([wave, -wave], axis=1).astype(np.float32)
        if MODE == "channels_first":
            audio = audio.T
        yield GenerationResult(audio=audio, sample_rate=SAMPLE_RATE)


def load(model_path, lazy=False, strict=True, **_):
    _record(
        load=model_path,
        offline=os.environ.get("HF_HUB_OFFLINE"),
        hf_home=os.environ.get("HF_HOME"),
        hub_cache=os.environ.get("HF_HUB_CACHE"),
        token_leaked=any(key.startswith("ACE_STEP_API_") for key in os.environ),
    )
    if MODE == "noisy":
        print("Fetching 11 files: 100%|##########|", flush=True)
    if MODE == "load_fail":
        raise FileNotFoundError("No safetensors found in the model directory")
    return Model(model_path)
