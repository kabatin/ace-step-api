"""Shared fixtures and helpers. Tests marked ``real`` run only with ACE_STEP_API_REAL_TESTS=1."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REAL_TESTS = os.environ.get("ACE_STEP_API_REAL_TESTS") == "1"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if REAL_TESTS:
        return
    skip = pytest.mark.skip(reason="real engine run: set ACE_STEP_API_REAL_TESTS=1")
    for item in items:
        if "real" in item.keywords:
            item.add_marker(skip)


def make_wav(
    path: Path, seconds: float = 0.25, rate: int = 48000, channels: int = 2, width: int = 2
) -> bytes:
    """A short silent PCM WAV."""
    import wave

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(width)
        wav.setframerate(rate)
        wav.writeframes(b"\x00" * int(seconds * rate) * channels * width)
    return path.read_bytes()


os.environ.setdefault("PYTHONUNBUFFERED", "1")
