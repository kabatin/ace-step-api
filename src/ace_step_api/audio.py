"""WAV checks with the standard library only: what a delivered track must be before a job
succeeds (PCM, stereo, 44.1 or 48 kHz, non-empty), and its real length."""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

ALLOWED_RATES = (44100, 48000)
CHANNELS = 2
SAMPLE_WIDTH = 2  # 16-bit PCM


class AudioError(ValueError):
    """The file is not a usable WAV (message is safe for clients)."""


@dataclass(frozen=True)
class WavInfo:
    channels: int
    sample_rate: int
    sample_width: int
    frames: int

    @property
    def seconds(self) -> float:
        return self.frames / self.sample_rate if self.sample_rate else 0.0


def inspect_wav(path: Path) -> WavInfo:
    try:
        with wave.open(str(path), "rb") as wav:
            info = WavInfo(
                channels=wav.getnchannels(),
                sample_rate=wav.getframerate(),
                sample_width=wav.getsampwidth(),
                frames=wav.getnframes(),
            )
    except (OSError, EOFError, wave.Error) as exc:
        raise AudioError(f"not a readable PCM WAV file ({type(exc).__name__})") from exc
    return info


def wav_problem(info: WavInfo) -> str | None:
    """Why a track cannot be delivered, or None when it is fine."""
    if info.channels != CHANNELS:
        return f"output has {info.channels} channel(s), expected stereo"
    if info.sample_rate not in ALLOWED_RATES:
        return f"output sample rate is {info.sample_rate} Hz"
    if info.sample_width != SAMPLE_WIDTH:
        return f"output is {info.sample_width * 8}-bit, expected 16-bit PCM"
    if info.frames <= 0:
        return "output is empty"
    return None
