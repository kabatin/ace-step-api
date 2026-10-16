"""A stand-in for ``soundfile``: 16-bit PCM WAV in and out, as numpy float32 frames.

The fake engine's record of a request (``<audio>.params.json``) follows the audio from the
file that was read to the file that is written, so tests can see what the engine was asked.
"""

import shutil
import wave

import numpy as np

_last_read: list[str] = []


def read(path, always_2d=False, dtype="float64"):
    with wave.open(path, "rb") as wav:
        channels, rate = wav.getnchannels(), wav.getframerate()
        raw = wav.readframes(wav.getnframes())
    data = np.frombuffer(raw, dtype="<i2").reshape(-1, channels).astype(dtype) / 32768.0
    _last_read[:] = [path]
    return data, rate


def write(path, data, rate, subtype=None):
    assert subtype == "PCM_16"
    pcm = np.clip(np.round(np.asarray(data) * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(path, "wb") as wav:
        wav.setnchannels(pcm.shape[1])
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm.tobytes())
    if _last_read:
        shutil.copyfile(_last_read[0] + ".params.json", path + ".params.json")
