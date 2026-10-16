import json
import os
import time
import uuid
import wave
from dataclasses import dataclass, field

from acestep import MODE


@dataclass
class GenerationParams:
    caption: str = ""
    lyrics: str = ""
    instrumental: bool = False
    vocal_language: str = "unknown"
    duration: float = -1.0
    inference_steps: int = 8
    shift: float = 1.0
    seed: int = -1


@dataclass
class GenerationConfig:
    batch_size: int = 2
    use_random_seed: bool = True
    seeds: list = field(default_factory=list)
    audio_format: str = "flac"


@dataclass
class GenerationResult:
    audios: list = field(default_factory=list)
    status_message: str = ""
    success: bool = True
    error: str = ""


def generate_music(dit, llm, params, config, save_dir=None, progress=None):
    if progress is not None:
        progress(0.1, "Phase 1")
        progress(0.52, desc="Generating music")
    if MODE == "noisy":
        print("Generating... 52%", flush=True)
    if MODE == "hang":
        time.sleep(3600)
    if MODE == "crash":
        os._exit(3)
    if MODE == "gen_error":
        return GenerationResult(success=False, error="MPS backend out of memory")
    # a record of what the engine was asked, next to the audio
    path = os.path.join(save_dir, f"{uuid.uuid4()}.wav")
    frames = int(params.duration * 480)
    # a full-scale square wave: the worker's limiter has to bring it under the ceiling
    square = (b"\xff\x7f\xff\x7f" * 50 + b"\x00\x80\x00\x80" * 50) * (frames // 100 + 1)
    with wave.open(path, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(48000)
        wav.writeframes(square[: frames * 4])
    with open(path + ".params.json", "w") as f:
        json.dump(
            {
                "caption": params.caption,
                "lyrics": params.lyrics,
                "instrumental": params.instrumental,
                "vocal_language": params.vocal_language,
                "duration": params.duration,
                "seed": params.seed,
                "seeds": config.seeds,
                "steps": params.inference_steps,
                "shift": params.shift,
            },
            f,
        )
    if progress is not None:
        progress(0.99, desc="Preparing audio data...")
    return GenerationResult(audios=[{"path": path}])
