"""Workflow ``ace-step-1.5``: style tags (and optionally lyrics) to music, delivered as WAV.

The client sends style tags (``prompt``), structured lyrics or none, a length in seconds and
1-4 tracks. Each track is generated with ``seed + i`` and checked again (PCM WAV, stereo,
44.1/48 kHz, non-empty) before the job succeeds; the result reports each track's real length.
"""

from __future__ import annotations

import re
import secrets
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ace_step_api.audio import ALLOWED_RATES, CHANNELS, AudioError, inspect_wav, wav_problem
from ace_step_api.backends.base import GenerationRequest, GenerationResult
from ace_step_api.workflows.base import Workflow, WorkflowFailedError, WorkflowOutput

WORKFLOW_ID = "ace-step-1.5"
OUTPUT_MEDIA_TYPE = "audio/wav"
MAX_PROMPT_CHARS = 1000
MAX_LYRICS_CHARS = 4000
MIN_DURATION_S = 10.0
MAX_DURATION_S = 240.0
DEFAULT_DURATION_S = 60.0
MAX_COUNT = 4
# ACE-Step draws its own random seeds from 0..2**32 - 1; track i uses seed + i
MAX_SEED = 2**32 - 1
# what ACE-Step itself reads: the DiT text encoder keeps 256 tokens of the prompt and 2048 of
# the lyrics (longer text is accepted here but silently cut by the engine)
PROMPT_TOKENS_READ = 256
LYRICS_TOKENS_READ = 2048

# C0 control characters and DEL; tab and newline are ordinary text.
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_KANA = re.compile(r"[぀-ヿㇰ-ㇿｦ-ﾟ]")
_HANGUL = re.compile(r"[ᄀ-ᇿ㄰-㆏가-힯]")


class AceStepParams(BaseModel):
    """POST body for ``ace-step-1.5``."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(
        min_length=1,
        max_length=MAX_PROMPT_CHARS,
        description="style tags, e.g. 'J-pop, upbeat, female vocal, synth, 128 BPM'",
    )
    lyrics: str = Field(
        default="",
        max_length=MAX_LYRICS_CHARS,
        description="structured lyrics ([verse], [chorus], ...); empty means instrumental",
    )
    duration_sec: float = Field(default=DEFAULT_DURATION_S, ge=MIN_DURATION_S, le=MAX_DURATION_S)
    instrumental: bool = Field(default=False, description="true ignores lyrics")
    seed: int | None = Field(
        default=None, ge=0, le=MAX_SEED, description="track i uses seed + i; random if omitted"
    )
    count: int = Field(default=1, ge=1, le=MAX_COUNT, description="tracks, one after another")

    @field_validator("prompt")
    @classmethod
    def _prompt(cls, value: str) -> str:
        if _CONTROL.search(value):
            raise ValueError("prompt contains control characters")
        stripped = value.strip()
        if not stripped:
            raise ValueError("prompt is blank")
        return stripped

    @field_validator("lyrics")
    @classmethod
    def _lyrics(cls, value: str) -> str:
        value = value.replace("\r\n", "\n")
        if _CONTROL.search(value):
            raise ValueError("lyrics contain control characters")
        return value.strip()

    @model_validator(mode="after")
    def _seed_range(self) -> AceStepParams:
        if self.seed is not None and self.seed + self.count - 1 > MAX_SEED:
            raise ValueError(f"seed + count - 1 must not exceed {MAX_SEED}")
        return self

    @property
    def is_instrumental(self) -> bool:
        return self.instrumental or not self.lyrics


def vocal_language(lyrics: str) -> str:
    """The sung language when the lyrics make it obvious, else "unknown" (the engine's
    language model then decides)."""
    if _KANA.search(lyrics):
        return "ja"
    if _HANGUL.search(lyrics):
        return "ko"
    return "unknown"


class AceStepWorkflow(Workflow):
    id = WORKFLOW_ID
    description = (
        "ACE-Step 1.5: style tags and optional lyrics to music, delivered as 16-bit stereo WAV"
    )
    output_media_type = OUTPUT_MEDIA_TYPE
    params_model = AceStepParams

    def __init__(self, *, typical_seconds_60s: float = 45.0) -> None:
        self.typical_seconds_60s = typical_seconds_60s

    def store_inputs(self, params: BaseModel, job_dir: Path) -> dict[str, Any]:
        p = cast(AceStepParams, params)
        instrumental = p.is_instrumental
        stored = p.model_dump()
        stored["instrumental"] = instrumental
        stored["lyrics"] = "" if instrumental else p.lyrics
        if p.seed is None:
            stored["seed"] = secrets.randbelow(MAX_SEED - MAX_COUNT + 2)
        return stored

    def build_request(
        self, job_id: str, params: Mapping[str, Any], job_dir: Path
    ) -> GenerationRequest:
        instrumental = bool(params["instrumental"])
        lyrics = "" if instrumental else params["lyrics"]
        return GenerationRequest(
            job_id=job_id,
            prompt=params["prompt"],
            lyrics=lyrics,
            instrumental=instrumental,
            duration_s=float(params["duration_sec"]),
            seed=params["seed"],
            count=params["count"],
            vocal_language="unknown" if instrumental else vocal_language(lyrics),
        )

    def finalize(
        self, job_id: str, params: Mapping[str, Any], job_dir: Path, raw: GenerationResult
    ) -> WorkflowOutput:
        count = params["count"]
        if len(raw.outputs) != count:
            raise WorkflowFailedError(
                "invalid_output",
                f"the engine delivered {len(raw.outputs)} of {count} tracks",
                retryable=True,
            )
        tracks: list[dict[str, Any]] = []
        finals: list[Path] = []
        for index, produced in enumerate(raw.outputs):
            try:
                info = inspect_wav(produced)
            except AudioError as exc:
                raise WorkflowFailedError("invalid_output", str(exc), retryable=True) from exc
            problem = wav_problem(info)
            if problem is not None:
                raise WorkflowFailedError("invalid_output", problem, retryable=True)
            tracks.append(
                {
                    "index": index,
                    "media_type": self.output_media_type,
                    "duration_sec": round(info.seconds, 3),
                    "sample_rate": info.sample_rate,
                    "channels": info.channels,
                    "seed_used": params["seed"] + index,
                }
            )
            finals.append(job_dir / f"output-{index}.wav")
        for produced, final in zip(raw.outputs, finals, strict=True):
            produced.replace(final)
        instrumental = bool(params["instrumental"])
        return WorkflowOutput(
            files=tuple(finals),
            result={
                "outputs": tracks,
                "seed_used": params["seed"],
                "details": {
                    "generation": {
                        "requested_duration_sec": params["duration_sec"],
                        "instrumental": instrumental,
                        "vocal_language": "unknown"
                        if instrumental
                        else vocal_language(params["lyrics"]),
                        "engine": raw.details,
                    }
                },
            },
        )

    def capabilities(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "output": {
                "media_type": self.output_media_type,
                "sample_rates": list(ALLOWED_RATES),
                "channels": CHANNELS,
                "sample_format": "pcm_s16le",
            },
            "prompt": {"max_chars": MAX_PROMPT_CHARS, "engine_reads_tokens": PROMPT_TOKENS_READ},
            "lyrics": {
                "max_chars": MAX_LYRICS_CHARS,
                "engine_reads_tokens": LYRICS_TOKENS_READ,
                "empty_means_instrumental": True,
            },
            "duration_sec": {
                "min": MIN_DURATION_S,
                "max": MAX_DURATION_S,
                "default": DEFAULT_DURATION_S,
            },
            "count": {"min": 1, "max": MAX_COUNT},
            "seed": {"min": 0, "max": MAX_SEED, "per_track": "seed + i"},
            "instrumental": True,
            "typical_seconds": {"60s_track": round(self.typical_seconds_60s)},
        }

    def smoke_params(self) -> dict[str, Any]:
        return {
            "prompt": "lo-fi piano, calm, short",
            "lyrics": "",
            "duration_sec": MIN_DURATION_S,
            "instrumental": True,
            "seed": 1,
            "count": 1,
        }
