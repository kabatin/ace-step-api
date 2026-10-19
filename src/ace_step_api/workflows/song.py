"""What the music workflows share: the POST body (style prompt, lyrics, length, instrumental,
seed, count), how it is stored, and how raw tracks become checked deliverables.

Each track is generated with ``seed + i`` and checked again (PCM WAV, stereo, one of the
workflow's sample rates, non-empty) before the job succeeds; the result reports each track's
real length. Empty lyrics mean instrumental.
"""

from __future__ import annotations

import re
import secrets
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Self, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ace_step_api.audio import ALLOWED_RATES, AudioError, inspect_wav, wav_problem
from ace_step_api.backends.base import GenerationResult
from ace_step_api.workflows.base import Workflow, WorkflowFailedError, WorkflowOutput

OUTPUT_MEDIA_TYPE = "audio/wav"
MAX_PROMPT_CHARS = 1000
MAX_LYRICS_CHARS = 4000
MIN_DURATION_S = 10.0
MAX_DURATION_S = 240.0
DEFAULT_DURATION_S = 60.0
MAX_COUNT = 4
# ACE-Step draws its own random seeds from 0..2**32 - 1; track i uses seed + i
MAX_SEED = 2**32 - 1

# C0 control characters and DEL; tab and newline are ordinary text.
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


class SongParams(BaseModel):
    """The POST body every music workflow takes."""

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
    def _seed_range(self) -> Self:
        if self.seed is not None and self.seed + self.count - 1 > MAX_SEED:
            raise ValueError(f"seed + count - 1 must not exceed {MAX_SEED}")
        return self

    @property
    def is_instrumental(self) -> bool:
        return self.instrumental or not self.lyrics


class SongWorkflow(Workflow):
    output_media_type: ClassVar[str] = OUTPUT_MEDIA_TYPE
    # what a delivered track may be
    sample_rates: ClassVar[tuple[int, ...]] = ALLOWED_RATES

    def __init__(self, *, typical_seconds_60s: float) -> None:
        self.typical_seconds_60s = typical_seconds_60s

    def generation_details(
        self, params: Mapping[str, Any], raw: GenerationResult
    ) -> dict[str, Any]:
        """``result.details.generation``: what was asked and what the engine reported."""
        return {
            "requested_duration_sec": params["duration_sec"],
            "instrumental": bool(params["instrumental"]),
            "engine": raw.details,
        }

    def store_inputs(self, params: BaseModel, job_dir: Path) -> dict[str, Any]:
        p = cast(SongParams, params)
        instrumental = p.is_instrumental
        stored = p.model_dump()
        stored["instrumental"] = instrumental
        stored["lyrics"] = "" if instrumental else p.lyrics
        if p.seed is None:
            stored["seed"] = secrets.randbelow(MAX_SEED - MAX_COUNT + 2)
        return stored

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
            if problem is None and info.sample_rate not in self.sample_rates:
                problem = f"output sample rate is {info.sample_rate} Hz"
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
        return WorkflowOutput(
            files=tuple(finals),
            result={
                "outputs": tracks,
                "seed_used": params["seed"],
                "details": {"generation": self.generation_details(params, raw)},
            },
        )

    def smoke_params(self) -> dict[str, Any]:
        return {
            "prompt": "lo-fi piano, calm, short",
            "lyrics": "",
            "duration_sec": MIN_DURATION_S,
            "instrumental": True,
            "seed": 1,
            "count": 1,
        }
