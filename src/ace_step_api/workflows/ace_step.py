"""Workflow ``ace-step-1.5``: style tags (and optionally lyrics) to music, delivered as WAV.

The client sends style tags (``prompt``), structured lyrics or none, a length in seconds and
1-4 tracks. Each track is generated with ``seed + i`` and checked again (PCM WAV, stereo,
44.1/48 kHz, non-empty) before the job succeeds; the result reports each track's real length.
The body and its rules are shared with the other music workflows (:mod:`.song`).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ace_step_api.audio import ALLOWED_RATES, CHANNELS
from ace_step_api.backends.base import GenerationRequest, GenerationResult
from ace_step_api.workflows.song import (
    DEFAULT_DURATION_S,
    MAX_COUNT,
    MAX_DURATION_S,
    MAX_LYRICS_CHARS,
    MAX_PROMPT_CHARS,
    MAX_SEED,
    MIN_DURATION_S,
    OUTPUT_MEDIA_TYPE,
    SongParams,
    SongWorkflow,
)

WORKFLOW_ID = "ace-step-1.5"
ENGINE = "acestep"
# what ACE-Step itself reads: the DiT text encoder keeps 256 tokens of the prompt and 2048 of
# the lyrics (longer text is accepted here but silently cut by the engine)
PROMPT_TOKENS_READ = 256
LYRICS_TOKENS_READ = 2048

_KANA = re.compile(r"[぀-ヿㇰ-ㇿｦ-ﾟ]")
_HANGUL = re.compile(r"[ᄀ-ᇿ㄰-㆏가-힯]")

__all__ = [
    "DEFAULT_DURATION_S",
    "ENGINE",
    "MAX_COUNT",
    "MAX_DURATION_S",
    "MAX_LYRICS_CHARS",
    "MAX_PROMPT_CHARS",
    "MAX_SEED",
    "MIN_DURATION_S",
    "OUTPUT_MEDIA_TYPE",
    "WORKFLOW_ID",
    "AceStepParams",
    "AceStepWorkflow",
    "vocal_language",
]


class AceStepParams(SongParams):
    """POST body for ``ace-step-1.5``."""


def vocal_language(lyrics: str) -> str:
    """The sung language when the lyrics make it obvious, else "unknown" (the engine's
    language model then decides)."""
    if _KANA.search(lyrics):
        return "ja"
    if _HANGUL.search(lyrics):
        return "ko"
    return "unknown"


class AceStepWorkflow(SongWorkflow):
    id = WORKFLOW_ID
    description = (
        "ACE-Step 1.5: style tags and optional lyrics to music, delivered as 16-bit stereo WAV"
    )
    engine = ENGINE
    params_model = AceStepParams

    def __init__(self, *, typical_seconds_60s: float = 45.0) -> None:
        super().__init__(typical_seconds_60s=typical_seconds_60s)

    def build_request(
        self, job_id: str, params: Mapping[str, Any], job_dir: Path
    ) -> GenerationRequest:
        instrumental = bool(params["instrumental"])
        lyrics = "" if instrumental else params["lyrics"]
        return GenerationRequest(
            job_id=job_id,
            engine=ENGINE,
            prompt=params["prompt"],
            lyrics=lyrics,
            instrumental=instrumental,
            duration_s=float(params["duration_sec"]),
            seed=params["seed"],
            count=params["count"],
            vocal_language="unknown" if instrumental else vocal_language(lyrics),
        )

    def generation_details(
        self, params: Mapping[str, Any], raw: GenerationResult
    ) -> dict[str, Any]:
        instrumental = bool(params["instrumental"])
        return {
            "requested_duration_sec": params["duration_sec"],
            "instrumental": instrumental,
            "vocal_language": "unknown" if instrumental else vocal_language(params["lyrics"]),
            "engine": raw.details,
        }

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
