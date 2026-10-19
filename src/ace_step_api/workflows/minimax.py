"""Workflow ``minimax-music3``: the ``ace-step-1.5`` body (style prompt, optional lyrics, length,
instrumental, seed, count) run by MiniMax Music 3, delivered as 16-bit 44.1 kHz stereo WAV.

The rules are those of ``ace-step-1.5`` (:mod:`.song`), including "empty lyrics mean
instrumental": the engine cannot write lyrics itself and refuses empty ones, so a body without
lyrics gets an instrumental track (the engine's ``[instrumental]`` marker), and the result's
``instrumental`` says so. ``duration_sec`` is an upper bound: the model may end a song early,
and the result reports each track's real length.

The engine refuses a caption + lyrics longer than 5000 tokens (it does not cut them); such a
job fails as ``invalid_params``, not retryable, with the engine's message. MiniMax-Music3's
licence asks a commercial product or service using it to display "MiniMax-Music3" in its UI
(``attribution`` in capabilities).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from ace_step_api.audio import CHANNELS
from ace_step_api.backends.base import GenerationRequest
from ace_step_api.workflows.song import (
    DEFAULT_DURATION_S,
    MAX_COUNT,
    MAX_DURATION_S,
    MAX_LYRICS_CHARS,
    MAX_PROMPT_CHARS,
    MAX_SEED,
    MIN_DURATION_S,
    SongParams,
    SongWorkflow,
)

WORKFLOW_ID = "minimax-music3"
ENGINE = "minimax"
SAMPLE_RATE = 44100
# the engine raises (does not cut) when the assembled caption + lyrics exceed this
MAX_TEXT_TOKENS = 5000
ATTRIBUTION = "MiniMax-Music3"


class MiniMaxParams(SongParams):
    """POST body for ``minimax-music3`` (the fields and rules of ``ace-step-1.5``)."""


class MiniMaxMusic3Workflow(SongWorkflow):
    id = WORKFLOW_ID
    description = (
        "MiniMax Music 3: style description and optional lyrics to a song or an instrumental, "
        "delivered as 16-bit 44.1 kHz stereo WAV"
    )
    engine = ENGINE
    params_model = MiniMaxParams
    sample_rates: ClassVar[tuple[int, ...]] = (SAMPLE_RATE,)

    def __init__(self, *, typical_seconds_60s: float = 580.0) -> None:
        super().__init__(typical_seconds_60s=typical_seconds_60s)

    def build_request(
        self, job_id: str, params: Mapping[str, Any], job_dir: Path
    ) -> GenerationRequest:
        instrumental = bool(params["instrumental"])
        return GenerationRequest(
            job_id=job_id,
            engine=ENGINE,
            prompt=params["prompt"],
            lyrics="" if instrumental else params["lyrics"],
            instrumental=instrumental,
            duration_s=float(params["duration_sec"]),
            seed=params["seed"],
            count=params["count"],
        )

    def capabilities(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "output": {
                "media_type": self.output_media_type,
                "sample_rates": [SAMPLE_RATE],
                "channels": CHANNELS,
                "sample_format": "pcm_s16le",
            },
            "prompt": {"max_chars": MAX_PROMPT_CHARS},
            "lyrics": {"max_chars": MAX_LYRICS_CHARS, "empty_means_instrumental": True},
            "prompt_and_lyrics_max_tokens": MAX_TEXT_TOKENS,
            "duration_sec": {
                "min": MIN_DURATION_S,
                "max": MAX_DURATION_S,
                "default": DEFAULT_DURATION_S,
                # the real length can be shorter: see result.outputs[].duration_sec
                "upper_bound": True,
            },
            "count": {"min": 1, "max": MAX_COUNT},
            "seed": {"min": 0, "max": MAX_SEED, "per_track": "seed + i"},
            "instrumental": True,
            "typical_seconds": {"60s_track": round(self.typical_seconds_60s)},
            "attribution": ATTRIBUTION,
        }
