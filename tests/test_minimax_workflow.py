"""Workflow ``minimax-music3``: the ace-step-1.5 body, the MiniMax engine, 44.1 kHz stereo."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ace_step_api.backends.base import GenerationPlan, GenerationResult
from ace_step_api.workflows.ace_step import AceStepParams, AceStepWorkflow
from ace_step_api.workflows.base import WorkflowFailedError
from ace_step_api.workflows.minimax import MiniMaxMusic3Workflow, MiniMaxParams
from tests.conftest import make_wav

LYRICS = "[verse]\n夜明けの街を 駆け抜けてく\n[chorus]\n光の中へ"


def store(params: dict, job_dir: Path) -> dict:
    workflow = MiniMaxMusic3Workflow()
    return workflow.store_inputs(MiniMaxParams.model_validate(params), job_dir)


def raw(outputs: tuple[Path, ...], count: int) -> GenerationResult:
    return GenerationResult(
        plan=GenerationPlan(count, 600.0),
        duration_s=1.0,
        outputs=outputs,
        details={"load_seconds": 3.1, "tracks": [{"index": 0, "seconds": 512.0, "gain_db": 0.0}]},
    )


def test_the_body_is_the_ace_step_body() -> None:
    ours, theirs = MiniMaxParams.model_json_schema(), AceStepParams.model_json_schema()
    assert ours["properties"] == theirs["properties"]
    assert MiniMaxParams.model_json_schema()["title"] == "MiniMaxParams"
    with pytest.raises(ValidationError):
        MiniMaxParams.model_validate({"prompt": "x", "steps": 30})
    with pytest.raises(ValidationError, match="seed \\+ count"):
        MiniMaxParams.model_validate({"prompt": "x", "seed": 2**32 - 2, "count": 4})


@pytest.mark.parametrize(
    ("params", "instrumental", "lyrics"),
    [
        # empty lyrics mean instrumental, as for ace-step-1.5 (the engine itself needs lyrics)
        ({"prompt": "x"}, True, ""),
        ({"prompt": "x", "lyrics": "", "instrumental": False}, True, ""),
        ({"prompt": "x", "lyrics": "  \n "}, True, ""),
        ({"prompt": "x", "lyrics": LYRICS, "instrumental": True}, True, ""),
        ({"prompt": "x", "lyrics": LYRICS}, False, LYRICS),
        ({"prompt": "x", "lyrics": "a\r\nb"}, False, "a\nb"),
    ],
)
def test_instrumental_rules(tmp_path: Path, params: dict, instrumental: bool, lyrics: str) -> None:
    stored = store(params | {"seed": 3}, tmp_path)
    assert (stored["instrumental"], stored["lyrics"]) == (instrumental, lyrics)
    request = MiniMaxMusic3Workflow().build_request("job_1", stored, tmp_path)
    assert (request.engine, request.instrumental, request.lyrics) == (
        "minimax",
        instrumental,
        lyrics,
    )


def test_build_request_maps_every_field(tmp_path: Path) -> None:
    stored = store(
        {"prompt": " warm pop ", "lyrics": LYRICS, "duration_sec": 90, "seed": 3, "count": 2},
        tmp_path,
    )
    request = MiniMaxMusic3Workflow().build_request("job_9", stored, tmp_path)
    assert (request.job_id, request.prompt, request.duration_s) == ("job_9", "warm pop", 90.0)
    assert (request.seed, request.count, request.lyrics) == (3, 2, LYRICS)


def test_random_seeds_leave_room_for_every_track(tmp_path: Path) -> None:
    for _ in range(20):
        stored = store({"prompt": "x", "count": 4}, tmp_path)
        assert stored["seed"] >= 0 and stored["seed"] + 3 <= 2**32 - 1


def test_finalize_reports_real_lengths(tmp_path: Path) -> None:
    stored = store({"prompt": "x", "lyrics": LYRICS, "duration_sec": 60, "seed": 5}, tmp_path)
    produced = tmp_path / "raw-0.wav"
    make_wav(produced, 22.9 / 100, rate=44100)  # the model may end early: delivered as is
    out = MiniMaxMusic3Workflow().finalize("job_1", stored, tmp_path, raw((produced,), 1))
    assert [f.name for f in out.files] == ["output-0.wav"] and not produced.exists()
    assert out.result == {
        "outputs": [
            {
                "index": 0,
                "media_type": "audio/wav",
                "duration_sec": 0.229,
                "sample_rate": 44100,
                "channels": 2,
                "seed_used": 5,
            }
        ],
        "seed_used": 5,
        "details": {
            "generation": {
                "requested_duration_sec": 60.0,
                "instrumental": False,
                "engine": {
                    "load_seconds": 3.1,
                    "tracks": [{"index": 0, "seconds": 512.0, "gain_db": 0.0}],
                },
            }
        },
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rate": 48000}, "48000 Hz"),
        ({"channels": 1}, "1 channel"),
        ({"width": 1}, "8-bit"),
        ({"seconds": 0.0}, "empty"),
    ],
)
def test_finalize_accepts_only_44k_stereo_pcm(tmp_path: Path, kwargs: dict, message: str) -> None:
    stored = store({"prompt": "x", "seed": 1}, tmp_path)
    produced = tmp_path / "raw-0.wav"
    make_wav(produced, **{"seconds": 0.5, "rate": 44100, **kwargs})
    with pytest.raises(WorkflowFailedError, match=message) as info:
        MiniMaxMusic3Workflow().finalize("job_1", stored, tmp_path, raw((produced,), 1))
    assert info.value.code == "invalid_output" and info.value.retryable
    assert not (tmp_path / "output-0.wav").exists()
    # 48 kHz is still fine for ace-step-1.5
    if kwargs == {"rate": 48000}:
        stored = AceStepWorkflow().store_inputs(AceStepParams(prompt="x", seed=1), tmp_path)
        assert AceStepWorkflow().finalize("job_1", stored, tmp_path, raw((produced,), 1)).files


def test_capabilities() -> None:
    caps = MiniMaxMusic3Workflow(typical_seconds_60s=580.4).capabilities()
    assert caps == {
        "id": "minimax-music3",
        "output": {
            "media_type": "audio/wav",
            "sample_rates": [44100],
            "channels": 2,
            "sample_format": "pcm_s16le",
        },
        "prompt": {"max_chars": 1000},
        "lyrics": {"max_chars": 4000, "empty_means_instrumental": True},
        "prompt_and_lyrics_max_tokens": 5000,
        "duration_sec": {"min": 10.0, "max": 240.0, "default": 60.0, "upper_bound": True},
        "count": {"min": 1, "max": 4},
        "seed": {"min": 0, "max": 2**32 - 1, "per_track": "seed + i"},
        "instrumental": True,
        "typical_seconds": {"60s_track": 580},
        "attribution": "MiniMax-Music3",
    }
    assert MiniMaxMusic3Workflow().capabilities()["typical_seconds"] == {"60s_track": 580}


def test_smoke_is_a_short_instrumental() -> None:
    workflow = MiniMaxMusic3Workflow()
    smoke = MiniMaxParams.model_validate(workflow.smoke_params())
    assert smoke.is_instrumental and smoke.duration_sec == 10.0 and smoke.count == 1
    assert workflow.engine == "minimax" and workflow.output_media_type == "audio/wav"
