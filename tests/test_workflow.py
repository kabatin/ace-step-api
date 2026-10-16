from pathlib import Path

import pytest
from pydantic import ValidationError

from ace_step_api.backends.base import GenerationPlan, GenerationResult
from ace_step_api.workflows.ace_step import (
    MAX_SEED,
    AceStepParams,
    AceStepWorkflow,
    vocal_language,
)
from ace_step_api.workflows.base import WorkflowFailedError
from tests.conftest import make_wav

LYRICS = "[verse]\n夜明けの街を 駆け抜けてく\n[chorus]\n光の中へ"


def store(params: dict, job_dir: Path) -> dict:
    workflow = AceStepWorkflow()
    return workflow.store_inputs(AceStepParams.model_validate(params), job_dir)


def test_defaults_and_stored_params(tmp_path: Path) -> None:
    stored = store({"prompt": "  J-pop, upbeat  ", "lyrics": LYRICS, "seed": 7}, tmp_path)
    assert stored == {
        "prompt": "J-pop, upbeat",
        "lyrics": LYRICS,
        "duration_sec": 60.0,
        "instrumental": False,
        "seed": 7,
        "count": 1,
    }


@pytest.mark.parametrize(
    ("params", "instrumental", "lyrics"),
    [
        ({"prompt": "x"}, True, ""),
        ({"prompt": "x", "lyrics": "   \n "}, True, ""),
        ({"prompt": "x", "lyrics": LYRICS, "instrumental": True}, True, ""),
        ({"prompt": "x", "lyrics": LYRICS}, False, LYRICS),
        ({"prompt": "x", "lyrics": "a\r\nb"}, False, "a\nb"),
    ],
)
def test_instrumental_rules(tmp_path: Path, params: dict, instrumental: bool, lyrics: str) -> None:
    stored = store(params, tmp_path)
    assert (stored["instrumental"], stored["lyrics"]) == (instrumental, lyrics)
    request = AceStepWorkflow().build_request("job_1", stored, tmp_path)
    assert request.instrumental is instrumental and request.lyrics == lyrics
    assert request.vocal_language == ("unknown" if instrumental else vocal_language(lyrics))


def test_random_seed_leaves_room_for_every_track(tmp_path: Path) -> None:
    for _ in range(50):
        stored = store({"prompt": "x", "count": 4}, tmp_path)
        assert stored["seed"] >= 0 and stored["seed"] + 3 <= MAX_SEED


def test_seed_range_includes_the_last_track() -> None:
    AceStepParams.model_validate({"prompt": "x", "seed": MAX_SEED})
    AceStepParams.model_validate({"prompt": "x", "seed": MAX_SEED - 3, "count": 4})
    with pytest.raises(ValidationError, match="seed \\+ count"):
        AceStepParams.model_validate({"prompt": "x", "seed": MAX_SEED - 2, "count": 4})


@pytest.mark.parametrize(
    ("lyrics", "language"),
    [
        (LYRICS, "ja"),
        ("[verse]\nｶﾀｶﾅ", "ja"),
        ("[verse]\n사랑해", "ko"),
        ("[verse]\nhello world", "unknown"),
        ("[verse]\n你好", "unknown"),
    ],
)
def test_vocal_language(lyrics: str, language: str) -> None:
    assert vocal_language(lyrics) == language


def test_build_request_maps_every_field(tmp_path: Path) -> None:
    stored = store(
        {"prompt": "rock", "lyrics": LYRICS, "duration_sec": 90, "seed": 3, "count": 2}, tmp_path
    )
    request = AceStepWorkflow().build_request("job_9", stored, tmp_path)
    assert (request.job_id, request.prompt, request.duration_s) == ("job_9", "rock", 90.0)
    assert (request.seed, request.count, request.vocal_language) == (3, 2, "ja")


def _raw(outputs: tuple[Path, ...], count: int) -> GenerationResult:
    return GenerationResult(
        plan=GenerationPlan(count, 10.0),
        duration_s=1.0,
        outputs=outputs,
        details={"load_seconds": 20.0, "tracks": [{"index": 0, "seconds": 9.5}]},
    )


def test_finalize_reports_real_lengths(tmp_path: Path) -> None:
    stored = store({"prompt": "x", "lyrics": LYRICS, "seed": 5, "count": 2}, tmp_path)
    raws = (tmp_path / "raw-0.wav", tmp_path / "raw-1.wav")
    make_wav(raws[0], 1.5)
    make_wav(raws[1], 2.0, rate=44100)
    out = AceStepWorkflow().finalize("job_1", stored, tmp_path, _raw(raws, 2))
    assert [f.name for f in out.files] == ["output-0.wav", "output-1.wav"]
    assert all(f.is_file() for f in out.files) and not any(r.exists() for r in raws)
    assert out.result["outputs"] == [
        {
            "index": 0,
            "media_type": "audio/wav",
            "duration_sec": 1.5,
            "sample_rate": 48000,
            "channels": 2,
            "seed_used": 5,
        },
        {
            "index": 1,
            "media_type": "audio/wav",
            "duration_sec": 2.0,
            "sample_rate": 44100,
            "channels": 2,
            "seed_used": 6,
        },
    ]
    generation = out.result["details"]["generation"]
    assert generation["vocal_language"] == "ja" and generation["engine"]["load_seconds"] == 20.0
    assert out.result["seed_used"] == 5


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"channels": 1}, "1 channel"),
        ({"rate": 22050}, "22050 Hz"),
        ({"width": 1}, "8-bit"),
        ({"seconds": 0.0}, "empty"),
    ],
)
def test_finalize_refuses_bad_audio(tmp_path: Path, kwargs: dict, message: str) -> None:
    stored = store({"prompt": "x", "seed": 1}, tmp_path)
    raw = tmp_path / "raw-0.wav"
    make_wav(raw, **{"seconds": 0.5, **kwargs})
    with pytest.raises(WorkflowFailedError, match=message) as info:
        AceStepWorkflow().finalize("job_1", stored, tmp_path, _raw((raw,), 1))
    assert info.value.retryable and not (tmp_path / "output-0.wav").exists()


def test_finalize_refuses_unreadable_and_missing_tracks(tmp_path: Path) -> None:
    stored = store({"prompt": "x", "seed": 1, "count": 2}, tmp_path)
    bad = tmp_path / "raw-0.wav"
    bad.write_text("not audio")
    with pytest.raises(WorkflowFailedError, match="1 of 2"):
        AceStepWorkflow().finalize("job_1", stored, tmp_path, _raw((bad,), 2))
    stored = store({"prompt": "x", "seed": 1}, tmp_path)
    with pytest.raises(WorkflowFailedError, match="readable PCM WAV"):
        AceStepWorkflow().finalize("job_1", stored, tmp_path, _raw((bad,), 1))


def test_capabilities_and_smoke_params() -> None:
    workflow = AceStepWorkflow(typical_seconds_60s=44.6)
    caps = workflow.capabilities()
    assert caps["typical_seconds"] == {"60s_track": 45}
    smoke = AceStepParams.model_validate(workflow.smoke_params())
    assert smoke.is_instrumental and smoke.duration_sec == 10.0
