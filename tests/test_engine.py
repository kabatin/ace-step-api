"""The resident engine and the ACE-Step backend, driving the real worker script against a fake
``acestep`` package (tests/fake_acestep) in a real child process."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from ace_step_api.backends.acestep import AceStepBackend, AceStepOptions
from ace_step_api.backends.base import GenerationRequest
from ace_step_api.backends.engine import (
    WORKER_SCRIPT,
    EngineConfig,
    EngineError,
    ResidentEngine,
    worker_command,
)
from ace_step_api.backends.process import PID_FILE
from ace_step_api.checks import Level
from ace_step_api.settings import SettingsError, load_settings

FAKE = Path(__file__).with_name("fake_acestep")


@pytest.fixture(autouse=True)
def fake_acestep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHONPATH", str(FAKE))
    monkeypatch.setenv("FAKE_ACESTEP_MODE", "ok")


def config(tmp_path: Path, **overrides: object) -> EngineConfig:
    values: dict[str, object] = {
        "python": Path(sys.executable),
        "project_root": tmp_path,
        "dit_model": "acestep-v15-turbo",
        "lm_model": "acestep-5Hz-lm-1.7B",
        "lm_backend": "vllm",
        "state_dir": tmp_path / "state",
        "idle_seconds": 60.0,
        "poll_seconds": 0.05,
        "start_timeout_s": 30.0,
        "stop_grace_s": 2.0,
    }
    return EngineConfig(**(values | overrides))  # type: ignore[arg-type]


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[ResidentEngine]:
    eng = ResidentEngine(config(tmp_path))
    yield eng
    eng.shutdown()


def track(tmp_path: Path, name: str = "raw-0.wav", **overrides: object) -> dict[str, object]:
    return {
        "caption": "J-pop, upbeat",
        "lyrics": "[verse]\nla",
        "instrumental": False,
        "vocal_language": "ja",
        "duration": 10.0,
        "seed": 7,
        "out": str(tmp_path / name),
    } | overrides


def far() -> float:
    return time.monotonic() + 30


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return (
        subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True
        ).stdout.strip()[:1]
        != "Z"
    )


def wait(predicate: Callable[[], bool], timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        time.sleep(0.02)


def test_worker_command_is_fixed(tmp_path: Path) -> None:
    argv = worker_command(config(tmp_path), 9)
    assert argv[:5] == [sys.executable, "-P", str(WORKER_SCRIPT), "--proto-fd", "9"]
    assert argv[5:] == [
        "--project-root",
        str(tmp_path),
        "--dit-model",
        "acestep-v15-turbo",
        "--lm-model",
        "acestep-5Hz-lm-1.7B",
        "--lm-backend",
        "vllm",
    ]


def test_tracks_reuse_one_loaded_worker(engine: ResidentEngine, tmp_path: Path) -> None:
    cancel = threading.Event()
    assert engine.acquire("job_1", cancel, far())
    assert engine.last_load_s is not None and engine.running()
    pid = engine.pid
    progress: list[float] = []
    first = engine.run_track(
        "job_1", track(tmp_path), cancel=cancel, deadline=far(), on_progress=progress.append
    )
    assert first.ok and first.reply["sample_rate"] == 48000 and first.reply["channels"] == 2
    assert first.reply["frames"] == 4800
    assert progress == [0.1, 0.52, 0.99]
    asked = json.loads((tmp_path / "raw-0.wav.params.json").read_text())
    assert asked == {
        "caption": "J-pop, upbeat",
        "lyrics": "[verse]\nla",
        "instrumental": False,
        "vocal_language": "ja",
        "duration": 10.0,
        "seed": 7,
        "seeds": [7],
        "steps": 8,
        "shift": 3.0,
    }
    engine.release(job_id="job_1")
    assert engine.acquire("job_2", cancel, far())
    assert engine.pid == pid and engine.last_load_s is None  # no second load
    second = engine.run_track(
        "job_2", track(tmp_path, "b.wav", instrumental=True, seed=8), cancel=cancel, deadline=far()
    )
    assert second.ok
    asked = json.loads((tmp_path / "b.wav.params.json").read_text())
    assert asked["lyrics"] == "[Instrumental]" and asked["instrumental"] and asked["seed"] == 8
    engine.release(job_id="job_2")
    # scratch directories the worker made next to the outputs are gone
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("ace-step-")) == []


def test_stdout_noise_does_not_break_the_protocol(
    engine: ResidentEngine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_ACESTEP_MODE", "noisy")
    cancel = threading.Event()
    assert engine.acquire("job_1", cancel, far())
    assert engine.run_track("job_1", track(tmp_path), cancel=cancel, deadline=far()).ok
    log = (tmp_path / "state" / "worker.log").read_text()
    assert "progress bar 100%" in log and "Generating... 52%" in log


def test_a_failed_load_raises_with_the_reason(
    engine: ResidentEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_ACESTEP_MODE", "load_fail")
    with pytest.raises(EngineError, match="checkpoint missing"):
        engine.acquire("job_1", threading.Event(), far())
    assert not engine.running()


def test_an_engine_error_keeps_the_worker(
    engine: ResidentEngine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_ACESTEP_MODE", "gen_error")
    cancel = threading.Event()
    assert engine.acquire("job_1", cancel, far())
    outcome = engine.run_track("job_1", track(tmp_path), cancel=cancel, deadline=far())
    assert not outcome.ok and "out of memory" in (outcome.failure or "")
    assert engine.running()


def test_cancel_and_timeout_stop_the_worker(
    engine: ResidentEngine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_ACESTEP_MODE", "hang")
    cancel = threading.Event()
    assert engine.acquire("job_1", cancel, far())
    pid = engine.pid
    assert pid is not None
    threading.Timer(0.5, cancel.set).start()
    outcome = engine.run_track("job_1", track(tmp_path), cancel=cancel, deadline=far())
    assert outcome.canceled and not outcome.ok
    assert not engine.running() and not alive(pid)
    assert not (tmp_path / "state" / PID_FILE).exists()

    assert engine.acquire("job_2", threading.Event(), far())
    pid = engine.pid
    assert pid is not None
    late = engine.run_track(
        "job_2", track(tmp_path), cancel=threading.Event(), deadline=time.monotonic() + 0.5
    )
    assert late.timed_out and not alive(pid)


def test_a_worker_that_dies_mid_track_is_a_failure(
    engine: ResidentEngine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_ACESTEP_MODE", "crash")
    cancel = threading.Event()
    assert engine.acquire("job_1", cancel, far())
    outcome = engine.run_track("job_1", track(tmp_path), cancel=cancel, deadline=far())
    assert not outcome.ok and "exited" in (outcome.failure or "")
    # the next job starts a fresh worker
    monkeypatch.setenv("FAKE_ACESTEP_MODE", "ok")
    engine.release(discard=True)
    assert engine.acquire("job_2", cancel, far())
    assert engine.run_track("job_2", track(tmp_path), cancel=cancel, deadline=far()).ok


def test_idle_and_busy_peers_stop_the_worker(tmp_path: Path) -> None:
    eng = ResidentEngine(config(tmp_path, idle_seconds=0.3))
    try:
        cancel = threading.Event()
        assert eng.acquire("job_1", cancel, far())
        time.sleep(0.6)
        assert eng.running(), "never stopped while in use"
        eng.release()
        wait(lambda: not eng.running())

        busy = threading.Event()
        eng2 = ResidentEngine(config(tmp_path, idle_seconds=60))
        eng2.set_peer_check(busy.is_set)
        try:
            assert eng2.acquire("job_2", cancel, far())
            eng2.release()
            time.sleep(0.2)
            assert eng2.running()
            busy.set()
            wait(lambda: not eng2.running())
        finally:
            eng2.shutdown()
    finally:
        eng.shutdown()


def test_resident_off_stops_after_each_job(tmp_path: Path) -> None:
    eng = ResidentEngine(config(tmp_path, idle_seconds=0))
    try:
        assert eng.acquire("job_1", threading.Event(), far())
        eng.release()
        assert not eng.running()
    finally:
        eng.shutdown()


def test_cancel_while_loading_and_shutdown(tmp_path: Path) -> None:
    eng = ResidentEngine(config(tmp_path))
    cancel = threading.Event()
    cancel.set()
    assert not eng.acquire("job_1", cancel, far())
    assert not eng.running()
    eng.shutdown()
    with pytest.raises(EngineError, match="stopping"):
        eng.acquire("job_2", threading.Event(), far())


def test_missing_python_is_an_engine_error(tmp_path: Path) -> None:
    eng = ResidentEngine(config(tmp_path, python=tmp_path / "nope" / "python"))
    with pytest.raises(EngineError, match="not an executable Python"):
        eng.acquire("job_1", threading.Event(), far())


def test_a_leftover_worker_is_reaped(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    proc = subprocess.Popen(
        [sys.executable, str(WORKER_SCRIPT), "--proto-fd", "1", "--project-root", str(tmp_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        (state / PID_FILE).write_text(str(proc.pid))
        message = ResidentEngine(config(tmp_path)).reap_leftover()
        assert message == f"stopped leftover acestep_worker.py (pid {proc.pid})"
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


# -- backend ---------------------------------------------------------------------------------


def checkout(tmp_path: Path, models: tuple[str, ...] | None = None) -> Path:
    root = tmp_path / "ACE-Step"
    (root / "acestep").mkdir(parents=True)
    (root / ".venv" / "bin").mkdir(parents=True)
    # a wrapper, not a symlink: the venv is found from the path the interpreter runs as
    python = root / ".venv" / "bin" / "python"
    python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    python.chmod(0o755)
    names = models or ("acestep-v15-turbo", "vae", "Qwen3-Embedding-0.6B", "acestep-5Hz-lm-1.7B")
    for name in names:
        (root / "checkpoints" / name).mkdir(parents=True)
        (root / "checkpoints" / name / "model.safetensors").write_bytes(b"x")
    return root


def backend_for(tmp_path: Path, **options: object) -> AceStepBackend:
    opts = AceStepOptions.model_validate({"project_root": str(checkout(tmp_path))} | options)
    return AceStepBackend(opts, state_dir=tmp_path / "engine", poll_seconds=0.05)


def request(**overrides: object) -> GenerationRequest:
    values: dict[str, object] = {
        "job_id": "job_1",
        "prompt": "J-pop",
        "lyrics": "[verse]\nla",
        "instrumental": False,
        "duration_s": 10.0,
        "seed": 41,
        "count": 2,
        "vocal_language": "ja",
    }
    return GenerationRequest(**(values | overrides))  # type: ignore[arg-type]


def test_backend_generates_every_track_with_seed_plus_i(tmp_path: Path) -> None:
    backend = backend_for(tmp_path)
    run_dir = tmp_path / "job"
    progress: list[float] = []
    try:
        raw = backend.generate(
            request(), run_dir, timeout_s=60, cancel=threading.Event(), on_progress=progress.append
        )
        assert raw.ok, raw.describe_failure()
        assert [p.name for p in raw.outputs] == ["raw-0.wav", "raw-1.wav"]
        seeds = [json.loads(Path(f"{p}.params.json").read_text())["seed"] for p in raw.outputs]
        assert seeds == [41, 42]
        assert raw.details["load_seconds"] is not None and len(raw.details["tracks"]) == 2
        assert progress == sorted(progress) and progress[-1] == pytest.approx(0.995)
        assert backend.engine.running()  # resident for the next job
        # a loaded engine plans without the load time
        assert (
            backend.plan(request()).estimate_seconds
            < backend_for(tmp_path / "b").plan(request()).estimate_seconds
        )
    finally:
        backend.shutdown()
    assert not backend.engine.running()


def test_backend_failure_discards_the_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_ACESTEP_MODE", "gen_error")
    backend = backend_for(tmp_path)
    try:
        raw = backend.generate(request(), tmp_path / "job", timeout_s=60, cancel=threading.Event())
        assert not raw.ok and "out of memory" in raw.describe_failure()
        assert not backend.engine.running()
        monkeypatch.setenv("FAKE_ACESTEP_MODE", "load_fail")
        raw = backend.generate(request(), tmp_path / "job2", timeout_s=60, cancel=threading.Event())
        assert (
            raw.failure_code == "engine_failed" and "checkpoint missing" in raw.describe_failure()
        )
        cancel = threading.Event()
        cancel.set()
        monkeypatch.setenv("FAKE_ACESTEP_MODE", "ok")
        raw = backend.generate(request(), tmp_path / "job3", timeout_s=60, cancel=cancel)
        assert raw.canceled and not raw.ok
    finally:
        backend.shutdown()


def test_backend_without_a_checkout(tmp_path: Path) -> None:
    backend = AceStepBackend(state_dir=tmp_path)
    raw = backend.generate(request(), tmp_path / "job", timeout_s=5, cancel=threading.Event())
    assert raw.failure_code == "backend_unavailable"
    assert [c.level for c in backend.health()] == [Level.FAIL]


def test_backend_health(tmp_path: Path) -> None:
    backend = backend_for(tmp_path)
    checks = {c.name: c for c in backend.health()}
    assert all(c.level is Level.OK for c in checks.values()), checks
    assert "models stay loaded 120 s" in checks["resident engine"].detail
    root = tmp_path / "ACE-Step"
    (root / "checkpoints" / "vae" / "model.safetensors").unlink()
    (root / ".venv" / "bin" / "python").unlink()
    checks = {c.name: c for c in backend.health()}
    assert checks["ACE-Step models"].failed and "vae" in checks["ACE-Step models"].detail
    assert checks["ACE-Step Python"].failed
    off = backend_for(tmp_path / "off", resident_idle_seconds=0)
    assert "off" in {c.name: c for c in off.health()}["resident engine"].detail
    other = AceStepBackend(
        AceStepOptions(project_root=tmp_path / "missing"), state_dir=tmp_path / "s"
    )
    assert {c.name: c for c in other.health()}["ACE-Step checkout"].failed


def test_worker_env_never_downloads(tmp_path: Path) -> None:
    backend = backend_for(tmp_path, hf_home="~/hf")
    env = backend.worker_env()
    assert env["HF_HUB_OFFLINE"] == "1" and env["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert env["HF_HOME"] == str(Path("~/hf").expanduser())
    assert "HF_HOME" not in backend_for(tmp_path / "x").worker_env()
    assert backend.python_path == tmp_path / "ACE-Step" / ".venv" / "bin" / "python"
    custom = backend_for(tmp_path / "y", python="/opt/py/bin/python3")
    assert custom.python_path == Path("/opt/py/bin/python3")


def test_typical_seconds_and_plan(tmp_path: Path) -> None:
    backend = backend_for(
        tmp_path, load_seconds=20, track_overhead_seconds=5, seconds_per_audio_second=0.5
    )
    assert backend.typical_seconds(60) == 35
    assert backend.plan(request(duration_s=60, count=2)).estimate_seconds == 90


def test_from_settings(tmp_path: Path) -> None:
    cfg = tmp_path / "c.toml"
    cfg.write_text(f'[backends.acestep]\nproject_root = "{tmp_path}"\nlm_backend = "pt"\n')
    settings = load_settings(env={"ACE_STEP_API_DATA_DIR": str(tmp_path)}, config_path=cfg)
    backend = AceStepBackend.from_settings(settings)
    assert backend.options.lm_backend == "pt" and backend.project_root == tmp_path
    assert backend.engine.config.state_dir == tmp_path / "engine"
    cfg.write_text('[backends.acestep]\nlm_backend = "cuda"\n')
    with pytest.raises(SettingsError, match=r"backends\.acestep"):
        AceStepBackend.from_settings(load_settings(env={}, config_path=cfg))
    cfg.write_text('[backends.acestep]\nmodels = "x"\n')
    with pytest.raises(SettingsError):
        AceStepBackend.from_settings(load_settings(env={}, config_path=cfg))


# -- the worker's peak limiter ---------------------------------------------------------------


def test_limiter_keeps_every_sample_under_the_ceiling() -> None:
    import numpy as np

    from ace_step_api.backends.acestep_worker import limit_peaks

    rate = 48000
    t = np.arange(rate * 2) / rate
    quiet = 0.3 * np.sin(2 * np.pi * 220 * t)
    loud = quiet.copy()
    loud[rate // 2 : rate // 2 + 50] = 1.6  # a short burst far over full scale
    audio = np.stack([loud, quiet], axis=1).astype(np.float32)
    limited, levels = limit_peaks(audio)
    ceiling = 10 ** (-1 / 20)
    assert float(np.abs(limited).max()) <= ceiling + 1e-6
    # one short burst: no overall gain change, the limiter takes it
    assert levels["gain_db"] == 0.0
    assert levels["limiter_db"] == pytest.approx(20 * np.log10(ceiling / 1.6), abs=0.1)
    # far from the burst the gain is back to unity: the rest of the track is untouched
    assert np.allclose(limited[rate + 1000 :], audio[rate + 1000 :], atol=1e-6)
    assert np.allclose(limited[: rate // 4], audio[: rate // 4], atol=1e-6)


def test_limiter_leaves_quiet_audio_alone() -> None:
    import numpy as np

    from ace_step_api.backends.acestep_worker import limit_peaks

    audio = (0.5 * np.ones((1000, 2))).astype(np.float32)
    limited, levels = limit_peaks(audio)
    assert levels == {"gain_db": 0.0, "limiter_db": 0.0} and np.array_equal(limited, audio)
    empty, none = limit_peaks(np.zeros((0, 2), dtype=np.float32))
    assert none == {"gain_db": 0.0, "limiter_db": 0.0} and empty.shape == (0, 2)


def test_a_hot_track_is_turned_down_not_squashed() -> None:
    import numpy as np

    from ace_step_api.backends.acestep_worker import limit_peaks

    t = np.arange(48000 * 3) / 48000
    hot = 1.5 * np.sin(2 * np.pi * 110 * t)
    audio = np.stack([hot, hot], axis=1).astype(np.float32)
    limited, levels = limit_peaks(audio)
    ceiling = 10 ** (-1 / 20)
    assert levels["gain_db"] == pytest.approx(20 * np.log10(ceiling / 1.5), abs=0.05)
    assert levels["limiter_db"] > -0.1
    # a scaled copy of the input: the waveform keeps its shape
    assert np.allclose(limited, audio * (ceiling / 1.5), atol=2e-3)


def test_tracks_written_by_the_worker_are_limited(engine: ResidentEngine, tmp_path: Path) -> None:
    import wave

    import numpy as np

    cancel = threading.Event()
    assert engine.acquire("job_1", cancel, far())
    outcome = engine.run_track("job_1", track(tmp_path), cancel=cancel, deadline=far())
    assert outcome.ok and outcome.reply["gain_db"] < 0 and outcome.reply["limiter_db"] <= 0
    with wave.open(str(tmp_path / "raw-0.wav"), "rb") as wav:
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    assert int(np.abs(pcm.astype(np.int32)).max()) <= round(32767 * 10 ** (-1 / 20)) + 1


@pytest.mark.parametrize(
    ("mlx_dit", "flag", "seen"),
    [(None, [], True), (False, ["--mlx-dit", "off"], False), (True, ["--mlx-dit", "on"], True)],
)
def test_the_mlx_switch_reaches_the_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mlx_dit: bool | None,
    flag: list[str],
    seen: bool,
) -> None:
    record = tmp_path / "init.json"
    monkeypatch.setenv("FAKE_ACESTEP_RECORD", str(record))
    cfg = config(tmp_path, mlx_dit=mlx_dit)
    assert worker_command(cfg, 9)[-len(flag) or len(worker_command(cfg, 9)) :] == (flag or [])
    eng = ResidentEngine(cfg)
    try:
        assert eng.acquire("job_1", threading.Event(), far())
    finally:
        eng.shutdown()
    assert json.loads(record.read_text()) == {
        "config_path": "acestep-v15-turbo",
        "use_mlx_dit": seen,
    }
    backend = backend_for(tmp_path / "b", mlx_dit=mlx_dit)
    assert backend.engine.config.mlx_dit is mlx_dit
