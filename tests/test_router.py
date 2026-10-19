"""The engine router: each request goes to its engine, one engine in memory at a time, and the
lifecycle calls reach every engine. Runs both real worker scripts against their fake engines."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from ace_step_api.backends import build_backend
from ace_step_api.backends.acestep import AceStepBackend, AceStepOptions
from ace_step_api.backends.base import GenerationRequest, MusicBackend
from ace_step_api.backends.minimax import MiniMaxBackend, MiniMaxOptions
from ace_step_api.backends.router import EngineRouter
from ace_step_api.settings import SettingsError, load_settings
from tests.fakes import FakeMusicBackend, acestep_checkout, minimax_runtime

FAKES = Path(__file__).parent


@pytest.fixture(autouse=True)
def fake_engines(monkeypatch: pytest.MonkeyPatch) -> None:
    paths = [str(FAKES / "fake_acestep"), str(FAKES / "fake_mlx_audio")]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(paths))
    monkeypatch.setenv("FAKE_ACESTEP_MODE", "ok")
    monkeypatch.setenv("FAKE_MINIMAX_MODE", "ok")


def engines(tmp_path: Path) -> EngineRouter:
    ace = AceStepBackend(
        AceStepOptions(project_root=acestep_checkout(tmp_path / "a")),
        state_dir=tmp_path / "engine",
        poll_seconds=0.05,
    )
    minimax = MiniMaxBackend(
        MiniMaxOptions.model_validate(minimax_runtime(tmp_path / "m")),
        state_dir=tmp_path / "engine" / "minimax",
        poll_seconds=0.05,
    )
    return EngineRouter({"acestep": ace, "minimax": minimax})


def request(engine: str, **overrides: object) -> GenerationRequest:
    values: dict[str, object] = {
        "job_id": "job_1",
        "engine": engine,
        "prompt": "lo-fi piano",
        "lyrics": "",
        "instrumental": True,
        "duration_s": 10.0,
        "seed": 1,
        "count": 1,
    }
    return GenerationRequest(**(values | overrides))  # type: ignore[arg-type]


def run(router: EngineRouter, engine: str, run_dir: Path) -> bool:
    raw = router.generate(request(engine), run_dir, timeout_s=60, cancel=threading.Event())
    return raw.ok


def test_one_engine_in_memory_at_a_time(tmp_path: Path) -> None:
    router = engines(tmp_path)
    ace, minimax = router.engines["acestep"], router.engines["minimax"]
    try:
        assert run(router, "acestep", tmp_path / "r1")
        assert router.engine_status("acestep").loaded
        assert not router.engine_status("minimax").loaded
        assert run(router, "minimax", tmp_path / "r2")
        # ACE-Step was unloaded before MiniMax ran: the two never share the memory
        assert router.engine_status("minimax").loaded
        assert not router.engine_status("acestep").loaded
        assert run(router, "acestep", tmp_path / "r3")
        assert not router.engine_status("minimax").loaded
        assert isinstance(ace, MusicBackend) and isinstance(minimax, MusicBackend)
    finally:
        router.shutdown()
    assert not router.engine_status("acestep").loaded


def test_each_engine_keeps_its_own_estimates_and_status(tmp_path: Path) -> None:
    router = engines(tmp_path)
    try:
        assert router.typical_seconds_for("acestep", 60) == 40.0
        assert router.typical_seconds_for("minimax", 60) == 580.0
        assert router.typical_seconds(60) == 580.0
        assert router.plan(request("minimax")).estimate_seconds == 30 + 105
        assert router.plan(request("acestep")).estimate_seconds == 40 + 15
        status = router.engine_status("nope")
        assert not status.available and status.reason == "unknown engine 'nope'"
        with pytest.raises(ValueError, match="unknown engine"):
            router.plan(request("nope"))
        with pytest.raises(ValueError, match="unknown engine"):
            router.typical_seconds_for("nope", 60)
        names = [check.name for check in router.health()]
        assert "ACE-Step models" in names and "MiniMax Music 3 model" in names
    finally:
        router.shutdown()


def test_an_unavailable_engine_leaves_the_other_working(tmp_path: Path) -> None:
    router = EngineRouter(
        {
            "acestep": AceStepBackend(state_dir=tmp_path / "engine"),  # not configured
            "minimax": MiniMaxBackend(
                MiniMaxOptions.model_validate(minimax_runtime(tmp_path)),
                state_dir=tmp_path / "engine" / "minimax",
                poll_seconds=0.05,
            ),
        }
    )
    try:
        status = router.engine_status("acestep")
        assert not status.available
        assert status.reason == "not configured (set project_root in [backends.acestep])"
        assert router.engine_status("minimax").available
        failed = router.generate(
            request("acestep"), tmp_path / "r1", timeout_s=60, cancel=threading.Event()
        )
        assert failed.failure_code == "backend_unavailable"
        assert run(router, "minimax", tmp_path / "r2")
    finally:
        router.shutdown()


def test_lifecycle_calls_reach_every_engine(tmp_path: Path) -> None:
    calls: list[str] = []

    class Recorder(FakeMusicBackend):
        def __init__(self, label: str) -> None:
            super().__init__()
            self.label = label

        def recover(self, run_dir: Path) -> str | None:
            calls.append(f"recover {self.label}")
            return f"stopped {self.label}"

        def startup(self) -> str | None:
            calls.append(f"startup {self.label}")
            return None

        def unload(self, reason: str) -> bool:
            calls.append(f"unload {self.label}: {reason}")
            return True

        def shutdown(self) -> None:
            calls.append(f"shutdown {self.label}")
            raise RuntimeError("stuck")

    a, b = Recorder("a"), Recorder("b")
    router = EngineRouter({"a": a, "b": b})
    assert router.recover(tmp_path) == "stopped a; stopped b"
    assert router.startup() is None
    check = lambda: False  # noqa: E731
    router.set_peer_check(check)
    assert a.peer_check is check and b.peer_check is check
    assert router.generate(request("b"), tmp_path / "r", timeout_s=5, cancel=threading.Event()).ok
    assert "unload a: engine b takes the GPU" in calls
    assert not any(call.startswith("unload b") for call in calls)
    assert [c.engine for c in b.calls] == ["b"] and not a.calls
    with pytest.raises(RuntimeError, match="stuck"):
        router.shutdown()
    assert calls.count("shutdown a") == calls.count("shutdown b") == 1  # b stopped even so
    with pytest.raises(ValueError, match="no engines"):
        EngineRouter({})


def test_build_backend_reads_each_table(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    runtime = minimax_runtime(tmp_path)
    config.write_text(
        f'data_dir = "{tmp_path / "data"}"\n'
        f'[backends.acestep]\nproject_root = "{tmp_path / "ACE-Step"}"\nlm_backend = "pt"\n'
        f'[backends.minimax]\npython = "{runtime["python"]}"\nhf_home = "{runtime["hf_home"]}"\n'
    )
    router = build_backend(load_settings(config_path=config, env={}))
    assert set(router.engines) == {"acestep", "minimax"}
    ace, minimax = router.engines["acestep"], router.engines["minimax"]
    assert isinstance(ace, AceStepBackend) and ace.options.lm_backend == "pt"
    assert isinstance(minimax, MiniMaxBackend)
    # ACE-Step keeps its pid file where 0.1.0 put it, so a restart still reaps its worker
    assert ace.engine.config.state_dir == tmp_path / "data" / "engine"
    assert minimax.engine.config.state_dir == tmp_path / "data" / "engine" / "minimax"
    assert router.engine_status("minimax").available
    assert router.engine_status("acestep").reason == "the ACE-Step checkout is missing"

    config.write_text("[backends.minimax]\nsteps = 8\n")
    with pytest.raises(SettingsError, match=r"\[backends.minimax\].*steps"):
        build_backend(load_settings(config_path=config, env={}))


@pytest.mark.parametrize("problem", ["still_running", "raises"])
def test_an_engine_that_cannot_be_unloaded_fails_the_job(tmp_path: Path, problem: str) -> None:
    class Stuck(FakeMusicBackend):
        def unload(self, reason: str) -> bool:
            if problem == "raises":
                raise RuntimeError("cannot stop")
            return False  # its worker survived SIGKILL: the memory is still taken

    other = FakeMusicBackend()
    router = EngineRouter({"stuck": Stuck(), "b": other})
    raw = router.generate(request("b"), tmp_path / "r", timeout_s=5, cancel=threading.Event())
    assert not raw.ok and raw.failure_code == "engine_unstoppable" and raw.retryable
    assert raw.failure == "engine stuck could not be stopped: engine b was not started"
    assert not other.calls  # nothing was loaded next to it
