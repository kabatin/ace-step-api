import logging
import subprocess
from pathlib import Path

import pytest

from ace_step_api import cli, doctor
from ace_step_api.backends.base import EngineStatus
from ace_step_api.checks import Check, Level
from ace_step_api.doctor import check_disk, check_macos, check_memory, check_power, run_doctor
from ace_step_api.jobs.store import JobStore
from ace_step_api.logs import JsonFormatter, event
from ace_step_api.settings import Settings, load_settings
from ace_step_api.workflows import build_registry
from tests.fakes import FakeMusicBackend, acestep_checkout, minimax_runtime

TOKEN = "test-token-0123456789-abcdefghijklmnop"


def fake_run(outputs: dict[str, str]):
    def run(cmd: list[str], timeout: float = 20):
        for key, stdout in outputs.items():
            if key in cmd[0]:
                return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")
        return None

    return run


@pytest.fixture
def healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    # CI also runs on Linux: pretend to be an Apple Silicon Mac
    monkeypatch.setattr(doctor, "check_platform", lambda: Check("platform", Level.OK, ""))
    monkeypatch.setattr(doctor, "check_macos", lambda: Check("macOS", Level.OK, "26.5"))
    monkeypatch.setattr(
        doctor,
        "_run",
        fake_run(
            {
                "pmset": "Now drawing from 'AC Power'",
                "memory_pressure": "System-wide memory free percentage: 64%",
            }
        ),
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return load_settings(
        env={"ACE_STEP_API_DATA_DIR": str(tmp_path / "data")}, config_path=tmp_path / "none.toml"
    )


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeMusicBackend:
    monkeypatch.setenv("ACE_STEP_API_CONFIG", str(tmp_path / "none.toml"))
    monkeypatch.setenv("ACE_STEP_API_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ACE_STEP_API_TOKEN", TOKEN)
    backend = FakeMusicBackend()
    monkeypatch.setattr(cli, "build_backend", lambda _settings: backend)
    return backend


# -- doctor ----------------------------------------------------------------------------


def by_name(checks: list[Check]) -> dict[str, Check]:
    return {c.name: c for c in checks}


def test_doctor_reports_backend_health(settings: Settings, healthy: None) -> None:
    emitted: list[Check] = []
    backend = FakeMusicBackend()
    checks = run_doctor(settings, backend, build_registry(backend), emit=emitted.append)
    assert emitted == checks
    names = by_name(checks)
    assert {"python", "fake backend", "disk", "power", "memory"} <= set(names)
    assert names["fake backend"].level is Level.OK
    backend.healthy = False
    assert by_name(run_doctor(settings, backend, build_registry(backend)))["fake backend"].failed


def test_macos_version_check() -> None:
    assert check_macos("26.5").level is Level.OK
    assert check_macos("13.6").level is Level.FAIL
    assert check_macos("").level is Level.WARN


def test_disk_check_walks_up_to_an_existing_dir(tmp_path: Path) -> None:
    check = check_disk(tmp_path / "not" / "yet" / "there")
    assert str(tmp_path) in check.detail


def test_power_and_memory_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        doctor,
        "_run",
        fake_run(
            {
                "pmset": "Now drawing from 'Battery Power'",
                "memory_pressure": "System-wide memory free percentage: 5%",
            }
        ),
    )
    assert check_power().level is Level.WARN
    assert check_memory().level is Level.WARN
    monkeypatch.setattr(doctor, "_run", fake_run({}))
    assert check_power().level is Level.WARN and check_memory().level is Level.WARN


def test_peer_checks_never_fail(
    settings: Settings, healthy: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ace_step_api.jobs.coordination import PeerHealth
    from ace_step_api.settings import PeerServer

    peer = PeerServer(name="vpipe-api", health_url="http://127.0.0.1:1/v1/health")
    with_peer = settings.model_copy(update={"coordinate_with": (peer,)})
    backend = FakeMusicBackend()
    check = by_name(run_doctor(with_peer, backend, build_registry(backend)))
    assert check["coordinate with vpipe-api"].level is Level.WARN
    assert "count as idle" in check["coordinate with vpipe-api"].detail

    def busy(_peer: PeerServer) -> PeerHealth:
        return PeerHealth(running=1, waiting=2)

    monkeypatch.setattr(doctor, "probe_health", busy)
    assert doctor.check_peer(peer).detail == "busy (running 1, waiting 2)"


def test_smoke_runs_a_short_job_per_workflow(settings: Settings, healthy: None) -> None:
    backend = FakeMusicBackend(wav=True, seconds=0.5)
    checks = by_name(run_doctor(settings, backend, build_registry(backend), smoke=True))
    smoke = checks["smoke ace-step-1.5"]
    assert smoke.level is Level.OK and "0.5 s WAV, 48000 Hz" in smoke.detail
    assert "0.5 s WAV, 44100 Hz" in checks["smoke minimax-music3"].detail
    assert [call.engine for call in backend.calls] == ["acestep", "minimax"]
    for call in backend.calls:
        assert call.instrumental and call.duration_s == 10.0 and call.count == 1
    assert not any((settings.data_dir / "smoke").iterdir())


def test_smoke_skips_only_the_unavailable_engine(settings: Settings, healthy: None) -> None:
    backend = FakeMusicBackend(wav=True, seconds=0.5)
    backend.statuses = {"minimax": EngineStatus(available=False, reason="not installed")}
    checks = by_name(run_doctor(settings, backend, build_registry(backend), smoke=True))
    assert checks["smoke ace-step-1.5"].level is Level.OK
    assert "smoke minimax-music3" not in checks
    assert [call.engine for call in backend.calls] == ["acestep"]


def test_memory_check_names_both_engines(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        doctor, "_run", fake_run({"memory_pressure": "System-wide memory free percentage: 64%"})
    )
    detail = check_memory().detail
    assert "~19 GB ACE-Step" in detail and "~24 GB MiniMax Music 3" in detail


def test_smoke_failure_is_reported(settings: Settings, healthy: None) -> None:
    backend = FakeMusicBackend()
    backend.mode = "fail"
    smoke = by_name(run_doctor(settings, backend, build_registry(backend), smoke=True))[
        "smoke ace-step-1.5"
    ]
    assert smoke.failed and "engine failed" in smoke.detail


def test_smoke_is_skipped_when_checks_fail(settings: Settings, healthy: None) -> None:
    backend = FakeMusicBackend()
    backend.healthy = False
    checks = run_doctor(settings, backend, build_registry(backend), smoke=True)
    assert not backend.calls and "smoke ace-step-1.5" not in by_name(checks)


def two_engines(tmp_path: Path, *, ace: bool, minimax: bool):
    """The real router; an engine that is not installed points at empty directories."""
    from ace_step_api.backends.acestep import AceStepBackend, AceStepOptions
    from ace_step_api.backends.minimax import MiniMaxBackend, MiniMaxOptions
    from ace_step_api.backends.router import EngineRouter

    ace_options = (
        AceStepOptions(project_root=acestep_checkout(tmp_path / "a")) if ace else AceStepOptions()
    )
    minimax_options = MiniMaxOptions.model_validate(
        minimax_runtime(tmp_path / "m")
        if minimax
        else {"python": str(tmp_path / "none" / "python"), "hf_home": str(tmp_path / "none")}
    )
    return EngineRouter(
        {
            "acestep": AceStepBackend(ace_options, state_dir=tmp_path / "engine"),
            "minimax": MiniMaxBackend(minimax_options, state_dir=tmp_path / "engine" / "minimax"),
        }
    )


@pytest.mark.parametrize("missing", ["acestep", "minimax"])
def test_a_missing_engine_is_a_warning_while_the_other_is_available(
    settings: Settings, healthy: None, tmp_path: Path, missing: str
) -> None:
    router = two_engines(tmp_path, ace=missing != "acestep", minimax=missing != "minimax")
    checks = run_doctor(settings, router, build_registry(router))
    assert not any(check.failed for check in checks)
    warned = [c for c in checks if c.level is Level.WARN and "workflow is unavailable" in c.detail]
    prefix = "ACE-Step" if missing == "acestep" else "MiniMax"
    assert warned and all(c.name.startswith(prefix) for c in warned)


def test_no_engine_at_all_is_a_failure(settings: Settings, healthy: None, tmp_path: Path) -> None:
    router = two_engines(tmp_path, ace=False, minimax=False)
    checks = by_name(run_doctor(settings, router, build_registry(router)))
    assert checks["ACE-Step checkout"].failed and checks["MiniMax Python"].failed
    assert checks["MiniMax Music 3 model"].failed


def test_doctor_exits_0_with_one_engine_and_1_with_none(
    env: FakeMusicBackend,
    healthy: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli, "build_backend", lambda _s: two_engines(tmp_path, ace=True, minimax=False)
    )
    assert cli.main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "all good" in out and "! MiniMax Python" in out
    none = tmp_path / "none-at-all"
    monkeypatch.setattr(
        cli, "build_backend", lambda _s: two_engines(none, ace=False, minimax=False)
    )
    assert cli.main(["doctor"]) == 1


# -- CLI -------------------------------------------------------------------------------


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    assert "ace-step-api 0.1.0" in capsys.readouterr().out


def test_workflows_and_config_hide_the_token(
    env: FakeMusicBackend, capsys: pytest.CaptureFixture[str]
) -> None:
    env.statuses = {"minimax": EngineStatus(available=False, reason="not installed")}
    assert cli.main(["workflows"]) == 0
    out = capsys.readouterr().out
    assert "ace-step-1.5" in out and "engine: acestep (available)" in out
    assert "minimax-music3" in out and "engine: minimax (unavailable: not installed)" in out
    assert cli.main(["config"]) == 0
    out = capsys.readouterr().out
    assert "token = ********" in out and TOKEN not in out and "port = 8768" in out


def test_doctor_exit_codes(
    env: FakeMusicBackend, healthy: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["doctor"]) == 0
    assert "all good" in capsys.readouterr().out
    env.healthy = False
    assert cli.main(["doctor"]) == 1


def test_settings_errors_exit_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ACE_STEP_API_CONFIG", str(tmp_path / "none.toml"))
    monkeypatch.setenv("ACE_STEP_API_HOST", "0.0.0.0")
    monkeypatch.delenv("ACE_STEP_API_TOKEN", raising=False)
    assert cli.main(["config"]) == 1
    assert "ACE_STEP_API_TOKEN" in capsys.readouterr().err


def test_serve_wires_uvicorn(env: FakeMusicBackend, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}

    def fake_run(app, **kwargs):
        seen.update(kwargs, app=app)

    monkeypatch.setattr("uvicorn.run", fake_run)
    assert cli.main(["serve", "--port", "9999"]) == 0
    assert seen["port"] == 9999 and seen["workers"] == 1 and seen["host"] == "127.0.0.1"
    assert seen["log_config"] is None


def test_serve_refuses_remote_bind_without_token(
    env: FakeMusicBackend, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("ACE_STEP_API_TOKEN")
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: pytest.fail("must not start"))
    assert cli.main(["serve", "--host", "0.0.0.0"]) == 1
    assert "ACE_STEP_API_TOKEN" in capsys.readouterr().err


def test_serve_fails_early_when_the_backend_cannot_generate(
    env: FakeMusicBackend, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    env.healthy = False
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: pytest.fail("must not start"))
    assert cli.main(["serve"]) == 1
    err = capsys.readouterr().err
    assert "no engine can generate" in err
    assert "acestep: model missing" in err and "minimax: model missing" in err


@pytest.mark.parametrize("down", ["acestep", "minimax"])
def test_serve_starts_with_one_engine_available(
    env: FakeMusicBackend,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    down: str,
) -> None:
    env.statuses = {down: EngineStatus(available=False, reason="not installed")}
    seen: dict = {}
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: seen.update(kw, app=app))
    assert cli.main(["serve"]) == 0 and "app" in seen
    captured = capsys.readouterr()
    assert f"warning: engine {down} is unavailable: not installed" in captured.err
    assert "workflows: ace-step-1.5, minimax-music3" in captured.out


def test_serve_refuses_second_instance(
    env: FakeMusicBackend, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    holder = JobStore(tmp_path / "data")
    holder.acquire_instance_lock()
    try:
        assert cli.main(["serve"]) == 1
        assert "already" in capsys.readouterr().err
    finally:
        holder.release_instance_lock()


# -- structured logs ---------------------------------------------------------------------


def test_json_log_lines() -> None:
    import json

    records: list[logging.LogRecord] = []

    class Keep(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = logging.getLogger("ace_step_api.test")
    logger.addHandler(Keep())
    logger.setLevel(logging.INFO)
    event(logger, "job.succeeded", job_id="job_1", total_seconds=3.5)
    line = json.loads(JsonFormatter().format(records[0]))
    assert line["event"] == "job.succeeded" and line["job_id"] == "job_1"
    assert line["total_seconds"] == 3.5 and line["level"] == "info"


def test_build_backend_routes_both_engines(tmp_path: Path) -> None:
    from ace_step_api.backends import build_backend
    from ace_step_api.backends.acestep import AceStepBackend
    from ace_step_api.backends.minimax import MiniMaxBackend

    base = {"ACE_STEP_API_DATA_DIR": str(tmp_path)}
    default = build_backend(load_settings(env=base, config_path=tmp_path / "x"))
    ace, minimax = default.engines["acestep"], default.engines["minimax"]
    assert isinstance(ace, AceStepBackend) and ace.project_root is None
    assert isinstance(minimax, MiniMaxBackend)  # its availability depends on this machine
    assert ace.health()[0].failed and not default.engine_status("acestep").available


def test_smoke_refuses_while_a_server_holds_the_data_dir(settings: Settings, healthy: None) -> None:
    holder = JobStore(settings.data_dir)
    holder.acquire_instance_lock()
    try:
        backend = FakeMusicBackend()
        smoke = by_name(run_doctor(settings, backend, build_registry(backend), smoke=True))[
            "smoke ace-step-1.5"
        ]
        assert smoke.failed and "server is running" in smoke.detail and not backend.calls
    finally:
        holder.release_instance_lock()


@pytest.mark.parametrize("host", ["::", "0.0.0.0", "192.168.0.2", "mac.local"])
def test_serve_host_override_needs_a_token(
    env: FakeMusicBackend,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    host: str,
) -> None:
    monkeypatch.delenv("ACE_STEP_API_TOKEN")
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: pytest.fail("must not start"))
    assert cli.main(["serve", "--host", host]) == 1
    assert "ACE_STEP_API_TOKEN" in capsys.readouterr().err


def test_serve_and_config_with_peers(
    env: FakeMusicBackend,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = tmp_path / "peers.toml"
    cfg.write_text(
        '[[coordinate_with]]\nname = "vpipe-api"\n'
        'health_url = "http://127.0.0.1:8765/v1/health"\ntoken = "peer-secret-value"\n'
    )
    cfg.chmod(0o600)
    monkeypatch.setenv("ACE_STEP_API_CONFIG", str(cfg))
    seen: dict = {}
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: seen.update(kw, app=app))
    assert cli.main(["serve"]) == 0
    out = capsys.readouterr().out
    assert "waits for: vpipe-api" in out and "peer-secret-value" not in out
    assert env.peer_check is not None  # the engine is told when a peer gets busy
    assert cli.main(["config"]) == 0
    out = capsys.readouterr().out
    assert "name = vpipe-api" in out and "token = ********" in out
    assert "peer-secret-value" not in out
