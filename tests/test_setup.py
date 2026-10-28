"""``ace-step-api setup`` against a fake machine: no network, no downloads, no launchctl."""

from __future__ import annotations

import json
import os
import plistlib
import signal
import stat
import tomllib
from collections.abc import Sequence
from functools import partial
from pathlib import Path

import pytest

from ace_step_api import cli
from ace_step_api.setup import command, pins
from ace_step_api.setup.command import Options, run_setup
from ace_step_api.setup.tools import SetupError
from ace_step_api.setup.watchdog import run_watched
from tests.setup_fakes import (
    MINIMAX_PACKAGE,
    Fake,
    FakeProc,
    ace_models,
    executable,
    mac,
    minimax_snapshot,
    minimax_venv,
)

TOKEN = "env-token-0123456789-abcdefghijklmnopq"
ALL = ("ace-step", "minimax-music3")


@pytest.fixture
def fake(tmp_path: Path) -> Fake:
    return Fake(home=tmp_path / "home")


def options(tmp_path: Path, *models: str, **kw: object) -> Options:
    return Options(
        models=models or ALL,
        explicit=bool(models),
        prefix=tmp_path / "opt",
        config=tmp_path / "config" / "config.toml",
        **kw,  # type: ignore[arg-type]
    )


def env(tmp_path: Path, **extra: str) -> dict[str, str]:
    return {"ACE_STEP_API_DATA_DIR": str(tmp_path / "data")} | extra


def setup(fake: Fake, tmp_path: Path, opts: Options, **kw: object):
    kw.setdefault("env", env(tmp_path))
    kw.setdefault("probe_machine", mac())
    return run_setup(opts, fake.tools, **kw)  # type: ignore[arg-type]


def status(report: command.Report) -> dict[str, str]:
    return {c.id: c.status for c in report.components}


def config(tmp_path: Path) -> dict:
    return tomllib.loads((tmp_path / "config" / "config.toml").read_text())


def tree(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*")) if root.exists() else []


# -- fake installers -----------------------------------------------------------------------------
def ace_effects(fake: Fake) -> None:
    def clone(cmd: Sequence[str], _cwd: Path | None) -> int:
        (Path(cmd[-1]) / "acestep").mkdir(parents=True)
        return 0

    def sync(_cmd: Sequence[str], cwd: Path | None) -> int:
        assert cwd is not None
        executable(cwd / ".venv" / "bin" / "python")
        return 0

    def download(_cmd: Sequence[str], cwd: Path | None) -> FakeProc:
        assert cwd is not None
        return FakeProc([lambda: ace_models(cwd)], 0)

    fake.effects |= {"git clone": clone, "uv sync": sync}
    fake.captures["rev-parse HEAD"] = pins.ACE_STEP_COMMIT
    fake.downloads.append(download)


def minimax_effects(fake: Fake, hf_home: Path) -> None:
    def venv(cmd: Sequence[str], _cwd: Path | None) -> int:
        minimax_venv(Path(cmd[-1]), package=False, hf=False)
        return 0

    def pip(cmd: Sequence[str], _cwd: Path | None) -> int:
        root = Path(cmd[cmd.index("--python") + 1]).parent.parent
        (root / MINIMAX_PACKAGE).parent.mkdir(parents=True)
        (root / MINIMAX_PACKAGE).write_text("")
        executable(root / "bin" / "hf")
        return 0

    def download(_cmd: Sequence[str], _cwd: Path | None) -> FakeProc:
        return FakeProc([lambda: minimax_snapshot(hf_home, ref=False)], 0)

    fake.effects |= {"uv venv": venv, "uv pip install": pip}
    fake.downloads.append(download)


def minimax_config(tmp_path: Path) -> Path:
    path = tmp_path / "config" / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'[backends.minimax]\nhf_home = "{tmp_path / "hf"}"\n')
    return tmp_path / "hf"


# -- arguments -----------------------------------------------------------------------------------
def test_models_default_to_every_component_and_reject_unknown_ids() -> None:
    assert command.parse_models(None) == ALL
    assert command.parse_models(" minimax-music3, minimax-music3 ") == ("minimax-music3",)
    with pytest.raises(SetupError, match="valid: ace-step, minimax-music3"):
        command.parse_models("ace-step,wan")


def test_unknown_model_exits_2_with_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["setup", "--models", "nope", "--json"]) == 2
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1
    assert json.loads(out[0])["ok"] is False


# -- check ---------------------------------------------------------------------------------------
def test_check_reports_sizes_and_changes_nothing(fake: Fake, tmp_path: Path) -> None:
    code, report = setup(fake, tmp_path, options(tmp_path, check=True))
    assert code == 0
    assert report.ok
    assert status(report) == {"ace-step": "missing", "minimax-music3": "missing"}
    sizes = {c.id: c.download_bytes for c in report.components}
    assert sizes["ace-step"] > pins.ACE_STEP_MODEL_BYTES
    assert sizes["minimax-music3"] > pins.MINIMAX_MODEL_BYTES
    assert fake.calls == []
    assert tree(tmp_path) == ["data"] or tree(tmp_path) == []
    payload = report.as_json()
    assert payload["action"] == "check"
    assert payload["service"] == {
        "installed": False,
        "label": "com.github.kabatin.ace-step-api",
        "healthy": False,
    }


def test_check_on_an_intel_mac_says_unsupported(fake: Fake, tmp_path: Path) -> None:
    _, report = setup(fake, tmp_path, options(tmp_path, check=True), probe_machine=mac(arm64=False))
    assert set(status(report).values()) == {"unsupported"}


def test_check_finds_an_install_the_config_does_not_name_yet(fake: Fake, tmp_path: Path) -> None:
    root = tmp_path / "opt" / pins.ACE_STEP_DIR
    (root / "acestep").mkdir(parents=True)
    executable(root / ".venv" / "bin" / "python")
    ace_models(root)
    _, report = setup(fake, tmp_path, options(tmp_path, "ace-step", check=True))
    assert status(report) == {"ace-step": "installed"}
    assert "add it to the config" in report.components[0].message


def test_interrupted_download_counts_as_missing(fake: Fake, tmp_path: Path) -> None:
    root = tmp_path / "opt" / pins.ACE_STEP_DIR
    (root / "acestep").mkdir(parents=True)
    executable(root / ".venv" / "bin" / "python")
    ace_models(root)
    partial_file = root / "checkpoints" / ".cache" / "huggingface" / "download" / "x.incomplete"
    partial_file.parent.mkdir(parents=True)
    partial_file.write_bytes(b"x")
    _, report = setup(fake, tmp_path, options(tmp_path, "ace-step", check=True))
    assert status(report) == {"ace-step": "missing"}
    assert "did not finish" in report.components[0].message


# -- install -------------------------------------------------------------------------------------
def test_ace_step_installs_pinned_and_writes_config(fake: Fake, tmp_path: Path) -> None:
    ace_effects(fake)
    code, report = setup(fake, tmp_path, options(tmp_path, "ace-step", yes=True))
    assert code == 0, report
    assert status(report) == {"ace-step": "installed"}
    root = tmp_path / "opt" / pins.ACE_STEP_DIR
    assert fake.commands("git clone") == [("git", "clone", pins.ACE_STEP_REPO, str(root))]
    assert ("git", "-C", str(root), "checkout", "--detach", pins.ACE_STEP_COMMIT) in fake.calls
    assert fake.commands("uv sync") == [
        ("uv", "sync", "--exclude-newer", pins.ACE_STEP_EXCLUDE_NEWER)
    ]
    download = fake.commands(str(root / ".venv" / "bin" / "python"))[0]
    # the weights at the tested revision, straight into checkpoints/ (not main)
    assert "snapshot_download" in download[2]
    assert download[3:] == (
        pins.ACE_STEP_MODEL_REPO,
        pins.ACE_STEP_MODEL_REVISION,
        str(root / "checkpoints"),
    )
    assert fake.envs[-1] is not None and fake.envs[-1]["HF_HUB_DISABLE_XET"] == "1"
    table = config(tmp_path)["backends"]["acestep"]
    assert table == {
        "project_root": str(root),
        "python": str(root / ".venv" / "bin" / "python"),
        "lm_backend": "pt",
        "hf_home": str(Path(pins.ACE_STEP_HF_HOME).expanduser()),
    }
    assert "minimax" not in config(tmp_path)["backends"]

    again = Fake(home=fake.home)
    code, report = setup(again, tmp_path, options(tmp_path, "ace-step", yes=True))
    assert (code, status(report)) == (0, {"ace-step": "already"})
    assert again.calls == []


def test_a_moved_commit_fails_the_component(fake: Fake, tmp_path: Path) -> None:
    ace_effects(fake)
    fake.captures["rev-parse HEAD"] = "0" * 40
    code, report = setup(fake, tmp_path, options(tmp_path, "ace-step", yes=True))
    assert (code, status(report)) == (1, {"ace-step": "failed"})
    assert pins.ACE_STEP_COMMIT in report.components[0].message
    assert fake.commands("uv sync") == []


def test_existing_checkout_with_lock_is_synced_frozen_and_left_at_its_version(
    fake: Fake, tmp_path: Path
) -> None:
    root = tmp_path / "mine"
    (root / "acestep").mkdir(parents=True)
    (root / "uv.lock").write_text("")
    cfg = tmp_path / "config" / "config.toml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(f'[backends.acestep]\nproject_root = "{root}"\nlm_backend = "mlx"\n')
    ace_effects(fake)
    fake.captures["rev-parse HEAD"] = "abc"
    code, report = setup(fake, tmp_path, options(tmp_path, "ace-step", yes=True))
    assert (code, status(report)) == (0, {"ace-step": "installed"})
    assert fake.commands("git clone") == []
    assert fake.commands("uv sync") == [("uv", "sync", "--frozen")]
    assert any("does not change its version" in line for line in fake.lines)
    table = config(tmp_path)["backends"]["acestep"]
    assert table["project_root"] == str(root)
    assert table["lm_backend"] == "mlx"  # kept


def test_minimax_alone_installs_venv_package_and_weights(fake: Fake, tmp_path: Path) -> None:
    hf_home = minimax_config(tmp_path)
    minimax_effects(fake, hf_home)
    code, report = setup(fake, tmp_path, options(tmp_path, "minimax-music3", yes=True))
    assert code == 0, report
    assert status(report) == {"minimax-music3": "installed"}
    venv = tmp_path / "opt" / pins.MINIMAX_DIR / ".venv"
    assert fake.commands("uv venv") == [("uv", "venv", "--python", "3.12", str(venv))]
    assert fake.commands("uv pip install")[0][-1] == "mlx-audio==0.5.9"
    hf = fake.commands(str(venv / "bin" / "hf"))[0]
    assert hf[1:] == (
        "download",
        pins.MINIMAX_MODEL_REPO,
        "--revision",
        pins.MINIMAX_MODEL_REVISION,
    )
    cache = hf_home / "hub" / ("models--" + pins.MINIMAX_MODEL_REPO.replace("/", "--"))
    assert (cache / "refs" / "main").read_text() == pins.MINIMAX_MODEL_REVISION
    assert config(tmp_path)["backends"] == {
        "minimax": {"hf_home": str(hf_home), "python": str(venv / "bin" / "python")}
    }
    assert fake.commands("git") == []


def test_both_install_and_one_failure_does_not_stop_the_other(fake: Fake, tmp_path: Path) -> None:
    hf_home = minimax_config(tmp_path)
    ace_effects(fake)
    fake.effects["git clone"] = lambda _cmd, _cwd: 128
    fake.downloads.clear()
    minimax_effects(fake, hf_home)
    code, report = setup(fake, tmp_path, options(tmp_path, yes=True))
    assert code == 1
    assert status(report) == {"ace-step": "failed", "minimax-music3": "installed"}
    assert not report.ok
    assert "acestep" not in config(tmp_path)["backends"]


def test_without_yes_and_no_terminal_nothing_happens(fake: Fake, tmp_path: Path) -> None:
    code, report = setup(fake, tmp_path, options(tmp_path))
    assert code == 3
    assert fake.calls == []
    fix = report.components[0].fix
    assert fix is not None and fix["hint"] == "rerun with --yes"
    assert not (tmp_path / "config").exists()


def test_the_prompt_shows_size_and_licences(fake: Fake, tmp_path: Path) -> None:
    asked: list[str] = []
    code, _ = setup(fake, tmp_path, options(tmp_path), confirm=lambda p: asked.append(p) or False)
    assert code == 3
    assert "MiniMax-Music3" in asked[0] and "MIT" in asked[0] and "GB" in asked[0]
    assert fake.calls == []

    ace_effects(fake)
    code, _ = setup(fake, tmp_path, options(tmp_path, "ace-step"), confirm=lambda _p: True)
    assert code == 0


def test_too_little_disk_fails_before_downloading(fake: Fake, tmp_path: Path) -> None:
    code, report = setup(
        fake, tmp_path, options(tmp_path, yes=True), probe_machine=mac(free_bytes=20 * 10**9)
    )
    assert code == 1
    assert fake.calls == []
    assert "free disk" in (report.error or "")


def test_install_on_an_intel_mac_reports_unsupported_and_exits_0_by_default(
    fake: Fake, tmp_path: Path
) -> None:
    code, report = setup(
        fake, tmp_path, options(tmp_path, yes=True), probe_machine=mac(arm64=False)
    )
    assert (code, report.ok) == (0, True)
    assert set(status(report).values()) == {"unsupported"}
    assert fake.calls == []


@pytest.mark.parametrize("check", [False, True])
def test_an_unsupported_component_asked_for_by_name_exits_1(
    fake: Fake, tmp_path: Path, check: bool
) -> None:
    opts = options(tmp_path, "ace-step", yes=True, check=check)
    code, report = setup(fake, tmp_path, opts, probe_machine=mac(arm64=False))
    assert (code, report.ok) == (1, False)
    assert status(report) == {"ace-step": "unsupported"}


def test_a_model_setup_cannot_install_is_unsupported_and_the_other_goes_on(
    fake: Fake, tmp_path: Path
) -> None:
    cfg = tmp_path / "config" / "config.toml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text('[backends.minimax]\nmodel = "someone/other-model"\n')
    ace_effects(fake)
    code, report = setup(fake, tmp_path, options(tmp_path, yes=True))
    assert status(report) == {"ace-step": "installed", "minimax-music3": "unsupported"}
    assert (code, report.ok) == (0, True)
    assert fake.commands("uv venv") == []


def test_exit_3_reports_not_ok(fake: Fake, tmp_path: Path) -> None:
    code, report = setup(fake, tmp_path, options(tmp_path))
    assert (code, report.ok, report.as_json()["ok"]) == (3, False, False)


def test_check_says_installed_for_a_working_engine(fake: Fake, tmp_path: Path) -> None:
    installed_minimax(tmp_path)
    _, report = setup(fake, tmp_path, options(tmp_path, check=True))
    assert status(report)["minimax-music3"] == "installed"


def test_missing_uv_is_a_fix_not_a_crash(fake: Fake, tmp_path: Path) -> None:
    fake.on_path = {"git"}
    _, report = setup(fake, tmp_path, options(tmp_path, "ace-step", yes=True))
    outcome = report.components[0]
    assert outcome.status == "failed"
    assert outcome.fix is not None and "uv" in outcome.fix["hint"]


# -- lan / port / coordinate_with ----------------------------------------------------------------
def test_lan_takes_the_env_token_and_never_reports_it(fake: Fake, tmp_path: Path) -> None:
    code, report = setup(
        fake,
        tmp_path,
        options(tmp_path, "minimax-music3", yes=True, lan=True, port=9000),
        env=env(tmp_path, ACE_STEP_API_TOKEN=TOKEN),
    )
    assert code == 1  # minimax has no fake installer here: only the config matters
    data = config(tmp_path)
    assert (data["host"], data["token"], data["port"]) == ("0.0.0.0", TOKEN, 9000)
    assert report.lan and report.token_set
    assert report.url == "http://127.0.0.1:9000"
    assert TOKEN not in json.dumps(report.as_json())
    assert all(TOKEN not in line for line in fake.lines)
    mode = (tmp_path / "config" / "config.toml").stat().st_mode
    assert stat.S_IMODE(mode) == 0o600
    assert stat.S_IMODE((tmp_path / "config").stat().st_mode) == 0o700


def test_lan_keeps_or_generates_a_token_and_rejects_a_weak_one(fake: Fake, tmp_path: Path) -> None:
    opts = options(tmp_path, "ace-step", check=False, yes=True, lan=True)
    fake.on_path = set()  # the component fails fast; the config is still written
    setup(fake, tmp_path, opts)
    first = config(tmp_path)["token"]
    assert len(first) >= 32
    setup(fake, tmp_path, opts)
    assert config(tmp_path)["token"] == first

    weak = "short"
    code, report = setup(fake, tmp_path, opts, env=env(tmp_path, ACE_STEP_API_TOKEN=weak))
    assert code == 1
    assert "token" in (report.error or "")
    assert weak not in (report.error or "")
    assert config(tmp_path)["token"] == first


def test_coordinate_with_is_deduped_and_takes_peer_tokens(fake: Fake, tmp_path: Path) -> None:
    fake.on_path = set()
    peers = (
        "vpipe-api=http://127.0.0.1:8765/v1/health",
        "klein-api=http://127.0.0.1:8767/v1/health",
    )
    opts = options(tmp_path, "ace-step", yes=True, coordinate_with=peers)
    peer_env = env(tmp_path, ACE_STEP_API_COORD_TOKEN_VPIPE_API="peer-token")
    setup(fake, tmp_path, opts, env=peer_env)
    setup(fake, tmp_path, opts, env=peer_env)
    entries = config(tmp_path)["coordinate_with"]
    assert [e["name"] for e in entries] == ["vpipe-api", "klein-api"]
    assert entries[0]["token"] == "peer-token"
    assert "token" not in entries[1]


# -- service -------------------------------------------------------------------------------------
def installed_minimax(tmp_path: Path) -> None:
    hf_home = minimax_config(tmp_path)
    minimax_venv(tmp_path / "opt" / pins.MINIMAX_DIR / ".venv")
    minimax_snapshot(hf_home)


def test_service_writes_the_plist_reloads_it_and_waits_for_health(
    fake: Fake, tmp_path: Path
) -> None:
    installed_minimax(tmp_path)
    fake.health = [None, None, 200]
    opts = options(tmp_path, "minimax-music3", service=True, label_suffix="verify")
    code, report = setup(fake, tmp_path, opts)
    assert code == 0, report
    label = "com.github.kabatin.ace-step-api.verify"
    plist = fake.home / "Library" / "LaunchAgents" / f"{label}.plist"
    data = plistlib.loads(plist.read_bytes())
    assert data["Label"] == label
    assert data["ProgramArguments"] == ["/opt/bin/ace-step-api", "serve"]
    assert data["KeepAlive"] == {"SuccessfulExit": False}
    assert data["EnvironmentVariables"]["ACE_STEP_API_CONFIG"] == str(opts.config)
    assert data["StandardErrorPath"].endswith("Logs/ace-step-api.verify.log")
    assert fake.commands("launchctl") == [
        ("launchctl", "bootout", f"gui/501/{label}"),
        ("launchctl", "print", f"gui/501/{label}"),
        ("launchctl", "bootstrap", "gui/501", str(plist)),
    ]
    assert report.service == {"installed": True, "label": label, "healthy": True}


def test_service_waits_for_bootout_and_retries_a_busy_bootstrap(fake: Fake, tmp_path: Path) -> None:
    installed_minimax(tmp_path)
    still_loaded = iter(["state = running", "state = running"])
    fake.captures["launchctl print"] = lambda: next(still_loaded, None)
    codes = iter([5, 37, 0])
    fake.effects["launchctl bootstrap"] = lambda _cmd, _cwd: next(codes)
    fake.health = [200]
    code, report = setup(fake, tmp_path, options(tmp_path, "minimax-music3", service=True))
    assert code == 0, report
    kinds = [c[1] for c in fake.commands("launchctl")]
    assert kinds == ["bootout", "print", "print", "print", "bootstrap", "bootstrap", "bootstrap"]


def test_service_bootstrap_gives_up_on_other_errors(fake: Fake, tmp_path: Path) -> None:
    installed_minimax(tmp_path)
    fake.effects["launchctl bootstrap"] = lambda _cmd, _cwd: 1
    code, report = setup(fake, tmp_path, options(tmp_path, "minimax-music3", service=True))
    assert code == 1
    assert len(fake.commands("launchctl bootstrap")) == 1
    assert "bootstrap" in (report.error or "")


def test_service_that_never_answers_fails(fake: Fake, tmp_path: Path) -> None:
    installed_minimax(tmp_path)
    code, report = setup(fake, tmp_path, options(tmp_path, "minimax-music3", service=True))
    assert code == 1
    assert report.service["healthy"] is False
    assert "did not answer" in (report.error or "")


def test_service_needs_the_installed_cli(fake: Fake, tmp_path: Path) -> None:
    installed_minimax(tmp_path)
    fake.on_path = {"git", "uv"}
    code, report = setup(fake, tmp_path, options(tmp_path, "minimax-music3", service=True))
    assert code == 1
    assert "not on PATH" in (report.error or "")
    assert fake.commands("launchctl") == []


# -- watchdog ------------------------------------------------------------------------------------
def grow(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(b"x")


def test_watchdog_interrupts_a_stall_and_resumes(fake: Fake, tmp_path: Path) -> None:
    watch = tmp_path / "w"
    stalled = FakeProc([partial(grow, watch / "a"), *[lambda: None] * 10], 0)
    finishing = FakeProc([partial(grow, watch / "a")], 0)
    procs = [stalled, finishing]
    fake.downloads = [lambda _c, _w: procs.pop(0)] * 2
    run_watched("x", ["dl"], watch, fake.tools, stall_s=30, poll_s=10)
    assert stalled.signals == [signal.SIGINT]
    assert procs == []
    assert any("stalled; resuming" in line for line in fake.lines)


def test_watchdog_does_not_retry_a_failure_that_wrote_nothing(fake: Fake, tmp_path: Path) -> None:
    fake.downloads = [lambda _c, _w: FakeProc([], 1)]
    with pytest.raises(SetupError, match="download failed"):
        run_watched("x", ["dl"], tmp_path / "w", fake.tools)


def test_watchdog_gives_up_after_its_restarts(fake: Fake, tmp_path: Path) -> None:
    watch = tmp_path / "w"
    fake.downloads = [lambda _c, _w: FakeProc([partial(grow, watch / "a")], 1)] * 3
    with pytest.raises(SetupError, match="after 2 restarts"):
        run_watched("x", ["dl"], watch, fake.tools, max_restarts=2)


# -- command line --------------------------------------------------------------------------------
def test_cli_json_is_the_only_stdout_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = Fake(home=tmp_path / "home")
    monkeypatch.setattr(cli, "Tools", lambda: fake.tools)
    monkeypatch.setattr(cli.setup, "run_setup", partial(run_setup, probe_machine=mac()))
    monkeypatch.setenv("ACE_STEP_API_CONFIG", str(tmp_path / "c.toml"))
    monkeypatch.setenv("ACE_STEP_API_TOKEN", TOKEN)
    monkeypatch.setenv("ACE_STEP_API_DATA_DIR", str(tmp_path / "data"))
    argv = ["setup", "--check", "--json", "--lan", "--prefix", str(tmp_path / "opt")]
    assert cli.main(argv) == 0
    captured = capsys.readouterr()
    lines = captured.out.strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["server"] == "ace-step-api"
    assert [c["id"] for c in payload["components"]] == list(ALL)
    assert TOKEN not in captured.out + captured.err
    assert not (tmp_path / "c.toml").exists()  # --check wrote nothing

    assert cli.main(["setup", "--check", "--prefix", str(tmp_path / "opt")]) == 0
    assert "minimax-music3" in capsys.readouterr().out


def test_cli_rejects_a_bad_port(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["setup", "--port", "0"]) == 2
    assert "--port" in capsys.readouterr().err


def test_default_prefix_is_under_local_opt() -> None:
    assert command.DEFAULT_PREFIX.expanduser() == Path(
        os.path.expanduser("~/.local/opt/ace-step-api")
    )


def test_child_output_goes_to_stderr_and_the_server_venv_does_not_leak(
    capfd: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from ace_step_api.setup import tools

    monkeypatch.setenv("VIRTUAL_ENV", "/server/.venv")
    assert tools._run(["sh", "-c", 'echo "out:${VIRTUAL_ENV:-none}"'], {"X": "1"}, None) == 0
    captured = capfd.readouterr()
    assert captured.out == ""
    assert "out:none" in captured.err
    assert tools._run(["/nonexistent/tool"], None, None) == 127


# -- usage errors: exit 2 with one complete JSON object --------------------------------------------
REPORT_KEYS = {
    "server",
    "version",
    "ok",
    "action",
    "url",
    "lan",
    "tokenSet",
    "configPath",
    "service",
    "machine",
    "components",
}


def usage_json(capsys: pytest.CaptureFixture[str], argv: list[str]) -> dict:
    assert cli.main(["setup", "--json", *argv]) == 2
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1
    payload = json.loads(out[0])
    assert payload.keys() >= REPORT_KEYS
    assert payload["ok"] is False and payload["error"]
    return payload


@pytest.mark.parametrize(
    "argv",
    [
        ["--label-suffix", "../x", "--port", "9000"],
        ["--label-suffix", "verify"],  # beside the real one it needs its own port
        ["--port", "nope"],  # argparse's own error
        ["--no-such-flag"],
        ["--coordinate-with", "vpipe-api=ftp://x/v1/health"],
        ["--coordinate-with", "no-equals"],
    ],
)
def test_usage_errors_give_one_json_object(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
) -> None:
    monkeypatch.setenv("ACE_STEP_API_CONFIG", str(tmp_path / "c.toml"))
    usage_json(capsys, argv)


def test_bad_lan_and_peer_tokens_in_the_environment_are_usage_errors(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ACE_STEP_API_CONFIG", str(tmp_path / "c.toml"))
    monkeypatch.setenv("ACE_STEP_API_TOKEN", "short")
    payload = usage_json(capsys, ["--lan"])
    assert "short" not in json.dumps(payload)
    monkeypatch.delenv("ACE_STEP_API_TOKEN")
    monkeypatch.setenv("ACE_STEP_API_COORD_TOKEN_VPIPE_API", "has space")
    payload = usage_json(capsys, ["--coordinate-with", "vpipe-api=http://127.0.0.1:8765/v1/health"])
    assert "has space" not in json.dumps(payload)
    assert not (tmp_path / "c.toml").exists()


def test_usage_errors_without_json_print_usage_and_exit_2(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["setup", "--no-such-flag"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "usage:" in captured.err and "--no-such-flag" in captured.err
