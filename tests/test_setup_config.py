"""The config writer: a TOML emitter that round-trips, and a merge that never deletes."""

from __future__ import annotations

import stat
import tomllib
from pathlib import Path

import pytest

from ace_step_api.setup import config_write
from ace_step_api.setup.config_write import Changes, dumps, merge, write
from ace_step_api.setup.tools import SetupError


def test_dumps_round_trips_the_subset_the_config_uses() -> None:
    data = {
        "host": "127.0.0.1",
        "port": 8768,
        "settle_seconds": 30.5,
        "flag": True,
        "tags": ["a", 'quote " and \\ back', "tab\tnew\nline"],
        "empty": [],
        "backends": {
            "acestep": {"project_root": "/Users/x/ACE Step", "mlx_dit": False},
            "minimax": {"model": "mlx-community/MiniMax-Music3-mxfp8"},
            "empty": {},
        },
        "odd key": "ü",
        "coordinate_with": [
            {"name": "vpipe-api", "health_url": "http://127.0.0.1:8765/v1/health"},
            {"name": "klein-api", "health_url": "http://x/v1/health", "token": "t"},
        ],
    }
    assert tomllib.loads(dumps(data)) == data


@pytest.mark.parametrize("value", [float("nan"), {1, 2}, [1, 2], None])
def test_dumps_refuses_what_it_cannot_write(value: object) -> None:
    with pytest.raises(SetupError):
        dumps({"x": value})


def test_merge_only_adds_missing_keys_and_returns_a_new_mapping() -> None:
    existing = {
        "port": 8768,
        "backends": {"acestep": {"project_root": "/mine", "dit_model": "x"}},
        "custom": {"kept": 1},
    }
    changes = Changes(backends={"acestep": {"project_root": "/new", "python": "/p"}})
    merged = merge(existing, changes, {})
    assert merged["backends"]["acestep"] == {
        "project_root": "/mine",
        "dit_model": "x",
        "python": "/p",
    }
    assert merged["custom"] == {"kept": 1}
    assert existing["backends"]["acestep"] == {"project_root": "/mine", "dit_model": "x"}


def test_peer_token_env_name() -> None:
    assert config_write.peer_token_env("vpipe-api") == "ACE_STEP_API_COORD_TOKEN_VPIPE_API"
    assert config_write.peer_token_env("a.b") == "ACE_STEP_API_COORD_TOKEN_A_B"


def test_bad_peer_url_is_refused() -> None:
    with pytest.raises(SetupError):
        merge({}, Changes(peers=(("p", "ftp://x/health"),)), {})
    with pytest.raises(SetupError):
        config_write.parse_peer("no-equals")


def test_write_is_private_atomic_and_only_when_changed(tmp_path: Path) -> None:
    path = tmp_path / "dir" / "config.toml"
    assert write(path, {"port": 9000}) is True
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    before = path.stat().st_mtime_ns
    assert write(path, {"port": 9000}) is False
    assert path.stat().st_mtime_ns == before
    assert sorted(p.name for p in path.parent.iterdir()) == ["config.toml"]


def test_write_refuses_a_config_the_server_would_reject(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("port = 9000\n")
    with pytest.raises(SetupError, match="invalid"):
        write(path, {"port": 9000, "host": "0.0.0.0"})  # LAN without a token
    assert path.read_text() == "port = 9000\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["config.toml"]
