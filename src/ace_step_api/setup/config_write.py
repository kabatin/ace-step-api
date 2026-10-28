"""Merge what setup learned into the config file ``settings.py`` reads.

Every existing key and value stays; only missing keys are added (``--lan`` and ``--port`` set
``host`` / ``token`` / ``port`` because they were asked for). The file is written atomically
(temp file + rename), mode 0600 in a 0700 directory, and only when something changed.
Python has no TOML writer: :func:`dumps` emits the subset the config uses.
"""

from __future__ import annotations

import contextlib
import math
import os
import re
import secrets
import tempfile
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ace_step_api.settings import ENV_PREFIX, PeerServer, Settings, SettingsError, load_settings
from ace_step_api.setup.tools import SetupError

_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")
_PEER = re.compile(r"^([A-Za-z0-9._-]{1,64})=(\S+)$")
LAN_HOST = "0.0.0.0"  # noqa: S104 - what --lan asks for; the token rule guards it


# -- TOML ----------------------------------------------------------------------------------------
def dumps(data: Mapping[str, Any]) -> str:
    """TOML for tables, arrays of tables and str/int/float/bool/list[str] values."""
    lines: list[str] = []
    _table(data, (), lines)
    return "\n".join(lines).lstrip("\n") + "\n"


def _table(data: Mapping[str, Any], path: tuple[str, ...], lines: list[str]) -> None:
    scalars = {k: v for k, v in data.items() if not _is_table(v) and not _is_table_array(v)}
    if path and (scalars or not any(_is_table(v) for v in data.values())):
        lines += ["", f"[{'.'.join(_key(p) for p in path)}]"]
    lines += [f"{_key(k)} = {_value(v)}" for k, v in scalars.items()]
    for key, value in data.items():
        if _is_table(value):
            _table(value, (*path, key), lines)
    for key, value in data.items():
        if _is_table_array(value):
            for item in value:
                lines += ["", f"[[{'.'.join(_key(p) for p in (*path, key))}]]"]
                lines += [f"{_key(k)} = {_value(v)}" for k, v in item.items()]


def _is_table(value: Any) -> bool:
    return isinstance(value, Mapping)


def _is_table_array(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(v, Mapping) for v in value)


def _key(key: str) -> str:
    return key if _BARE_KEY.fullmatch(key) else _string(key)


def _string(value: str) -> str:
    out = []
    for char in value:
        if char in ('"', "\\"):
            out.append("\\" + char)
        elif ord(char) < 0x20 or ord(char) == 0x7F:
            out.append(f"\\u{ord(char):04x}")
        else:
            out.append(char)
    return '"' + "".join(out) + '"'


def _value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SetupError(f"cannot write the number {value} to the config")
        return repr(value)
    if isinstance(value, str):
        return _string(value)
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return "[" + ", ".join(_string(v) for v in value) + "]"
    raise SetupError(f"cannot write a {type(value).__name__} to the config")


# -- merge ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Changes:
    backends: dict[str, dict[str, Any]] = field(default_factory=dict)
    lan: bool = False
    port: int | None = None
    peers: tuple[tuple[str, str], ...] = ()  # (name, health_url)


def read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise SetupError(f"cannot read the config {path}: {exc}") from exc


def parse_peer(spec: str) -> tuple[str, str]:
    match = _PEER.fullmatch(spec)
    if match is None:
        raise SetupError(f"--coordinate-with wants NAME=HEALTH_URL, got {spec!r}")
    return match.group(1), match.group(2)


def peer_token_env(name: str) -> str:
    return f"{ENV_PREFIX}COORD_TOKEN_" + re.sub(r"[^A-Za-z0-9]", "_", name).upper()


def merge(existing: Mapping[str, Any], changes: Changes, env: Mapping[str, str]) -> dict[str, Any]:
    """A new mapping: ``existing`` plus the missing keys ``changes`` asks for."""
    data = dict(existing)
    backends = {name: dict(table) for name, table in data.get("backends", {}).items()}
    for name, keys in changes.backends.items():
        backends[name] = keys | backends.get(name, {})
    if backends:
        data["backends"] = backends
    if changes.port is not None:
        data["port"] = changes.port
    if changes.lan:
        data["host"] = LAN_HOST
        data["token"] = _lan_token(data.get("token"), env)
    if changes.peers:
        data["coordinate_with"] = _peers(data.get("coordinate_with", []), changes.peers, env)
    return data


def check_lan_env(env: Mapping[str, str]) -> None:
    """``--lan``: a token in the environment must meet the server's rule (checked up front)."""
    supplied = env.get(ENV_PREFIX + "TOKEN", "").strip()
    if supplied:
        _check_token(supplied, f"{ENV_PREFIX}TOKEN")


def check_peer(name: str, url: str, env: Mapping[str, str]) -> None:
    """A ``--coordinate-with`` entry as the server would read it (URL, token from the env)."""
    token = env.get(peer_token_env(name), "").strip()
    entry = {"name": name, "health_url": url} | ({"token": token} if token else {})
    try:
        PeerServer.model_validate(entry)
    except ValidationError as exc:  # field and rule only: never the offending value
        problem = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        raise SetupError(f"--coordinate-with {name}: {problem}") from exc


def _check_token(token: str, source: str) -> None:
    try:
        Settings.model_validate({"host": LAN_HOST, "token": token})
    except ValidationError as exc:  # the message never contains the token
        raise SetupError(
            f"{source} does not meet the token rule",
            hint="use at least 32 printable ASCII characters without spaces (openssl rand -hex 32)",
        ) from exc


def _lan_token(current: Any, env: Mapping[str, str]) -> str:
    supplied = env.get(ENV_PREFIX + "TOKEN", "").strip()
    token = supplied or (current if isinstance(current, str) else "") or secrets.token_urlsafe(32)
    _check_token(token, f"{ENV_PREFIX}TOKEN" if supplied else "the config's token")
    return token


def _peers(
    current: list[Any], wanted: tuple[tuple[str, str], ...], env: Mapping[str, str]
) -> list[dict[str, Any]]:
    peers = [dict(peer) for peer in current]
    names = {peer.get("name") for peer in peers}
    for name, url in wanted:
        token = env.get(peer_token_env(name), "").strip()
        if name in names:
            peers = [
                {"token": token} | peer if peer.get("name") == name and token else peer
                for peer in peers
            ]
            continue
        check_peer(name, url, env)
        peers.append({"name": name, "health_url": url} | ({"token": token} if token else {}))
        names.add(name)
    return peers


# -- write ---------------------------------------------------------------------------------------
def write(path: Path, data: Mapping[str, Any]) -> bool:
    """Write ``data`` if it differs from the file; True when the file changed."""
    if path.is_file() and read(path) == data:
        return False
    text = dumps(data)
    if tomllib.loads(text) != data:  # the emitter must round-trip whatever it was given
        raise SetupError("internal error: the config would not read back the same")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        path.parent.chmod(0o700)
    fd, tmp = tempfile.mkstemp(prefix=".config-", suffix=".toml", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        Path(tmp).chmod(0o600)
        _validate(Path(tmp))
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return True


def _validate(path: Path) -> None:
    try:
        load_settings(config_path=path)  # the environment may hold the token
    except SettingsError as exc:
        raise SetupError(f"the merged config would be invalid: {exc}") from exc
