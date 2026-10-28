"""What setup does to the machine, behind one injectable object (tests pass fakes).

Child processes write to stderr only: with ``--json`` stdout carries the report alone.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

HTTP_TIMEOUT_S = 2.0
# never forwarded to the engines' own tools: the server's virtual environment would leak in
_DROPPED_ENV = ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "UV_PROJECT_ENVIRONMENT")


class SetupError(RuntimeError):
    """A setup step cannot continue; the message tells the user what to do."""

    def __init__(self, message: str, *, hint: str = "", command: str = "", human: bool = False):
        super().__init__(message)
        self.hint = hint
        self.command = command
        self.human = human


class Proc(Protocol):
    def poll(self) -> int | None: ...
    def send_signal(self, sig: int) -> None: ...
    def wait(self, timeout: float | None = None) -> int: ...
    def kill(self) -> None: ...


def child_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in _DROPPED_ENV}
    return env | dict(extra or {})


def _run(cmd: Sequence[str], env: Mapping[str, str] | None, cwd: Path | None) -> int:
    try:
        return subprocess.run(
            list(cmd), env=child_env(env), cwd=cwd, stdout=sys.stderr, check=False
        ).returncode
    except OSError as exc:
        sys.stderr.write(f"cannot run {cmd[0]}: {exc}\n")
        return 127


def _capture(cmd: Sequence[str]) -> str | None:
    try:
        result = subprocess.run(list(cmd), capture_output=True, text=True, check=False, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _spawn(cmd: Sequence[str], env: Mapping[str, str] | None, cwd: Path | None) -> Proc:
    return subprocess.Popen(list(cmd), env=child_env(env), cwd=cwd, stdout=sys.stderr)


def _emit(line: str) -> None:
    sys.stderr.write(line + "\n")
    sys.stderr.flush()


def http_status(url: str, token: str | None = None) -> int | None:
    """GET ``url`` straight (never through a proxy: the request may carry the token)."""
    request = urllib.request.Request(url)  # noqa: S310 - the URL is built from our own port
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=HTTP_TIMEOUT_S) as response:
            return int(response.status)
    except urllib.error.HTTPError as exc:
        return exc.code
    except (OSError, ValueError):
        return None


@dataclass(frozen=True)
class Tools:
    run: Callable[[Sequence[str], Mapping[str, str] | None, Path | None], int] = _run
    capture: Callable[[Sequence[str]], str | None] = _capture
    spawn: Callable[[Sequence[str], Mapping[str, str] | None, Path | None], Proc] = _spawn
    which: Callable[[str], str | None] = shutil.which
    http_status: Callable[[str, str | None], int | None] = http_status
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    emit: Callable[[str], None] = _emit
    home: Path = field(default_factory=Path.home)
    uid: int = field(default_factory=os.getuid)

    def has_tool(self, name: str) -> bool:
        return self.which(name) is not None
