"""Process helpers for the engine worker: the environment it gets, stopping its process group
(SIGINT first, then SIGKILL), and reaping a worker a crashed server left behind (pid file)."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path

PID_FILE = "backend.pid"
# `ps -o lstart` is a fixed-width "Tue Oct  6 21:43:10 2026"
_LSTART_WIDTH = 24
PID_START_SLACK_S = 10.0
# After SIGKILL, how long to wait for a GPU process to finish tearing down before giving up.
KILL_WAIT_S = 120.0


def child_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment for tools we start: ours, minus ace-step-api's own settings (the token)."""
    base = {key: value for key, value in os.environ.items() if not key.startswith("ACE_STEP_API_")}
    return {**base, **(extra or {})}


def stop_group(proc: subprocess.Popen[bytes], grace_s: float) -> None:
    """Ctrl-C first (lets the engine release the GPU cleanly), then SIGKILL the group.

    The group is SIGKILLed at the end in any case, so helpers the engine started cannot
    outlive it.
    """
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGINT)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=grace_s)
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=KILL_WAIT_S)


def reap_orphan(run_dir: Path, expected_name: str) -> str | None:
    """Kill a process group left by a server that died mid-run, if it is still ours.

    Pids are reused, so the pid is only trusted when the live process is ``expected_name``
    and started when the pid file was written; anything else is left alone.
    """
    pid_file = run_dir / PID_FILE
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
        written = pid_file.stat().st_mtime
    except (OSError, ValueError):
        return None
    finally:
        pid_file.unlink(missing_ok=True)
    if pid <= 1:
        return None
    info = process_info(pid)
    if info is None:
        return None
    command, started = info
    # the program is argv[0], or the script run through its interpreter (``python -P x.py``)
    names = [Path(token).name for token in command.split()[:3]]
    if expected_name not in names or abs(started - written) > PID_START_SLACK_S:
        return None
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        return None
    return f"stopped leftover {expected_name} (pid {pid})"


def process_info(pid: int) -> tuple[str, float] | None:
    """``(command line, start time as epoch seconds)`` of a live process, else None."""
    try:
        result = subprocess.run(
            ["ps", "-o", "lstart=,command=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            env={**os.environ, "LC_ALL": "C"},
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    line = result.stdout.strip()
    if result.returncode != 0 or len(line) <= _LSTART_WIDTH:
        return None
    try:
        started = time.mktime(time.strptime(line[:_LSTART_WIDTH], "%a %b %d %H:%M:%S %Y"))
    except ValueError:
        return None
    return line[_LSTART_WIDTH:].strip(), started
