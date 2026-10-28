"""Run a download under a watchdog: when the target directory stops growing, interrupt it
(Ctrl-C keeps Hugging Face's ``.incomplete`` files) and start it again, resuming.

A connection left bound to an old address (network switch) or a trickle would otherwise stall
for hours. An attempt that fails without having written anything is not retried: that is a
real error (no network, wrong repository), not a stall.
"""

from __future__ import annotations

import os
import signal
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from ace_step_api.setup.tools import Proc, SetupError, Tools

STALL_S = 600.0
POLL_S = 15.0
MAX_RESTARTS = 30
STOP_GRACE_S = 30.0


def dir_size(root: Path) -> int:
    total = 0
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            try:
                total += (Path(dirpath) / name).stat().st_size
            except FileNotFoundError:
                continue  # renamed from .incomplete while walking
    return total


def run_watched(
    name: str,
    cmd: Sequence[str],
    watch: Path,
    tools: Tools,
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    stall_s: float = STALL_S,
    poll_s: float = POLL_S,
    max_restarts: int = MAX_RESTARTS,
) -> None:
    for attempt in range(max_restarts + 1):
        before = dir_size(watch)
        code, stalled = _attempt(cmd, watch, tools, env, cwd, stall_s, poll_s)
        if code == 0 and not stalled:
            return
        grew = dir_size(watch) > before
        if not stalled and not grew:
            raise SetupError(f"{name}: download failed (exit {code}); see the output above")
        reason = "stalled" if stalled else f"stopped (exit {code})"
        tools.emit(f"{name}: {reason}; resuming ({attempt + 1}/{max_restarts})")
    raise SetupError(
        f"{name}: download did not finish after {max_restarts} restarts",
        hint="check the network, then run setup again: it resumes",
    )


def _attempt(
    cmd: Sequence[str],
    watch: Path,
    tools: Tools,
    env: Mapping[str, str] | None,
    cwd: Path | None,
    stall_s: float,
    poll_s: float,
) -> tuple[int, bool]:
    proc = tools.spawn(cmd, env, cwd)
    last, idle_since = dir_size(watch), tools.clock()
    try:
        while (code := proc.poll()) is None:
            tools.sleep(poll_s)
            size = dir_size(watch)
            if size != last:
                last, idle_since = size, tools.clock()
            elif tools.clock() - idle_since >= stall_s:
                return _stop(proc), True
        return code, False
    except BaseException:  # Ctrl-C on setup itself: leave no download running behind it
        _stop(proc)
        raise


def _stop(proc: Proc) -> int:
    proc.send_signal(signal.SIGINT)
    try:
        return proc.wait(timeout=STOP_GRACE_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        return proc.wait()
