"""A resident engine worker process, so consecutive jobs skip the model load.

Each engine's worker (:mod:`ace_step_api.backends.acestep_worker`,
:mod:`ace_step_api.backends.minimax_worker`) runs in that engine's own Python and keeps its
models loaded. A worker holds many GB of unified memory, so:

- it is started lazily by the first job and kept while jobs keep coming;
- it is stopped after ``idle_seconds`` without a job, as soon as another GPU server on the Mac
  is seen busy while it idles, after a job that did not succeed (canceled, timed out, failed:
  a fresh process is the only way to be sure the GPU is released), when another engine of this
  server takes the GPU, and on shutdown.

It is a child in its own process group with a pid file in ``state_dir``, so a restarted
ace-step-api stops a worker left behind by a crashed one (pid + script name + start time must
match). Requests go to its stdin; replies come back on a private pipe (``--proto-fd``).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import queue
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from ace_step_api.backends.process import PID_FILE, child_env, reap_orphan, stop_group
from ace_step_api.logs import event

log = logging.getLogger("ace_step_api.engine")

LOG_FILE = "worker.log"
MAX_REPLY_BYTES = 64 * 1024

PeerCheck = Callable[[], bool]
ProgressFn = Callable[[float], None]


@dataclass(frozen=True)
class EngineConfig:
    # short engine name for logs ("acestep", "minimax")
    name: str
    python: Path
    # the worker script (runs in ``python``) and its engine-specific arguments
    script: Path
    args: tuple[str, ...]
    # working directory of the worker
    cwd: Path
    # pid file and worker log; one directory per engine
    state_dir: Path
    idle_seconds: float
    poll_seconds: float
    start_timeout_s: float = 600.0
    stop_grace_s: float = 15.0
    env: Mapping[str, str] = field(default_factory=dict)


def worker_command(config: EngineConfig, proto_fd: int) -> list[str]:
    # -P: the script's own directory (this package) must not shadow the engine's modules
    return [str(config.python), "-P", str(config.script), "--proto-fd", str(proto_fd), *config.args]


@dataclass(frozen=True)
class TrackOutcome:
    ok: bool
    seconds: float
    canceled: bool = False
    timed_out: bool = False
    failure: str | None = None
    reply: dict[str, Any] = field(default_factory=dict)
    # the engine refused the input (its worker is fine and kept)
    input_error: bool = False
    # stopping the worker after a cancel or timeout failed: it still holds the GPU
    unstoppable: bool = False


class EngineError(RuntimeError):
    """The worker could not be started (the message is operator-facing)."""


class EngineUnstoppableError(EngineError):
    """An earlier worker survived SIGKILL and still holds its memory: nothing may start."""


UNSTOPPABLE = "the engine worker could not be stopped (it survived SIGKILL)"


class _Worker:
    """One live worker process and the thread that reads its replies."""

    def __init__(self, proc: subprocess.Popen[bytes], replies: IO[bytes]) -> None:
        self.proc = proc
        self.messages: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._reader = threading.Thread(
            target=self._read, args=(replies,), name="ace-step-api-replies", daemon=True
        )
        self._reader.start()

    def _read(self, replies: IO[bytes]) -> None:
        with replies:
            for raw in replies:
                if len(raw) > MAX_REPLY_BYTES:
                    continue
                with contextlib.suppress(ValueError):
                    message = json.loads(raw)
                    if isinstance(message, dict):
                        self.messages.put(message)
        self.messages.put(None)  # the worker closed its end: it exited

    def send(self, message: Mapping[str, Any]) -> bool:
        stdin = self.proc.stdin
        if stdin is None or self.proc.poll() is not None:
            return False
        try:
            stdin.write((json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8"))
            stdin.flush()
        except (BrokenPipeError, OSError):
            return False
        return True

    def alive(self) -> bool:
        return self.proc.poll() is None


class ResidentEngine:
    """Owns at most one worker. Thread-safe: the queue's worker thread runs tracks, a monitor
    thread stops the process when idle or when another GPU server needs the memory."""

    def __init__(self, config: EngineConfig, *, clock: Callable[[], float] = time.monotonic):
        self.config = config
        self._clock = clock
        self._lock = threading.RLock()
        self._worker: _Worker | None = None
        # a worker that outlived SIGKILL (stuck in the GPU driver) until it is finally gone
        self._stuck: subprocess.Popen[bytes] | None = None
        # the worker has said "ready" (models in memory); read without the lock by health
        self._ready = False
        self._in_use = False
        self._last_used = clock()
        self._peer_busy: PeerCheck | None = None
        self._closed = threading.Event()
        self._monitor: threading.Thread | None = None
        self.last_load_s: float | None = None

    # -- state -------------------------------------------------------------------------
    def set_peer_check(self, check: PeerCheck | None) -> None:
        self._peer_busy = check

    @property
    def pid(self) -> int | None:
        with self._lock:
            return self._worker.proc.pid if self._worker is not None else None

    @property
    def stuck_pid(self) -> int | None:
        stuck = self._stuck
        return stuck.pid if stuck is not None else None

    def running(self) -> bool:
        """A loaded worker is up. Lock-free: the lock is held for a whole model load or stop,
        and health checks must not wait for that."""
        worker = self._worker
        return self._ready and worker is not None and worker.alive()

    # -- use ---------------------------------------------------------------------------
    def acquire(self, job_id: str, cancel: threading.Event, deadline: float) -> bool:
        """Make sure a loaded worker is up and mark it in use. Returns False when the load was
        canceled or ran past ``deadline``; raises :class:`EngineError` when it failed."""
        with self._lock:
            if self._closed.is_set():
                raise EngineError("ace-step-api is stopping")
            if not self._stuck_gone_locked():
                raise EngineUnstoppableError(
                    f"{UNSTOPPABLE}: pid {self.stuck_pid} still holds the GPU; restart the server "
                    "if it does not exit"
                )
            self._in_use = True
            self.last_load_s = None
            if self._worker is not None and self._worker.alive():
                return True
            if self._worker is not None:
                self._stop_locked("the worker exited", job_id=job_id, level=logging.WARNING)
                self._in_use = True
            return self._start_locked(job_id, cancel, deadline)

    def release(self, *, discard: bool = False, job_id: str | None = None) -> None:
        with self._lock:
            self._in_use = False
            self._last_used = self._clock()
            if discard:
                self._stop_locked("the job did not succeed", job_id=job_id)
            elif self.config.idle_seconds <= 0:
                self._stop_locked("resident mode is off", job_id=job_id)

    def stop(self, reason: str) -> bool:
        """Stop the worker (waiting for a stop already under way). True once no worker of this
        engine holds memory; False while one survives SIGKILL."""
        with self._lock:
            return self._stop_locked(reason) and self._stuck_gone_locked()

    def shutdown(self) -> None:
        self._closed.set()
        self.stop("ace-step-api is stopping")

    def reap_leftover(self) -> str | None:
        """Stop a worker a crashed ace-step-api left running (our pid file, our script)."""
        return reap_orphan(self.config.state_dir, self.config.script.name)

    def run_track(
        self,
        job_id: str,
        request: Mapping[str, Any],
        *,
        cancel: threading.Event,
        deadline: float,
        on_progress: ProgressFn | None = None,
    ) -> TrackOutcome:
        """Send one ``generate`` request to the acquired worker and wait for its answer. On
        cancel or timeout the worker is stopped (that is what frees the GPU)."""
        started = time.monotonic()
        with self._lock:
            worker = self._worker
        if worker is None or not worker.send({"cmd": "generate", "id": job_id, **request}):
            return TrackOutcome(False, 0.0, failure="the engine worker is not running")
        while True:
            if cancel.is_set():
                if not self.stop("the job was canceled"):
                    return self._unstoppable(started)
                return TrackOutcome(False, time.monotonic() - started, canceled=True)
            if time.monotonic() > deadline:
                if not self.stop("the job timed out"):
                    return self._unstoppable(started)
                return TrackOutcome(False, time.monotonic() - started, timed_out=True)
            try:
                message = worker.messages.get(timeout=0.5)
            except queue.Empty:
                continue
            seconds = time.monotonic() - started
            if message is None:
                return TrackOutcome(False, seconds, failure="the engine worker exited mid-track")
            kind = message.get("event")
            if kind == "progress" and on_progress is not None:
                value = message.get("value")
                if isinstance(value, (int, float)):
                    with contextlib.suppress(Exception):
                        on_progress(float(value))
            elif kind == "done" and message.get("id") == job_id:
                return TrackOutcome(True, seconds, reply=message)
            elif kind == "error" and message.get("id") == job_id:
                reason = str(message.get("message") or "the engine reported an error")
                return TrackOutcome(
                    False, seconds, failure=reason[:2000], input_error=message.get("input") is True
                )

    @staticmethod
    def _unstoppable(started: float) -> TrackOutcome:
        return TrackOutcome(
            False, time.monotonic() - started, failure=UNSTOPPABLE, unstoppable=True
        )

    # -- process -----------------------------------------------------------------------
    def _start_locked(self, job_id: str, cancel: threading.Event, deadline: float) -> bool:
        config = self.config
        if not os.access(config.python, os.X_OK):
            raise EngineError(f"{config.python} is not an executable Python")
        config.state_dir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        read_fd, write_fd = os.pipe()
        try:
            with (config.state_dir / LOG_FILE).open("ab") as worker_log:
                proc = subprocess.Popen(
                    worker_command(config, write_fd),
                    cwd=config.cwd,
                    stdin=subprocess.PIPE,
                    stdout=worker_log,
                    stderr=subprocess.STDOUT,
                    pass_fds=(write_fd,),
                    start_new_session=True,  # own process group: a stop reaches children too
                    env=child_env({"PYTHONUNBUFFERED": "1", **config.env}),
                )
        except OSError as exc:
            os.close(read_fd)
            raise EngineError(f"cannot start the engine worker: {type(exc).__name__}") from exc
        finally:
            os.close(write_fd)
        (config.state_dir / PID_FILE).write_text(str(proc.pid), encoding="utf-8")
        self._worker = _Worker(proc, os.fdopen(read_fd, "rb"))
        self._ready = False
        event(log, "engine.starting", engine=config.name, job_id=job_id, pid=proc.pid)
        while True:
            if cancel.is_set():
                self._stop_locked("the job was canceled while loading", job_id=job_id)
                return False
            if time.monotonic() > min(deadline, started + config.start_timeout_s):
                self._stop_locked("the models did not load in time", job_id=job_id)
                return False
            try:
                message = self._worker.messages.get(timeout=0.5)
            except queue.Empty:
                continue
            if message is None or message.get("event") == "fatal":
                detail = (message or {}).get("message") or "the worker exited while loading"
                self._stop_locked("load failed", job_id=job_id, level=logging.WARNING)
                raise EngineError(f"the engine could not load its models: {detail}")
            if message.get("event") == "ready":
                self._ready = True
                self.last_load_s = round(time.monotonic() - started, 2)
                self._last_used = self._clock()
                event(
                    log,
                    "engine.ready",
                    engine=config.name,
                    job_id=job_id,
                    pid=proc.pid,
                    load_seconds=self.last_load_s,
                    device=message.get("device"),
                    lm_backend=message.get("lm_backend"),
                )
                self._ensure_monitor()
                return True

    def _stuck_gone_locked(self) -> bool:
        """True unless a worker that survived SIGKILL is still alive."""
        stuck = self._stuck
        if stuck is None:
            return True
        if stuck.poll() is None:
            return False
        self._stuck = None
        pid_file = self.config.state_dir / PID_FILE
        with contextlib.suppress(OSError):
            if pid_file.read_text(encoding="utf-8").strip() == str(stuck.pid):
                pid_file.unlink()
        event(log, "engine.unstuck", engine=self.config.name, pid=stuck.pid)
        return True

    def _stop_locked(
        self, reason: str, *, job_id: str | None = None, level: int = logging.INFO
    ) -> bool:
        """Stop the worker; True when its process group is gone (or there was none)."""
        worker, self._worker = self._worker, None
        self._ready = False
        self._in_use = False
        if worker is None:
            return True
        proc = worker.proc
        if proc.poll() is None:
            worker.send({"cmd": "quit"})
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=2)
        stop_group(proc, self.config.stop_grace_s)
        if proc.stdin is not None:
            with contextlib.suppress(OSError):
                proc.stdin.close()
        gone = proc.poll() is not None
        if gone:
            (self.config.state_dir / PID_FILE).unlink(missing_ok=True)
        else:
            # keep its pid file (a restart reaps it) and refuse every load while it lives
            self._stuck = proc
        event(
            log,
            "engine.stopped" if gone else "engine.unstoppable",
            level=level if gone else logging.CRITICAL,
            engine=self.config.name,
            job_id=job_id,
            pid=proc.pid,
            reason=reason,
        )
        return gone

    # -- monitor -----------------------------------------------------------------------
    def _ensure_monitor(self) -> None:
        if self._monitor is None or not self._monitor.is_alive():
            self._monitor = threading.Thread(
                target=self._watch, name=f"ace-step-api-{self.config.name}", daemon=True
            )
            self._monitor.start()

    def _watch(self) -> None:
        """While the worker idles: stop it after ``idle_seconds``, or at once when another
        GPU server is busy. Never touches a worker that is in use."""
        while not self._closed.wait(self.config.poll_seconds):
            with self._lock:
                if self._worker is None:
                    return
                if self._in_use:
                    continue
                if self._clock() - self._last_used >= self.config.idle_seconds:
                    self._stop_locked("idle")
                    return
            check = self._peer_busy
            busy = False
            if check is not None:
                with contextlib.suppress(Exception):  # a failing check must not kill this
                    busy = check()
            if busy:
                with self._lock:
                    if self._worker is not None and not self._in_use:
                        self._stop_locked("another GPU server is busy")
                        return
