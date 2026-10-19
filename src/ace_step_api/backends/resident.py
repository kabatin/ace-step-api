"""What every engine backend shares: a resident worker (:mod:`ace_step_api.backends.engine`) in
the engine's own Python, one track per request with ``seed + i``, and the job loop around it.

An engine backend only says how to start its worker (``engine_config``), what it is missing
(``unavailable_reason``, ``checks``), what one track request looks like (``track_request``)
and how long a track takes (``typical_seconds``).
"""

from __future__ import annotations

import logging
import threading
import time
from abc import abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ValidationError

from ace_step_api.backends.base import (
    EngineStatus,
    GenerationPlan,
    GenerationRequest,
    GenerationResult,
    MusicBackend,
    ProgressCallback,
)
from ace_step_api.backends.engine import (
    EngineConfig,
    EngineError,
    EngineUnstoppableError,
    ResidentEngine,
    TrackOutcome,
)
from ace_step_api.checks import Check
from ace_step_api.logs import event
from ace_step_api.settings import Settings, SettingsError

# the job error code for input the engine refuses (as for a body that fails validation)
REJECTED_INPUT = "invalid_params"


class ResidentBackend(MusicBackend):
    supports_cancel: ClassVar[bool] = True  # the worker is killed; the GPU is released
    # values of the worker's ``done`` reply kept per track in the result details
    track_detail_keys: ClassVar[tuple[str, ...]] = ("gain_db",)

    def __init__(self, *, load_seconds: float) -> None:
        self.load_seconds = load_seconds
        self.engine = ResidentEngine(self.engine_config())
        self.log = logging.getLogger(f"ace_step_api.{self.engine.config.name}")

    @abstractmethod
    def engine_config(self) -> EngineConfig:
        """How to start this engine's worker."""

    @abstractmethod
    def unavailable_reason(self) -> str | None:
        """Why the engine cannot take jobs (shown to clients: short, no paths), or None."""

    @abstractmethod
    def checks(self) -> list[Check]:
        """Install, Python and model files (for doctor and the startup log)."""

    @abstractmethod
    def track_request(self, request: GenerationRequest, seed: int, out: Path) -> dict[str, Any]:
        """The worker's ``generate`` request for one track."""

    def _checked_reason(self) -> str | None:
        """``unavailable_reason``, with a file system error (permissions, a file deleted while
        checked) making the engine unavailable instead of failing the health check."""
        try:
            return self.unavailable_reason()
        except OSError as exc:
            return f"cannot check its install ({type(exc).__name__})"

    def engine_status(self, engine: str) -> EngineStatus:
        reason = self._checked_reason()
        return EngineStatus(available=reason is None, loaded=self.engine.running(), reason=reason)

    def health(self) -> list[Check]:
        return self.checks()

    # -- planning --------------------------------------------------------------------
    def plan(self, request: GenerationRequest) -> GenerationPlan:
        load = 0.0 if self.engine.running() else self.load_seconds
        estimate = load + self.typical_seconds(request.duration_s) * request.count
        return GenerationPlan(request.count, estimate_seconds=round(estimate, 1))

    # -- generation ------------------------------------------------------------------
    def generate(
        self,
        request: GenerationRequest,
        run_dir: Path,
        *,
        timeout_s: float,
        cancel: threading.Event,
        on_progress: ProgressCallback | None = None,
    ) -> GenerationResult:
        plan = self.plan(request)
        started = time.monotonic()
        deadline = started + timeout_s
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return GenerationResult(plan=plan, duration_s=0.0, failure=f"cannot prepare: {exc}")
        reason = self._checked_reason()
        if reason is not None:
            return GenerationResult(
                plan=plan, duration_s=0.0, failure=reason, failure_code="backend_unavailable"
            )
        result: GenerationResult | None = None
        try:
            result = self._run(request, plan, run_dir, started, deadline, cancel, on_progress)
            return result
        finally:
            # a canceled or failed track may leave the GPU busy: never reuse that worker; one
            # that refused the input finished cleanly and is kept
            keep = result is not None and (result.ok or result.failure_code == REJECTED_INPUT)
            self.engine.release(discard=not keep, job_id=request.job_id)

    def _run(
        self,
        request: GenerationRequest,
        plan: GenerationPlan,
        run_dir: Path,
        started: float,
        deadline: float,
        cancel: threading.Event,
        on_progress: ProgressCallback | None,
    ) -> GenerationResult:
        def elapsed() -> float:
            return time.monotonic() - started

        try:
            loaded = self.engine.acquire(request.job_id, cancel, deadline)
        except EngineError as exc:
            code = (
                "engine_unstoppable" if isinstance(exc, EngineUnstoppableError) else "engine_failed"
            )
            return GenerationResult(
                plan=plan, duration_s=elapsed(), failure=str(exc), failure_code=code
            )
        if not loaded:
            return GenerationResult(
                plan=plan,
                duration_s=elapsed(),
                canceled=cancel.is_set(),
                timed_out=not cancel.is_set(),
            )
        details: dict[str, Any] = {"load_seconds": self.engine.last_load_s, "tracks": []}
        outputs: list[Path] = []
        for index in range(request.count):
            out = run_dir / f"raw-{index}.wav"
            outcome = self.engine.run_track(
                request.job_id,
                self.track_request(request, request.seed + index, out.resolve()),
                cancel=cancel,
                deadline=deadline,
                on_progress=_scaled(on_progress, index, request.count),
            )
            details["tracks"].append(
                {
                    "index": index,
                    "seconds": round(outcome.seconds, 2),
                    **{key: outcome.reply.get(key) for key in self.track_detail_keys},
                }
            )
            if not outcome.ok:
                return GenerationResult(
                    plan=plan,
                    duration_s=elapsed(),
                    outputs=tuple(outputs),
                    canceled=outcome.canceled,
                    timed_out=outcome.timed_out,
                    failure=_failure(outcome),
                    failure_code=_failure_code(outcome),
                    retryable=not outcome.input_error,
                    details=details,
                )
            if not out.is_file() or out.stat().st_size == 0:
                return GenerationResult(
                    plan=plan,
                    duration_s=elapsed(),
                    outputs=tuple(outputs),
                    failure=f"track {index}: the engine wrote no audio",
                    details=details,
                )
            outputs.append(out)
            event(
                self.log,
                "engine.track",
                engine=self.engine.config.name,
                job_id=request.job_id,
                index=index,
                seconds=round(outcome.seconds, 2),
            )
        return GenerationResult(
            plan=plan, duration_s=elapsed(), outputs=tuple(outputs), details=details
        )

    # -- lifecycle -------------------------------------------------------------------
    def unload(self, reason: str) -> bool:
        # always through the lock: a stop already under way (idle, busy peer) holds it until the
        # process is gone, and the next engine must not load before that
        return self.engine.stop(reason)

    def startup(self) -> str | None:
        return self.engine.reap_leftover()

    def shutdown(self) -> None:
        self.engine.shutdown()

    def set_peer_check(self, check: Callable[[], bool] | None) -> None:
        self.engine.set_peer_check(check)


def _failure(outcome: TrackOutcome) -> str | None:
    if outcome.input_error:
        return f"the engine rejected the input: {outcome.failure}"
    return outcome.failure


def _failure_code(outcome: TrackOutcome) -> str:
    if outcome.input_error:
        return REJECTED_INPUT
    return "engine_unstoppable" if outcome.unstoppable else "generation_failed"


def _scaled(
    on_progress: ProgressCallback | None, index: int, count: int
) -> Callable[[float], None] | None:
    if on_progress is None:
        return None

    def report(fraction: float) -> None:
        on_progress(min(max((index + fraction) / count, 0.0), 1.0))

    return report


def parse_options[T: BaseModel](model: type[T], settings: Settings, name: str) -> T:
    """Validate the ``[backends.<name>]`` table of the config file."""
    try:
        return model.model_validate(dict(settings.backend_options(name)))
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise SettingsError(f"invalid [backends.{name}] options: {problems}") from exc
