"""The backend contract: how a music engine is driven.

A backend receives a backend-neutral :class:`GenerationRequest` and returns raw WAV files. It
chooses everything engine-specific — model files, steps, shift, language model — itself. It
never sees HTTP, client schemas or client-chosen paths, and its own output (logs, error text)
never reaches clients unredacted. Swapping the backend must not change the HTTP contract.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from ace_step_api.checks import Check

# Backend-reported progress over the whole job, 0.0..1.0.
ProgressCallback = Callable[[float], None]


@dataclass(frozen=True)
class GenerationRequest:
    job_id: str
    # which engine makes the tracks ("acestep", "minimax"): set by the workflow
    engine: str
    # style tags / description of the music
    prompt: str
    # structured lyrics ([verse], [chorus], ...); empty when instrumental
    lyrics: str
    instrumental: bool
    duration_s: float
    # track i uses seed + i
    seed: int
    count: int
    # ISO 639-1 code of the sung language, or "unknown" (the engine then decides)
    vocal_language: str = "unknown"


@dataclass(frozen=True)
class EngineStatus:
    # installed with its model files present: jobs for it are accepted
    available: bool
    # its worker is up with the models in memory (false while idle: loaded on the next job)
    loaded: bool = False
    # why it is not available (shown to clients: short, no paths)
    reason: str | None = None


@dataclass(frozen=True)
class GenerationPlan:
    count: int
    estimate_seconds: float


@dataclass(frozen=True)
class GenerationResult:
    plan: GenerationPlan
    duration_s: float
    # one fresh, non-empty WAV per track, in order; complete only when every track succeeded
    outputs: tuple[Path, ...] = ()
    canceled: bool = False
    timed_out: bool = False
    # operator-facing reason (may contain paths: the queue redacts it before storing)
    failure: str | None = None
    failure_code: str = "generation_failed"
    # False when trying again cannot help (the engine refused the input)
    retryable: bool = True
    # engine timings (load, per track), JSON-safe
    details: dict[str, Any] = field(default_factory=dict)
    log_path: Path | None = None

    @property
    def ok(self) -> bool:
        return (
            len(self.outputs) == self.plan.count
            and not self.canceled
            and not self.timed_out
            and self.failure is None
        )

    def describe_failure(self) -> str:
        if self.canceled:
            return "canceled"
        if self.timed_out:
            return f"timed out after {self.duration_s:.0f}s"
        return self.failure or "no output written"


class MusicBackend(ABC):
    name: ClassVar[str]
    # True when ``cancel`` really stops the engine (frees the GPU), not just the bookkeeping.
    supports_cancel: ClassVar[bool]

    @abstractmethod
    def typical_seconds(self, duration_s: float) -> float:
        """Expected seconds to make one track of ``duration_s`` with the models loaded."""

    @abstractmethod
    def plan(self, request: GenerationRequest) -> GenerationPlan:
        """How long the whole job should take (all tracks, a model load included)."""

    @abstractmethod
    def health(self) -> list[Check]:
        """Can this backend generate right now (engine installed, model files present)?"""

    @abstractmethod
    def generate(
        self,
        request: GenerationRequest,
        run_dir: Path,
        *,
        timeout_s: float,
        cancel: threading.Event,
        on_progress: ProgressCallback | None = None,
    ) -> GenerationResult:
        """Run one job (all tracks). Must return (not raise) for engine failures, cancel,
        timeout."""

    def typical_seconds_for(self, engine: str, duration_s: float) -> float:
        """``typical_seconds`` of one engine (backends with several engines pick it)."""
        return self.typical_seconds(duration_s)

    def engine_status(self, engine: str) -> EngineStatus:
        """Whether ``engine`` can take jobs now and whether its models are in memory."""
        return EngineStatus(available=True)

    def unload(self, reason: str) -> bool:
        """Give back the memory of models kept loaded between jobs (no job is running).
        False when a worker could not be stopped and still holds its memory."""
        return True

    def recover(self, run_dir: Path) -> str | None:
        """Clean up after a run the server lost track of (restart). Returns what was done."""
        return None

    def startup(self) -> str | None:
        """Called once when the queue starts: clean up what a crashed server left (beyond
        single runs). Returns what was done."""
        return None

    def shutdown(self) -> None:
        """Called when the queue stops: release long-lived engine resources."""
        return None

    def set_peer_check(self, check: Callable[[], bool] | None) -> None:
        """``check()`` is True while another GPU server on the machine is busy (backends that
        hold GPU memory between jobs give it back then)."""
        return None
