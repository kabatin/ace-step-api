"""ACE-Step 1.5 backend: a resident worker (:mod:`ace_step_api.backends.engine`) running the
ACE-Step Python API from an ACE-Step checkout, one track per request with ``seed + i``.

Facts this module relies on (ACE-Step 1.5 on the target Mac; both the official checkout and the
Apple Silicon port expose the same Python API):
- the turbo DiT with 8 steps and shift 3.0 (ACE-Step's own recommendation), and the 5 Hz
  language model planning the song first ("thinking");
- the checkout's ``checkpoints/`` holds ``acestep-v15-turbo``, ``vae``,
  ``Qwen3-Embedding-0.6B`` and the language model; the worker never downloads
  (``HF_HUB_OFFLINE=1``);
- output is 48 kHz stereo; the worker writes it as 16-bit PCM WAV.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ace_step_api.backends.base import (
    GenerationPlan,
    GenerationRequest,
    GenerationResult,
    MusicBackend,
    ProgressCallback,
)
from ace_step_api.backends.engine import EngineConfig, EngineError, ResidentEngine
from ace_step_api.checks import Check, Level
from ace_step_api.logs import event
from ace_step_api.settings import Settings, SettingsError

log = logging.getLogger("ace_step_api.acestep")

BACKEND_NAME = "acestep"
MODEL_FILE = "model.safetensors"
# measured on the target Mac (M5, 32 GB, the Apple Silicon port): loads took 19-38 s, a 60 s
# track 32-36 s and a 240 s track 121 s once loaded (README "Measurements"); kept conservative
LOAD_SECONDS = 40.0
TRACK_OVERHEAD_SECONDS = 10.0
SECONDS_PER_AUDIO_SECOND = 0.5


class AceStepOptions(BaseModel):
    """``[backends.acestep]`` in the config file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # an ACE-Step 1.5 checkout with its own virtual environment and checkpoints/
    project_root: Path | None = None
    python: Path | None = None  # default: <project_root>/.venv/bin/python
    dit_model: str = Field(default="acestep-v15-turbo", pattern=r"^[A-Za-z0-9._-]{1,128}$")
    lm_model: str = Field(default="acestep-5Hz-lm-1.7B", pattern=r"^[A-Za-z0-9._-]{1,128}$")
    lm_backend: str = Field(default="vllm", pattern=r"^(vllm|pt|mlx)$")
    # DiT and VAE on MLX (true) or PyTorch (false); unset: the engine's default
    mlx_dit: bool | None = None
    # HF_HOME for the worker (Hugging Face caches, code downloaded with remote models)
    hf_home: Path | None = None
    # keep the models loaded between jobs this long (0: load for every job)
    resident_idle_seconds: float = Field(default=120.0, ge=0.0, le=3600.0)
    start_timeout_s: float = Field(default=600.0, ge=10.0, le=3600.0)
    stop_grace_s: float = Field(default=15.0, ge=1.0, le=120.0)
    load_seconds: float = Field(default=LOAD_SECONDS, gt=0.0, le=3600.0)
    track_overhead_seconds: float = Field(default=TRACK_OVERHEAD_SECONDS, ge=0.0, le=3600.0)
    seconds_per_audio_second: float = Field(default=SECONDS_PER_AUDIO_SECOND, gt=0.0, le=60.0)


class AceStepBackend(MusicBackend):
    name: ClassVar[str] = BACKEND_NAME
    supports_cancel: ClassVar[bool] = True  # the worker is killed; the GPU is released

    def __init__(
        self,
        options: AceStepOptions | None = None,
        *,
        state_dir: Path,
        poll_seconds: float = 5.0,
    ) -> None:
        self.options = options or AceStepOptions()
        self.engine = ResidentEngine(
            EngineConfig(
                python=self.python_path,
                project_root=self.project_root or Path(),
                dit_model=self.options.dit_model,
                lm_model=self.options.lm_model,
                lm_backend=self.options.lm_backend,
                mlx_dit=self.options.mlx_dit,
                state_dir=state_dir,
                idle_seconds=self.options.resident_idle_seconds,
                poll_seconds=poll_seconds,
                start_timeout_s=self.options.start_timeout_s,
                stop_grace_s=self.options.stop_grace_s,
                env=self.worker_env(),
            )
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> AceStepBackend:
        try:
            options = AceStepOptions.model_validate(dict(settings.backend_options(BACKEND_NAME)))
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
            )
            raise SettingsError(f"invalid [backends.{BACKEND_NAME}] options: {problems}") from exc
        return cls(
            options,
            state_dir=settings.data_dir / "engine",
            poll_seconds=settings.coordination_poll_seconds,
        )

    # -- paths -----------------------------------------------------------------------
    @property
    def project_root(self) -> Path | None:
        root = self.options.project_root
        return root.expanduser() if root is not None else None

    @property
    def python_path(self) -> Path:
        if self.options.python is not None:
            return self.options.python.expanduser()
        return (self.project_root or Path()) / ".venv" / "bin" / "python"

    def worker_env(self) -> dict[str, str]:
        env = {
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "GRADIO_ANALYTICS_ENABLED": "False",
            "ACESTEP_DOWNLOAD_SOURCE": "huggingface",
            "TOKENIZERS_PARALLELISM": "false",
        }
        if self.options.hf_home is not None:
            env["HF_HOME"] = str(self.options.hf_home.expanduser())
        return env

    def required_models(self) -> list[str]:
        return [self.options.dit_model, "vae", "Qwen3-Embedding-0.6B", self.options.lm_model]

    # -- planning --------------------------------------------------------------------
    def typical_seconds(self, duration_s: float) -> float:
        o = self.options
        return o.track_overhead_seconds + o.seconds_per_audio_second * duration_s

    def plan(self, request: GenerationRequest) -> GenerationPlan:
        load = 0.0 if self.engine.running() else self.options.load_seconds
        estimate = load + self.typical_seconds(request.duration_s) * request.count
        return GenerationPlan(request.count, estimate_seconds=round(estimate, 1))

    # -- health ----------------------------------------------------------------------
    def health(self) -> list[Check]:
        root = self.project_root
        if root is None:
            return [
                Check(
                    "ACE-Step checkout",
                    Level.FAIL,
                    f"set project_root in [backends.{BACKEND_NAME}] (an ACE-Step 1.5 checkout)",
                )
            ]
        checks = [
            Check(
                "ACE-Step checkout",
                Level.OK if (root / "acestep").is_dir() else Level.FAIL,
                str(root) if (root / "acestep").is_dir() else f"{root}/acestep not found",
            ),
            Check(
                "ACE-Step Python",
                Level.OK if os.access(self.python_path, os.X_OK) else Level.FAIL,
                str(self.python_path)
                if os.access(self.python_path, os.X_OK)
                else f"{self.python_path} not found (run `uv sync` in the checkout)",
            ),
        ]
        checkpoints = root / "checkpoints"
        missing = [
            name
            for name in self.required_models()
            if not any((checkpoints / name).glob("*.safetensors"))
        ]
        checks.append(
            Check(
                "ACE-Step models",
                Level.FAIL if missing else Level.OK,
                f"missing in {checkpoints}: {', '.join(missing)}"
                if missing
                else f"{self.options.dit_model} + {self.options.lm_model} (+ vae, text encoder)",
            )
        )
        idle = self.options.resident_idle_seconds
        checks.append(
            Check(
                "resident engine",
                Level.OK,
                f"models stay loaded {idle:.0f} s after a job" if idle else "off (load per job)",
            )
        )
        return checks

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
        if self.project_root is None:
            return GenerationResult(
                plan=plan,
                duration_s=0.0,
                failure="no ACE-Step checkout configured",
                failure_code="backend_unavailable",
            )
        result: GenerationResult | None = None
        try:
            result = self._run(request, plan, run_dir, started, deadline, cancel, on_progress)
            return result
        finally:
            # a canceled or failed track may leave the GPU busy: never reuse that worker
            ok = result is not None and result.ok
            self.engine.release(discard=not ok, job_id=request.job_id)

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
            return GenerationResult(
                plan=plan, duration_s=elapsed(), failure=str(exc), failure_code="engine_failed"
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
                {
                    "caption": request.prompt,
                    "lyrics": request.lyrics,
                    "instrumental": request.instrumental,
                    "vocal_language": request.vocal_language,
                    "duration": request.duration_s,
                    "seed": request.seed + index,
                    "out": str(out.resolve()),
                },
                cancel=cancel,
                deadline=deadline,
                on_progress=_scaled(on_progress, index, request.count),
            )
            details["tracks"].append(
                {
                    "index": index,
                    "seconds": round(outcome.seconds, 2),
                    "gain_db": outcome.reply.get("gain_db"),
                    "limiter_db": outcome.reply.get("limiter_db"),
                }
            )
            if not outcome.ok:
                return GenerationResult(
                    plan=plan,
                    duration_s=elapsed(),
                    outputs=tuple(outputs),
                    canceled=outcome.canceled,
                    timed_out=outcome.timed_out,
                    failure=outcome.failure,
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
                log,
                "engine.track",
                job_id=request.job_id,
                index=index,
                seconds=round(outcome.seconds, 2),
            )
        return GenerationResult(
            plan=plan, duration_s=elapsed(), outputs=tuple(outputs), details=details
        )

    def startup(self) -> str | None:
        return self.engine.reap_leftover()

    def shutdown(self) -> None:
        self.engine.shutdown()

    def set_peer_check(self, check: Callable[[], bool] | None) -> None:
        self.engine.set_peer_check(check)


def _scaled(
    on_progress: ProgressCallback | None, index: int, count: int
) -> Callable[[float], None] | None:
    if on_progress is None:
        return None

    def report(fraction: float) -> None:
        on_progress(min(max((index + fraction) / count, 0.0), 1.0))

    return report
