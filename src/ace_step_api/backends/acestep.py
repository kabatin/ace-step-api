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

import os
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from ace_step_api.backends.base import GenerationRequest
from ace_step_api.backends.engine import EngineConfig
from ace_step_api.backends.resident import ResidentBackend, parse_options
from ace_step_api.checks import Check, Level
from ace_step_api.settings import Settings

BACKEND_NAME = "acestep"
WORKER_SCRIPT = Path(__file__).with_name("acestep_worker.py")
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


def acestep_args(
    project_root: Path, dit_model: str, lm_model: str, lm_backend: str, mlx_dit: bool | None
) -> tuple[str, ...]:
    """The ACE-Step worker's own arguments (after ``--proto-fd``)."""
    return (
        "--project-root",
        str(project_root),
        "--dit-model",
        dit_model,
        "--lm-model",
        lm_model,
        "--lm-backend",
        lm_backend,
        *(() if mlx_dit is None else ("--mlx-dit", "on" if mlx_dit else "off")),
    )


class AceStepBackend(ResidentBackend):
    name: ClassVar[str] = BACKEND_NAME
    track_detail_keys: ClassVar[tuple[str, ...]] = ("gain_db", "limiter_db")

    def __init__(
        self,
        options: AceStepOptions | None = None,
        *,
        state_dir: Path,
        poll_seconds: float = 5.0,
    ) -> None:
        self.options = options or AceStepOptions()
        self.state_dir = state_dir
        self.poll_seconds = poll_seconds
        super().__init__(load_seconds=self.options.load_seconds)

    @classmethod
    def from_settings(cls, settings: Settings) -> AceStepBackend:
        return cls(
            parse_options(AceStepOptions, settings, BACKEND_NAME),
            # where 0.1.0 kept it: a restart after the upgrade still reaps a leftover worker
            state_dir=settings.data_dir / "engine",
            poll_seconds=settings.coordination_poll_seconds,
        )

    def engine_config(self) -> EngineConfig:
        o = self.options
        root = self.project_root or Path()
        return EngineConfig(
            name=BACKEND_NAME,
            python=self.python_path,
            script=WORKER_SCRIPT,
            args=acestep_args(root, o.dit_model, o.lm_model, o.lm_backend, o.mlx_dit),
            cwd=root,
            state_dir=self.state_dir,
            idle_seconds=o.resident_idle_seconds,
            poll_seconds=self.poll_seconds,
            start_timeout_s=o.start_timeout_s,
            stop_grace_s=o.stop_grace_s,
            env=self.worker_env(),
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

    def missing_models(self, root: Path) -> list[str]:
        checkpoints = root / "checkpoints"
        return [
            name
            for name in self.required_models()
            if not any((checkpoints / name).glob("*.safetensors"))
        ]

    # -- planning --------------------------------------------------------------------
    def typical_seconds(self, duration_s: float) -> float:
        o = self.options
        return o.track_overhead_seconds + o.seconds_per_audio_second * duration_s

    def track_request(self, request: GenerationRequest, seed: int, out: Path) -> dict[str, Any]:
        return {
            "caption": request.prompt,
            "lyrics": request.lyrics,
            "instrumental": request.instrumental,
            "vocal_language": request.vocal_language,
            "duration": request.duration_s,
            "seed": seed,
            "out": str(out),
        }

    # -- health ----------------------------------------------------------------------
    def unavailable_reason(self) -> str | None:
        root = self.project_root
        if root is None:
            return f"not configured (set project_root in [backends.{BACKEND_NAME}])"
        if not (root / "acestep").is_dir():
            return "the ACE-Step checkout is missing"
        if not os.access(self.python_path, os.X_OK):
            return "the ACE-Step Python is missing (run `uv sync` in the checkout)"
        missing = self.missing_models(root)
        if missing:
            return f"model files missing: {', '.join(missing)}"
        return None

    def checks(self) -> list[Check]:
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
        missing = self.missing_models(root)
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
                "ACE-Step resident engine",
                Level.OK,
                f"models stay loaded {idle:.0f} s after a job" if idle else "off (load per job)",
            )
        )
        return checks
