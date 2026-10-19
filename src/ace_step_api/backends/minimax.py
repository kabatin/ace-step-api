"""MiniMax Music 3 backend: a resident worker (:mod:`ace_step_api.backends.minimax_worker`) in
the Python of a virtual environment with mlx-audio, one track per request with ``seed + i``.

Facts this module relies on (mlx-audio 0.5.9, MLX 0.32, ``mlx-community/MiniMax-Music3-mxfp8``,
measured outside this server on the target Mac, M5 32 GB):
- ``mlx_audio.music.load(<repo>)`` reads the weights from the Hugging Face cache in ``HF_HOME``
  (offline); the model needs ``config.json``, ``tokenizer/`` and every weight shard (without the
  tokenizer it silently falls back to a toy text encoding);
- lyrics are required (``[instrumental]`` for none); 30 steps is the default and the maximum;
  the duration is an upper bound (0 < d <= 360) and the model may end early (62 s asked, 22.9 s
  made); caption + lyrics may not exceed 5000 tokens;
- output is 44.1 kHz stereo float, clipped to [-1, 1];
- ~8.3-9.4 s of compute per second of audio (15 s: 125 s, 62 s: 584 s); ``load`` returns in
  ~2.7 s, the first generation pays for paging the weights in; peak memory ~24 GB.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from ace_step_api.backends.base import GenerationRequest
from ace_step_api.backends.engine import EngineConfig
from ace_step_api.backends.resident import ResidentBackend, parse_options
from ace_step_api.checks import Check, Level
from ace_step_api.settings import Settings

BACKEND_NAME = "minimax"
WORKER_SCRIPT = Path(__file__).with_name("minimax_worker.py")
DEFAULT_PYTHON = Path("~/.local/opt/minimax-music3/.venv/bin/python")
DEFAULT_HF_HOME = Path("~/.local/share/minimax-music3/hf")
DEFAULT_MODEL = "mlx-community/MiniMax-Music3-mxfp8"
# what mlx-audio needs to run the model (relative to the snapshot)
MODEL_PACKAGE = Path("mlx_audio") / "music" / "models" / "minimax_music3" / "__init__.py"
CONFIG_FILE = "config.json"
TOKENIZER_FILE = "tokenizer/tokenizer.json"
WEIGHT_INDEX = "model.safetensors.index.json"
# load() returns in ~2.7 s, but the first track pages ~14 GB of weights in: kept generous
LOAD_SECONDS = 30.0
TRACK_OVERHEAD_SECONDS = 10.0
# measured 8.3 (15 s track) to 9.4 (62 s track) s of compute per second of audio
SECONDS_PER_AUDIO_SECOND = 9.5


class MiniMaxOptions(BaseModel):
    """``[backends.minimax]`` in the config file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # the Python of a virtual environment with mlx-audio (MiniMax Music 3 support)
    python: Path = DEFAULT_PYTHON
    # HF_HOME the weights were downloaded into (the worker never downloads)
    hf_home: Path = DEFAULT_HF_HOME
    # a Hugging Face repo id in that cache
    model: str = Field(
        default=DEFAULT_MODEL,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$",
    )
    # keep the model loaded between jobs this long (0: load for every job)
    resident_idle_seconds: float = Field(default=120.0, ge=0.0, le=3600.0)
    start_timeout_s: float = Field(default=300.0, ge=10.0, le=3600.0)
    stop_grace_s: float = Field(default=15.0, ge=1.0, le=120.0)
    load_seconds: float = Field(default=LOAD_SECONDS, gt=0.0, le=3600.0)
    track_overhead_seconds: float = Field(default=TRACK_OVERHEAD_SECONDS, ge=0.0, le=3600.0)
    seconds_per_audio_second: float = Field(default=SECONDS_PER_AUDIO_SECOND, gt=0.0, le=60.0)


class MiniMaxBackend(ResidentBackend):
    name: ClassVar[str] = BACKEND_NAME
    track_detail_keys: ClassVar[tuple[str, ...]] = ("gain_db",)

    def __init__(
        self,
        options: MiniMaxOptions | None = None,
        *,
        state_dir: Path,
        poll_seconds: float = 5.0,
    ) -> None:
        self.options = options or MiniMaxOptions()
        self.state_dir = state_dir
        self.poll_seconds = poll_seconds
        super().__init__(load_seconds=self.options.load_seconds)

    @classmethod
    def from_settings(cls, settings: Settings) -> MiniMaxBackend:
        return cls(
            parse_options(MiniMaxOptions, settings, BACKEND_NAME),
            state_dir=settings.data_dir / "engine" / BACKEND_NAME,
            poll_seconds=settings.coordination_poll_seconds,
        )

    def engine_config(self) -> EngineConfig:
        o = self.options
        return EngineConfig(
            name=BACKEND_NAME,
            python=self.python_path,
            script=WORKER_SCRIPT,
            args=("--model", o.model),
            # nothing is read from the working directory: the worker's own state dir
            cwd=self.state_dir,
            state_dir=self.state_dir,
            idle_seconds=o.resident_idle_seconds,
            poll_seconds=self.poll_seconds,
            start_timeout_s=o.start_timeout_s,
            stop_grace_s=o.stop_grace_s,
            env=self.worker_env(),
        )

    # -- paths -----------------------------------------------------------------------
    @property
    def python_path(self) -> Path:
        return self.options.python.expanduser()

    @property
    def hf_home(self) -> Path:
        return self.options.hf_home.expanduser()

    @property
    def model_cache(self) -> Path:
        return self.hf_home / "hub" / ("models--" + self.options.model.replace("/", "--"))

    def worker_env(self) -> dict[str, str]:
        return {
            "HF_HOME": str(self.hf_home),
            # set explicitly: an HF_HUB_CACHE in the server's environment must not win
            "HF_HUB_CACHE": str(self.hf_home / "hub"),
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }

    # -- planning --------------------------------------------------------------------
    def typical_seconds(self, duration_s: float) -> float:
        o = self.options
        return o.track_overhead_seconds + o.seconds_per_audio_second * duration_s

    def track_request(self, request: GenerationRequest, seed: int, out: Path) -> dict[str, Any]:
        return {
            "caption": request.prompt,
            "lyrics": request.lyrics,
            "instrumental": request.instrumental,
            "duration": request.duration_s,
            "seed": seed,
            # what the worker's progress estimate counts against
            "expected_s": round(self.typical_seconds(request.duration_s), 1),
            "out": str(out),
        }

    # -- health ----------------------------------------------------------------------
    def _site_packages(self) -> list[Path] | None:
        """The interpreter's venv site-packages, or None when it is not in a venv (then
        mlx-audio cannot be checked without running it)."""
        venv = self.python_path.parent.parent  # not resolved: a venv python is a symlink
        if not (venv / "pyvenv.cfg").is_file():
            return None
        return sorted(venv.glob("lib/python3*/site-packages"))

    def _runtime_problem(self) -> str | None:
        if not os.access(self.python_path, os.X_OK):
            return "the MiniMax Music 3 runtime is not installed (no Python)"
        site = self._site_packages()
        if site is not None and not any((s / MODEL_PACKAGE).is_file() for s in site):
            return "mlx-audio with MiniMax Music 3 is not installed in its Python"
        return None

    def _snapshot(self) -> Path | None:
        try:
            commit = (self.model_cache / "refs" / "main").read_text(encoding="utf-8").strip()
        except OSError:
            return None
        snapshot = self.model_cache / "snapshots" / commit
        return snapshot if commit and "/" not in commit and snapshot.is_dir() else None

    def _model_problem(self) -> str | None:
        snapshot = self._snapshot()
        if snapshot is None:
            return f"{self.options.model} is not downloaded"
        required = (CONFIG_FILE, TOKENIZER_FILE)
        missing = [name for name in required if not (snapshot / name).is_file()]
        index = snapshot / WEIGHT_INDEX
        if index.is_file():
            shards = _shards(index, index.stat().st_mtime_ns)
            if shards is None:
                missing.append(WEIGHT_INDEX)
            else:
                missing += [name for name in shards if not (snapshot / name).is_file()]
        elif not any(snapshot.glob("*.safetensors")):
            missing.append("*.safetensors")
        return f"model files missing: {', '.join(missing)}" if missing else None

    def unavailable_reason(self) -> str | None:
        return self._runtime_problem() or self._model_problem()

    def checks(self) -> list[Check]:
        python = self.python_path
        runtime = self._runtime_problem()
        checks = [
            Check(
                "MiniMax Python",
                Level.OK if os.access(python, os.X_OK) else Level.FAIL,
                str(python) if os.access(python, os.X_OK) else f"{python} not found",
            )
        ]
        site = self._site_packages()
        if site is None:
            checks.append(
                Check("MiniMax mlx-audio", Level.WARN, "not checked: its Python is not in a venv")
            )
        else:
            installed = runtime is None or not runtime.startswith("mlx-audio")
            checks.append(
                Check(
                    "MiniMax mlx-audio",
                    Level.OK if installed else Level.FAIL,
                    "MiniMax Music 3 support found"
                    if installed
                    else f"{runtime} ({python.parent.parent})",
                )
            )
        model = self._model_problem()
        checks.append(
            Check(
                "MiniMax Music 3 model",
                Level.FAIL if model else Level.OK,
                f"{model or self.options.model} in {self.hf_home}",
            )
        )
        idle = self.options.resident_idle_seconds
        checks.append(
            Check(
                "MiniMax resident engine",
                Level.OK,
                f"models stay loaded {idle:.0f} s after a job" if idle else "off (load per job)",
            )
        )
        return checks


@lru_cache(maxsize=8)
def _shards(index: Path, mtime_ns: int) -> tuple[str, ...] | None:
    """The weight files a sharded checkpoint's index names (cached: health reads it often)."""
    try:
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        names = {str(name) for name in weight_map.values()}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    # only plain file names next to the index: anything else is not a checkpoint we can trust
    if not names or any(Path(name).name != name or name in (".", "..") for name in names):
        return None
    return tuple(sorted(names))
