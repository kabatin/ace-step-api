"""Component ``minimax-music3``: a Python 3.12 venv with mlx-audio and the MiniMax Music 3
weights (``mlx-community/MiniMax-Music3-mxfp8``, ~13 GB) in a Hugging Face cache.

The venv goes to ``<prefix>/minimax-music3/.venv`` unless the config names a ``python``; the
weights go to the backend's ``hf_home`` (default ``~/.local/share/minimax-music3/hf``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ace_step_api.backends.minimax import (
    DEFAULT_MODEL,
    MiniMaxBackend,
    MiniMaxOptions,
)
from ace_step_api.setup import pins
from ace_step_api.setup.component import Context, Plan, missing_keys
from ace_step_api.setup.tools import SetupError
from ace_step_api.setup.watchdog import dir_size, run_watched


class MiniMaxComponent:
    id = "minimax-music3"
    backend = "minimax"
    licence = pins.MINIMAX_LICENCE

    def _options(self, ctx: Context) -> tuple[MiniMaxOptions, MiniMaxOptions]:
        raw = ctx.backends.get(self.backend, {})
        try:
            current = MiniMaxOptions.model_validate(raw)
        except ValidationError as exc:
            raise SetupError(
                f"[backends.{self.backend}] in the config is invalid",
                hint=f"fix it: {exc.errors()[0]['msg']}",
            ) from exc
        venv = ctx.prefix / pins.MINIMAX_DIR / ".venv"
        python = current.python if "python" in raw else venv / "bin" / "python"
        return current, current.model_copy(update={"python": python})

    def plan(self, ctx: Context) -> Plan:
        current, target = self._options(ctx)
        if _backend(current).unavailable_reason() is None:
            return Plan(working=True, reason=None)
        if target.model != DEFAULT_MODEL:
            return Plan(
                working=False, reason=f"setup installs {DEFAULT_MODEL} only", unsupported=True
            )
        wanted: dict[str, Any] = {
            "python": str(target.python.expanduser()),
            "hf_home": str(target.hf_home.expanduser()),
        }
        config = missing_keys(ctx.backends.get(self.backend, {}), wanted)
        backend = _backend(target)
        problem = backend.unavailable_reason()
        if problem is None:
            return Plan(working=True, reason=None, config=config)
        download = disk = 0
        if backend._runtime_problem() is not None:
            download += pins.MINIMAX_ENV_BYTES
            disk += pins.MINIMAX_ENV_BYTES
        if backend._model_problem() is not None:
            download += max(0, pins.MINIMAX_MODEL_BYTES - dir_size(backend.model_cache))
        return Plan(
            working=False,
            reason=problem,
            download_bytes=download,
            disk_bytes=download + disk,
            config=config,
        )

    def install(self, ctx: Context) -> None:
        _, target = self._options(ctx)
        tools = ctx.tools
        if not tools.has_tool("uv"):
            raise SetupError("uv not found", hint="install it: brew install uv")
        backend = _backend(target)
        python = target.python.expanduser()
        venv = python.parent.parent
        if not (venv / "pyvenv.cfg").is_file():
            if python.exists():
                raise SetupError(
                    f"{python} is not in a virtual environment",
                    hint="point [backends.minimax] python at a venv, or remove the key",
                )
            tools.emit(f"minimax-music3: creating {venv} (Python {pins.MINIMAX_PYTHON_VERSION})")
            cmd = ["uv", "venv", "--python", pins.MINIMAX_PYTHON_VERSION, str(venv)]
            if tools.run(cmd, None, None) != 0:
                raise SetupError("uv venv for MiniMax Music 3 failed; see the output above")
        if backend._runtime_problem() is not None:
            tools.emit(f"minimax-music3: installing {pins.MINIMAX_MLX_AUDIO}")
            cmd = [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                "--exclude-newer",
                pins.MINIMAX_EXCLUDE_NEWER,
                pins.MINIMAX_MLX_AUDIO,
            ]
            if tools.run(cmd, None, None) != 0:
                raise SetupError("installing mlx-audio failed; see the output above")
        if backend._model_problem() is not None:
            self._download(backend, venv, ctx)
        problem = backend.unavailable_reason()
        if problem is not None:
            raise SetupError(f"MiniMax Music 3 still not usable after setup: {problem}")

    def _download(self, backend: MiniMaxBackend, venv: Path, ctx: Context) -> None:
        hf = venv / "bin" / "hf"
        if not hf.is_file():
            raise SetupError(f"{hf} not found: mlx-audio did not bring huggingface_hub's CLI")
        ctx.tools.emit(f"minimax-music3: downloading {pins.MINIMAX_MODEL_REPO} (~13 GB)")
        backend.model_cache.mkdir(parents=True, exist_ok=True)
        cmd = [str(hf), "download", pins.MINIMAX_MODEL_REPO, "--revision"]
        run_watched(
            "minimax-music3",
            [*cmd, pins.MINIMAX_MODEL_REVISION],
            backend.model_cache,
            ctx.tools,
            env={
                "HF_HOME": str(backend.hf_home),
                "HF_HUB_CACHE": str(backend.hf_home / "hub"),
                "HF_HUB_DISABLE_XET": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
            },
        )
        # a download by commit writes no refs/main, and the server finds the snapshot by it
        if backend._model_problem() is not None:
            ref = backend.model_cache / "refs" / "main"
            ref.parent.mkdir(parents=True, exist_ok=True)
            ref.write_text(pins.MINIMAX_MODEL_REVISION, encoding="utf-8")


def _backend(options: MiniMaxOptions) -> MiniMaxBackend:
    return MiniMaxBackend(options, state_dir=Path("/nonexistent"))
