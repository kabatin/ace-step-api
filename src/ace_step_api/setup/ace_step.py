"""Component ``ace-step``: the ACE-Step 1.5 Apple Silicon port, its environment and models.

Clones the pinned commit under ``--prefix``, ``uv sync`` in it, then fetches
``ACE-Step/Ace-Step1.5`` at a pinned revision into ``checkpoints/`` (~10 GB) with the checkout's own
huggingface_hub: the same ``snapshot_download(local_dir=...)`` as its ``download_main_model``,
which only knows main (and then finds the files complete). A path the
config file already names (``project_root``, ``python``, ``hf_home``) is used instead of the
default under the prefix: setup never rewrites a key that is there.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ace_step_api.backends.acestep import AceStepBackend, AceStepOptions
from ace_step_api.setup import pins
from ace_step_api.setup.component import Context, Plan, missing_keys
from ace_step_api.setup.tools import SetupError
from ace_step_api.setup.watchdog import dir_size, run_watched

# argv: repo, revision, checkpoints dir; Hugging Face resumes the .incomplete files it left
DOWNLOAD_SCRIPT = (
    "import sys\n"
    "from huggingface_hub import snapshot_download\n"
    "snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])\n"
)
UNTOUCHED = "using the existing checkout {root} at {head} (setup does not change its version)"


class AceStepComponent:
    id = "ace-step"
    backend = "acestep"
    licence = pins.ACE_STEP_LICENCE

    def _options(self, ctx: Context) -> tuple[AceStepOptions, AceStepOptions]:
        """(what the config says now, where setup installs)."""
        raw = ctx.backends.get(self.backend, {})
        try:
            current = AceStepOptions.model_validate(raw)
        except ValidationError as exc:
            raise SetupError(
                f"[backends.{self.backend}] in the config is invalid",
                hint=f"fix it: {exc.errors()[0]['msg']}",
            ) from exc
        root = current.project_root or ctx.prefix / pins.ACE_STEP_DIR
        target = current.model_copy(
            update={
                "project_root": root,
                "python": current.python or root / ".venv" / "bin" / "python",
                "lm_backend": raw.get("lm_backend", pins.ACE_STEP_LM_BACKEND),
                "hf_home": current.hf_home or Path(pins.ACE_STEP_HF_HOME),
            }
        )
        return current, target

    def _wanted(self, target: AceStepOptions) -> dict[str, Any]:
        return {
            "project_root": str(_expand(target.project_root)),
            "python": str(_expand(target.python)),
            "lm_backend": target.lm_backend,
            "hf_home": str(_expand(target.hf_home)),
        }

    def plan(self, ctx: Context) -> Plan:
        current, target = self._options(ctx)
        if _problem(current) is None:
            return Plan(working=True, reason=None)
        raw = ctx.backends.get(self.backend, {})
        config = missing_keys(raw, self._wanted(target))
        problem = _problem(target)
        if problem is None:
            return Plan(working=True, reason=None, config=config)
        root = _expand(target.project_root)
        python = _expand(target.python)
        download = disk = 0
        if not (root / "acestep").is_dir():
            download += pins.ACE_STEP_CLONE_BYTES
        if not python.is_file():
            download += pins.ACE_STEP_ENV_BYTES
            disk += pins.ACE_STEP_ENV_BYTES  # uv's cache and the environment
        if _models_incomplete(target):
            left = max(0, pins.ACE_STEP_MODEL_BYTES - dir_size(root / "checkpoints"))
            download += left
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
        for tool, hint in (("git", "xcode-select --install"), ("uv", "brew install uv")):
            if not tools.has_tool(tool):
                raise SetupError(f"{tool} not found", hint=f"install it: {hint}", command=hint)
        root = _expand(target.project_root)
        python = _expand(target.python)
        self._checkout(root, ctx)
        if not python.is_file():
            self._sync(root, ctx)
        if _models_incomplete(target):
            tools.emit(f"ace-step: downloading {pins.ACE_STEP_MODEL_REPO} (~10 GB)")
            run_watched(
                "ace-step",
                [
                    str(python),
                    "-c",
                    DOWNLOAD_SCRIPT,
                    pins.ACE_STEP_MODEL_REPO,
                    pins.ACE_STEP_MODEL_REVISION,
                    str(root / "checkpoints"),
                ],
                root / "checkpoints",
                tools,
                env={
                    "HF_HOME": str(_expand(target.hf_home)),
                    # plain HTTPS: Xet ran at ~1.4 MB/s on the target Mac, HTTPS 4.5 minutes
                    "HF_HUB_DISABLE_XET": "1",
                    "HF_HUB_DISABLE_TELEMETRY": "1",
                },
                cwd=root,
            )
        problem = _problem(target)
        if problem is not None:
            raise SetupError(f"ACE-Step still not usable after setup: {problem}")

    def _checkout(self, root: Path, ctx: Context) -> None:
        tools = ctx.tools
        if (root / "acestep").is_dir():
            head = tools.capture(["git", "-C", str(root), "rev-parse", "HEAD"]) or "unknown"
            if head != pins.ACE_STEP_COMMIT:
                tools.emit("ace-step: " + UNTOUCHED.format(root=root, head=head[:12]))
            return
        if root.exists() and any(root.iterdir()):
            raise SetupError(
                f"{root} exists but is not an ACE-Step checkout",
                hint="move it away (or set project_root to a checkout), then run setup again",
            )
        root.parent.mkdir(parents=True, exist_ok=True)
        tools.emit(f"ace-step: cloning {pins.ACE_STEP_REPO} at {pins.ACE_STEP_COMMIT[:12]}")
        if tools.run(["git", "clone", pins.ACE_STEP_REPO, str(root)], None, None) != 0:
            raise SetupError("git clone of ACE-Step failed", hint="check the network")
        checkout = ["git", "-C", str(root), "checkout", "--detach", pins.ACE_STEP_COMMIT]
        if tools.run(checkout, None, None) != 0:
            raise SetupError(f"commit {pins.ACE_STEP_COMMIT} not found in {pins.ACE_STEP_REPO}")
        head = tools.capture(["git", "-C", str(root), "rev-parse", "HEAD"])
        if head != pins.ACE_STEP_COMMIT:
            raise SetupError(f"{root} is at {head}, expected {pins.ACE_STEP_COMMIT}")

    def _sync(self, root: Path, ctx: Context) -> None:
        # a lock next to the checkout (the measured one, or our first sync's) is used as is
        if (root / "uv.lock").is_file():
            cmd = ["uv", "sync", "--frozen"]
        else:
            cmd = ["uv", "sync", "--exclude-newer", pins.ACE_STEP_EXCLUDE_NEWER]
        ctx.tools.emit(f"ace-step: {' '.join(cmd)} (PyTorch etc., a few minutes)")
        if ctx.tools.run(cmd, None, root) != 0:
            raise SetupError("uv sync in the ACE-Step checkout failed; see the output above")


def _expand(path: Path | None) -> Path:
    return (path or Path()).expanduser()


def _backend(options: AceStepOptions) -> AceStepBackend:
    return AceStepBackend(options, state_dir=Path("/nonexistent"))


def _models_incomplete(options: AceStepOptions) -> bool:
    root = _expand(options.project_root)
    if _backend(options).missing_models(root):
        return True
    # an interrupted download leaves .incomplete files behind finished ones
    return next((root / "checkpoints" / ".cache").rglob("*.incomplete"), None) is not None


def _problem(options: AceStepOptions) -> str | None:
    if options.project_root is None:
        return "not configured"
    reason = _backend(options).unavailable_reason()
    if reason is None and _models_incomplete(options):
        return "a model download did not finish"
    return reason
