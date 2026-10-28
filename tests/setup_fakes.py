"""A fake machine for ``setup``: commands are recorded and their effects simulated on disk."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from ace_step_api.setup import pins
from ace_step_api.setup.machine import Machine
from ace_step_api.setup.tools import Tools

ACE_MODELS = ("acestep-v15-turbo", "vae", "Qwen3-Embedding-0.6B", "acestep-5Hz-lm-1.7B")
SITE = Path("lib") / "python3.12" / "site-packages"
MINIMAX_PACKAGE = SITE / "mlx_audio" / "music" / "models" / "minimax_music3" / "__init__.py"
Effect = Callable[[Sequence[str], Path | None], int]


def executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    return path


def ace_models(root: Path) -> None:
    for name in ACE_MODELS:
        (root / "checkpoints" / name).mkdir(parents=True, exist_ok=True)
        (root / "checkpoints" / name / "model.safetensors").write_bytes(b"x" * 10)


def minimax_snapshot(hf_home: Path, *, ref: bool = True) -> Path:
    cache = hf_home / "hub" / ("models--" + pins.MINIMAX_MODEL_REPO.replace("/", "--"))
    snapshot = cache / "snapshots" / pins.MINIMAX_MODEL_REVISION
    (snapshot / "tokenizer").mkdir(parents=True, exist_ok=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "tokenizer" / "tokenizer.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"x" * 10)
    if ref:
        (cache / "refs").mkdir(parents=True, exist_ok=True)
        (cache / "refs" / "main").write_text(pins.MINIMAX_MODEL_REVISION)
    return cache


def minimax_venv(venv: Path, *, package: bool = True, hf: bool = True) -> Path:
    python = executable(venv / "bin" / "python")
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n")
    if package:
        (venv / MINIMAX_PACKAGE).parent.mkdir(parents=True, exist_ok=True)
        (venv / MINIMAX_PACKAGE).write_text("")
    if hf:
        executable(venv / "bin" / "hf")
    return python


class FakeProc:
    """A download: each poll writes a little (or nothing, when stalled)."""

    def __init__(self, steps: list[Callable[[], None]], code: int) -> None:
        self.steps = steps
        self.code = code
        self.signals: list[int] = []
        self.returncode: int | None = None

    def poll(self) -> int | None:
        if self.returncode is not None:
            return self.returncode
        if self.steps:
            self.steps.pop(0)()
            return None
        self.returncode = self.code
        return self.code

    def send_signal(self, sig: int) -> None:
        self.signals.append(sig)
        self.returncode = 130

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode if self.returncode is not None else self.code

    def kill(self) -> None:
        self.returncode = -9


@dataclass
class Fake:
    """Holds the recorded calls; ``tools`` is the injectable Tools."""

    home: Path
    effects: dict[str, Effect] = field(default_factory=dict)
    captures: dict[str, str | None] = field(default_factory=dict)
    downloads: list[Callable[[Sequence[str], Path | None], FakeProc]] = field(default_factory=list)
    on_path: set[str] = field(default_factory=lambda: {"git", "uv", "ace-step-api"})
    health: list[int | None] = field(default_factory=list)
    calls: list[tuple[str, ...]] = field(default_factory=list)
    envs: list[Mapping[str, str] | None] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)
    now: float = 0.0

    def run(self, cmd: Sequence[str], env: Mapping[str, str] | None, cwd: Path | None) -> int:
        self.calls.append(tuple(cmd))
        self.envs.append(env)
        for key, effect in self.effects.items():
            if " ".join(cmd).startswith(key):
                return effect(cmd, cwd)
        return 0

    def capture(self, cmd: Sequence[str]) -> str | None:
        joined = " ".join(cmd)
        for key, out in self.captures.items():
            if key in joined:
                return out
        return None

    def spawn(self, cmd: Sequence[str], env: Mapping[str, str] | None, cwd: Path | None):
        self.calls.append(tuple(cmd))
        self.envs.append(env)
        return self.downloads.pop(0)(cmd, cwd)

    def which(self, name: str) -> str | None:
        return f"/opt/bin/{name}" if name in self.on_path else None

    def http_status(self, url: str, token: str | None) -> int | None:
        return self.health.pop(0) if self.health else None

    def sleep(self, seconds: float) -> None:
        self.now += seconds

    @property
    def tools(self) -> Tools:
        return Tools(
            run=self.run,
            capture=self.capture,
            spawn=self.spawn,
            which=self.which,
            http_status=self.http_status,
            sleep=self.sleep,
            clock=lambda: self.now,
            emit=self.lines.append,
            home=self.home,
            uid=501,
        )

    def commands(self, prefix: str) -> list[tuple[str, ...]]:
        return [c for c in self.calls if " ".join(c).startswith(prefix)]


MAC = Machine(
    arm64=True, macos="26.5", ram_gb=32, free_bytes=10**12, tools={"git": True, "uv": True}
)


def mac(**changes: object) -> Callable[[Path, Tools], Machine]:
    machine = replace(MAC, **changes)  # type: ignore[arg-type]
    return lambda _prefix, _tools: machine
