"""Facts about this Mac that decide what setup can install."""

from __future__ import annotations

import platform
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ace_step_api.setup.tools import Tools

TOOLS = ("git", "uv")


@dataclass(frozen=True)
class Machine:
    arm64: bool
    macos: str | None
    ram_gb: int | None
    free_bytes: int
    tools: dict[str, bool]

    @property
    def supported_reason(self) -> str | None:
        if not self.arm64 or self.macos is None:
            return "needs an Apple Silicon Mac"
        return None

    def as_json(self) -> dict[str, Any]:
        return {
            "arm64": self.arm64,
            "macos": self.macos,
            "ramGb": self.ram_gb,
            "freeBytes": self.free_bytes,
            "tools": dict(self.tools),
        }


def existing_parent(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if candidate.exists():
            return candidate
    return Path("/")


def probe(
    prefix: Path,
    tools: Tools,
    *,
    machine: Callable[[], str] = platform.machine,
    mac_ver: Callable[[], str] = lambda: platform.mac_ver()[0],
    free_bytes: Callable[[Path], int] = lambda p: shutil.disk_usage(p).free,
) -> Machine:
    memsize = tools.capture(["sysctl", "-n", "hw.memsize"])
    ram_gb = round(int(memsize) / 2**30) if memsize and memsize.isdigit() else None
    # a Python running under Rosetta reports x86_64 on Apple Silicon: ask the kernel too
    arm64 = machine() == "arm64" or tools.capture(["sysctl", "-n", "hw.optional.arm64"]) == "1"
    return Machine(
        arm64=arm64,
        macos=mac_ver() or None,
        ram_gb=ram_gb,
        free_bytes=free_bytes(existing_parent(prefix)),
        tools={name: tools.has_tool(name) for name in TOOLS},
    )
