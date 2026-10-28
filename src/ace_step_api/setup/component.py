"""What every installable component tells setup: where it is, whether it works, what it costs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ace_step_api.setup.tools import Tools


@dataclass(frozen=True)
class Context:
    prefix: Path
    # the config file's own [backends.*] tables (not the environment's view of them)
    backends: dict[str, dict[str, Any]]
    tools: Tools


@dataclass(frozen=True)
class Plan:
    """Where a component is (or goes) and what installing it would take."""

    working: bool  # the engine can generate from here
    reason: str | None  # why not, for the report
    download_bytes: int = 0
    disk_bytes: int = 0
    # [backends.<name>] keys the config needs for the server to find this install
    config: dict[str, Any] = field(default_factory=dict)


class Component(Protocol):
    id: str
    backend: str
    licence: str

    def plan(self, ctx: Context) -> Plan: ...

    def install(self, ctx: Context) -> None: ...


def missing_keys(existing: dict[str, Any], wanted: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in wanted.items() if key not in existing}
