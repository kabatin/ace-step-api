"""Built-in workflows. Registration is explicit: add new workflows to ``build_registry``."""

from __future__ import annotations

from ace_step_api.backends.base import MusicBackend
from ace_step_api.workflows.ace_step import AceStepWorkflow
from ace_step_api.workflows.base import Workflow, WorkflowRegistry


def build_registry(backend: MusicBackend) -> WorkflowRegistry:
    workflows: list[Workflow] = [
        AceStepWorkflow(typical_seconds_60s=backend.typical_seconds(60.0)),
    ]
    return WorkflowRegistry(workflows)


__all__ = ["WorkflowRegistry", "build_registry"]
