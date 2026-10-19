"""Built-in workflows. Registration is explicit: add new workflows to ``build_registry``."""

from __future__ import annotations

from ace_step_api.backends.base import MusicBackend
from ace_step_api.workflows import ace_step, minimax
from ace_step_api.workflows.base import WorkflowRegistry


def build_registry(backend: MusicBackend) -> WorkflowRegistry:
    def typical_60s(engine: str) -> float:
        return backend.typical_seconds_for(engine, 60.0)

    return WorkflowRegistry(
        [
            ace_step.AceStepWorkflow(typical_seconds_60s=typical_60s(ace_step.ENGINE)),
            minimax.MiniMaxMusic3Workflow(typical_seconds_60s=typical_60s(minimax.ENGINE)),
        ]
    )


__all__ = ["WorkflowRegistry", "build_registry"]
