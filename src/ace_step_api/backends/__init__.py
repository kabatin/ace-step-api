"""Backends: the engines that actually make the music (ACE-Step 1.5, MiniMax Music 3), behind
one router."""

from __future__ import annotations

from ace_step_api.backends.base import MusicBackend
from ace_step_api.backends.router import EngineRouter
from ace_step_api.settings import Settings


def build_backend(settings: Settings) -> EngineRouter:
    from ace_step_api.backends.acestep import AceStepBackend
    from ace_step_api.backends.minimax import MiniMaxBackend

    return EngineRouter(
        {
            AceStepBackend.name: AceStepBackend.from_settings(settings),
            MiniMaxBackend.name: MiniMaxBackend.from_settings(settings),
        }
    )


__all__ = ["EngineRouter", "MusicBackend", "build_backend"]
