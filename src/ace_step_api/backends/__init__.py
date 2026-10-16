"""Backends: the engines that actually make the music (ACE-Step 1.5)."""

from __future__ import annotations

from ace_step_api.backends.base import MusicBackend
from ace_step_api.settings import Settings


def build_backend(settings: Settings) -> MusicBackend:
    from ace_step_api.backends.acestep import AceStepBackend

    return AceStepBackend.from_settings(settings)


__all__ = ["MusicBackend", "build_backend"]
