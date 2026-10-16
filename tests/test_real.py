"""A real ACE-Step generation through the workflow (opt-in: ACE_STEP_API_REAL_TESTS=1).

Uses the server's own configuration (``$ACE_STEP_API_CONFIG`` or
~/.config/ace-step-api/config.toml), so it runs the checkout and models the server would use.
Not part of CI. Stop the server first: both would load the models.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from ace_step_api.audio import inspect_wav
from ace_step_api.backends import build_backend
from ace_step_api.settings import load_settings
from ace_step_api.workflows import build_registry

pytestmark = pytest.mark.real


def test_real_generation(tmp_path: Path) -> None:
    settings = load_settings()
    backend = build_backend(settings)
    failed = [c for c in backend.health() if c.failed]
    assert not failed, failed
    workflow = next(iter(build_registry(backend)))
    params = workflow.params_model.model_validate(
        workflow.smoke_params() | {"lyrics": "[verse]\nla la la", "instrumental": False}
    )
    try:
        stored = workflow.store_inputs(params, tmp_path)
        request = workflow.build_request("job_real", stored, tmp_path)
        raw = backend.generate(request, tmp_path, timeout_s=900, cancel=threading.Event())
        assert raw.ok, raw.describe_failure()
        output = workflow.finalize("job_real", stored, tmp_path, raw)
    finally:
        backend.shutdown()
    info = inspect_wav(output.files[0])
    assert info.channels == 2 and info.sample_rate in (44100, 48000)
    assert 8.0 <= info.seconds <= 12.0
