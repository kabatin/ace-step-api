"""A real generation through each workflow (opt-in: ACE_STEP_API_REAL_TESTS=1).

Uses the server's own configuration (``$ACE_STEP_API_CONFIG`` or
~/.config/ace-step-api/config.toml), so it runs the engines and models the server would use.
A workflow whose engine is not installed is skipped. Not part of CI. Stop the server first:
both would load the models. MiniMax Music 3 takes ~2 minutes for its 10 s track.
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

# (sample rates, shortest acceptable length): MiniMax Music 3 may end a song early
EXPECTED = {"ace-step-1.5": ((44100, 48000), 8.0), "minimax-music3": ((44100,), 1.0)}


@pytest.mark.parametrize("workflow_id", sorted(EXPECTED))
def test_real_generation(tmp_path: Path, workflow_id: str) -> None:
    settings = load_settings()
    backend = build_backend(settings)
    workflow = build_registry(backend).get(workflow_id)
    assert workflow is not None and workflow.engine is not None
    status = backend.engine_status(workflow.engine)
    if not status.available:
        pytest.skip(f"{workflow.engine}: {status.reason}")
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
    rates, shortest = EXPECTED[workflow_id]
    info = inspect_wav(output.files[0])
    assert info.channels == 2 and info.sample_rate in rates
    assert shortest <= info.seconds <= 12.0
