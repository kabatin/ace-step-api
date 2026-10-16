import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ace_step_api.api.app import create_app
from ace_step_api.jobs.queue import JobQueue
from ace_step_api.jobs.store import JobStore
from ace_step_api.workflows.ace_step import AceStepWorkflow
from ace_step_api.workflows.base import WorkflowRegistry
from tests.fakes import EchoWorkflow, FakeMusicBackend

TOKEN = "test-token-0123456789-abcdefghijklmnop"
MUSIC = "/v1/workflows/ace-step-1.5/jobs"


def make_client(
    tmp_path: Path,
    backend: FakeMusicBackend,
    token: str | None = None,
    max_body: int = 1 << 20,
    waits_for: list[str] | None = None,
) -> TestClient:
    registry = WorkflowRegistry([EchoWorkflow(), AceStepWorkflow()])
    queue = JobQueue(
        JobStore(tmp_path), registry, backend, max_waiting=1, timeout_factor=2, retention_days=7
    )
    app = create_app(queue, registry, token=token, max_body_bytes=max_body, waits_for=waits_for)
    return TestClient(app, base_url="http://127.0.0.1")


@pytest.fixture
def backend() -> FakeMusicBackend:
    return FakeMusicBackend()


@pytest.fixture
def client(tmp_path: Path, backend: FakeMusicBackend) -> Iterator[TestClient]:
    with make_client(tmp_path, backend) as c:
        yield c
    backend.release.set()


def poll(client: TestClient, job_id: str, want: str, timeout: float = 5) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/v1/jobs/{job_id}").json()
        if body["status"] == want:
            return body
        time.sleep(0.02)
    raise AssertionError(f"job never reached {want}")


def test_health_matches_vpipe_api_shape(client: TestClient) -> None:
    expected = {
        "status": "ok",
        "version": "0.1.0",
        "running": 0,
        "waiting": 0,
        "max_waiting": 1,
        "waiting_for_peer": 0,
    }
    assert client.get("/v1/health").json() == expected
    assert client.get("/healthz").json() == expected


def test_capabilities_match_the_contract(tmp_path: Path, backend: FakeMusicBackend) -> None:
    with make_client(tmp_path, backend, waits_for=["vpipe-api"]) as c:
        body = c.get("/v1/capabilities").json()
    assert body["version"] == "0.1.0"
    assert body["queue"] == {"max_running": 1, "max_waiting": 1}
    assert body["coordination"] == {"waits_for": ["vpipe-api"]}
    music = next(w for w in body["workflows"] if w["id"] == "ace-step-1.5")
    assert music == {
        "id": "ace-step-1.5",
        "output": {
            "media_type": "audio/wav",
            "sample_rates": [44100, 48000],
            "channels": 2,
            "sample_format": "pcm_s16le",
        },
        "prompt": {"max_chars": 1000, "engine_reads_tokens": 256},
        "lyrics": {
            "max_chars": 4000,
            "engine_reads_tokens": 2048,
            "empty_means_instrumental": True,
        },
        "duration_sec": {"min": 10.0, "max": 240.0, "default": 60.0},
        "count": {"min": 1, "max": 4},
        "seed": {"min": 0, "max": 2**32 - 1, "per_track": "seed + i"},
        "instrumental": True,
        "typical_seconds": {"60s_track": 45},
    }
    text = str(body).lower()
    for secret in ("safetensors", "steps", "shift", "turbo", "1.7b", "checkpoint", "mlx"):
        assert secret not in text


def test_no_coordination_is_an_empty_list(client: TestClient) -> None:
    assert client.get("/v1/capabilities").json()["coordination"] == {"waits_for": []}


def test_workflows_and_openapi(client: TestClient) -> None:
    flows = {w["id"]: w for w in client.get("/v1/workflows").json()["workflows"]}
    assert set(flows) == {"echo", "ace-step-1.5"}
    props = flows["ace-step-1.5"]["params_schema"]["properties"]
    assert set(props) == {"prompt", "lyrics", "duration_sec", "instrumental", "seed", "count"}
    assert flows["ace-step-1.5"]["output_media_type"] == "audio/wav"
    spec = client.get("/openapi.json").json()
    route = spec["paths"][MUSIC]["post"]
    schema_ref = route["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    assert schema_ref.endswith("AceStepParams")
    assert {"422", "429"} <= set(route["responses"])
    assert client.get("/docs").status_code == 200


def test_full_job_flow(client: TestClient) -> None:
    accepted = client.post("/v1/workflows/echo/jobs", json={"text": "hi"})
    assert accepted.status_code == 202
    job = accepted.json()
    assert set(job) == {"id", "workflow", "status", "created_at"}
    assert (job["workflow"], job["status"]) == ("echo", "queued")
    done = poll(client, job["id"], "succeeded")
    assert done["result"] == {"length": 2} and done["error"] is None
    assert done["progress"] == 1.0 and done["cancel_requested"] is False
    assert done["waiting_for"] is None and done["queue_position"] is None
    assert {
        "queue_seconds",
        "coordination_seconds",
        "backend_seconds",
        "postprocess_seconds",
        "total_seconds",
    } <= set(done["timings"])
    assert done["timings"]["coordination_seconds"] == 0.0
    out = client.get(f"/v1/jobs/{job['id']}/output")
    assert out.status_code == 200 and out.text == "HI"
    assert out.headers["content-type"].startswith("text/plain")
    assert client.get(f"/v1/jobs/{job['id']}/output?index=0").text == "HI"
    missing = client.get(f"/v1/jobs/{job['id']}/output?index=1")
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "not_found"
    assert client.get(f"/v1/jobs/{job['id']}/output?index=-1").status_code == 404
    assert client.get(f"/v1/jobs/{job['id']}/output?index=x").status_code == 422
    assert client.delete(f"/v1/jobs/{job['id']}").json()["error"]["code"] == "conflict"


def test_failed_job_over_http(client: TestClient, backend: FakeMusicBackend) -> None:
    backend.mode = "fail"
    job = client.post("/v1/workflows/echo/jobs", json={"text": "x"}).json()
    body = poll(client, job["id"], "failed")
    assert body["error"]["code"] == "generation_failed" and body["error"]["retryable"]
    assert body["result"] is None
    # 409 before the job succeeded, whatever the index
    assert client.get(f"/v1/jobs/{job['id']}/output").status_code == 409
    assert client.get(f"/v1/jobs/{job['id']}/output?index=7").status_code == 409


def test_errors_use_the_envelope(client: TestClient) -> None:
    missing = client.get("/v1/jobs/job_nope")
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "not_found"
    assert client.get("/v1/jobs/job_nope/output").status_code == 404
    assert client.delete("/v1/jobs/job_nope").status_code == 404
    assert client.post("/v1/workflows/nope/jobs", json={}).status_code == 404
    rejected = client.post("/v1/workflows/echo/jobs", json={"text": "x", "reject": True})
    assert rejected.status_code == 422 and "rejected" in rejected.json()["error"]["message"]
    assert rejected.json()["error"]["details"] == [{"loc": ["body", "reject"], "msg": "no"}]
    assert client.get("/v1/nothing-here").json()["error"]["code"] == "not_found"


@pytest.mark.parametrize(
    "body",
    [
        {"prompt": "x", "negative_prompt": "noise"},
        {"prompt": "x", "steps": 8},
        {"prompt": "x", "shift": 3.0},
        {"prompt": "x", "model": "acestep-v15-turbo"},
        {"prompt": "x", "lm_model": "acestep-5Hz-lm-4B"},
        {"prompt": "x", "bpm": 128},
        {"prompt": "x", "output_path": "/tmp/x"},
        {"prompt": " "},
        {"prompt": ""},
        {"prompt": "x" * 1001},
        {"prompt": "x\x00y"},
        {"lyrics": "la la"},
        {"prompt": "x", "lyrics": "x" * 4001},
        {"prompt": "x", "lyrics": "a\x07b"},
        {"prompt": "x", "duration_sec": 9.9},
        {"prompt": "x", "duration_sec": 240.1},
        {"prompt": "x", "duration_sec": "long"},
        {"prompt": "x", "instrumental": "maybe"},
        {"prompt": "x", "seed": -1},
        {"prompt": "x", "seed": 2**32},
        {"prompt": "x", "seed": 2**32 - 2, "count": 3},
        {"prompt": "x", "count": 0},
        {"prompt": "x", "count": 5},
    ],
)
def test_request_validation(client: TestClient, body: dict) -> None:
    response = client.post(MUSIC, json=body)
    error = response.json()["error"]
    assert response.status_code == 422 and error["code"] == "invalid_params"
    assert all(set(d) == {"loc", "msg"} for d in error["details"] or [])


@pytest.mark.parametrize(
    "body",
    [
        {"prompt": "x"},
        {"prompt": "x", "duration_sec": 10},
        {"prompt": "x", "duration_sec": 240},
        {"prompt": "x", "duration_sec": 60.5},
        {"prompt": "x", "lyrics": "", "instrumental": False},
        {"prompt": "x", "lyrics": "[verse]\nla\n\t[chorus]\nla", "seed": 2**32 - 4, "count": 4},
        {"prompt": "x" * 1000, "lyrics": "y" * 4000, "instrumental": True, "count": 4},
    ],
)
def test_bounds_are_accepted(client: TestClient, body: dict) -> None:
    assert client.post(MUSIC, json=body).status_code == 202


def test_busy_returns_429_with_retry_after(client: TestClient, backend: FakeMusicBackend) -> None:
    backend.block = True
    first = client.post("/v1/workflows/echo/jobs", json={"text": "a"}).json()
    assert backend.started.wait(2)
    second = client.post("/v1/workflows/echo/jobs", json={"text": "b"}).json()
    running = client.get(f"/v1/jobs/{first['id']}").json()
    queued = client.get(f"/v1/jobs/{second['id']}").json()
    assert (running["status"], running["queue_position"]) == ("running", None)
    assert (queued["status"], queued["queue_position"]) == ("queued", 1)
    assert client.get("/v1/health").json()["running"] == 1
    busy = client.post("/v1/workflows/echo/jobs", json={"text": "c"})
    assert busy.status_code == 429
    assert busy.json()["error"] == {
        "code": "busy",
        "message": "the GPU slot and the waiting queue are full",
        "retryable": True,
        "details": None,
    }
    assert int(busy.headers["retry-after"]) >= 30
    assert client.get(f"/v1/jobs/{first['id']}/output").status_code == 409
    assert client.delete(f"/v1/jobs/{second['id']}").json()["status"] == "canceled"
    stopped = client.delete(f"/v1/jobs/{first['id']}")
    assert stopped.status_code == 200 and stopped.json()["status"] == "canceled"


def test_cancel_of_a_slow_stop_is_202(
    client: TestClient, backend: FakeMusicBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("ace_step_api.jobs.queue.CANCEL_WAIT_S", 0.2)
    backend.mode = "ignore_cancel"
    backend.block = True
    job = client.post("/v1/workflows/echo/jobs", json={"text": "a"}).json()
    assert backend.started.wait(2)
    pending = client.delete(f"/v1/jobs/{job['id']}")
    assert pending.status_code == 202
    assert pending.json()["status"] == "running" and pending.json()["cancel_requested"]
    backend.release.set()
    final = poll(client, job["id"], "canceled")
    assert final["result"] is None
    assert client.get(f"/v1/jobs/{job['id']}/output").status_code == 409


def test_bearer_auth_everywhere(tmp_path: Path, backend: FakeMusicBackend) -> None:
    with make_client(tmp_path, backend, token=TOKEN) as c:
        for path in ("/healthz", "/v1/health", "/v1/capabilities", "/openapi.json", "/docs"):
            denied = c.get(path)
            assert denied.status_code == 401 and denied.headers["www-authenticate"] == "Bearer"
            assert denied.json()["error"]["code"] == "unauthorized"
        assert c.post(MUSIC, json={}).status_code == 401
        assert c.get("/v1/health", headers={"Authorization": "Bearer wrong"}).status_code == 401
        ok = c.get("/v1/health", headers={"Authorization": f"Bearer {TOKEN}"})
        assert ok.status_code == 200


def test_body_limits(tmp_path: Path, backend: FakeMusicBackend) -> None:
    with make_client(tmp_path, backend, max_body=100) as c:
        big = c.post("/v1/workflows/echo/jobs", json={"text": "x" * 500})
        assert big.status_code == 413 and big.json()["error"]["code"] == "payload_too_large"
        chunked = c.post(
            "/v1/workflows/echo/jobs",
            content=iter([b'{"text":"x"}']),
            headers={"Content-Type": "application/json"},
        )
        assert chunked.status_code == 411
        assert chunked.json()["error"]["code"] == "length_required"
