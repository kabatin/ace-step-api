"""The MiniMax Music 3 backend, driving the real worker script against a fake ``mlx_audio``
package (tests/fake_mlx_audio) in a real child process."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from ace_step_api.backends import engine as engine_module
from ace_step_api.backends.base import GenerationRequest
from ace_step_api.backends.minimax import (
    DEFAULT_HF_HOME,
    DEFAULT_MODEL,
    DEFAULT_PYTHON,
    WORKER_SCRIPT,
    MiniMaxBackend,
    MiniMaxOptions,
)
from ace_step_api.backends.process import PID_FILE
from ace_step_api.checks import Level
from ace_step_api.jobs.queue import TIMEOUT_MARGIN_S
from ace_step_api.settings import Settings, SettingsError, load_settings
from tests.fakes import MINIMAX_SHARDS, minimax_runtime, python_wrapper

FAKE = Path(__file__).with_name("fake_mlx_audio")
CEILING = 10 ** (-1 / 20)


@pytest.fixture(autouse=True)
def fake_mlx_audio(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("PYTHONPATH", str(FAKE))
    monkeypatch.setenv("FAKE_MINIMAX_MODE", "ok")
    monkeypatch.delenv("FAKE_MINIMAX_DELAY", raising=False)
    record = tmp_path / "engine-record.jsonl"
    monkeypatch.setenv("FAKE_MINIMAX_RECORD", str(record))
    return record


def recorded(record: Path) -> list[dict]:
    return [json.loads(line) for line in record.read_text().splitlines()]


def asked(record: Path) -> list[dict]:
    """What the engine was asked (loads and generations; not the cache releases)."""
    return [entry for entry in recorded(record) if "clear_cache" not in entry]


def backend_for(tmp_path: Path, **options: object) -> MiniMaxBackend:
    opts = MiniMaxOptions.model_validate(minimax_runtime(tmp_path) | options)
    return MiniMaxBackend(opts, state_dir=tmp_path / "engine" / "minimax", poll_seconds=0.05)


def request(**overrides: object) -> GenerationRequest:
    values: dict[str, object] = {
        "job_id": "job_1",
        "engine": "minimax",
        "prompt": "warm acoustic pop, 96 BPM",
        "lyrics": "",
        "instrumental": True,
        "duration_s": 10.0,
        "seed": 41,
        "count": 2,
    }
    return GenerationRequest(**(values | overrides))  # type: ignore[arg-type]


def read_wav(path: Path) -> tuple[int, int, np.ndarray]:
    with wave.open(str(path), "rb") as wav:
        assert wav.getsampwidth() == 2
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        return wav.getframerate(), wav.getnchannels(), pcm.reshape(-1, wav.getnchannels())


def generate(backend: MiniMaxBackend, tmp_path: Path, **overrides: object):
    return backend.generate(
        request(**overrides), tmp_path / "job", timeout_s=60, cancel=threading.Event()
    )


# -- generation through the real worker ---------------------------------------------------------


def test_every_track_gets_its_seed_and_instrumental_lyrics(
    tmp_path: Path, fake_mlx_audio: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ACE_STEP_API_TOKEN", "x" * 40)  # must never reach the worker
    backend = backend_for(tmp_path)
    try:
        assert not backend.engine_status("minimax").loaded
        raw = generate(backend, tmp_path)
        assert raw.ok, raw.describe_failure()
        assert [p.name for p in raw.outputs] == ["raw-0.wav", "raw-1.wav"]
        load, *tracks = asked(fake_mlx_audio)
        assert load == {
            "load": DEFAULT_MODEL,
            "offline": "1",
            "hf_home": str(tmp_path / "hf"),
            "hub_cache": str(tmp_path / "hf" / "hub"),
            "token_leaked": False,
        }
        assert tracks == [
            {
                "text": "warm acoustic pop, 96 BPM",
                "lyrics": "[instrumental]",
                "duration": 10.0,
                "steps": 30,
                "seed": seed,
            }
            for seed in (41, 42)
        ]
        rate, channels, pcm = read_wav(raw.outputs[0])
        assert (rate, channels, len(pcm)) == (44100, 2, 4410)
        assert raw.details["load_seconds"] is not None
        assert raw.details["tracks"] == [
            {"index": 0, "seconds": raw.details["tracks"][0]["seconds"], "gain_db": 0.0},
            {"index": 1, "seconds": raw.details["tracks"][1]["seconds"], "gain_db": 0.0},
        ]
        assert backend.engine_status("minimax").loaded  # resident for the next job
        assert backend.plan(request()).estimate_seconds == 2 * backend.typical_seconds(10.0)
    finally:
        backend.shutdown()
    assert not backend.engine_status("minimax").loaded


def test_lyrics_reach_the_engine(tmp_path: Path, fake_mlx_audio: Path) -> None:
    backend = backend_for(tmp_path)
    try:
        lyrics = "[verse]\n夜明けの街を\n[chorus]\n光の中へ"
        raw = generate(backend, tmp_path, lyrics=lyrics, instrumental=False, count=1, seed=7)
        assert raw.ok, raw.describe_failure()
        track = asked(fake_mlx_audio)[1]
        assert (track["lyrics"], track["seed"]) == (lyrics, 7)
        # blank lyrics are never sent to the engine (it refuses them)
        assert generate(backend, tmp_path, lyrics=" \n", instrumental=False, count=1).ok
        assert asked(fake_mlx_audio)[2]["lyrics"] == "[instrumental]"
    finally:
        backend.shutdown()


@pytest.mark.parametrize("mode", ["channels_first", "early"])
def test_the_shape_and_length_the_engine_returns_are_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.setenv("FAKE_MINIMAX_MODE", mode)
    backend = backend_for(tmp_path)
    try:
        raw = generate(backend, tmp_path, count=1, duration_s=30.0)
        assert raw.ok, raw.describe_failure()
        rate, channels, pcm = read_wav(raw.outputs[0])
        # (2, N) is turned into (N, 2); a track that ends early is delivered as it is
        assert (rate, channels, len(pcm)) == (44100, 2, 4410 if mode == "early" else 13230)
        assert pcm[25, 0] == -pcm[25, 1] != 0
    finally:
        backend.shutdown()


def test_a_hot_track_is_turned_down_as_a_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_MINIMAX_MODE", "hot")
    backend = backend_for(tmp_path)
    try:
        raw = generate(backend, tmp_path, count=1)
        assert raw.ok, raw.describe_failure()
        assert raw.details["tracks"][0]["gain_db"] == pytest.approx(-1.0, abs=0.01)
        _, _, pcm = read_wav(raw.outputs[0])
        peak = int(np.abs(pcm.astype(np.int32)).max())
        assert round(32767 * CEILING) - 2 <= peak <= round(32767 * CEILING) + 1
    finally:
        backend.shutdown()


def test_progress_is_estimated_from_elapsed_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_MINIMAX_DELAY", "2.5")
    # 10 s of audio is expected to take 2 s: the ticker (every second) reports ~0.5, then holds
    # at 0.95; 2.5 s leaves a slow runner time for at least one tick
    backend = backend_for(tmp_path, track_overhead_seconds=0, seconds_per_audio_second=0.2)
    progress: list[float] = []
    try:
        raw = backend.generate(
            request(count=1),
            tmp_path / "job",
            timeout_s=60,
            cancel=threading.Event(),
            on_progress=progress.append,
        )
        assert raw.ok, raw.describe_failure()
        assert progress and progress == sorted(progress)
        assert progress[0] > 0.0 and progress[-1] <= 0.95
    finally:
        backend.shutdown()


def test_engine_errors_discard_the_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = backend_for(tmp_path)
    try:
        monkeypatch.setenv("FAKE_MINIMAX_MODE", "gen_error")
        raw = generate(backend, tmp_path)
        assert not raw.ok and "out of memory" in raw.describe_failure()
        assert raw.failure_code == "generation_failed" and not backend.engine.running()
        monkeypatch.setenv("FAKE_MINIMAX_MODE", "load_fail")
        raw = generate(backend, tmp_path)
        assert raw.failure_code == "engine_failed" and "No safetensors" in raw.describe_failure()
        monkeypatch.setenv("FAKE_MINIMAX_MODE", "crash")
        raw = generate(backend, tmp_path)
        assert not raw.ok and "exited" in raw.describe_failure()
    finally:
        backend.shutdown()


def test_cancel_and_timeout_stop_the_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_MINIMAX_MODE", "hang")
    backend = backend_for(tmp_path, stop_grace_s=2)
    try:
        cancel = threading.Event()
        threading.Timer(1.0, cancel.set).start()
        raw = backend.generate(request(), tmp_path / "job", timeout_s=60, cancel=cancel)
        assert raw.canceled and not raw.ok and not backend.engine.running()
        assert not (tmp_path / "engine" / "minimax" / PID_FILE).exists()
        started = time.monotonic()
        raw = backend.generate(request(), tmp_path / "job", timeout_s=2, cancel=threading.Event())
        assert raw.timed_out and time.monotonic() - started < 30
        assert not backend.engine.running()
    finally:
        backend.shutdown()


def test_stdout_noise_does_not_break_the_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_MINIMAX_MODE", "noisy")
    backend = backend_for(tmp_path)
    try:
        assert generate(backend, tmp_path, count=1).ok
    finally:
        backend.shutdown()
    log = (tmp_path / "engine" / "minimax" / "worker.log").read_text()
    assert "Fetching 11 files" in log and "generating frames 52%" in log


def test_a_leftover_worker_is_reaped(tmp_path: Path) -> None:
    backend = backend_for(tmp_path)
    state = tmp_path / "engine" / "minimax"
    state.mkdir(parents=True)
    proc = subprocess.Popen(
        [sys.executable, str(WORKER_SCRIPT), "--proto-fd", "1", "--model", DEFAULT_MODEL],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        (state / PID_FILE).write_text(str(proc.pid))
        assert backend.startup() == f"stopped leftover minimax_worker.py (pid {proc.pid})"
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


# -- availability, configuration, estimates ------------------------------------------------------


def test_health_when_everything_is_installed(tmp_path: Path) -> None:
    backend = backend_for(tmp_path)
    assert backend.unavailable_reason() is None
    status = backend.engine_status("minimax")
    assert status.available and status.reason is None and not status.loaded
    checks = {c.name: c for c in backend.health()}
    assert all(c.level is Level.OK for c in checks.values()), checks
    assert set(checks) == {
        "MiniMax Python",
        "MiniMax mlx-audio",
        "MiniMax Music 3 model",
        "MiniMax resident engine",
    }
    assert "models stay loaded 120 s" in checks["MiniMax resident engine"].detail
    assert DEFAULT_MODEL in checks["MiniMax Music 3 model"].detail


def snapshot(tmp_path: Path) -> Path:
    cache = tmp_path / "hf" / "hub" / "models--mlx-community--MiniMax-Music3-mxfp8"
    return cache / "snapshots" / "d00a12c3"


@pytest.mark.parametrize(
    ("damage", "reason", "check"),
    [
        (
            lambda t: (t / "minimax-music3" / ".venv" / "bin" / "python").unlink(),
            "the MiniMax Music 3 runtime is not installed (no Python)",
            "MiniMax Python",
        ),
        (
            lambda t: (
                t / "minimax-music3/.venv/lib/python3.12/site-packages/mlx_audio/music/models"
                "/minimax_music3/__init__.py"
            ).unlink(),
            "mlx-audio with MiniMax Music 3 is not installed in its Python",
            "MiniMax mlx-audio",
        ),
        (
            lambda t: (snapshot(t) / MINIMAX_SHARDS[1]).unlink(),
            f"model files missing: {MINIMAX_SHARDS[1]}",
            "MiniMax Music 3 model",
        ),
        (
            lambda t: (snapshot(t) / "tokenizer" / "tokenizer.json").unlink(),
            "model files missing: tokenizer/tokenizer.json",
            "MiniMax Music 3 model",
        ),
        (
            lambda t: (snapshot(t) / "model.safetensors.index.json").write_text("{"),
            "model files missing: model.safetensors.index.json",
            "MiniMax Music 3 model",
        ),
        (
            lambda t: (snapshot(t).parent.parent / "refs" / "main").unlink(),
            f"{DEFAULT_MODEL} is not downloaded",
            "MiniMax Music 3 model",
        ),
    ],
)
def test_what_is_missing_makes_it_unavailable(
    tmp_path: Path, damage, reason: str, check: str
) -> None:
    backend = backend_for(tmp_path)
    damage(tmp_path)
    assert backend.unavailable_reason() == reason
    status = backend.engine_status("minimax")
    assert not status.available and status.reason == reason
    assert str(tmp_path) not in reason  # reasons are shown to clients: no paths
    failed = [c.name for c in backend.health() if c.failed]
    assert failed == [check]
    # an unavailable engine never starts a worker
    raw = generate(backend, tmp_path)
    assert raw.failure_code == "backend_unavailable" and raw.failure == reason
    assert not backend.engine.running()


def test_a_python_outside_a_venv_is_not_searched_for_mlx_audio(tmp_path: Path) -> None:
    plain = python_wrapper(tmp_path / "plain" / "bin" / "python")
    backend = backend_for(tmp_path, python=str(plain))
    assert backend.unavailable_reason() is None
    checks = {c.name: c for c in backend.health()}
    assert checks["MiniMax mlx-audio"].level is Level.WARN


def test_option_defaults() -> None:
    options = MiniMaxOptions()
    assert options.python == DEFAULT_PYTHON == Path("~/.local/opt/minimax-music3/.venv/bin/python")
    assert options.hf_home == DEFAULT_HF_HOME == Path("~/.local/share/minimax-music3/hf")
    assert options.model == DEFAULT_MODEL == "mlx-community/MiniMax-Music3-mxfp8"
    assert options.resident_idle_seconds == 120.0
    backend = MiniMaxBackend(state_dir=Path("/nowhere"))
    assert backend.python_path == DEFAULT_PYTHON.expanduser()
    assert backend.hf_home == DEFAULT_HF_HOME.expanduser()


def test_worker_command_and_env(tmp_path: Path) -> None:
    backend = backend_for(tmp_path, model="mlx-community/MiniMax-Music3-4bit")
    config = backend.engine.config
    assert config.name == "minimax" and config.script == WORKER_SCRIPT
    assert config.script.name == "minimax_worker.py" and config.script.is_file()
    assert config.args == ("--model", "mlx-community/MiniMax-Music3-4bit")
    assert config.python == tmp_path / "minimax-music3" / ".venv" / "bin" / "python"
    assert config.cwd == config.state_dir == tmp_path / "engine" / "minimax"
    assert config.env == {
        "HF_HOME": str(tmp_path / "hf"),
        "HF_HUB_CACHE": str(tmp_path / "hf" / "hub"),
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    # the 4-bit repo is not in the cache
    assert backend.unavailable_reason() == "mlx-community/MiniMax-Music3-4bit is not downloaded"


def test_from_settings(tmp_path: Path) -> None:
    cfg = tmp_path / "c.toml"
    cfg.write_text(f'[backends.minimax]\nhf_home = "{tmp_path}/hf"\nresident_idle_seconds = 0\n')
    settings = load_settings(env={"ACE_STEP_API_DATA_DIR": str(tmp_path)}, config_path=cfg)
    backend = MiniMaxBackend.from_settings(settings)
    assert backend.hf_home == tmp_path / "hf" and backend.options.resident_idle_seconds == 0
    assert backend.engine.config.state_dir == tmp_path / "engine" / "minimax"
    for bad in ('model = "../../etc"', "steps = 50", "seconds_per_audio_second = 0"):
        cfg.write_text(f"[backends.minimax]\n{bad}\n")
        with pytest.raises(SettingsError, match=r"backends\.minimax"):
            MiniMaxBackend.from_settings(load_settings(env={}, config_path=cfg))


def test_estimates_cover_the_measured_times(tmp_path: Path) -> None:
    backend = backend_for(tmp_path)
    assert backend.typical_seconds(60) == 580.0
    # measured outside the server: 125 s for 15 s, 584 s for 62 s (compute only)
    assert backend.typical_seconds(15) >= 125 and backend.typical_seconds(62) >= 584
    longest = backend.plan(request(duration_s=240.0, count=4)).estimate_seconds
    assert longest == 30 + 4 * (10 + 9.5 * 240)
    # the job timeout (estimate x factor + margin) leaves the longest job room to spare, even
    # if long tracks ran at 1.4x the worst measured rate (9.4 s per second of audio)
    timeout = longest * Settings().job_timeout_factor + TIMEOUT_MARGIN_S
    assert timeout > 1.4 * 4 * 240 * 9.4 + 30


# -- review findings -------------------------------------------------------------------------------


def test_unload_waits_for_a_stop_in_progress(tmp_path: Path) -> None:
    """A stop holds the engine lock while the process dies (``running()`` is already False):
    unloading must wait for it, or the other engine would load next to a dying one."""
    backend = backend_for(tmp_path)
    with backend.engine._lock:  # what an idle or peer-busy stop holds
        unloading = threading.Thread(target=backend.unload, args=("engine acestep takes the GPU",))
        unloading.start()
        unloading.join(0.3)
        assert unloading.is_alive()
    unloading.join(5)
    assert not unloading.is_alive()


def test_an_unreadable_install_is_unavailable_not_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = backend_for(tmp_path)

    def denied() -> str | None:
        raise PermissionError(13, "Permission denied", str(tmp_path / "hf"))

    monkeypatch.setattr(backend, "unavailable_reason", denied)
    status = backend.engine_status("minimax")
    assert not status.available
    assert status.reason == "cannot check its install (PermissionError)"
    raw = generate(backend, tmp_path)
    assert raw.failure_code == "backend_unavailable" and str(tmp_path) not in (raw.failure or "")


@pytest.mark.parametrize("shard", ["../../etc/passwd", "/abs/model.safetensors", "sub/x"])
def test_shard_names_from_the_index_must_be_plain_files(tmp_path: Path, shard: str) -> None:
    backend = backend_for(tmp_path)
    index = snapshot(tmp_path) / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"a": MINIMAX_SHARDS[0], "b": shard}}))
    assert backend.unavailable_reason() == "model files missing: model.safetensors.index.json"


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("too_long", "The assembled prompt has 5123 tokens; the maximum is 5000"),
        ("no_frames", "MiniMax Music 3 generated zero audio frames"),
    ],
)
def test_input_the_engine_rejects_keeps_the_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, message: str
) -> None:
    backend = backend_for(tmp_path)
    try:
        assert generate(backend, tmp_path, count=1).ok
        pid = backend.engine.pid
        # the fake engine reads its mode when the worker starts: restart it in that mode
        monkeypatch.setenv("FAKE_MINIMAX_MODE", mode)
        backend.unload("test")
        raw = generate(backend, tmp_path)
        assert (raw.failure_code, raw.retryable) == ("invalid_params", False)
        assert raw.failure == f"the engine rejected the input: ValueError: {message}"
        assert len(raw.details["tracks"]) == 1  # stopped at the first track
        # the ~24 GB worker is kept: the input was the problem, not the engine
        assert backend.engine.running() and backend.engine.pid not in (None, pid)
        rejected_pid = backend.engine.pid
        again = generate(backend, tmp_path, count=1)
        assert again.failure_code == "invalid_params" and backend.engine.pid == rejected_pid
        assert again.details["load_seconds"] is None  # no reload
    finally:
        backend.shutdown()


def test_other_engine_errors_still_discard_the_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_MINIMAX_MODE", "gen_error")
    backend = backend_for(tmp_path)
    try:
        raw = generate(backend, tmp_path)
        assert (raw.failure_code, raw.retryable) == ("generation_failed", True)
        assert not backend.engine.running()
    finally:
        backend.shutdown()


@pytest.mark.parametrize("api", ["core", "metal", "none"])
def test_mlx_buffer_cache_is_released_after_each_track(
    tmp_path: Path, fake_mlx_audio: Path, monkeypatch: pytest.MonkeyPatch, api: str
) -> None:
    monkeypatch.setenv("FAKE_MLX_CACHE_API", api)
    backend = backend_for(tmp_path)
    try:
        assert generate(backend, tmp_path, count=2).ok
    finally:
        backend.shutdown()
    released = [e["clear_cache"] for e in recorded(fake_mlx_audio) if "clear_cache" in e]
    assert released == ([] if api == "none" else [api, api])


def test_a_worker_that_survives_sigkill_blocks_every_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A process that outlives SIGKILL (stuck in the GPU driver) still holds its memory: no
    engine may load next to it, and a restart reaps it (its pid file is kept)."""
    monkeypatch.setenv("FAKE_MINIMAX_MODE", "hang")
    real_stop_group = engine_module.stop_group
    monkeypatch.setattr(engine_module, "stop_group", lambda proc, grace_s: None)
    backend = backend_for(tmp_path, stop_grace_s=1)
    stuck: int | None = None
    try:
        cancel = threading.Event()
        threading.Timer(1.0, cancel.set).start()
        raw = backend.generate(request(), tmp_path / "job", timeout_s=60, cancel=cancel)
        assert (raw.failure_code, raw.canceled) == ("engine_unstoppable", False)
        assert "could not be stopped" in (raw.failure or "")
        stuck = backend.engine.stuck_pid
        assert stuck is not None
        assert backend.unload("engine acestep takes the GPU") is False
        again = generate(backend, tmp_path)  # not even its own engine starts a second worker
        assert again.failure_code == "engine_unstoppable" and backend.engine.pid is None
        assert (tmp_path / "engine" / "minimax" / PID_FILE).read_text() == str(stuck)
    finally:
        if stuck is not None:
            os.killpg(stuck, signal.SIGKILL)
        monkeypatch.setattr(engine_module, "stop_group", real_stop_group)
    # once it is gone, the engine is usable again
    deadline = time.monotonic() + 10
    while backend.engine.stuck_pid is not None and time.monotonic() < deadline:
        backend.unload("check")
        time.sleep(0.1)
    assert backend.unload("check") is True
    monkeypatch.setenv("FAKE_MINIMAX_MODE", "ok")
    try:
        assert generate(backend, tmp_path, count=1).ok
    finally:
        backend.shutdown()
