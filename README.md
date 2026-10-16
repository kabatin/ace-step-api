# ace-step-api

**Make music with ACE-Step 1.5 on your Apple Silicon Mac, as an HTTP job API.**

`ace-step-api` is a small, safe job server around [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5).
Send style tags, lyrics (or none) and a length, get back 1–4 WAV tracks:

```
POST /v1/workflows/ace-step-1.5/jobs   → 202 {id}
GET  /v1/jobs/{id}                      → queued → running (progress) → succeeded
GET  /v1/jobs/{id}/output?index=N       → audio/wav
```

It speaks the same job protocol as [klein-api](https://github.com/kabatin/klein-api),
[wan-api](https://github.com/kabatin/wan-api) and [vpipe-api](https://github.com/kabatin/vpipe-api)
(job lifecycle, error envelope, `429` + `Retry-After`, `Idempotency-Key`), so a client that drives them
needs only a different workflow id, body and output type. The full contract is [docs/api.md](docs/api.md).

- **Closed workflow.** Clients send `prompt` (style tags), `lyrics`, `duration_sec`, `instrumental`, a
  seed and a count — never model names, steps, shift, BPM overrides or paths.
- **Checked output.** Every track is a 16-bit PCM stereo WAV (48 kHz), peak-limited to −1 dBFS, read back
  before the job succeeds; the result reports each track's real length.
- **One GPU slot, honest backpressure.** One job at a time (its tracks run one after another), a small
  waiting queue, `429 busy` with `Retry-After` when full.
- **Plays nicely with other GPU servers.** With `coordinate_with`, a job starts only once vpipe-api /
  klein-api report idle (and not before `settle_seconds` after one was last seen busy). While waiting it
  yields, so klein-api — which can wait for ace-step-api in turn — goes first instead of deadlocking.
- **No needless model loads.** The models stay loaded in a resident worker between jobs and are released
  when idle (120 s) or as soon as another GPU server is busy: ACE-Step holds ~19 GB of unified memory.
- **Restart-safe.** Jobs live on disk; queued jobs run again, interrupted ones fail as retryable, and a
  worker left behind by a crashed server is stopped.

**The same seed does not reproduce a track.** ACE-Step's language model samples the song plan, so two
runs with identical inputs differ. Make a few takes and pick one.

## Requirements

- Apple Silicon Mac, macOS 14+, 32 GB of memory recommended (a loaded engine peaks at ~19 GB).
- Python 3.12+ and [uv](https://docs.astral.sh/uv/) for ace-step-api itself.
- An [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) checkout with its own environment
  (`uv sync` there: Python 3.11–3.12, PyTorch, MLX) and these models in its `checkpoints/`
  (the `ACE-Step/Ace-Step1.5` repository on Hugging Face, ~10 GB):

| folder | what | size |
|---|---|---|
| `acestep-v15-turbo` | the music model (DiT, turbo) | 4.8 GB |
| `acestep-5Hz-lm-1.7B` | the language model that plans the song | 3.7 GB |
| `Qwen3-Embedding-0.6B` | text encoder | 1.2 GB |
| `vae` | audio decoder | 0.3 GB |

ace-step-api runs that checkout's Python as a separate worker process; the two environments never mix.

### Measurements

On the target Mac (MacBook Pro M5, 32 GB, macOS 26.5), through this server's backend, turbo model,
1.7B language model, one track per request. Peak memory is macOS's `phys_footprint` of the worker,
sampled every second (a lower bound).

| engine checkout | first job (load + track) | next 60 s tracks | 240 s track | peak memory |
|---|---|---|---|---|
| Apple Silicon port `a97f051` (PyTorch on MPS, LM `pt`) | 55 s (21 + 34) | **32–34 s** | 121 s | **19 GB** |
| official `ca1e85f`, MLX DiT/VAE/LM (its default) | 127 s (48 + 79) | 66–70 s | — | 37 GB |
| official, PyTorch DiT + PyTorch LM | 118 s (25 + 93) | 126 s | — | — |
| official, MLX DiT + PyTorch LM | 127 s (34 + 94) | 122 s | — | — |

Loads varied between 19 and 38 s. The machine was also running a virtual machine that held ~23 GB, so
swap was in use for every run (16–28 GB); the official checkout's larger footprint made that worse.
**The server here runs the port** (`project_root` points at it; the uv.lock it was measured with is kept
next to it). Switching to the official checkout is `project_root` + `lm_backend = "mlx"` + a restart.

The port's own MLX path for the language model fails to load its checkpoint with the mlx-lm it
resolves (`Received 310 parameters not in model`) and falls back to PyTorch; `lm_backend = "pt"`
skips that attempt.

## Setup on the Mac

```sh
# 1. the engine (anywhere; this is the layout the examples use). Either checkout works, see
#    "Measurements": the official one is maintained and pinned, the port was 2x faster here.
mkdir -p ~/.local/opt/ace-step && cd ~/.local/opt/ace-step
git clone https://github.com/ace-step/ACE-Step-1.5 && cd ACE-Step-1.5
#   or: git clone https://github.com/clockworksquirrel/ace-step-apple-silicon && cd ace-step-apple-silicon
uv sync                   # the port has no uv.lock: the first sync writes one, keep it
HF_HUB_DISABLE_XET=1 uv run python -c 'from acestep.model_downloader import download_main_model; print(download_main_model(prefer_source="huggingface"))'

# 2. the server
uv tool install git+https://github.com/kabatin/ace-step-api
```

`HF_HUB_DISABLE_XET=1` is optional: on the target Mac the default (Xet) transfer ran at ~1.4 MB/s while
plain HTTPS took 4.5 minutes for the 10 GB. The server itself never downloads anything: its worker runs
with `HF_HUB_OFFLINE=1`.

```toml
# ~/.config/ace-step-api/config.toml   (chmod 600: it holds tokens)
host = "0.0.0.0"          # LAN; requires a token
port = 8768
token = "…openssl rand -hex 32…"

[backends.acestep]
project_root = "~/.local/opt/ace-step/ace-step-apple-silicon"   # or …/ACE-Step-1.5
lm_backend = "pt"                                               # "mlx" for the official checkout
hf_home = "~/.local/share/ace-step-api/hf"   # keep the worker's Hugging Face caches apart

# wait for the other GPU servers on this Mac before each job
[[coordinate_with]]
name = "vpipe-api"
health_url = "http://127.0.0.1:8765/v1/health"
token = "…vpipe-api's bearer token…"

[[coordinate_with]]
name = "klein-api"
health_url = "http://127.0.0.1:8767/v1/health"
token = "…klein-api's bearer token…"
```

### `doctor`

```sh
ace-step-api doctor           # platform, macOS, Python, config permissions, checkout, engine Python,
                              # models, disk, power, memory, and every coordinate_with server
ace-step-api doctor --smoke   # + one real 10-second track through the workflow (stop the server first)
```

### `serve`

```sh
ace-step-api serve            # http://127.0.0.1:8768   (OpenAPI: /openapi.json, docs: /docs)
```

`serve` checks the checkout and the models first and exits with a clear message if it cannot generate.

### Run at login (launchd)

[`examples/com.github.kabatin.ace-step-api.plist`](examples/com.github.kabatin.ace-step-api.plist):

```sh
cp examples/com.github.kabatin.ace-step-api.plist ~/Library/LaunchAgents/   # then edit the paths
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.github.kabatin.ace-step-api.plist
launchctl bootout gui/$(id -u)/com.github.kabatin.ace-step-api                # stop
```

Logs are JSON lines on stderr (job id, state transitions, queue/coordination/backend/postprocess/total
seconds, model loads, failure category) — never prompts, lyrics, audio or tokens. The engine worker's own
output goes to `data_dir/engine/worker.log`.

## Configuration

`~/.config/ace-step-api/config.toml` (or `$ACE_STEP_API_CONFIG`); every scalar can also be set as
`ACE_STEP_API_<NAME>`.

| key | default | |
|---|---|---|
| `host` / `port` | `127.0.0.1` / `8768` | a non-loopback host **requires** `token` |
| `token` | — | when set, every request needs `Authorization: Bearer <token>` (≥ 32 printable ASCII) |
| `max_waiting` | `2` | jobs allowed to wait behind the running one |
| `retention_days` | `7` | finished jobs and outputs are deleted after this |
| `data_dir` | `~/.local/share/ace-step-api` | `jobs/<id>/` (record, outputs), `engine/` (worker log, pid) |
| `job_timeout_factor` | `3.0` | a job is stopped after `estimate × factor + 5 min` |
| `max_body_mb` | `64` | request size limit |
| `coordinate_with` | `[]` | `[[coordinate_with]]` tables: `name`, `health_url`, optional `token` (file only) |
| `settle_seconds` | `30` | after a coordinated server was seen busy, a job starts no sooner than this |
| `coordination_poll_seconds` | `5` | how often a busy server is asked again |

`[backends.acestep]`:

| key | default | |
|---|---|---|
| `project_root` | — | the ACE-Step 1.5 checkout (required) |
| `python` | `<project_root>/.venv/bin/python` | the engine's interpreter |
| `dit_model` / `lm_model` | `acestep-v15-turbo` / `acestep-5Hz-lm-1.7B` | folders in `checkpoints/` |
| `lm_backend` | `vllm` | `mlx`, `pt` (PyTorch), or `vllm` (switched to MLX on Apple Silicon) |
| `mlx_dit` | engine default | `true`/`false`: DiT and VAE on MLX or PyTorch (official checkout only) |
| `hf_home` | — | `HF_HOME` for the worker |
| `resident_idle_seconds` | `120` | keep the models loaded this long after a job (`0`: load per job) |
| `start_timeout_s` | `600` | give up loading after this |
| `stop_grace_s` | `15` | Ctrl-C grace before SIGKILL |
| `load_seconds`, `track_overhead_seconds`, `seconds_per_audio_second` | measured | estimate model (`load + (overhead + rate × duration) × count`) |

### Coordination with other GPU servers

When `coordinate_with` is set, the job at the head of the queue starts only after a check in which
**every** listed server's `/v1/health` reports `running == 0 && waiting == 0` (and `outside_runs == 0`
when present). If one was seen busy within the last `settle_seconds`, the job checks again
`settle_seconds` after that sighting (clients often send follow-up jobs). Until then it is `queued` with
`waiting_for: "<name>"`. A server that cannot be reached counts as idle and is logged as a warning. A
running job is never paused. Redirects are not followed, so a peer token only ever goes to its own
`health_url`. Time spent waiting shows up as `timings.coordination_seconds`.

**Yielding.** While its head job waits for a peer, ace-step-api's own `/v1/health` reports the queue as
`waiting_for_peer` and `waiting: 0`. klein-api can therefore list ace-step-api in its `coordinate_with`
without the two waiting for each other forever: klein-api goes first, ace-step-api follows. vpipe-api
counts only its own `vpipe --launch` processes, so it never waits for ace-step-api.

### Resident engine

The first job starts the worker (`<python> -P acestep_worker.py …` in the checkout, its own process
group, stdin for requests and a private pipe for replies, so the engine's progress bars on stdout can
never corrupt the protocol). It loads the DiT, VAE, text encoder and language model once; later jobs
reuse them. The worker is stopped:

- after `resident_idle_seconds` (120) without a job;
- immediately when a `coordinate_with` server is seen busy while it idles (checked every
  `coordination_poll_seconds`) — this gives the ~19 GB back to vpipe-api or klein-api;
- after a job that did not succeed (canceled, timed out, failed), and on shutdown.

A worker left behind by a crashed ace-step-api is stopped at the next start (pid file in
`data_dir/engine/`, matched by pid, script name and start time).

### Output

Each track comes from the engine as float audio (before any file format could clip it) and is kept
under a −1 dBFS ceiling before it is written as 16-bit PCM:

1. **Gain.** If more than 0.1 % of its 5 ms blocks would exceed the ceiling, the whole track is turned
   down until only that share does. The port's output often runs hot all the way through (a 240 s
   instrumental peaked ~5 dB over the ceiling in 86 % of its 100 ms frames); limiting all of it would
   pump and distort.
2. **Limiter** for the rest: one block of look-ahead, one block of hold, linear release within 0.2 s.

Audio that stays under the ceiling is left untouched; loudness is not normalised otherwise. Without
this, peaks above full scale were clipped for good (up to ~1000 clipped samples in a 60-second
instrumental). Measured: gain −0.4 … −3.5 dB, limiter −0.7 … −2.1 dB on 60 s tracks; the official
checkout normalises its own output to −1 dBFS, so there both stay at 0. Each track's values are in
`result.details.generation.engine.tracks[]`.

## API examples

Full reference: [docs/api.md](docs/api.md); more in [examples/curl.md](examples/curl.md).

```sh
API=http://127.0.0.1:8768
AUTH=()                                   # or: AUTH=(-H "Authorization: Bearer $ACE_STEP_API_TOKEN")

JOB=$(curl -s "${AUTH[@]}" -X POST "$API/v1/workflows/ace-step-1.5/jobs" \
        -H 'Content-Type: application/json' -H 'Idempotency-Key: song-001' \
        -d '{"prompt": "J-pop, upbeat, female vocal, synth, 128 BPM",
             "lyrics": "[verse]\n夜明けの街を 駆け抜けてく\n[chorus]\n光の中へ 手を伸ばして",
             "duration_sec": 60, "seed": 42, "count": 2}' | jq -r .id)

curl -s "${AUTH[@]}" "$API/v1/jobs/$JOB" | jq '{status, progress, queue_position, waiting_for, timings}'
curl -s "${AUTH[@]}" -o take-0.wav "$API/v1/jobs/$JOB/output?index=0"
curl -s "${AUTH[@]}" -o take-1.wav "$API/v1/jobs/$JOB/output?index=1"
curl -s "${AUTH[@]}" -X DELETE "$API/v1/jobs/$JOB"          # cancel (200 canceled / 202 stopping)
```

## Security

- Binds `127.0.0.1` by default. A non-loopback `host` without a `token` refuses to start.
- Without a token, only loopback `Host` headers are served (blocks DNS rebinding) and proxy headers are
  refused. With a token, every endpoint (including `/docs`, `/openapi.json`, `/healthz`) needs it.
- **No TLS.** On a LAN the token, the lyrics and the audio travel in clear text: use a trusted network or
  a TLS reverse proxy (keep the token set). `doctor` warns if a config file holding tokens is readable by
  others.
- Body limit (`max_body_mb`; any `Transfer-Encoding` is refused with 411), at most 64 concurrent
  connections, control characters refused in prompts and lyrics, strict job ids, no client-chosen paths,
  models or settings. The engine worker runs without a shell and without ace-step-api's environment (the
  token), offline (`HF_HUB_OFFLINE=1`), and receives prompts as JSON on a pipe. Client-facing errors never
  contain paths or internals. Job files are private to the server's user (umask 077). Do not add CORS:
  the API relies on browsers not being able to send cross-site JSON.

## Verification status

Checked on the target Mac (2026-10-17) with `doctor`, the real-engine test, and the measurements above
through the backend; the protocol, cancel/timeout, crash, idle release, the MLX switch and the limiter
are covered by tests against a fake engine. Output: 48 kHz stereo 16-bit WAV, exactly the requested
length (60.0 s, 240.0 s), no clipped samples, peaks at −1.0 dBFS.

Lyrics: a Whisper (large-v3-turbo) transcript of a 60 s Japanese track got the first and last lines
right and garbled the middle ones; listen before you rely on intelligibility. The same seed gives a
different track every time.

Not checked: 44.1 kHz output (the engine always produced 48 kHz), `count` 3–4 in one job, durations
between 60 and 240 s other than those two, the official checkout's 240 s time.

## Development

```sh
uv sync
uv run pytest            # fake engine (tests/fake_acestep); 80%+ coverage gate; `real` tests are skipped
uv run ruff check && uv run ruff format --check && uv run pyright
ACE_STEP_API_REAL_TESTS=1 uv run pytest -m real --no-cov   # on the Mac with the engine installed
```

`tests/test_engine.py` runs the real worker script in a child process against a fake `acestep` package,
so the protocol, cancel/timeout, crashes, idle release and the limiter are covered without a GPU.

## License

Apache-2.0 (this repository; it ships no weights and no ACE-Step code). ACE-Step 1.5's code and the
`ACE-Step/Ace-Step1.5` weights are MIT-licensed; check the license of every model you use.
