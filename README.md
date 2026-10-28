# ace-step-api

**Make music with ACE-Step 1.5 or MiniMax Music 3 on your Apple Silicon Mac, as an HTTP job API.**

`ace-step-api` is a small, safe job server around two music engines:
[ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) (workflow `ace-step-1.5`) and
[MiniMax Music 3](https://huggingface.co/MiniMaxAI/MiniMax-Music3) through
[mlx-audio](https://github.com/Blaizzy/mlx-audio) (workflow `minimax-music3`). Both take the same body:
send style tags, lyrics (or none) and a length, get back 1–4 WAV tracks:

```
POST /v1/workflows/ace-step-1.5/jobs     → 202 {id}
POST /v1/workflows/minimax-music3/jobs   → 202 {id}
GET  /v1/jobs/{id}                        → queued → running (progress) → succeeded
GET  /v1/jobs/{id}/output?index=N         → audio/wav
```

It speaks the same job protocol as [klein-api](https://github.com/kabatin/klein-api),
[wan-api](https://github.com/kabatin/wan-api) and [vpipe-api](https://github.com/kabatin/vpipe-api)
(job lifecycle, error envelope, `429` + `Retry-After`, `Idempotency-Key`), so a client that drives them
needs only a different workflow id, body and output type. The full contract is [docs/api.md](docs/api.md).

- **Closed workflow.** Clients send `prompt` (style tags), `lyrics`, `duration_sec`, `instrumental`, a
  seed and a count — never model names, steps, shift, BPM overrides or paths.
- **Checked output.** Every track is a 16-bit PCM stereo WAV (ACE-Step 48 kHz, MiniMax 44.1 kHz), kept
  under −1 dBFS, read back before the job succeeds; the result reports each track's real length.
- **Two engines, one queue.** Each workflow names its engine; jobs of both share the one GPU slot, and
  only one engine is in memory at a time (the other is unloaded before a job starts). If one engine is
  not installed, its workflow answers `503 workflow_unavailable` and `/v1/health` says why; the other
  keeps working. The server starts when at least one engine is available.
- **One GPU slot, honest backpressure.** One job at a time (its tracks run one after another), a small
  waiting queue, `429 busy` with `Retry-After` when full.
- **Plays nicely with other GPU servers.** With `coordinate_with`, a job starts only once vpipe-api /
  klein-api report idle (and not before `settle_seconds` after one was last seen busy). While waiting it
  yields, so klein-api — which can wait for ace-step-api in turn — goes first instead of deadlocking.
- **No needless model loads.** The models stay loaded in a resident worker between jobs and are released
  when idle (120 s), when the other engine takes the GPU, or as soon as another GPU server is busy:
  ACE-Step holds ~19 GB of unified memory, MiniMax Music 3 ~24 GB.
- **Restart-safe.** Jobs live on disk; queued jobs run again, interrupted ones fail as retryable, and a
  worker left behind by a crashed server is stopped.

**The same seed does not reproduce a track.** ACE-Step's language model samples the song plan, so two
runs with identical inputs differ. Make a few takes and pick one. (Whether MiniMax Music 3 repeats a
track for the same seed has not been checked.)

**MiniMax Music 3 is slow.** It runs ~9 s per second of audio on the target Mac (a 60 s track ≈ 10
minutes, a 240 s track over half an hour) and peaks at ~24 GB. Its `duration_sec` is an upper bound:
the model may end a song early (a 62 s request once gave 22.9 s); the result reports the real length.

## Requirements

- Apple Silicon Mac, macOS 14+, 32 GB of memory recommended (a loaded engine peaks at ~19 GB ACE-Step,
  ~24 GB MiniMax Music 3; only one is loaded at a time).
- Python 3.12+ and [uv](https://docs.astral.sh/uv/) for ace-step-api itself.
- For `ace-step-1.5`: an [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) checkout with its own environment
  (`uv sync` there: Python 3.11–3.12, PyTorch, MLX) and these models in its `checkpoints/`
  (the `ACE-Step/Ace-Step1.5` repository on Hugging Face, ~10 GB):

| folder | what | size |
|---|---|---|
| `acestep-v15-turbo` | the music model (DiT, turbo) | 4.8 GB |
| `acestep-5Hz-lm-1.7B` | the language model that plans the song | 3.7 GB |
| `Qwen3-Embedding-0.6B` | text encoder | 1.2 GB |
| `vae` | audio decoder | 0.3 GB |

- For `minimax-music3`: a virtual environment with [mlx-audio](https://github.com/Blaizzy/mlx-audio)
  0.5.9+ (MiniMax Music 3 support; Python 3.12, MLX 0.32) and the `mlx-community/MiniMax-Music3-mxfp8`
  weights (~13 GB: three shards, the tokenizer, the config) in a Hugging Face cache.

ace-step-api runs each engine's Python as a separate worker process; the environments never mix.
Either engine is optional: a workflow whose engine is missing is reported unavailable and the other
keeps working. `doctor` shows the missing engine's checks as warnings while another engine is
available, and fails (exit 1) only when no engine can generate.

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

**MiniMax Music 3** (`mlx-community/MiniMax-Music3-mxfp8`, mlx-audio 0.5.9, MLX 0.32.4, 30 steps,
same Mac). **Measured outside this server** (mlx-audio's Python API directly), not yet through it:

| track asked | compute | per second of audio | notes |
|---|---|---|---|
| 15 s | 125 s | 8.3 s | |
| 62 s | 584 s | 9.4 s | |
| 62 s | — | — | once ended at 22.9 s (the length is an upper bound) |

`load()` returned in ~2.7 s; the first generation also pays for paging the weights in. Peak memory
~24 GB. The server's estimate is `30 s (load) + (10 + 9.5 × duration) s` per track — ~10 minutes for
60 s, ~38 minutes for 240 s. Lengths beyond 62 s were not measured; if the cost grows faster than
linearly, the default `job_timeout_factor` still leaves room (see *Configuration*).

## Setup on the Mac

The quick way: install the server, then let it install the engines.

```sh
uv tool install git+https://github.com/kabatin/ace-step-api
ace-step-api setup --check            # what is installed, what is missing, how much it downloads
ace-step-api setup                    # both engines (asks once before downloading ~25 GB)
ace-step-api setup --models minimax-music3 --yes   # one engine, no question
ace-step-api setup --lan --service    # also: listen on the LAN (with a token), run at login
```

| option | what it does |
|---|---|
| `--models ace-step,minimax-music3` | the components to install (default: both; either works alone) |
| `--check` | change nothing; report the Mac, each component (`installed` / `missing` / `unsupported`) and its download size |
| `--yes` | do not ask before downloading (without a terminal, setup asks for this and exits 3) |
| `--json` | one JSON object on stdout (progress stays on stderr) |
| `--prefix DIR` | where the checkout and the venv go (default `~/.local/opt/ace-step-api`) |
| `--lan` | `host = "0.0.0.0"` and a token: `ACE_STEP_API_TOKEN` if set, else the config's, else a new one (never printed) |
| `--port N` | `port` in the config |
| `--coordinate-with NAME=HEALTH_URL` | a `[[coordinate_with]]` entry (repeatable); its token from `ACE_STEP_API_COORD_TOKEN_<NAME>` |
| `--service` | a launchd agent `com.github.kabatin.ace-step-api` running `serve`, then waits for `/v1/health` |
| `--label-suffix S` | a second agent (`….S`, its own log) beside the real one, for trying things out (needs `--port`) |

What it installs (pinned in [`src/ace_step_api/setup/pins.py`](src/ace_step_api/setup/pins.py)):

- `ace-step`: the Apple Silicon port at `a97f051` (the checkout measured above) cloned into
  `<prefix>/ace-step-apple-silicon`, `uv sync` resolved as of 2026-10-17 (the port has no uv.lock; the
  first sync writes one), then `ACE-Step/Ace-Step1.5` at a fixed revision (~10 GB, MIT) into its
  `checkpoints/` (what its `download_main_model` does, which can only fetch the latest).
- `minimax-music3`: a Python 3.12 venv in `<prefix>/minimax-music3/.venv` with `mlx-audio==0.5.9`, then
  `hf download mlx-community/MiniMax-Music3-mxfp8` at a fixed revision (~13 GB) into
  `~/.local/share/minimax-music3/hf`. **Licence:** commercial use is allowed if the product shows
  "MiniMax-Music3" prominently; over USD 20 million annual revenue needs MiniMax's permission.

Setup is idempotent (an engine that already works is left alone) and resumes an interrupted download
(a download that stops growing for 10 minutes is restarted). It checks free disk first. It adds the
keys the server needs to `~/.config/ace-step-api/config.toml` (or `$ACE_STEP_API_CONFIG`) and never
changes or removes one that is there: a `project_root` or `python` already in the config is where it
installs, and a symlinked config is written through to its target. It also runs when the config is
one the server rejects (e.g. a LAN `host` without a token), so `setup --lan` can repair it.

Exit codes: 0 ok, 1 failed, 2 bad arguments (with `--json`, still one JSON object), 3 needs a
confirmation. A component this Mac cannot run (an Intel Mac, a `[backends.minimax] model` setup cannot
fetch) is reported as `unsupported`; that fails the run only when it was named in `--models`.

Or by hand:

```sh
# 1. ACE-Step (anywhere; this is the layout the examples use). Either checkout works, see
#    "Measurements": the official one is maintained and pinned, the port was 2x faster here.
mkdir -p ~/.local/opt/ace-step && cd ~/.local/opt/ace-step
git clone https://github.com/ace-step/ACE-Step-1.5 && cd ACE-Step-1.5
#   or: git clone https://github.com/clockworksquirrel/ace-step-apple-silicon && cd ace-step-apple-silicon
uv sync                   # the port has no uv.lock: the first sync writes one, keep it
HF_HUB_DISABLE_XET=1 uv run python -c 'from acestep.model_downloader import download_main_model; print(download_main_model(prefer_source="huggingface"))'

# 2. MiniMax Music 3 (the defaults of [backends.minimax] point here)
uv venv --python 3.12 ~/.local/opt/minimax-music3/.venv
uv pip install --python ~/.local/opt/minimax-music3/.venv/bin/python "mlx-audio>=0.5.9"
HF_HOME=~/.local/share/minimax-music3/hf ~/.local/opt/minimax-music3/.venv/bin/hf download mlx-community/MiniMax-Music3-mxfp8

# 3. the server
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

# optional: the defaults match step 2 above
[backends.minimax]
python = "~/.local/opt/minimax-music3/.venv/bin/python"
hf_home = "~/.local/share/minimax-music3/hf"

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
ace-step-api doctor           # platform, macOS, Python, config permissions, each engine's install and
                              # models, disk, power, memory, and every coordinate_with server
ace-step-api doctor --smoke   # + one real 10-second instrumental per available workflow, one after the
                              # other (stop the server first; ~1 min ACE-Step, ~2–3 min MiniMax)
ace-step-api workflows        # each workflow, its engine, and whether that engine is available
```

### `serve`

```sh
ace-step-api serve            # http://127.0.0.1:8768   (OpenAPI: /openapi.json, docs: /docs)
```

`serve` checks each engine's install and models first. It exits with a clear message if no engine can
generate, and starts with a warning when only one can (the other workflow answers
`503 workflow_unavailable`).

### Run at login (launchd)

`ace-step-api setup --service` writes and loads the agent. Or by hand,
[`examples/com.github.kabatin.ace-step-api.plist`](examples/com.github.kabatin.ace-step-api.plist):

```sh
cp examples/com.github.kabatin.ace-step-api.plist ~/Library/LaunchAgents/   # then edit the paths
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.github.kabatin.ace-step-api.plist
launchctl bootout gui/$(id -u)/com.github.kabatin.ace-step-api                # stop
```

Logs are JSON lines on stderr (job id, state transitions, queue/coordination/backend/postprocess/total
seconds, model loads, failure category) — never prompts, lyrics, audio or tokens. The engine workers' own
output goes to `data_dir/engine/worker.log` (ACE-Step) and `data_dir/engine/minimax/worker.log`
(MiniMax Music 3).

## Configuration

`~/.config/ace-step-api/config.toml` (or `$ACE_STEP_API_CONFIG`); every scalar can also be set as
`ACE_STEP_API_<NAME>`.

| key | default | |
|---|---|---|
| `host` / `port` | `127.0.0.1` / `8768` | a non-loopback host **requires** `token` |
| `token` | — | when set, every request needs `Authorization: Bearer <token>` (≥ 32 printable ASCII) |
| `max_waiting` | `2` | jobs allowed to wait behind the running one |
| `retention_days` | `7` | finished jobs and outputs are deleted after this |
| `data_dir` | `~/.local/share/ace-step-api` | `jobs/<id>/` (record, outputs), `engine/` (ACE-Step worker log, pid), `engine/minimax/` (MiniMax's) |
| `job_timeout_factor` | `3.0` | a job is stopped after `estimate × factor + 5 min` (see below) |
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

`[backends.minimax]` (every key optional):

| key | default | |
|---|---|---|
| `python` | `~/.local/opt/minimax-music3/.venv/bin/python` | a venv Python with mlx-audio (MiniMax Music 3 support) |
| `hf_home` | `~/.local/share/minimax-music3/hf` | `HF_HOME` holding the weights (the worker runs offline) |
| `model` | `mlx-community/MiniMax-Music3-mxfp8` | Hugging Face repo id in that cache |
| `resident_idle_seconds` | `120` | keep the model loaded this long after a job (`0`: load per job) |
| `start_timeout_s` | `300` | give up loading after this |
| `stop_grace_s` | `15` | Ctrl-C grace before SIGKILL |
| `load_seconds` / `track_overhead_seconds` / `seconds_per_audio_second` | `30` / `10` / `9.5` | estimate model, as above |

The MiniMax workflow is available when that Python exists, the venv has mlx-audio's MiniMax Music 3
package, and the cache holds the repo's `config.json`, `tokenizer/` and every weight shard its index
names. A missing piece makes `minimax-music3` unavailable with the reason in `/v1/health`; ACE-Step is
unaffected (and vice versa).

**Job timeouts.** A job is stopped after `estimate × job_timeout_factor + 300 s`. The longest MiniMax
job (4 × 240 s) is estimated at 30 + 4 × (10 + 9.5 × 240) = 9190 s (~2.6 h); with the default factor 3
it is stopped only after ~7.7 h — at the worst measured rate (9.4 s/s) it needs ~2.5 h. At factor 1
the only margin is the 300 s, so keep the default (or ≥ 1.5) when MiniMax is used for long tracks.

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
0.1.4 and later can list ace-step-api too and yields the same way: when both have jobs waiting, the first
final check to pass starts its job and the other sees `running: 1` and waits. Both start only if the two
final checks overlap, a window as long as one check.

### Resident engine

The first job of an engine starts its worker (`<python> -P acestep_worker.py …` in the ACE-Step
checkout, `<python> -P minimax_worker.py --model …` for MiniMax; each in its own process group, stdin
for requests and a private pipe for replies, so the engine's progress bars on stdout can never corrupt
the protocol). It loads the models once; later jobs reuse them. A worker is stopped:

- after `resident_idle_seconds` (120) without a job;
- before a job of the other engine starts: the two engines never share the memory (~19 + ~24 GB would
  not fit a 32 GB Mac);
- immediately when a `coordinate_with` server is seen busy while it idles (checked every
  `coordination_poll_seconds`) — this gives the memory back to vpipe-api or klein-api;
- after a job that did not succeed (canceled, timed out, failed), and on shutdown.

A worker left behind by a crashed ace-step-api is stopped at the next start (pid file in
`data_dir/engine/` for ACE-Step — where 0.1.0 kept it — and `data_dir/engine/minimax/`, matched by
pid, script name and start time).

MiniMax Music 3 reports no progress while it generates, so its job `progress` is an estimate: time
spent over the track's expected time, at most 0.95 per track until the track is done. Long tracks may
run past the (uncalibrated, linear) estimate; progress then holds at 0.95 until the track finishes.

After every MiniMax track the worker releases MLX's buffer cache, so an idle worker keeps only the
model. Input the engine refuses (prompt + lyrics over 5000 tokens, a model that makes no audio) fails
the job as `invalid_params`, not retryable, and keeps the worker loaded. A worker that survives SIGKILL
(stuck in the GPU driver) blocks every load, of either engine, until it exits: such jobs fail as
`engine_unstoppable`.

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

MiniMax Music 3 clips its own float output to [−1, 1]. If a track peaks above −1 dBFS, the whole track is
turned down to −1 dBFS (one gain, no limiter: the waveform keeps its shape); `gain_db` per track is in
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

# the same body on MiniMax Music 3 (no lyrics: an instrumental; ~10 minutes per 60 s track)
curl -s "${AUTH[@]}" -X POST "$API/v1/workflows/minimax-music3/jobs" \
     -H 'Content-Type: application/json' \
     -d '{"prompt": "warm acoustic pop, 96 BPM, soft piano and guitar", "instrumental": true, "duration_sec": 60}'
curl -s "${AUTH[@]}" "$API/v1/health" | jq .workflows    # available / loaded / reason per workflow
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
  models or settings. The engine workers run without a shell and without ace-step-api's environment (the
  token), offline (`HF_HUB_OFFLINE=1`), and receive prompts as JSON on a pipe. Client-facing errors never
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

**MiniMax Music 3 through this server: not yet verified on the real engine.** The worker, the router
(unloading ACE-Step first), availability checks and the workflow are covered by tests against a fake
`mlx_audio`; on the target Mac, `ace-step-api workflows` with the server's configuration reports both
engines available (install and weights found), and the worker's array conversion and WAV writing were
run in the MiniMax venv on a tiny array (CPU; the model was not loaded). Still to
check with `doctor --smoke` / the real-engine test: a full generation, its time and memory through the
server, the first-generation load cost, how often the progress estimate updates while MLX computes, a
240 s track, and seed reproducibility.

## Development

```sh
uv sync
uv run pytest            # fake engines (tests/fake_acestep, tests/fake_mlx_audio); 80%+ coverage gate;
                         # `real` tests are skipped
uv run ruff check && uv run ruff format --check && uv run pyright
ACE_STEP_API_REAL_TESTS=1 uv run pytest -m real --no-cov   # on the Mac with the engine installed
```

`tests/test_engine.py` and `tests/test_minimax.py` run the real worker scripts in child processes
against a fake `acestep` package and a fake `mlx_audio`, so the protocol, cancel/timeout, crashes, idle
release, the limiter and MiniMax's shape/level handling are covered without a GPU;
`tests/test_router.py` runs both together (one engine in memory at a time).

## License

Apache-2.0 (this repository; it ships no weights and no engine code). ACE-Step 1.5's code and the
`ACE-Step/Ace-Step1.5` weights are MIT-licensed; check the license of every model you use.

**MiniMax Music 3** weights (and the `mlx-community` conversion) are under the
[MiniMax-Music3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-Music3/blob/main/LICENSE).
In short (read the licence itself):

- commercial use is allowed;
- a commercial product or service that uses it must **prominently display "MiniMax-Music3" in its
  user interface** (`/v1/capabilities` gives it as the workflow's `attribution`);
- above US$20 million aggregate yearly revenue (with affiliates), a separate prior written
  authorisation from MiniMax is needed;
- offering generation to third parties (a product, service or hosted service) requires reasonable
  safeguards against uses and outputs that violate the licence, e.g. infringing third-party rights;
- the acceptable-use policy (Exhibit A) forbids, among others, infringing uses, and asks that content
  published to a public environment be clearly and prominently disclosed as machine-generated.
