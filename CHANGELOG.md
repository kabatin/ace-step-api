# Changelog

## Unreleased

Second engine: MiniMax Music 3, ported from the multi-engine design of sfx-api.

- Workflow `minimax-music3`: the `ace-step-1.5` body and rules (prompt ≤ 1000 chars, lyrics ≤ 4000,
  `instrumental`, `duration_sec` 10–240, `count` 1–4, track *i* uses `seed + i`) run by MiniMax Music 3
  (`mlx-community/MiniMax-Music3-mxfp8` through mlx-audio, 30 steps). Empty lyrics mean instrumental, as
  for `ace-step-1.5` (the engine gets its `[instrumental]` marker). `duration_sec` is an upper bound:
  a track that ends early succeeds with its real length. Output checked as 16-bit PCM 44.1 kHz stereo.
  Capabilities add `upper_bound`, `prompt_and_lyrics_max_tokens` (5000) and `attribution`
  (`MiniMax-Music3`, which the licence asks commercial products to display).
- Backend `minimax` (`[backends.minimax]`: `python`, `hf_home`, `model`, `resident_idle_seconds`,
  `start_timeout_s`, `stop_grace_s`, estimate knobs): a resident worker in the MiniMax venv's Python
  (`-P`, offline, same line protocol), progress estimated from elapsed time (the engine reports none),
  a hot track turned down to −1 dBFS as a whole (no limiter). Estimates `30 + (10 + 9.5 × duration)`
  s per job track, from ~8.3–9.4 s per second of audio measured outside the server; peak ~24 GB.
- Engine router: each request names its engine; one engine in memory at a time (the other is unloaded
  before a job starts); startup, shutdown, recover and the peer check reach both.
- `/v1/health` gains `workflows: {id: {available, loaded, reason}}` (additive). A workflow whose engine
  is not installed or lacks model files answers `503 workflow_unavailable`; the other keeps working.
  `serve` starts when at least one engine is available (warning for the other), `doctor --smoke` runs one
  short instrumental per available workflow, `workflows` shows each engine's availability.
- ACE-Step: unchanged contract (body, limits, results, seed rule, capabilities). Its worker state stays
  in `data_dir/engine/`; MiniMax's is `data_dir/engine/minimax/`. ACE-Step now refuses a job up front
  (`backend_unavailable`) when its checkout, Python or models are missing, instead of failing the load.
  The resident engine's `loaded` state is read without waiting for a model load in progress; a file
  system error while checking an install makes its workflow unavailable instead of failing health.
- A worker that survives SIGKILL is tracked: no engine loads next to it (the job fails as
  `engine_unstoppable`, retryable) until it exits; its pid file is kept so a restart reaps it.
- Input an engine refuses (MiniMax: prompt + lyrics over 5000 tokens, zero audio frames) fails the job
  as `invalid_params`, not retryable, and keeps the worker loaded. Job errors can now be
  non-retryable when the backend says so.
- The MiniMax worker releases MLX's buffer cache after every track.
- `doctor` reports a missing engine as warnings while another engine is available; it fails only when
  no engine can generate.

## 0.1.0 — 2026-10-17

First version, built on the job server of [klein-api](https://github.com/kabatin/klein-api) 0.1.0.

- HTTP job API per [docs/api.md](docs/api.md): health (vpipe-api / klein-api compatible, plus
  `waiting_for_peer`), capabilities, workflow listing with JSON Schemas, typed submit route, job status
  with progress over all tracks, queue position, `waiting_for`, estimate and timings (incl.
  `coordination_seconds`, `load_seconds`), per-index WAV download, cancel. One error envelope.
- Workflow `ace-step-1.5`: `prompt` (style tags), `lyrics` (empty = instrumental), `duration_sec`
  10–240, `instrumental`, `seed` (track *i* uses `seed + i`, 0 … 2³²−1), `count` 1–4; every track read
  back as 16-bit PCM stereo WAV before the job succeeds, real lengths in the result; sung language taken
  from the lyrics' script.
- Backend `acestep`: a resident worker in the ACE-Step checkout's own Python (`-P`, own process group,
  requests on stdin, replies on a private pipe, offline), released when idle, when a coordinated server
  is busy, after failed jobs and on shutdown; leftovers reaped after a crash. Works with the official
  checkout and the Apple Silicon port (`mlx_dit`, `lm_backend` switches).
- Output level: whole-track gain when a track runs hot, then a −1 dBFS look-ahead limiter.
- Coordination with vpipe-api and klein-api; a head job waiting for a peer is reported as
  `waiting_for_peer`, so two servers that wait for each other cannot deadlock.
