# Changelog

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
