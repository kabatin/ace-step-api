# ace-step-api HTTP API (v1)

Request and response bodies are JSON (UTF-8) unless noted. The machine-readable schema is served at
`/openapi.json`; interactive docs at `/docs`. The shapes deliberately match
[wan-api](https://github.com/kabatin/wan-api) and [vpipe-api](https://github.com/kabatin/vpipe-api)
(same job lifecycle, error envelope, `429` + `Retry-After`, `Idempotency-Key`), so a client that drives
them needs only a different workflow id, body and output type.

Two workflows, one engine each, on Apple Silicon: `ace-step-1.5`
([ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5)) and `minimax-music3`
([MiniMax Music 3](https://huggingface.co/MiniMaxAI/MiniMax-Music3)). They take the same body. **Nothing
about the engines (model files, steps, shift, language model, backends) is part of this contract.**

## Authentication

Identical to wan-api:

- If the server has a token configured (`ACE_STEP_API_TOKEN`, ≥ 32 printable ASCII characters), **every**
  request — including `/healthz`, `/docs` and `/openapi.json` — must send `Authorization: Bearer <token>`;
  otherwise `401`.
- A server bound to a non-loopback address refuses to start without a token.
- Without a token the server only answers requests whose `Host` is `127.0.0.1`, `localhost` or `[::1]`
  and that carry no proxy headers (`Forwarded`, `X-Forwarded-*`, `X-Real-IP`).
- There is no TLS. On a LAN the token, the lyrics and the audio travel in clear text.

## Error envelope

```json
{ "error": { "code": "busy", "message": "human readable", "retryable": true, "details": null } }
```

| HTTP | `code` | `retryable` | When |
|---|---|---|---|
| 401 | `unauthorized` | false | missing / wrong bearer token |
| 403 | `forbidden_host` / `proxy_requires_token` | false | token-less server reached via a foreign Host or a proxy |
| 404 | `not_found` | false | unknown workflow, job, output index or path |
| 409 | `conflict` | false | output requested before the job succeeded, cancel of a finished job |
| 409 | `idempotency_conflict` | false | `Idempotency-Key` reused with different params |
| 409 | `idempotency_in_flight` | true | another request with the same key is being accepted right now |
| 411 | `length_required` | false | any `Transfer-Encoding` — send a plain `Content-Length` |
| 413 | `payload_too_large` | false | request body over `max_body_mb` (default 64 MB) |
| 422 | `invalid_params` | false | params failed validation; `details` lists `{loc, msg}` (inputs are never echoed) |
| 429 | `busy` | true | the GPU slot and the waiting queue are full. `Retry-After` header (seconds) is set |
| 500 | `internal` | true | unexpected server error (details only in the server log) |
| 503 | `insufficient_storage` | true | less than 2 GB free in `data_dir`; nothing was queued |
| 503 | `workflow_unavailable` | false | this workflow's engine is not installed or its model files are missing (the message says which; `/v1/health` → `workflows`); the other workflow keeps working |

## Endpoints

### `GET /healthz` and `GET /v1/health`

```json
{ "status": "ok", "version": "0.1.0", "running": 1, "waiting": 0, "max_waiting": 2, "waiting_for_peer": 0,
  "workflows": {
    "ace-step-1.5":   { "available": true,  "loaded": false, "reason": null },
    "minimax-music3": { "available": true,  "loaded": true,  "reason": null }
  } }
```

`503` with `"status": "degraded"` if the worker has stopped. `running`/`waiting` have the same meaning
as in vpipe-api, with one exception: while the job at the head of the queue is itself waiting for
another GPU server, the queued jobs are counted in `waiting_for_peer` instead of `waiting` (see
*Coordination*).

`workflows` (added after 0.1.0; the other fields are unchanged) has one entry per workflow:
`available` — its engine is installed with its model files, so jobs are accepted (otherwise a submit
gets `503 workflow_unavailable` and `reason` says why, without paths); `loaded` — its models are in
memory right now (`false` while idle is normal: the next job loads them; at most one engine is loaded
at a time). One unavailable workflow does not make `status` `degraded`.

### `GET /v1/capabilities`

What a client may ask for, without engine details. **This is the source of truth for client
descriptors** — a client must not assume values that are not listed here.

```json
{
  "version": "0.1.0",
  "queue": { "max_running": 1, "max_waiting": 2 },
  "workflows": [{
    "id": "ace-step-1.5",
    "output": { "media_type": "audio/wav", "sample_rates": [44100, 48000], "channels": 2,
                "sample_format": "pcm_s16le" },
    "prompt": { "max_chars": 1000, "engine_reads_tokens": 256 },
    "lyrics": { "max_chars": 4000, "engine_reads_tokens": 2048, "empty_means_instrumental": true },
    "duration_sec": { "min": 10.0, "max": 240.0, "default": 60.0 },
    "count": { "min": 1, "max": 4 },
    "seed": { "min": 0, "max": 4294967295, "per_track": "seed + i" },
    "instrumental": true,
    "typical_seconds": { "60s_track": 33 }
  }, {
    "id": "minimax-music3",
    "output": { "media_type": "audio/wav", "sample_rates": [44100], "channels": 2,
                "sample_format": "pcm_s16le" },
    "prompt": { "max_chars": 1000 },
    "lyrics": { "max_chars": 4000, "empty_means_instrumental": true },
    "prompt_and_lyrics_max_tokens": 5000,
    "duration_sec": { "min": 10.0, "max": 240.0, "default": 60.0, "upper_bound": true },
    "count": { "min": 1, "max": 4 },
    "seed": { "min": 0, "max": 4294967295, "per_track": "seed + i" },
    "instrumental": true,
    "typical_seconds": { "60s_track": 580 },
    "attribution": "MiniMax-Music3"
  }],
  "coordination": { "waits_for": ["vpipe-api", "klein-api"] }
}
```

`typical_seconds` is per 60-second track with the models already loaded, and only a hint. ACE-Step
reads only the first `engine_reads_tokens` tokens of the prompt and the lyrics; longer text is accepted
but the rest is ignored by the engine (a token is roughly one Japanese character or a few English
letters). MiniMax Music 3 reads all of it but refuses a prompt + lyrics over
`prompt_and_lyrics_max_tokens` (the job then fails with the engine's message). `upper_bound`: the
track can be shorter than `duration_sec`. `attribution`: the name MiniMax-Music3's licence asks a
commercial product or service to display in its UI.

### `GET /v1/workflows`

Lists workflows with the JSON Schema of their POST body (`params_schema`).

### `POST /v1/workflows/ace-step-1.5/jobs` and `POST /v1/workflows/minimax-music3/jobs`

Body = the params below (the same for both workflows), `Content-Type: application/json`.
Optional header `Idempotency-Key: <1–128 of [A-Za-z0-9._:-]>`.

| field | type | default | notes |
|---|---|---|---|
| `prompt` | string, 1–1000 chars | — | Style tags, e.g. `"J-pop, upbeat, female vocal, synth, 128 BPM"`. No control characters (tab/newline are fine). |
| `lyrics` | string, 0–4000 chars | `""` | Structured lyrics with `[verse]`, `[chorus]`, … tags. Empty means instrumental (both workflows). CRLF is turned into LF. |
| `duration_sec` | number 10–240 | `60` | Requested length (for `minimax-music3` an upper bound). The real length of each track is reported in the result. |
| `instrumental` | bool | `false` | `true` ignores `lyrics` and makes a track without vocals. |
| `seed` | int 0 … 2³²−1 \| null | random | Track *i* (0-based) uses `seed + i`; `seed + count − 1` must not exceed 2³²−1. |
| `count` | int 1–4 | `1` | Number of tracks. They run one after another inside the one job. |

For `ace-step-1.5` the sung language is taken from the lyrics: kana → Japanese, hangul → Korean;
otherwise the engine's language model decides. Anything else in the body — model names, steps, shift,
BPM, keys, file paths — is rejected with `422`. A workflow whose engine is unavailable answers
`503 workflow_unavailable` (after the body was validated) and queues nothing.

**Empty lyrics with `instrumental: false`** give an instrumental track on both workflows, and
`result.details.generation.instrumental` is `true`. ace-step-1.5 always worked this way; MiniMax Music 3
cannot write lyrics itself and refuses empty ones, so the server sends it its `[instrumental]` marker
instead of failing the job or rejecting a body the other workflow accepts. To get vocals, send lyrics.

**The same seed does not give the same track twice** on `ace-step-1.5`. The engine's language model
samples the song plan (structure, melody codes) and the result differs from run to run even with an
identical seed and prompt. Make several tracks and pick one. (Not yet checked for `minimax-music3`.)

Success is `202 Accepted`:

```json
{ "id": "job_01J9ZK3N2Q8V7W6X5Y4Z3A2B1C", "workflow": "ace-step-1.5", "status": "queued", "created_at": "2026-10-16T12:00:00Z" }
```

`429 busy`: nothing was queued; try again after `Retry-After` seconds. Answered before the body is read.

**Idempotency** works exactly as in wan-api: same key + same params → the existing job with `200 OK`;
same key + different params → `409 idempotency_conflict`; concurrent → `409 idempotency_in_flight`.

### `GET /v1/jobs/{job_id}`

```json
{
  "id": "job_…",
  "workflow": "ace-step-1.5",
  "status": "succeeded",
  "progress": 1.0,
  "queue_position": null,
  "waiting_for": null,
  "created_at": "…", "started_at": "…", "finished_at": "…",
  "estimate_seconds": 108.0,
  "cancel_requested": false,
  "timings": { "queue_seconds": 0.8, "coordination_seconds": 0.0, "backend_seconds": 66.2,
               "load_seconds": 25.1, "postprocess_seconds": 0.1, "total_seconds": 67.3 },
  "result": {
    "outputs": [
      { "index": 0, "media_type": "audio/wav", "duration_sec": 60.0, "sample_rate": 48000,
        "channels": 2, "seed_used": 12345 },
      { "index": 1, "media_type": "audio/wav", "duration_sec": 60.0, "sample_rate": 48000,
        "channels": 2, "seed_used": 12346 }
    ],
    "seed_used": 12345,
    "details": { "generation": { "requested_duration_sec": 60.0, "instrumental": false,
                                 "vocal_language": "ja", "engine": { "load_seconds": 25.1,
                                 "tracks": [{ "index": 0, "seconds": 33.0 }, { "index": 1, "seconds": 32.8 }] } } }
  },
  "error": null
}
```

- `status`: `queued` → `running` → `succeeded` | `failed` | `canceled` (one "l").
- `running` only while the job owns the GPU slot. A job that is first in line but is waiting for another
  GPU user on the machine (see *Coordination*) stays `queued` with `waiting_for: "vpipe-api"`.
- `progress`: `0.0..0.99` while running, over all tracks of the job; `1.0` when succeeded; `null` when
  unknown. For `minimax-music3` it is an estimate from elapsed time (the engine reports nothing until a
  track is done): at most 0.95 per track until that track is finished.
- `queue_position`: 1-based while `queued`, else `null`.
- `result.seed_used` is the seed of track 0 (wan-api/vpipe-api compatible); each output has its own.
- `timings.load_seconds` is present only when the job had to load the models (first job, or after the
  engine was released).
- `result.details.generation` for `minimax-music3` is `{requested_duration_sec, instrumental, engine}`
  (no `vocal_language`); `engine.tracks[]` has `index`, `seconds` and `gain_db`.
- `error` (only when `failed`): `{code, message, retryable}`; e.g. `generation_failed`, `timeout`,
  `invalid_output`, `server_restarted`, `server_stopped`, `engine_failed` (the engine could not
  load), `backend_unavailable` (its engine went missing after the job was queued), and:
  - `invalid_params`, **not retryable**: the engine refused the input after the job started — for
    `minimax-music3` a prompt + lyrics over 5000 tokens, or a model that ended before making any
    audio. The message starts with "the engine rejected the input:". Change the input before
    submitting again. The engine stays loaded.
  - `engine_unstoppable`, retryable: an engine worker survived SIGKILL (stuck in the GPU driver) and
    still holds its memory. Nothing is loaded next to it: every job fails this way until that
    process exits (the server checks again for each job) or the server is restarted (which stops it).
    The server itself keeps running.
- Restart behaviour is identical to wan-api (`server_stopped` / `server_restarted`, retryable; queued
  jobs run again).

### `GET /v1/jobs/{job_id}/output` and `GET /v1/jobs/{job_id}/output?index=N`

The WAV (`audio/wav`) of track `N` (default `0`). `409 conflict` unless the job has succeeded,
`404 not_found` for an index outside `0 … count-1`. Outputs are kept for `retention_days` (default 7).

Every output is checked before the job succeeds: a readable 16-bit PCM WAV, stereo, not empty, at one
of the workflow's `output.sample_rates` (`ace-step-1.5`: 44.1 or 48 kHz; `minimax-music3`: 44.1 kHz).
Its real length is `result.outputs[N].duration_sec`.

### `DELETE /v1/jobs/{job_id}`

As in wan-api: `200` `canceled`, `202` while the engine is stopping, `409` if already finished.

## Coordination with other GPU servers

The Mac has one GPU and one pool of unified memory; ACE-Step with its models loaded takes ~19 GB of
it, MiniMax Music 3 ~24 GB, and only one of them is loaded at a time. When `coordinate_with` is configured (a list of `{name, health_url, token}`), the job at the head of
the queue starts only after a check in which **every** listed server reports idle:
`running == 0 && waiting == 0` (and `outside_runs == 0` when present). The server remembers the time
it last saw any listed server busy. If none was seen busy within the last `settle_seconds` (default
30), a single all-idle check is enough; otherwise the job waits until `settle_seconds` after that
observation and checks again (clients often submit follow-up jobs right after one finishes). While
waiting the job stays `queued` with `waiting_for` set. An unreachable server counts as idle (logged as
a warning). Coordination is checked only before a job starts; a running job is never paused.

**Two servers that wait for each other.** klein-api can list ace-step-api in its own
`coordinate_with`. To keep the pair from waiting for each other forever, ace-step-api yields: while its
head job is waiting for a peer, `/v1/health` reports those queued jobs as `waiting_for_peer` and
`waiting: 0`, so klein-api sees it idle and goes first. A running ace-step-api job reports
`running: 1`, which makes klein-api wait.

vpipe-api does not read other servers' health: it only counts its own `vpipe --launch` processes. It
therefore never waits for ace-step-api; ace-step-api waits for it.

**Memory.** Between jobs the models stay loaded for `resident_idle_seconds` (default 120) so that
follow-up tracks skip the load (~25 s for ACE-Step). An engine is released at once when a job of the
other engine starts, when a listed server is seen busy while it idles, after any job that did not
succeed, and on shutdown.

## Workflow `ace-step-1.5`

- Output: WAV, 16-bit PCM, stereo, 48 kHz (44.1 kHz is allowed by the contract).
- Peaks are kept under −1 dBFS before the track is written (the engine's float output can exceed full
  scale, which 16-bit PCM would clip for good): a track that runs hot is first turned down as a whole,
  then a look-ahead limiter takes the rare remaining peaks. Loudness is otherwise left as the engine
  made it.
- The engine runs the turbo model and plans each song with its language model first. The settings are
  the server's; clients only choose the fields above.
- One job runs at a time; its tracks run one after another.

## Workflow `minimax-music3`

- Output: WAV, 16-bit PCM, stereo, 44.1 kHz.
- Same body and rules as `ace-step-1.5` (above), including "empty lyrics mean instrumental".
- `duration_sec` is an upper bound: the model may end a song early (a 62 s request once gave 22.9 s).
  The job succeeds; each track's real length is in `result.outputs[N].duration_sec`.
- The engine refuses a prompt + lyrics over 5000 tokens (it does not cut them); such a job fails
  with `invalid_params` (not retryable). A token is roughly one Japanese character or a few English letters, so
  only very long Japanese lyrics come near it.
- Slow: ~9 s of compute per second of audio on the target Mac (measured outside the server: 15 s in
  125 s, 62 s in 584 s), so a 60 s track takes ~10 minutes and `count: 4` × 240 s a few hours.
  `estimate_seconds` uses 30 + (10 + 9.5 × duration) × count; long tracks may run past it (progress
  then holds at 0.95 until the track is done).
- Peaks are kept under −1 dBFS by turning the whole track down (one gain, no limiter); the model itself
  clips its output to full scale.
- The engine runs its own settings (30 steps); clients only choose the fields above.
- Licence: MiniMax-Music3 Community License. Commercial use is allowed; a commercial product or service
  must prominently display "MiniMax-Music3" in its UI (`attribution` in capabilities); above US$20M
  yearly revenue a written authorisation from MiniMax is needed; services offering generation to third
  parties need safeguards against infringing uses and outputs; content published to a public
  environment must be disclosed as machine-generated (acceptable-use policy).
