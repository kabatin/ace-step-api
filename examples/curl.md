# curl examples

```sh
API=http://127.0.0.1:8768
AUTH=()                                   # or: AUTH=(-H "Authorization: Bearer $ACE_STEP_API_TOKEN")
```

## Is it up? What can it do?

```sh
curl -s "${AUTH[@]}" "$API/healthz"
# {"status":"ok","version":"0.1.1","running":0,"waiting":0,"max_waiting":2,"waiting_for_peer":0,
#  "workflows":{"ace-step-1.5":{…},"minimax-music3":{…}}}
curl -s "${AUTH[@]}" "$API/v1/capabilities" | jq '{queue, coordination, w: .workflows[] | {id, duration_sec, count, typical_seconds}}'
```

## A song with Japanese lyrics, two takes

```sh
jq -n --arg lyrics $'[verse]\n夜明けの街を 駆け抜けてく\nまだ知らない 景色を探して\n[chorus]\n光の中へ 手を伸ばして\n今 この瞬間を 走り出そう' '{
  prompt: "J-pop, upbeat, female vocal, synth, 128 BPM",
  lyrics: $lyrics,
  duration_sec: 60,
  seed: 42,
  count: 2
}' > body.json
curl -s "${AUTH[@]}" -X POST "$API/v1/workflows/ace-step-1.5/jobs" \
  -H 'Content-Type: application/json' -H 'Idempotency-Key: song-001' --data-binary @body.json
# → 202 {"id":"job_…","workflow":"ace-step-1.5","status":"queued","created_at":"…"}
# Sending the same Idempotency-Key + body again returns the same job (200), never a duplicate.
```

The same seed does not reproduce a track: make a few and pick one.

## Instrumental

```sh
curl -s "${AUTH[@]}" -X POST "$API/v1/workflows/ace-step-1.5/jobs" \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "lo-fi hip hop, mellow piano, vinyl crackle, 80 BPM", "instrumental": true, "duration_sec": 90}'
# an empty "lyrics" (the default) also means instrumental
```

## Poll, download, cancel

```sh
JOB=job_…
curl -s "${AUTH[@]}" "$API/v1/jobs/$JOB" | jq '{status, progress, queue_position, waiting_for, timings, error}'
curl -s "${AUTH[@]}" "$API/v1/jobs/$JOB" | jq '.result.outputs[] | {index, duration_sec, seed_used}'
curl -s "${AUTH[@]}" -o take-0.wav "$API/v1/jobs/$JOB/output"           # index 0; 409 until succeeded
curl -s "${AUTH[@]}" -o take-1.wav "$API/v1/jobs/$JOB/output?index=1"   # 404 outside 0 … count-1
curl -s "${AUTH[@]}" -X DELETE "$API/v1/jobs/$JOB"                       # 200 canceled / 202 stopping
```

## Waiting for another GPU server

With `[[coordinate_with]]` configured, the first job in line stays queued while e.g. vpipe-api is busy:

```json
{ "status": "queued", "queue_position": 1, "waiting_for": "vpipe-api", "progress": null }
```

Meanwhile `/v1/health` reports the queue as `waiting_for_peer`, not `waiting`, so servers that wait
for ace-step-api (klein-api) go first instead of waiting for each other.

## Handling `429 busy`

One job runs and `max_waiting` (default 2) wait. When full, the server answers before reading the body:

```
HTTP/1.1 429 Too Many Requests
Retry-After: 30
{"error":{"code":"busy","message":"the GPU slot and the waiting queue are full","retryable":true,"details":null}}
```

Wait `Retry-After` seconds and submit again. Nothing was queued.

## MiniMax Music 3

The same body on the other workflow. It is slow (~10 minutes per 60 s track) and its `duration_sec` is an
upper bound: check `duration_sec` of each output.

```sh
curl -s "${AUTH[@]}" "$API/v1/health" | jq .workflows
# {"ace-step-1.5":{"available":true,"loaded":false,"reason":null},
#  "minimax-music3":{"available":true,"loaded":false,"reason":null}}

jq -n --arg lyrics $'[verse]\nMorning light across the room\n[chorus]\nSing with me' '{
  prompt: "warm acoustic pop, 96 BPM, intimate female vocal",
  lyrics: $lyrics,
  duration_sec: 60,
  seed: 7
}' > song.json
curl -s "${AUTH[@]}" -X POST "$API/v1/workflows/minimax-music3/jobs" \
  -H 'Content-Type: application/json' --data-binary @song.json
# → 202 …; 503 {"error":{"code":"workflow_unavailable",…}} if MiniMax is not installed on the server

curl -s "${AUTH[@]}" -X POST "$API/v1/workflows/minimax-music3/jobs" \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "lo-fi hip hop, mellow piano, vinyl crackle, 80 BPM", "instrumental": true, "duration_sec": 30}'
```

A commercial product using these tracks must display "MiniMax-Music3" in its UI (see the README's
*License*).

