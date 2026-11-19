# Feature Store PIT Lab

A compact Python backend demo for one hard ML-platform problem: making training-time features match what a model could actually have seen at prediction time. It stores immutable, timestamped feature events in SQLite and builds point-in-time snapshots through FastAPI.

## Why point-in-time correctness matters

A feature can arrive late or be corrected after a label was observed. Joining on the latest database row leaks future information into training data and makes offline scores look better than production behavior. This demo keeps event time separate from ingestion time and always selects the newest event at or before each sample's `as_of` timestamp.

## What it demonstrates

- Idempotent event ingestion: an identical `event_id` is a no-op; reusing it with different content returns a conflict.
- Event-time retrieval: delayed events are placed according to when they happened, not when they arrived.
- Deterministic ties: events at the same timestamp are resolved by event ID.
- Training-set materialization: each labeled sample receives an as-of feature snapshot, with missing features represented as `null`.
- Optional freshness limits: stale latest values become missing rather than silently falling back to older data.
- Timezone-aware inputs normalized to UTC; indexed SQLite history and bounded API requests.

## Run locally

Python 3.11+ is required.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
uvicorn feature_store.api:app --reload
```

The interactive API is at http://127.0.0.1:8000/docs. Store a feature event:

```bash
curl -X POST http://127.0.0.1:8000/events \
  -H 'Content-Type: application/json' \
  -d '{"event_id":"e-1","entity_id":"customer-42","feature_name":"plan_tier","value":"pro","event_time":"2026-01-01T12:00:00Z"}'
```

Read the state that would have been available at a past time:

```bash
curl 'http://127.0.0.1:8000/entities/customer-42/features?as_of=2026-01-02T00%3A00%3A00Z&feature_names=plan_tier'
```

`POST /training-data` accepts a list of labeled samples, requested feature names and an optional `max_age_seconds`. Set `FEATURE_STORE_DB` to choose a SQLite file; the default is `feature-store.db` in the current directory. Run `pytest -q` for tests.

## Design boundaries

This is an educational, single-process SQLite implementation using synthetic examples. It does not include distributed serving, a streaming broker, feature definitions/ownership, online cache, access control, schema migration, model training or a production-scale point-in-time join engine. Feature values are limited to JSON scalar types, and the tie-break rule is deterministic rather than a claim about business precedence. Do not use it for real customer data.
