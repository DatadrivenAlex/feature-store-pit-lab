"""SQLite-backed event history and point-in-time feature retrieval."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from typing import Any


def connect() -> sqlite3.Connection:
    db = sqlite3.connect(os.environ.get("FEATURE_STORE_DB", "feature-store.db"))
    db.row_factory = sqlite3.Row
    return db


def init_db() -> None:
    with closing(connect()) as db, db:
        db.execute(
            """CREATE TABLE IF NOT EXISTS feature_events (
                event_id TEXT PRIMARY KEY,
                entity_id TEXT NOT NULL,
                feature_name TEXT NOT NULL,
                value_json TEXT NOT NULL,
                event_time TEXT NOT NULL,
                ingested_at TEXT NOT NULL
            )"""
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_feature_history "
            "ON feature_events(entity_id, feature_name, event_time DESC, event_id DESC)"
        )


def timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must include a timezone")
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def ingest(
    *, event_id: str, entity_id: str, feature_name: str, value: Any, event_time: datetime
) -> tuple[dict[str, Any], bool]:
    init_db()
    event_time_text = timestamp(event_time)
    value_text = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    ingested_at = datetime.now(UTC).isoformat(timespec="microseconds")
    with closing(connect()) as db, db:
        existing = db.execute(
            "SELECT * FROM feature_events WHERE event_id = ?", (event_id,)
        ).fetchone()
        if existing is not None:
            same = (
                existing["entity_id"] == entity_id
                and existing["feature_name"] == feature_name
                and existing["value_json"] == value_text
                and existing["event_time"] == event_time_text
            )
            if not same:
                raise ValueError("event_id already exists with different content")
            return dict(existing), False
        db.execute(
            """INSERT INTO feature_events
               (event_id, entity_id, feature_name, value_json, event_time, ingested_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (event_id, entity_id, feature_name, value_text, event_time_text, ingested_at),
        )
        row = db.execute("SELECT * FROM feature_events WHERE event_id = ?", (event_id,)).fetchone()
    result = dict(row)
    result["value"] = json.loads(result.pop("value_json"))
    return result, True


def retrieve(
    *,
    entity_id: str,
    as_of: datetime,
    feature_names: list[str] | None = None,
    max_age_seconds: int | None = None,
) -> dict[str, Any]:
    """Return each feature's latest event at or before as_of, excluding stale values."""
    init_db()
    cutoff = timestamp(as_of)
    with closing(connect()) as db:
        if feature_names:
            placeholders = ",".join("?" for _ in feature_names)
            rows = db.execute(
                f"""SELECT * FROM feature_events
                    WHERE entity_id = ? AND feature_name IN ({placeholders})
                      AND event_time <= ?
                    ORDER BY feature_name, event_time DESC, event_id DESC""",
                (entity_id, *feature_names, cutoff),
            ).fetchall()
        else:
            rows = db.execute(
                """SELECT * FROM feature_events
                   WHERE entity_id = ? AND event_time <= ?
                   ORDER BY feature_name, event_time DESC, event_id DESC""",
                (entity_id, cutoff),
            ).fetchall()

    selected: dict[str, sqlite3.Row] = {}
    for row in rows:
        selected.setdefault(row["feature_name"], row)
    names = feature_names if feature_names is not None else sorted(selected)
    result: dict[str, Any] = {name: None for name in names}
    cutoff_dt = datetime.fromisoformat(cutoff)
    for name, row in selected.items():
        event_dt = datetime.fromisoformat(row["event_time"])
        age = (cutoff_dt - event_dt).total_seconds()
        if max_age_seconds is None or age <= max_age_seconds:
            result[name] = json.loads(row["value_json"])
    return result
