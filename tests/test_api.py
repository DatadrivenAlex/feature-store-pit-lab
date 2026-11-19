import pytest
from fastapi.testclient import TestClient

from feature_store.api import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FEATURE_STORE_DB", str(tmp_path / "features.sqlite"))
    with TestClient(app) as test_client:
        yield test_client


def event(event_id, name, value, time, entity="customer-1"):
    return {
        "event_id": event_id,
        "entity_id": entity,
        "feature_name": name,
        "value": value,
        "event_time": time,
    }


def test_health_and_ingest(client):
    assert client.get("/health").json() == {"status": "ok"}
    response = client.post("/events", json=event("e1", "plan_tier", "pro", "2026-01-01T12:00:00Z"))
    assert response.status_code == 201
    assert response.json()["created"] is True
    assert response.json()["value"] == "pro"


def test_point_in_time_uses_event_time_not_arrival_order(client):
    client.post("/events", json=event("early", "score", 0.2, "2026-01-01T00:00:00Z"))
    client.post("/events", json=event("future", "score", 0.9, "2026-01-03T00:00:00Z"))
    # A delayed event arrives after the newer event but belongs before the cutoff.
    client.post("/events", json=event("late", "score", 0.5, "2026-01-02T00:00:00Z"))
    response = client.get(
        "/entities/customer-1/features",
        params={"as_of": "2026-01-02T12:00:00Z", "feature_names": "score"},
    )
    assert response.json()["features"] == {"score": 0.5}


def test_timezone_normalization_and_tie_break_are_deterministic(client):
    client.post("/events", json=event("z", "rank", 2, "2026-02-01T01:00:00+01:00"))
    client.post("/events", json=event("a", "rank", 1, "2026-02-01T00:00:00Z"))
    features = client.get(
        "/entities/customer-1/features", params={"as_of": "2026-02-01T00:00:00Z"}
    ).json()["features"]
    assert features == {"rank": 2}


def test_idempotent_ingest_and_conflicting_event_id(client):
    payload = event("same", "active", True, "2026-01-01T00:00:00Z")
    assert client.post("/events", json=payload).status_code == 201
    duplicate = client.post("/events", json=payload)
    assert duplicate.status_code == 201
    assert duplicate.json()["created"] is False
    assert client.post("/events", json={**payload, "value": False}).status_code == 409


def test_training_data_prevents_future_feature_leakage(client):
    client.post("/events", json=event("old", "balance", 10, "2026-03-01T00:00:00Z"))
    client.post("/events", json=event("new", "balance", 99, "2026-03-03T00:00:00Z"))
    payload = {
        "features": ["balance", "tenure_months"],
        "samples": [
            {
                "sample_id": "s1",
                "entity_id": "customer-1",
                "label": "retained",
                "as_of": "2026-03-02T00:00:00Z",
            }
        ],
    }
    result = client.post("/training-data", json=payload).json()
    assert result["rows"][0]["features"] == {"balance": 10, "tenure_months": None}


def test_max_age_does_not_fall_back_to_older_value(client):
    client.post("/events", json=event("old", "risk", 0.1, "2026-04-01T00:00:00Z"))
    client.post("/events", json=event("new", "risk", 0.4, "2026-04-09T00:00:00Z"))
    result = client.get(
        "/entities/customer-1/features",
        params={
            "as_of": "2026-04-10T00:00:00Z",
            "feature_names": "risk",
            "max_age_seconds": 86400,
        },
    ).json()
    assert result["features"] == {"risk": 0.4}
    stale = client.get(
        "/entities/customer-1/features",
        params={
            "as_of": "2026-04-11T00:00:00Z",
            "feature_names": "risk",
            "max_age_seconds": 86400,
        },
    ).json()
    assert stale["features"] == {"risk": None}


def test_validation_rejects_naive_timestamps_and_bad_features(client):
    assert (
        client.post(
            "/events", json=event("bad-time", "value", 1, "2026-01-01T00:00:00")
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/events", json=event("bad-name", "Risk Score", 1, "2026-01-01T00:00:00Z")
        ).status_code
        == 422
    )
    assert (
        client.get(
            "/entities/customer-1/features", params={"as_of": "2026-01-01T00:00:00"}
        ).status_code
        == 422
    )
