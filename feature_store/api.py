"""HTTP API for an educational event-time feature store."""

import math
import re
from datetime import datetime
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from feature_store.store import ingest, retrieve

app = FastAPI(title="Feature Store PIT Lab", version="0.1.0")
_NAME = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")


class FeatureEvent(BaseModel):
    event_id: str = Field(min_length=1, max_length=128)
    entity_id: str = Field(min_length=1, max_length=128)
    feature_name: str = Field(min_length=1, max_length=64)
    value: str | int | float | bool
    event_time: datetime

    @field_validator("feature_name")
    @classmethod
    def valid_feature_name(cls, value: str) -> str:
        if not _NAME.fullmatch(value):
            raise ValueError("feature_name must be lowercase snake-case style")
        return value

    @field_validator("event_time")
    @classmethod
    def aware_event_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("event_time must include a timezone")
        return value

    @field_validator("value")
    @classmethod
    def finite_number(cls, value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("numeric feature values must be finite")
        return value


class TrainingSample(BaseModel):
    sample_id: str = Field(min_length=1, max_length=128)
    entity_id: str = Field(min_length=1, max_length=128)
    label: str = Field(min_length=1, max_length=256)
    as_of: datetime

    @field_validator("as_of")
    @classmethod
    def aware_as_of(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("as_of must include a timezone")
        return value


class TrainingSetRequest(BaseModel):
    features: list[str] = Field(min_length=1, max_length=100)
    samples: list[TrainingSample] = Field(min_length=1, max_length=500)
    max_age_seconds: int | None = Field(default=None, ge=0, le=31_536_000)

    @field_validator("features")
    @classmethod
    def valid_features(cls, values: list[str]) -> list[str]:
        if len(set(values)) != len(values):
            raise ValueError("features must be unique")
        if any(not _NAME.fullmatch(value) for value in values):
            raise ValueError("feature names must be lowercase snake-case style")
        return values


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/events", status_code=201)
def create_event(event: FeatureEvent) -> dict[str, Any]:
    try:
        result, created = ingest(
            event_id=event.event_id,
            entity_id=event.entity_id,
            feature_name=event.feature_name,
            value=event.value,
            event_time=event.event_time,
        )
    except ValueError as error:
        if "already exists" in str(error):
            raise HTTPException(status_code=409, detail=str(error)) from error
        raise HTTPException(status_code=422, detail=str(error)) from error
    result["created"] = created
    return result


@app.get("/entities/{entity_id}/features")
def get_entity_features(
    entity_id: str,
    as_of: datetime,
    feature_names: Annotated[list[str] | None, Query()] = None,
    max_age_seconds: Annotated[int | None, Query(ge=0, le=31_536_000)] = None,
) -> dict[str, Any]:
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise HTTPException(status_code=422, detail="as_of must include a timezone")
    if feature_names and any(not _NAME.fullmatch(name) for name in feature_names):
        raise HTTPException(status_code=422, detail="invalid feature name")
    if feature_names and len(set(feature_names)) != len(feature_names):
        raise HTTPException(status_code=422, detail="feature_names must be unique")
    return {
        "entity_id": entity_id,
        "as_of": as_of.isoformat(),
        "features": retrieve(
            entity_id=entity_id,
            as_of=as_of,
            feature_names=feature_names,
            max_age_seconds=max_age_seconds,
        ),
    }


@app.post("/training-data")
def materialize_training_data(request: TrainingSetRequest) -> dict[str, Any]:
    rows = []
    for sample in request.samples:
        rows.append(
            {
                "sample_id": sample.sample_id,
                "entity_id": sample.entity_id,
                "label": sample.label,
                "as_of": sample.as_of.isoformat(),
                "features": retrieve(
                    entity_id=sample.entity_id,
                    as_of=sample.as_of,
                    feature_names=request.features,
                    max_age_seconds=request.max_age_seconds,
                ),
            }
        )
    return {"features": request.features, "rows": rows}
