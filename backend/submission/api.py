from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.submission.macro_specialist import (
    Citation,
    DataRetrievalError,
    SERIES_REGISTRY,
    MacroSpecialist,
    MacroMetric,
)

DEFAULT_CORS_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    metrics: list[MacroMetric] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    selectedSeries: str | None = None


class SeriesPoint(BaseModel):
    date: str
    value: float


class SeriesResponse(BaseModel):
    key: str
    series_id: str
    description: str
    points: list[SeriesPoint]
    citation: Citation


class SeriesMetadata(BaseModel):
    key: str
    series_id: str
    description: str


def _cors_origins() -> list[str]:
    raw_origins = os.environ.get("CORS_ORIGINS")
    if not raw_origins:
        return list(DEFAULT_CORS_ORIGINS)
    return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]


@lru_cache(maxsize=1)
def get_specialist() -> MacroSpecialist:
    return MacroSpecialist()


def _citation_for_series(series_key: str) -> Citation:
    cfg = SERIES_REGISTRY[series_key]
    return Citation(
        source="FRED - Federal Reserve Bank of St. Louis",
        series_id=cfg.fred_id,
        description=cfg.description,
        url=f"https://fred.stlouisfed.org/series/{cfg.fred_id}",
    )


def _series_key_from_citations(citations: list[Citation]) -> str | None:
    if not citations:
        return None

    first_series_id = citations[0].series_id
    for key, cfg in SERIES_REGISTRY.items():
        if cfg.fred_id == first_series_id:
            return key
    return None


def _error_detail(error: Exception) -> dict[str, Any]:
    if isinstance(error, DataRetrievalError):
        return {
            "message": str(error),
            "issues": [issue.model_dump() for issue in error.issues],
        }

    return {
        "message": "The macro specialist backend could not complete the request.",
        "error": str(error),
    }


app = FastAPI(title="Macro Specialist API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/series", response_model=list[SeriesMetadata])
def list_series() -> list[SeriesMetadata]:
    return [
        SeriesMetadata(key=key, series_id=cfg.fred_id, description=cfg.description)
        for key, cfg in SERIES_REGISTRY.items()
    ]


@app.get("/series/{series_key}", response_model=SeriesResponse)
def get_series(
    series_key: str,
    limit: int = Query(default=60, ge=1, le=500),
) -> SeriesResponse:
    if series_key not in SERIES_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown series key: {series_key}")

    try:
        specialist = get_specialist()
        df = specialist.get_series_df(series_key, limit=limit).dropna(subset=["value"])
        points = [
            SeriesPoint(date=str(row["date"]), value=float(row["value"]))
            for _, row in df.iterrows()
        ]
        cfg = SERIES_REGISTRY[series_key]
        return SeriesResponse(
            key=series_key,
            series_id=cfg.fred_id,
            description=cfg.description,
            points=points,
            citation=_citation_for_series(series_key),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_error_detail(exc)) from exc


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Query cannot be empty.")

    try:
        result = get_specialist().ask(query)
        return AskResponse(
            answer=result.answer,
            citations=result.citations,
            metrics=result.metrics,
            warnings=result.warnings,
            selectedSeries=_series_key_from_citations(result.citations),
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_error_detail(exc)) from exc
