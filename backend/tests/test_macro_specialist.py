import json

import pandas as pd
import pytest

from backend.submission import macro_specialist as subject
from backend.submission.macro_specialist import (
    DataRetrievalError,
    MacroSpecialist,
    RetrievalResult,
)


class FakeFredResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_fetch_fred_parses_observations_and_numeric_values(monkeypatch):
    def fake_urlopen(url: str, timeout: int):
        assert "series_id=CPIAUCSL" in url
        assert timeout == 15
        return FakeFredResponse({
            "observations": [
                {"date": "2025-01-01", "value": "320.1"},
                {"date": "2024-12-01", "value": "."},
            ],
        })

    monkeypatch.setattr(subject.urllib.request, "urlopen", fake_urlopen)

    df = subject._fetch_fred("CPIAUCSL", limit=2, api_key="test-key")

    assert list(df.columns) == ["date", "value"]
    assert df.loc[0, "value"] == pytest.approx(320.1)
    assert pd.isna(df.loc[1, "value"])


def test_fetch_fred_raises_on_fred_error(monkeypatch):
    monkeypatch.setattr(
        subject.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: FakeFredResponse({"error_message": "bad api key"}),
    )

    with pytest.raises(RuntimeError, match="bad api key"):
        subject._fetch_fred("GDP", limit=1, api_key="bad-key")


def test_retrieve_series_guardrails_route_geography_and_gdp_growth(monkeypatch):
    specialist = MacroSpecialist()
    monkeypatch.setattr(specialist, "_embed_texts", lambda _texts: (_ for _ in ()).throw(RuntimeError("offline")))

    japan_results = specialist.retrieve_series("What is the unemployment rate in Japan?", top_k=2)
    growth_results = specialist.retrieve_series("Make a plot of US GDP growth over the past 10 years.", top_k=2)

    assert japan_results[0].series_key == "JAPAN_UNEMPLOYMENT"
    assert "guardrail" in japan_results[0].source
    assert growth_results[0].series_key == "REAL_GDP"
    assert "guardrail" in growth_results[0].source


def test_series_metrics_compute_real_gdp_growth_and_cpi_inflation():
    specialist = MacroSpecialist()
    real_gdp = pd.DataFrame({
        "date": ["2024-01-01", "2024-04-01", "2024-07-01", "2024-10-01", "2025-01-01"],
        "value": [100.0, 101.0, 102.0, 103.0, 105.0],
    })
    cpi = pd.DataFrame({
        "date": [f"2024-{month:02d}-01" for month in range(1, 13)] + ["2025-01-01"],
        "value": [100 + month for month in range(13)],
    })

    real_gdp_metrics = {metric.metric_key: metric for metric in specialist._series_metrics("REAL_GDP", real_gdp)}
    cpi_metrics = {metric.metric_key: metric for metric in specialist._series_metrics("CPI", cpi)}

    assert real_gdp_metrics["real_gdp_level"].value == pytest.approx(105.0)
    assert real_gdp_metrics["real_gdp_qoq_annualized_growth"].value == pytest.approx((((105 / 103) ** 4) - 1) * 100)
    assert real_gdp_metrics["real_gdp_yoy_growth"].value == pytest.approx(5.0)
    assert cpi_metrics["us_cpi_yoy_inflation"].value == pytest.approx(12.0)


def test_build_context_returns_citations_metrics_and_no_warnings(monkeypatch):
    specialist = MacroSpecialist()
    monkeypatch.setattr(
        specialist,
        "retrieve_series",
        lambda _query: [RetrievalResult(series_key="UNEMPLOYMENT", score=1.0, source="test")],
    )
    monkeypatch.setattr(
        specialist,
        "_load_series",
        lambda _key, limit=24: pd.DataFrame({
            "date": ["2025-01-01", "2025-02-01", "2025-03-01"],
            "value": [4.1, 4.2, 4.3],
        }),
    )

    context, citations, metrics, warnings = specialist._build_context("What is unemployment?")

    assert "Derived metrics" in context
    assert citations[0].series_id == "UNRATE"
    assert metrics[0].metric_key == "us_unemployment_rate"
    assert metrics[0].value == pytest.approx(4.3)
    assert warnings == []


def test_build_context_fails_closed_when_no_series_data_loads(monkeypatch):
    specialist = MacroSpecialist()
    monkeypatch.setattr(
        specialist,
        "retrieve_series",
        lambda _query: [RetrievalResult(series_key="CPI", score=1.0, source="test")],
    )
    monkeypatch.setattr(
        specialist,
        "_load_series",
        lambda _key, limit=24: pd.DataFrame({"date": ["2025-01-01"], "value": [float("nan")]}),
    )

    with pytest.raises(DataRetrievalError) as exc:
        specialist._build_context("What is inflation?")

    assert "No FRED data" in str(exc.value)
    assert exc.value.issues[0].series_key == "CPI"
