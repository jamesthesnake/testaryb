from fastapi.testclient import TestClient

from backend.submission import api
from backend.submission.macro_specialist import Citation, MacroAnalysisResponse, MacroMetric


class StubSpecialist:
    def ask(self, query: str) -> MacroAnalysisResponse:
        assert query == "What is the current GDP of the US?"
        return MacroAnalysisResponse(
            answer="US nominal GDP is 30,000.00 billions of dollars, SAAR.",
            citations=[
                Citation(
                    source="FRED - Federal Reserve Bank of St. Louis",
                    series_id="GDP",
                    description="US Nominal Gross Domestic Product (Billions of Dollars, SAAR)",
                    url="https://fred.stlouisfed.org/series/GDP",
                )
            ],
            metrics=[
                MacroMetric(
                    metric_key="nominal_gdp_level",
                    label="US nominal GDP level",
                    series_key="GDP",
                    series_id="GDP",
                    value=30000.0,
                    unit="billions of dollars, SAAR",
                    date="2025-01-01",
                    calculation="Latest FRED GDP observation.",
                    description="US Nominal Gross Domestic Product (Billions of Dollars, SAAR)",
                )
            ],
            warnings=[],
        )


def test_ask_endpoint_returns_answer_metrics_citations_and_selected_series(monkeypatch):
    monkeypatch.setattr(api, "get_specialist", lambda: StubSpecialist())
    client = TestClient(api.app)

    response = client.post("/ask", json={"query": "  What is the current GDP of the US?  "})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("US nominal GDP")
    assert body["selectedSeries"] == "GDP"
    assert body["citations"][0]["series_id"] == "GDP"
    assert body["metrics"][0]["metric_key"] == "nominal_gdp_level"
    assert body["warnings"] == []


def test_ask_endpoint_rejects_empty_query():
    client = TestClient(api.app)

    response = client.post("/ask", json={"query": "   "})

    assert response.status_code == 422
    assert response.json()["detail"] == "Query cannot be empty."


def test_series_endpoint_rejects_unknown_series():
    client = TestClient(api.app)

    response = client.get("/series/UNKNOWN")

    assert response.status_code == 404
    assert response.json()["detail"] == "Unknown series key: UNKNOWN"
