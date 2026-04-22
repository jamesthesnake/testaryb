# AI Macro Specialist — Submission README
**James Hennessy**

---

## Project Overview

A full-stack AI macroeconomics assistant. A React dashboard shows live FRED economic indicators; a chat panel lets you ask natural-language questions and get data-grounded answers with citations and derived metrics.

**Stack:** Python / FastAPI (backend) · React + TypeScript + Vite (frontend) · OpenAI GPT-4o + text-embedding-3-small · FRED API · Pinecone (vector storage)

---

## Setup

### Prerequisites

- Python 3.12+, [Poetry](https://python-poetry.org/)
- Node.js 18+, [pnpm](https://pnpm.io/)
- API keys for OpenAI, FRED, and Pinecone (provided)

### 1. Configure environment variables

Create `backend/.env` (or copy `.env` at the repo root into `backend/`):

```env
OPENAI_API_KEY=sk-...
FRED_API_KEY=...
PINECONE_API_KEY=pcsk_...
```

### 2. Install backend dependencies

```bash
cd backend
poetry install
```

### 3. Start the backend API

Run from the **repo root** so the `backend` package is importable:

```bash
cd takehome-macro-specialist-main
PYTHONPATH="$(pwd)" poetry --directory backend run uvicorn backend.submission.api:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.

### 4. Install and start the frontend

```bash
cd frontend
pnpm install
pnpm dev
```

Open `http://localhost:5173`. The Vite dev server proxies `/ask`, `/health`, and `/series` to the backend automatically.

### 5. Run tests

**Backend:**
```bash
cd backend
PYTHONPATH="$(pwd)/.." poetry run pytest
```

**Frontend:**
```bash
cd frontend
pnpm test
```

---

## API Reference

All endpoints are served by the FastAPI backend on port 8000.

### `GET /health`
Liveness check.
```json
{ "status": "ok" }
```

### `GET /series`
List all available FRED series.
```json
[
  { "key": "GDP", "series_id": "GDP", "description": "US Nominal Gross Domestic Product..." },
  ...
]
```

### `GET /series/{series_key}?limit=60`
Fetch time-series data for a named series. `limit` controls the number of observations (default 60, max 500).

**Series keys:** `GDP`, `REAL_GDP`, `CPI`, `EUROZONE_INFLATION`, `UNEMPLOYMENT`, `JAPAN_UNEMPLOYMENT`, `FEDERAL_FUNDS_RATE`, `USD_EUR`

```json
{
  "key": "UNEMPLOYMENT",
  "series_id": "UNRATE",
  "description": "US Unemployment Rate (Percent)",
  "points": [{ "date": "2025-01-01", "value": 4.1 }, ...],
  "citation": { "source": "FRED...", "series_id": "UNRATE", "url": "https://fred.stlouisfed.org/series/UNRATE" }
}
```

### `POST /ask`
Ask a macroeconomic question. Returns a GPT-4o answer grounded in live FRED data.

**Request:**
```json
{ "query": "What is the current unemployment rate in Japan?" }
```

**Response:**
```json
{
  "answer": "As of March 2025, Japan's unemployment rate stands at 2.5%...",
  "citations": [{ "source": "FRED...", "series_id": "LRUN64TTJPM156S", "url": "..." }],
  "metrics": [{ "metric_key": "japan_unemployment_rate", "value": 2.5, "unit": "%", "date": "2025-03-01", ... }],
  "warnings": [],
  "selectedSeries": "JAPAN_UNEMPLOYMENT"
}
```

---

## Design Approach

### MacroSpecialist class

`MacroSpecialist` is the central orchestrator. For each query it:

1. **Retrieves** the most relevant FRED series via embedding similarity (Pinecone) with deterministic guardrails for geography/topic disambiguation (e.g. "Japan" + "unemployment" always routes to `JAPAN_UNEMPLOYMENT` regardless of embedding score).
2. **Fetches** live observations from the FRED API and computes derived metrics (YoY growth, QoQ annualized growth, inflation rate) deterministically in Python — these are not left to the LLM.
3. **Builds** a structured context string containing the derived metrics and recent observations, passed to GPT-4o as grounding data.
4. **Returns** the answer with citations (FRED URLs) and the computed metrics so the frontend can display them independently of the prose answer.

### Retrieval system

Series metadata (FRED ID, description, keywords, search text) is consolidated in `SERIES_REGISTRY`, a dict of `FREDSeriesConfig` dataclasses. At startup, each entry is embedded with `text-embedding-3-small` and stored in a Pinecone serverless index (`takehome-macro-james-hennessy`). Subsequent cold starts skip re-embedding by checking `total_vector_count`. When Pinecone is unavailable, the system falls back to in-memory cosine similarity, and further to lexical overlap scoring.

### Accuracy and reliability

Derived metrics (growth rates, inflation) are computed with explicit formulas in Python before the LLM sees any data. GPT-4o is instructed to treat these as authoritative and not re-derive them. FRED data returning non-numeric values (FRED uses `"."` for missing observations) is coerced with `pd.to_numeric(..., errors="coerce")` and dropped before any calculation. The `DataRetrievalError` model captures per-series failures and surfaces them as structured warnings so the LLM can acknowledge missing data rather than hallucinate it.

### Frontend

A single-page React app with two panels: an economic indicator dashboard (8 cards + a line chart) and an analyst chat. The dashboard pre-fetches all series in parallel on load. After each `/ask` response, the chart switches to the `selectedSeries` returned by the backend — routing logic lives in one place (the backend), not duplicated in the frontend.

---

## Assumptions and Design Decisions

- **FRED only.** The spec lists FRED, World Bank, and IMF as options; FRED alone covers all seven required questions and has a clean REST API.
- **8 series.** GDP, Real GDP, CPI, Eurozone Inflation, US Unemployment, Japan Unemployment, Federal Funds Rate, and USD/EUR — sufficient to answer every question in the spec with geography-aware guardrail routing.
- **Deterministic metrics over LLM arithmetic.** Growth rates and inflation are pre-computed so answers are reproducible and verifiable against the cited FRED data.
- **Pinecone serverless on AWS us-east-1.** Index is created automatically on first startup if it does not exist. The index name is `takehome-macro-james-hennessy`.
- **No streaming.** GPT-4o responses are returned in full; streaming would require SSE plumbing on both ends for limited UX gain at this latency.

---

## Potential Future Improvements

- **Broader data coverage.** Add World Bank / IMF series for emerging markets; make `SERIES_REGISTRY` database-backed so new series can be added without a deploy.
- **Evaluation harness.** Instrument answer quality with a reference QA set (the seven spec questions make a natural starting point) scored by GPT-4o-as-judge, tracked over time and across models.
- **Streaming responses.** SSE from FastAPI + incremental rendering in React would noticeably improve perceived latency for long answers.
- **Report generation.** Extend `MacroSpecialist.ask` with a `generate_report` method that structures multi-series analysis as a PDF or structured JSON, using the Instructor library already in the dependency tree.
- **Rate limiting and auth.** Add per-IP rate limiting (e.g. `slowapi`) and API key auth before any public deployment.
- **Richer chart interactions.** Replace the custom SVG `LineChart` with Recharts or Vega-Lite to get tooltips, zoom, and multi-series overlays without maintaining drawing math.
