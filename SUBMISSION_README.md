# AI Macro Specialist
**James Hennessy**

---

## Overview

This project is a full-stack AI macroeconomics assistant built around a central `MacroSpecialist` class that retrieves live economic data from FRED and uses GPT-4o to answer natural-language questions with precision and citation. A React dashboard visualises eight key indicators in real time; a chat panel lets users ask anything from "What is the current Fed funds rate?" to "Walk me through US unemployment over the past 20 years."

The core design philosophy is that the LLM should narrate, not calculate. Growth rates, inflation, and level metrics are computed deterministically in Python before GPT-4o sees any data. This keeps answers reproducible and directly verifiable against the cited FRED sources.

**Stack:** Python 3.12 / FastAPI · React 18 + TypeScript + Vite · OpenAI GPT-4o + text-embedding-3-small · FRED API · Pinecone serverless

---

## Setup

### Prerequisites

- Python 3.12+, [Poetry](https://python-poetry.org/)
- Node.js 18+, [pnpm](https://pnpm.io/)
- API keys for OpenAI, FRED, and Pinecone (provided)

### 1. Environment variables

Place a `.env` file in the repo root:

```env
OPENAI_API_KEY=sk-...
FRED_API_KEY=...
PINECONE_API_KEY=pcsk_...
```

### 2. Backend

```bash
cd backend
poetry install
```

Start from the **repo root** so the `backend` package is on the path:

```bash
PYTHONPATH="$(pwd)" poetry --directory backend run uvicorn backend.submission.api:app --reload --port 8000
```

### 3. Frontend

```bash
cd frontend
pnpm install
pnpm dev
```

Open `http://localhost:5173`. The Vite dev server proxies all API calls to the backend.

### 4. Tests

```bash
# Backend
PYTHONPATH="$(pwd)" poetry --directory backend run pytest

# Frontend
cd frontend && pnpm test
```

### 5. Evaluation harness

```bash
PYTHONPATH="$(pwd)" poetry --directory backend run python -m backend.submission.eval
```

Runs the seven canonical spec questions through the full pipeline, scores each response with a GPT-4o judge, and writes a timestamped JSON report to `backend/eval_results/`. Supports `--model` and `--judge-model` flags to compare across models.

---

## API Reference

### `GET /health`
```json
{ "status": "ok" }
```

### `GET /series`
Returns metadata for all available FRED series.
```json
[{ "key": "GDP", "series_id": "GDP", "description": "US Nominal Gross Domestic Product..." }]
```

### `GET /series/{series_key}?limit=60`
Returns time-series data for a named series (`limit` default 60, max 500).

**Keys:** `GDP` · `REAL_GDP` · `CPI` · `EUROZONE_INFLATION` · `UNEMPLOYMENT` · `JAPAN_UNEMPLOYMENT` · `FEDERAL_FUNDS_RATE` · `USD_EUR`

```json
{
  "key": "UNEMPLOYMENT",
  "series_id": "UNRATE",
  "description": "US Unemployment Rate (Percent)",
  "points": [{ "date": "2025-01-01", "value": 4.1 }],
  "citation": { "source": "FRED...", "series_id": "UNRATE", "url": "https://fred.stlouisfed.org/series/UNRATE" }
}
```

### `POST /ask`
```json
{ "query": "What is the current unemployment rate in Japan?" }
```
```json
{
  "answer": "As of February 2026, Japan's unemployment rate stands at 2.80%...",
  "citations": [{ "series_id": "LRUN64TTJPM156S", "url": "https://fred.stlouisfed.org/series/LRUN64TTJPM156S" }],
  "metrics": [{ "metric_key": "japan_unemployment_rate", "value": 2.80, "unit": "%", "date": "2026-02-01" }],
  "warnings": [],
  "selectedSeries": "JAPAN_UNEMPLOYMENT"
}
```

---

## Design Approach

### The MacroSpecialist pipeline

The `MacroSpecialist` class is the single orchestrator for every query. Its pipeline has four stages:

**1. Retrieval.** The query is embedded with `text-embedding-3-small` and compared against pre-indexed series documents stored in a Pinecone serverless index (`takehome-macro-james-hennessy`). The top-k matches are surfaced. Before returning them, a set of deterministic guardrails run over the raw query text: if the query mentions "Japan" and any labour market term, `JAPAN_UNEMPLOYMENT` is promoted regardless of its embedding score. Similarly, "eurozone"+"inflation" routes to `EUROZONE_INFLATION` rather than the US CPI. This two-layer approach means common geographic disambiguation never depends on the embedding model getting it right.

**2. Data fetching.** Rather than using a fixed observation window, the system infers a time horizon from the query — detecting patterns like "past 10 years", "since 2005", or "decade" — and adjusts the FRED fetch limit accordingly. Each series has an `obs_per_year` field (4 for quarterly GDP, 12 for monthly unemployment, 1 for annual Eurozone CPI) so the fetch scales correctly by frequency. For trend questions, observations are sampled evenly across the full horizon rather than taken from the recent tail, ensuring the LLM sees data spanning the whole period rather than just the last few months.

**3. Context building.** Derived metrics — year-on-year growth, quarter-on-quarter annualised growth, inflation rate — are computed in Python with explicit formulas before GPT-4o sees any numbers. The system prompt instructs the model to treat these as authoritative. This is the most important reliability decision in the project: LLMs are unreliable at arithmetic, and pre-computing the metrics means a wrong answer is traceable to a data issue, not a model hallucination.

**4. Response assembly.** GPT-4o produces the prose. The response object bundles the answer with structured citations (FRED URLs, series IDs) and the computed metrics separately, so the frontend can display them independently of the text. The `selectedSeries` field tells the frontend which chart to surface — this routing decision lives entirely in the backend so the two sides can never diverge.

### Retrieval system architecture

All series metadata lives in a single `SERIES_REGISTRY` dict of `FREDSeriesConfig` dataclasses. Each entry bundles the FRED series ID, description, keyword list, semantic search text, and frequency — replacing the three parallel module-level dicts that are the natural starting point for this kind of project but quickly become a maintenance hazard as series are added. On first startup, each config entry is embedded and upserted into Pinecone; subsequent cold starts skip the embed step entirely by checking the index's vector count. When Pinecone is unavailable the system degrades to in-memory cosine similarity, then to lexical overlap scoring — so the product still works in offline or CI environments.

### Frontend

The React frontend is intentionally thin. It pre-fetches all eight series in parallel on load, renders them as indicator cards and a custom SVG line chart, and sends queries to the backend. Critically, it does not contain any series routing logic — an earlier version duplicated the backend's guardrail heuristics in a `findSeriesForQuery` function that would have silently drifted as the backend evolved. The `selectedSeries` field in the `/ask` response is the single source of truth for which chart to highlight after a query.

---

## Challenges

**LLM hallucination on numeric data.** The most significant challenge was preventing the model from fabricating figures. Early prompting experiments where the LLM was given raw FRED observations and asked to compute growth rates produced plausible-but-wrong numbers. The solution — pre-computing all metrics deterministically and presenting them as facts rather than inputs — eliminated this class of error entirely and made answers directly auditable.

**FRED data quality.** FRED represents missing observations as the string `"."` rather than null. Without explicit coercion (`pd.to_numeric(..., errors="coerce")`), these silently become NaN and can propagate into growth rate calculations. Handling this correctly and tracking per-series retrieval failures in a structured `DataRetrievalError` model — rather than letting a single series failure abort the whole response — was important for robustness.

**Time horizon mismatch.** A fixed observation window (the obvious starting point) breaks badly on trend questions. "What has been US unemployment over the past 20 years?" with a 24-observation cap gives the model roughly two years of data. The fix required inferring the time horizon from the query, scaling the fetch limit by series frequency, and — critically — sampling observations evenly across the full range rather than from the tail, so the model actually sees the 2008 crisis and the COVID spike rather than just the last few months.

**Keeping routing in one place.** The frontend naturally wants to update the chart when a query comes in, which creates pressure to duplicate the backend's series selection logic in the browser. Resisting this and instead returning `selectedSeries` from the API keeps the two sides decoupled and ensures the chart always reflects what the backend actually used, not what the frontend guessed.

---

## Evaluation

An evaluation harness (`backend/submission/eval.py`) runs the seven canonical spec questions through the full pipeline and scores each response using a GPT-4o judge on five dimensions: overall quality (1–5), groundedness, citation accuracy, and presence of specific numeric values with units and dates.

Results on the current build:

| Question | Score |
|---|---|
| Current US GDP | 5/5 |
| Eurozone inflation rate | 5/5 |
| Japan unemployment rate | 5/5 |
| Federal Reserve interest rate | 5/5 |
| USD/EUR exchange rate | 5/5 |
| US GDP growth over 10 years | 5/5 |
| US unemployment trend over 20 years | 5/5 |
| **Mean** | **5.0 / 5** |

The harness supports `--model` and `--judge-model` flags, writes timestamped JSON to `backend/eval_results/`, and can be run in CI to track quality across model upgrades.

---

## Design Decisions and Rationale

- **FRED as the primary data source.** The project uses FRED only, even though the spec mentions World Bank and IMF as possible alternatives. This is a deliberate scope decision: FRED covers all required questions, has a stable API, and returns data in a format that is fast to integrate and easy to verify. In a time-bounded take-home, depth and reliability on one source is more valuable than shallow support for several.
- **A registry-driven series model.** All supported indicators live in a single `SERIES_REGISTRY` configuration object. Each entry contains the FRED id, description, search metadata, and observation frequency. This keeps the system maintainable as new series are added: retrieval, metric calculation, citations, and API exposure all build from the same source of truth instead of duplicated constants.
- **Deterministic metrics before LLM generation.** Growth rates, inflation rates, and latest levels are computed in Python before the prompt is assembled. This is the most important reliability decision in the project. LLMs are good at explanation and synthesis, but not consistently trustworthy for arithmetic. By separating calculation from narration, the output becomes reproducible, auditable, and easier to test.
- **Hybrid retrieval instead of pure semantic search.** Retrieval combines embeddings with deterministic guardrails. Pinecone is used when available, the system falls back to in-memory embedding similarity if needed, and finally to lexical overlap scoring if embeddings are unavailable. On top of that, geographic rules such as Eurozone inflation and Japan unemployment are explicitly promoted. This design trades a small amount of handcrafted logic for a large gain in correctness and resilience.
- **Query-aware time horizons.** The amount of FRED data fetched is inferred from the question. Queries such as "past 10 years" or "since 2005" trigger longer windows, scaled by the frequency of the series. Observations are then sampled across the full span rather than from the recent tail. This avoids a common failure mode where a long-horizon question is answered using only recent data.
- **Structured API responses, not prose-only answers.** The `/ask` endpoint returns `answer`, `citations`, `metrics`, `warnings`, and `selectedSeries` as separate fields. This makes the frontend more robust because it can render citations, metric chips, and chart selection directly from structured data instead of trying to parse meaning out of free text.
- **Backend-owned chart routing.** The backend decides which series best matches a user query and returns that decision as `selectedSeries`. The frontend does not try to re-implement the routing logic. This prevents drift between the UI and the backend reasoning path, and ensures the highlighted chart always corresponds to the data actually used in the answer.
- **A thin frontend by design.** The React app focuses on presenting dashboard cards, charts, chat responses, and citations. It does not own macroeconomic logic. Keeping the domain reasoning in the backend reduces duplication, makes the frontend easier to reason about, and keeps future backend improvements from requiring matching browser-side rewrites.
- **No streaming in the first version.** Responses are returned as a single payload. Streaming would improve perceived latency slightly, but it would also require SSE or WebSocket support on the backend plus more complex incremental state handling in the frontend. For this stage of the project, that complexity does not pay for itself.
- **Pinecone as an accelerator, not a hard dependency.** The Pinecone index is created automatically on first startup, but the application is still designed to function when Pinecone is unavailable. That makes local development, CI, and demos less brittle, while still preserving the retrieval quality benefits of vector search when the service is present.

---

## Potential Future Improvements

- **Broader data coverage.** Add World Bank and IMF series for emerging markets. Make `SERIES_REGISTRY` database-backed so new series can be added via an admin interface rather than a code change and redeploy.
- **Streaming responses.** SSE from FastAPI with incremental rendering in React would meaningfully improve perceived latency on longer analytical questions.
- **Report generation.** A `generate_report` method on `MacroSpecialist` could produce structured multi-series analysis as a PDF or JSON, using the Instructor library already in the dependency tree for typed output.
- **Rate limiting and auth.** Per-IP rate limiting (`slowapi`) and API key authentication are the minimum required before any public deployment. Each `/ask` request makes two OpenAI API calls; an unthrottled endpoint will hit rate limits quickly under load.
- **Richer chart interactions.** The custom SVG `LineChart` works well but replacing it with Recharts or Vega-Lite would add tooltips, zoom, and multi-series overlays without maintaining the coordinate math manually.
