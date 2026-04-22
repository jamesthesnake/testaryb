# Macro Specialist Backend

FastAPI serves the JSON API used by the React frontend. Streamlit is still
available in `submission/app.py` as a separate UI.

## Setup

From the repository root:

```bash
cd backend
poetry install
```

Create a root `.env` file or export these variables before running the API:

```bash
OPENAI_API_KEY=...
FRED_API_KEY=...
```

## Run The API

Run from the repository root so imports resolve correctly:

```bash
PYTHONPATH="$(pwd)" poetry --directory backend run uvicorn backend.submission.api:app --reload --port 8000
```

The React dev server proxies `/ask`, `/health`, and `/series` to
`http://localhost:8000`.

## Run Tests

From the repository root:

```bash
poetry --directory backend run pytest backend/tests
```

## Endpoints

- `GET /health` returns API liveness.
- `POST /ask` accepts `{ "query": "..." }` and returns an answer, citations,
  and the selected FRED series key.
- `GET /series` lists available FRED series.
- `GET /series/{series_key}?limit=60` returns chart-ready observations.
