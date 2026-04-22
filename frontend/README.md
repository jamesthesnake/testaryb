# Takehome Frontend

The frontend uses Vite + React. It calls the FastAPI backend through `/ask`.

## Using

- From the repo root, start the backend first:
  `PYTHONPATH="$(pwd)" poetry --directory backend run uvicorn backend.submission.api:app --reload --port 8000`
- Run the frontend dev server: `pnpm run dev`
- Edit: `src/App.tsx`

The Vite dev server proxies `/ask`, `/health`, and `/series` to
`http://localhost:8000`. For a deployed frontend, set `VITE_API_BASE_URL` to
the backend origin.
