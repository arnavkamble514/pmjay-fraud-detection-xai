# ClaimGuard web app

React + Vite + Tailwind frontend for the FastAPI server in `stage2/api.py`.

## Run (demo)

```
cd stage2
python api.py
```

Open http://localhost:8000. The server serves the built frontend from `frontend/dist`.
The first start scores every claim out-of-fold (about a minute) and caches the result in
`stage2/data/review_scores.json`.

## Develop

```
cd frontend
npm install
npm run dev        # http://localhost:5173, calls the API on :8000
npm run build      # writes frontend/dist, which api.py serves
```

Reviewer decisions are appended to `stage2/data/review_log.jsonl`.
