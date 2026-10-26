# Backend

FastAPI service: JWT auth + cough-audio evaluation pipeline (XGBoost gate → PyTorch model).

## Setup
```
cd backend
python -m venv venv
venv\Scripts\pip install -r requirements.txt
venv\Scripts\uvicorn app.main:app --reload
```

## Env vars (optional)
- `JWT_SECRET` — token signing secret (default: dev value, change in prod)
- `XGB_MODEL_PATH` — default `models/xgb_model.json` (placeholder, not uploaded yet)
- `TORCH_MODEL_PATH` — default `models/cough_model.pth` (placeholder)
- `BACKEND_DB` — sqlite path (default `backend/app.db`)

## Routes
- `POST /signup`, `POST /login` → `{"token": ...}`
- `GET /me`
- `POST /evaluate` (bearer token, multipart `file`) → resamples upload to 16 kHz mono wav → XGBoost gate (`is_success` is a MOCK, always passes) → PyTorch model (placeholder stub) → row saved to SQLite
- `GET /evaluations`, `GET /evaluations/{id}` — user's history
- `GET /export` — JSON dump of user's evals

Wav files are never persisted — only prediction, score and metadata.
