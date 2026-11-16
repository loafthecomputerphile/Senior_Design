# Slide 7 — Backend (FastAPI)
**Screenshot:** `backend/app/routes_eval.py` — the /evaluate pipeline (audio → 16kHz WAV → XGBoost gate → PyTorch model → SQLite)

Pipeline as a one-line diagram: **upload → gate (422 reject) → model → store → history/export**

Verbally: auth endpoints, smoke tests all pass (`backend/test_smoke.py`), model stubs pending artifact export, server-side storage moves client-side in M3.
