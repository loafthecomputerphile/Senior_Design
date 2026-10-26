from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes_auth import router as auth_router
from .routes_eval import router as eval_router

app = FastAPI(title="Cough Classifier Backend")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])  # ponytail: dev CORS, restrict before deploy

app.include_router(auth_router)
app.include_router(eval_router)
