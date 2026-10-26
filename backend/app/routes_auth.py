from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from .auth import hash_password, verify_password, make_token, get_current_user
from .db import get_conn

router = APIRouter()

class Credentials(BaseModel):
    username: str
    password: str

@router.post("/signup")
def signup(creds: Credentials):
    if len(creds.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    conn = get_conn()
    try:
        conn.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)",
                     (creds.username, hash_password(creds.password)))
        conn.commit()
    except Exception:
        raise HTTPException(409, "Username already taken")
    row = conn.execute("SELECT id FROM users WHERE username = ?", (creds.username,)).fetchone()
    return {"token": make_token(row["id"])}

@router.post("/login")
def login(creds: Credentials):
    row = get_conn().execute("SELECT id, password_hash FROM users WHERE username = ?", (creds.username,)).fetchone()
    if row is None or not verify_password(creds.password, row["password_hash"]):
        raise HTTPException(401, "Invalid username or password")
    return {"token": make_token(row["id"])}

@router.get("/me")
def me(user=Depends(get_current_user)):
    return {"id": user["id"], "username": user["username"]}
