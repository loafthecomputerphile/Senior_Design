import os, jwt, bcrypt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from .db import get_conn

# ponytail: default dev secret; set JWT_SECRET in prod
SECRET = os.environ.get("JWT_SECRET", "dev-secret-change-me")
bearer = HTTPBearer()

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())

def make_token(user_id: int) -> str:
    return jwt.encode({"sub": str(user_id)}, SECRET, algorithm="HS256")

def get_current_user(creds: HTTPAuthorizationCredentials = Depends(bearer)):
    try:
        user_id = int(jwt.decode(creds.credentials, SECRET, algorithms=["HS256"])["sub"])
    except Exception:
        raise HTTPException(401, "Invalid token")
    row = get_conn().execute("SELECT id, username FROM users WHERE id = ?", (user_id,)).fetchone()
    if row is None:
        raise HTTPException(401, "User not found")
    return row
