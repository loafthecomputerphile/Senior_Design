import sqlite3, os, threading

# ponytail: global connection + lock; fine at this scale
_local = threading.local()
DB_PATH = os.environ.get("BACKEND_DB", os.path.join(os.path.dirname(__file__), "..", "app.db"))

def get_conn():
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id),
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            prediction TEXT NOT NULL,
            score REAL NOT NULL,
            xgb_gate_success INTEGER NOT NULL,
            metadata TEXT NOT NULL
        )""")
        conn.commit()
        _local.conn = conn
    return conn
