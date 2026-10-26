import json
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from .auth import get_current_user
from .audio import to_wav_16k
from .models_loader import evaluate_xgboost, is_success, run_torch_model
from .db import get_conn

router = APIRouter()

@router.post("/evaluate")
def evaluate(file: UploadFile = File(...), user=Depends(get_current_user)):
    raw = file.file.read()
    if not raw:
        raise HTTPException(400, "Empty upload")
    wav = to_wav_16k(raw)

    xgb_result = evaluate_xgboost(wav)
    if not is_success(xgb_result):
        raise HTTPException(422, "XGBoost gate rejected the sample")

    result = run_torch_model(wav)

    metadata = json.dumps({"filename": file.filename, "xgb_score": xgb_result["score"]})
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO evaluations (user_id, prediction, score, xgb_gate_success, metadata) VALUES (?, ?, ?, ?, ?)",
        (user["id"], result["prediction"], result["score"], 1, metadata),
    )
    conn.commit()
    return {"id": cur.lastrowid, "prediction": result["prediction"], "score": result["score"]}

def _row_to_dict(r):
    return {"id": r["id"], "created_at": r["created_at"], "prediction": r["prediction"],
            "score": r["score"], "xgb_gate_success": bool(r["xgb_gate_success"]), "metadata": json.loads(r["metadata"])}

@router.get("/evaluations")
def list_evals(user=Depends(get_current_user)):
    rows = get_conn().execute("SELECT * FROM evaluations WHERE user_id = ? ORDER BY id", (user["id"],)).fetchall()
    return [_row_to_dict(r) for r in rows]

@router.get("/evaluations/{eval_id}")
def get_eval(eval_id: int, user=Depends(get_current_user)):
    r = get_conn().execute("SELECT * FROM evaluations WHERE id = ? AND user_id = ?", (eval_id, user["id"])).fetchone()
    if r is None:
        raise HTTPException(404, "Evaluation not found")
    return _row_to_dict(r)

@router.get("/export")
def export(user=Depends(get_current_user)):
    rows = get_conn().execute("SELECT * FROM evaluations WHERE user_id = ? ORDER BY id", (user["id"],)).fetchall()
    return {"user": user["username"], "evaluations": [_row_to_dict(r) for r in rows]}
