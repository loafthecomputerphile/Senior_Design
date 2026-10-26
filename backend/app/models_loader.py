import os

XGB_MODEL_PATH = os.environ.get("XGB_MODEL_PATH", os.path.join(os.path.dirname(__file__), "..", "models", "xgb_model.json"))
TORCH_MODEL_PATH = os.environ.get("TORCH_MODEL_PATH", os.path.join(os.path.dirname(__file__), "..", "models", "cough_model.pth"))

_xgb = "UNLOADED"
_torch = "UNLOADED"

def get_xgb():
    global _xgb
    if _xgb == "UNLOADED":
        import xgboost as xgb
        _xgb = xgb.Booster()
        _xgb.load_model(XGB_MODEL_PATH)  # ponytail: placeholder path, file not uploaded yet
    return _xgb

def get_torch():
    global _torch
    if _torch == "UNLOADED":
        import torch
        _torch = torch.load(TORCH_MODEL_PATH, map_location="cpu", weights_only=False)  # ponytail: placeholder path
        _torch.eval()
    return _torch

def evaluate_xgboost(wav: bytes) -> dict:
    """Gate model. Returns dict the success fn evaluates.

    ponytail: model file not uploaded yet — returns placeholder scores.
    Swap in real feature extraction + booster.predict once available.
    """
    try:
        return {"score": float(get_xgb().predict([[0.0]])[0])}  # placeholder path
    except Exception:
        return {"score": 0.0}  # ponytail: placeholder until model file exists

def is_success(xgb_result: dict) -> bool:
    """MOCK FUNCTION — doesn't actually evaluate anything yet. Always passes."""
    return True  # ponytail: mock gate

def run_torch_model(wav: bytes) -> dict:
    """PyTorch cough classifier. ponytail: placeholder path — model not wired yet,
    returns a stub prediction so the pipeline is testable end-to-end."""
    try:
        model = get_torch()
        # TODO: real preprocessing + model(wav_tensor) once architecture is wired
        return {"prediction": "placeholder", "score": 0.5}
    except Exception:
        return {"prediction": "placeholder", "score": 0.5}
