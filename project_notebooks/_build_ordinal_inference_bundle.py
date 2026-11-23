"""One-off builder for ordinal_inference_bundle.zip (not bundled)."""
from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "project_notebooks"
OUT = REPO / "ordinal_inference_bundle.zip"

MODULES = [
    "modules/__init__.py",
    "modules/dataset/__init__.py",
    "modules/dataset/make_dataset.py",
    "modules/nn/__init__.py",
    "modules/nn/base_nn.py",
    "modules/nn/coral.py",
    "modules/preprocessing/__init__.py",
    "modules/preprocessing/audio_process.py",
    "modules/preprocessing/cough_verify.py",
]
STATES = [
    "run_state_3class/ordinal_oof_state.pt",
    "run_state_3class/ensemble_state.pt",
    "run_state_3class/final_best_test_variant.pt",
    "run_state_3class/final_best_coral_tune.pt",
    "run_state_3class/final_ensemble_soft_tuned.pt",
]

C1 = '''import os

_DATASET = os.path.join("data", "final_processed_dataset.npz")
assert os.path.exists(_DATASET), (
    f"Dataset not found: {_DATASET}. Extract the bundle and place "
    f"project_notebooks/data/final_processed_dataset.npz at this path."
)
dataset: NpzDataPipeline = NpzDataPipeline(_DATASET)
device: torch.types.Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device
'''

C2 = '''import os

RUN_STATE_DIR: str = "run_state_3class"
os.makedirs(RUN_STATE_DIR, exist_ok=True)
os.makedirs("tables_3class", exist_ok=True)
os.makedirs("figures_3class", exist_ok=True)
'''

C3 = '''# ---- Shared constants (inference only) ----
N_FOLDS: int = 5
SEED: int = 42  # identical split across all experiments
BATCH_SIZE: int = 128

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
'''

C9 = '''# ==== Model names (no model construction in this bundle) ====
from modules.nn import BranchedCoughCNN
from modules.nn.base_nn import ResNetModel

CV_MODELS: list[str] = ["default", "branched", "resnet"]
'''

C10 = '''# ---- Helpers retained for recorded-level aggregation ----
def _make_model(model_name: str, num_classes: int = 1) -> nn.Module:
    """Model factory by string (unused in inference-only bundle)."""
    if model_name == "default":
        return DefaultCNN(num_classes)
    if model_name == "branched":
        return BranchedCoughCNN(num_classes)
    if model_name == "resnet":
        return ResNetModel(num_classes, freeze_backbone=True)
    raise ValueError(f"unknown model_name {model_name!r} (default|branched|resnet)")


def aggregate_recordings(oof: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Mean cumulative probs per recording; predicted rank via 0.5 crossings."""
    order: dict[str, list[int]] = {}
    for i, p in enumerate(oof["path"]):
        order.setdefault(p, []).append(i)
    rec_true: list[int] = []
    rec_p1: list[float] = []
    rec_p2: list[float] = []
    rec_paths: list[str] = []
    for path, idxs in order.items():
        ranks = {int(oof["true_rank"][i]) for i in idxs}
        assert len(ranks) == 1, f"recording {path} spans ranks {ranks}"
        rec_paths.append(path)
        rec_true.append(ranks.pop())
        rec_p1.append(float(np.mean(oof["p1"][idxs])))
        rec_p2.append(float(np.mean(oof["p2"][idxs])))
    rec_p1, rec_p2 = np.asarray(rec_p1), np.asarray(rec_p2)
    rec_pred = (rec_p1 > 0.5).astype(int) + (rec_p2 > 0.5).astype(int)
    return {"path": np.asarray(rec_paths, dtype=object), "true": np.asarray(rec_true, dtype=int),
            "pred": rec_pred, "p1": rec_p1, "p2": rec_p2}
'''

C18 = '''# ---- OOF state load (no re-inference, no checkpoint loading) ----
import glob
import os
import re

import numpy as np
import polars as pl
import torch

STATE_PATH = "run_state_3class/ordinal_oof_state.pt"
HIST_COLS: list[str] = ["train_loss", "train_acc", "train_F1", "train_auc_r1", "train_auc_r2",
                        "val_loss", "val_acc", "val_F1", "val_auc_r1", "val_auc_r2"]
_EXPECTED_SEGS = len(cv_pool)


def _discover_from_csv() -> dict[str, dict[int, str]]:
    pat = re.compile(r"cv__(.+?)__f(\\d+)_experiment_stats\\.csv$")
    found: dict[str, dict[int, str]] = {}
    for p in sorted(glob.glob("tables_3class/cv__*__f*_experiment_stats.csv")):
        m = pat.search(os.path.basename(p))
        if m:
            found.setdefault(m.group(1), {})[int(m.group(2))] = p
    return found


def _cv_rows_from(csv_map: dict[str, dict[int, str]]) -> list[dict]:
    rows: list[dict] = []
    for mn, folds in csv_map.items():
        if mn not in CV_MODELS:
            continue
        for fi, csv in sorted(folds.items()):
            df = pl.read_csv(csv)
            if not len(df):
                continue
            h = {k: df[k].to_list() for k in HIST_COLS}
            ep = int(np.argmin(h["val_loss"]))
            rows.append({"model": mn, "fold": fi, "epochs": len(h["val_loss"]),
                         "val_loss": round(h["val_loss"][ep], 4),
                         "val_acc": round(h["val_acc"][ep], 4),
                         "val_F1": round(h["val_F1"][ep], 4),
                         "train_auc_r1": round(h["train_auc_r1"][ep], 4),
                         "train_auc_r2": round(h["train_auc_r2"][ep], 4),
                         "val_auc_r1": round(h["val_auc_r1"][ep], 4),
                         "val_auc_r2": round(h["val_auc_r2"][ep], 4)})
    return rows


_st = torch.load(STATE_PATH, weights_only=False, map_location="cpu")
oof_all: dict[str, dict[str, np.ndarray]] = dict(_st.get("oof_all", {}))
cv_rows: list[dict] = list(_st.get("cv_rows", []))
csv_map = _discover_from_csv()
if not cv_rows:
    cv_rows = _cv_rows_from(csv_map)
pl.DataFrame(cv_rows).write_csv("tables_3class/cv_ordinal_folds.csv")

missing = [mn for mn in CV_MODELS if mn not in oof_all or
           len(oof_all[mn]["p1"]) != _EXPECTED_SEGS]
assert not missing, f"incomplete OOF (expected {_EXPECTED_SEGS} segs/model): {missing}"
print(f"=> OOF state covers: {sorted(oof_all)}; {_EXPECTED_SEGS} segments per model")
'''

C22 = '''# ---- Static artifacts (shipped, not regenerated) ----
# The following were produced from per-fold test probabilities that only existed
# in the excluded model checkpoints, so this bundle ships them as PNG/CSV files:
#   figures_3class/best_cvfold_test_giant.png
#   figures_3class/best_f1_fold_test_giant.png
#   tables_3class/best_cvfold_test_eval.csv
#   tables_3class/best_f1_fold_test_eval.csv
# All other included figures/CSVs are regenerated by the cells below.
'''

C25 = '''# ---- Ensemble: cached probabilities, no checkpoint loading ----
import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, roc_auc_score

cv_models = CV_MODELS
N_ENS = len(cv_models)
ENS_RANK_NAMES: list[str] = ["low", "med", "high"]
TOP_N_MODELS: int = 10


def _class_scores(rec: dict) -> dict[str, np.ndarray]:
    p1, p2 = np.clip(rec["p1"], 0.0, 1.0), np.clip(rec["p2"], 0.0, 1.0)
    return {"low": 1.0 - p1, "med": p1 - p2, "high": p2}


def _ens_extra(rec: dict, pred: np.ndarray) -> dict:
    t = rec["true"]
    accs = {f"acc_{n}": round(100 * float(np.mean(pred[t == k] == k)), 2)
            for k, n in enumerate(ENS_RANK_NAMES)}
    scores = _class_scores(rec)
    aucs = {f"auc_{n}": round(float(roc_auc_score((t == k).astype(int), s)), 4)
            if len(np.unique(t == k)) > 1 else None
            for k, (n, s) in enumerate(scores.items())}
    auc_vals = [v for v in aucs.values() if v is not None]
    return {"acc_total": round(100 * float(np.mean(pred == t)), 2), **accs,
            "auc_macro_ovr": round(float(np.mean(auc_vals)), 4) if auc_vals else None, **aucs}


def _ens_row(model_name: str, rec: dict, pred: np.ndarray | None = None) -> tuple[dict, dict]:
    p = rec["pred"] if pred is None else pred
    m = ordinal_oof_metrics(rec["true"], p, rec["p1"], rec["p2"])
    row = {"model": model_name, "n_recordings": len(rec["true"]),
           **{k: (round(v, 4) if v is not None else None) for k, v in m.items()},
           **_ens_extra(rec, p)}
    return row, m


def _predict_at(rec: dict, t1: float, t2: float) -> np.ndarray:
    return (rec["p1"] > t1).astype(int) + (rec["p2"] > t2).astype(int)


def _vote(votes: np.ndarray) -> np.ndarray:
    out = np.empty(votes.shape[1], dtype=int)
    for j in range(votes.shape[1]):
        counts = np.bincount(votes[:, j], minlength=3)
        out[j] = int(np.flatnonzero(counts == counts.max())[0])
    return out


_est = torch.load("run_state_3class/ensemble_state.pt", weights_only=False, map_location="cpu")
ens_val_rec = _est["val_rec"]
ens_test_rec = _est["test_rec"]
val_family_recs = _est["val_family_recs"]
test_model_recs = _est["test_model_recs"]
top_folds = _est.get("top_folds") or []
t1 = float(_est["best_thresholds"]["t1"])
t2 = float(_est["best_thresholds"]["t2"])
assert len(val_family_recs) == N_ENS and len(test_model_recs) == TOP_N_MODELS

ens_val_pred_t = _predict_at(ens_val_rec, t1, t2)
ens_test_pred_t = _predict_at(ens_test_rec, t1, t2)
val_hard_05 = _vote(np.stack([_predict_at(r, 0.5, 0.5) for r in val_family_recs]))
test_hard_05 = _vote(np.stack([_predict_at(r, 0.5, 0.5) for r in test_model_recs]))
val_hard_t = _vote(np.stack([_predict_at(r, t1, t2) for r in val_family_recs]))
test_hard_t = _vote(np.stack([_predict_at(r, t1, t2) for r in test_model_recs]))

_suffix = f" (top-{TOP_N_MODELS} wrec)"
ens_val_row_s, _ = _ens_row("val soft @0.5", ens_val_rec)
ens_val_row_st, _ = _ens_row(f"val soft @t={t1:.2f}/{t2:.2f}", ens_val_rec, ens_val_pred_t)
ens_val_row_h, _ = _ens_row("val hard @0.5", ens_val_rec, val_hard_05)
ens_val_row_ht, _ = _ens_row(f"val hard @t={t1:.2f}/{t2:.2f}", ens_val_rec, val_hard_t)
ens_test_row_s, _ = _ens_row(f"test soft @0.5{_suffix}", ens_test_rec)
ens_test_row_st, _ = _ens_row(f"test soft @t={t1:.2f}/{t2:.2f}{_suffix}", ens_test_rec, ens_test_pred_t)
ens_test_row_h, _ = _ens_row(f"test hard @0.5{_suffix}", ens_test_rec, test_hard_05)
ens_test_row_ht, _ = _ens_row(f"test hard @t={t1:.2f}/{t2:.2f}{_suffix}", ens_test_rec, test_hard_t)

all_rows = [ens_val_row_s, ens_val_row_st, ens_val_row_h, ens_val_row_ht,
            ens_test_row_s, ens_test_row_st, ens_test_row_h, ens_test_row_ht]
pl.DataFrame(all_rows).write_csv("tables_3class/ensemble_eval.csv")
_best_state = torch.load("run_state_3class/final_best_test_variant.pt",
                         weights_only=False, map_location="cpu")
pl.DataFrame([_best_state["best_row"]]).write_csv("tables_3class/final_best_test_variant.csv")
print(f"=> ensemble rows rebuilt from cache at t={t1:.2f}/{t2:.2f}")

# ---- figures ----
_tt = f"t={t1:.2f}/{t2:.2f}"
_RC_KEYS = ("font.size", "axes.titlesize", "axes.labelsize",
            "xtick.labelsize", "ytick.labelsize", "legend.fontsize", "figure.titlesize")
_rc_old = {k: plt.rcParams[k] for k in _RC_KEYS}
plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13,
                     "xtick.labelsize": 11, "ytick.labelsize": 11,
                     "legend.fontsize": 11, "figure.titlesize": 18})

_cases = [("val soft @0.5", ens_val_rec, ens_val_rec["pred"]),
          (f"val soft {_tt}", ens_val_rec, ens_val_pred_t),
          ("val hard @0.5", ens_val_rec, val_hard_05),
          (f"val hard {_tt}", ens_val_rec, val_hard_t),
          ("test soft @0.5", ens_test_rec, ens_test_rec["pred"]),
          (f"test soft {_tt}", ens_test_rec, ens_test_pred_t),
          ("test hard @0.5", ens_test_rec, test_hard_05),
          (f"test hard {_tt}", ens_test_rec, test_hard_t)]
fig = plt.figure(figsize=(16, 11))
gs = fig.add_gridspec(3, 4, hspace=0.38, wspace=0.28)
for ax, (title, rec, pred) in zip(
        [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(4)], _cases):
    cm = confusion_matrix(rec["true"], pred, labels=[0, 1, 2])
    ConfusionMatrixDisplay(cm, display_labels=ENS_RANK_NAMES).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
for ax, title, rec in [(fig.add_subplot(gs[2, 0:2]), f"val soft (OOF x{N_ENS})", ens_val_rec),
                       (fig.add_subplot(gs[2, 2:4]), f"test soft (top-{TOP_N_MODELS})", ens_test_rec)]:
    for r in range(3):
        ax.hist(rec["p2"][rec["true"] == r], bins=30, alpha=0.5, range=(0, 1),
                label=ENS_RANK_NAMES[r])
    ax.set_title(title); ax.set_xlabel("P(rank>med)"); ax.set_ylabel("recordings")
    ax.legend(fontsize=8)
fig.suptitle(f"Ensemble diagnostics - soft/hard voting ({_tt}, top-{TOP_N_MODELS})", fontsize=18)
fig.savefig("figures_3class/ensemble_combined_diagnostics.png", dpi=150, bbox_inches="tight")
plt.show(fig); plt.close(fig)

_comp_rows: list[tuple[str, float]] = []
if "eval_rows" in globals() and eval_rows:
    _comp_rows += [(f"{r['model']} (OOF)", r["macro_f1"]) for r in eval_rows]
_comp_rows += [(r["model"], r["macro_f1"]) for r in all_rows]
fig2 = plt.figure(figsize=(19, 11))
gs2 = fig2.add_gridspec(1, 2, width_ratios=[0.9, 2.4], wspace=0.22)
_ax_f1 = fig2.add_subplot(gs2[0, 0])
_inner = gs2[0, 1].subgridspec(2, 1, hspace=0.45)
_ax_acc = fig2.add_subplot(_inner[0])
_ax_auc = fig2.add_subplot(_inner[1])
_vals = [c[1] for c in _comp_rows]
_cols = ["tab:blue"] * (len(_vals) - 8) + ["tab:cyan"] * 4 + ["tab:red", "tab:red", "tab:orange", "tab:orange"]
_ax_f1.barh([c[0] for c in _comp_rows], _vals, color=_cols)
_ax_f1.set_xlabel("macro F1 (recording level)")
_ax_f1.set_xlim(0, max(_vals) * 1.25 or 1.0)
_ax_f1.set_title("single-model OOF vs ensemble")
_ax_f1.grid(axis="x", alpha=0.3)
for i, v in enumerate(_vals):
    _ax_f1.text(v, i, f" {v:.2f}", va="center", fontsize=10)

_acc_keys = ["acc_total", "acc_low", "acc_med", "acc_high"]
_auc_keys = ["auc_macro_ovr", "auc_low", "auc_med", "auc_high"]
_metric_rows = [("v s @0.5", ens_val_row_s, "blue"),
                (f"v s {_tt}", ens_val_row_st, "cyan"),
                ("v h @0.5", ens_val_row_h, "steelblue"),
                (f"v h {_tt}", ens_val_row_ht, "teal"),
                ("t s @0.5", ens_test_row_s, "red"),
                (f"t s {_tt}", ens_test_row_st, "orange"),
                ("t h @0.5", ens_test_row_h, "brown"),
                (f"t h {_tt}", ens_test_row_ht, "pink")]
_x = np.arange(4)
_w = 0.1
for gi, (lbl, row, col) in enumerate(_metric_rows):
    _ax_acc.bar(_x + (gi - 3.5) * _w, [row[k] for k in _acc_keys], _w, label=lbl, color=col)
    _ax_auc.bar(_x + (gi - 3.5) * _w,
                [row[k] if row[k] is not None else np.nan for k in _auc_keys],
                _w, label=lbl, color=col)
for ax, keys, title, ylim in [(_ax_acc, _acc_keys, "accuracy (%) - total + per class", (0, 105)),
                              (_ax_auc, _auc_keys, "AUC - macro OvR + per class", (0.4, 1.02))]:
    ax.set_xticks(_x); ax.set_xticklabels(["total", "low", "med", "high"])
    ax.set_title(title); ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=10, ncol=2)
    for gi, (lbl, row, _) in enumerate(_metric_rows):
        for xi, k in enumerate(keys):
            v = row[k]
            if v is not None:
                ax.text(xi + (gi - 3.5) * _w, v, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=9, rotation=90)
fig2.suptitle(f"Ensemble metrics - F1 / accuracy / AUC ({_tt}, top-{TOP_N_MODELS})", fontsize=18)
fig2.savefig("figures_3class/ensemble_combined_metrics.png", dpi=150, bbox_inches="tight")
plt.show(fig2); plt.close(fig2)
plt.rcParams.update(_rc_old)
print("=> ensemble_combined_diagnostics.png + ensemble_combined_metrics.png written")
'''

C27 = '''# ---- CORAL thresholds from cached probabilities only ----
import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, roc_auc_score

RANK_NAMES: list[str] = ["low", "med", "high"]
HIGH_RECALL_FLOOR: float = 0.0


def _logit(x):
    x = np.clip(np.asarray(x, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def _predict(rec: dict, t1: float, t2: float) -> np.ndarray:
    return (rec["p1"] > t1).astype(int) + (rec["p2"] > t2).astype(int)


def _coral_tune_biases(biases: np.ndarray, t1: float, t2: float) -> np.ndarray:
    return np.array([float(biases[0]) + float(_logit(t1)),
                     float(biases[1]) + float(_logit(t2))])


def _predict_coral(rec: dict, tuned_b: np.ndarray, base_b: np.ndarray) -> np.ndarray:
    d1, d2 = float(tuned_b[0] - base_b[0]), float(tuned_b[1] - base_b[1])
    p1 = 1.0 / (1.0 + np.exp(-(_logit(rec["p1"]) - d1)))
    p2 = 1.0 / (1.0 + np.exp(-(_logit(rec["p2"]) - d2)))
    return (p1 > 0.5).astype(int) + (p2 > 0.5).astype(int)


def _predict_learned(rec: dict, biases: np.ndarray) -> np.ndarray:
    b1, b2 = float(biases[0]), float(biases[1])
    s1 = _logit(rec["p1"]) + b1
    s2 = _logit(rec["p2"]) + b2
    return (s1 > b1).astype(int) + (s2 > b2).astype(int)


def _tune(val_rec: dict) -> tuple[list[tuple], tuple]:
    cands: list[tuple] = []
    for d1 in np.arange(-2.0, 2.01, 0.25):
        for d2 in np.arange(-2.0, 2.01, 0.25):
            a = 1.0 / (1.0 + np.exp(-d1))
            b = 1.0 / (1.0 + np.exp(-d2))
            p = _predict(val_rec, a, b)
            rh = float(np.mean(p[val_rec["true"] == 2] == 2))
            f1 = float(f1_score(val_rec["true"], p, average="macro", zero_division=0))
            acc = 100 * float(np.mean(p == val_rec["true"]))
            cands.append((round(f1, 4), round(a, 2), round(b, 2),
                          round(rh, 4), round(acc, 2),
                          round(float(d1), 2), round(float(d2), 2)))
    q = [c for c in cands if c[3] >= HIGH_RECALL_FLOOR] or cands
    ranked = sorted(q, key=lambda c: (-c[0], -c[3]))
    return ranked, ranked[0]


def _full(rec: dict, pred: np.ndarray) -> tuple[dict, np.ndarray]:
    t = rec["true"]
    cm = confusion_matrix(t, pred, labels=[0, 1, 2])
    recalls = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    precisions = cm.diagonal() / cm.sum(axis=0).clip(min=1)
    f1s = f1_score(t, pred, average=None, labels=[0, 1, 2], zero_division=0)
    m = ordinal_oof_metrics(t, pred, rec["p1"], rec["p2"])
    p1, p2 = np.clip(rec["p1"], 0, 1), np.clip(rec["p2"], 0, 1)
    scores = {"low": 1 - p1, "med": p1 - p2, "high": p2}
    aucs = {f"auc_{n}": (round(float(roc_auc_score((t == k).astype(int), s)), 4)
                        if len(np.unique(t == k)) > 1 else None)
            for k, (n, s) in enumerate(scores.items())}
    out = {"acc_total": round(100 * float(np.mean(pred == t)), 2),
           "macro_f1": round(float(f1s.mean()), 4)}
    for k, n in enumerate(RANK_NAMES):
        out[f"acc_{n}"] = round(100 * float(np.mean(pred[t == k] == k)), 2)
        out[f"recall_{n}"] = round(float(recalls[k]), 4)
        out[f"precision_{n}"] = round(float(precisions[k]), 4)
        out[f"f1_{n}"] = round(float(f1s[k]), 4)
    out.update({k: (round(v, 4) if v is not None else None) for k, v in m.items()})
    out.update(aucs)
    return out, cm


_st = torch.load("run_state_3class/ordinal_oof_state.pt", weights_only=False, map_location="cpu")
_ens = torch.load("run_state_3class/ensemble_state.pt", weights_only=False, map_location="cpu")
_best = torch.load("run_state_3class/final_best_test_variant.pt",
                   weights_only=False, map_location="cpu")
_tune_csv = pl.read_csv("tables_3class/coral_threshold_per_model_test_eval.csv")


def _bias_for(mn: str) -> np.ndarray:
    sub = _tune_csv.filter((pl.col("model") == mn) & (pl.col("variant") == "learned_coral"))
    assert len(sub), f"no learned_b1/b2 row for {mn}"
    row = sub.row(0, named=True)
    return np.array([float(row["learned_b1"]), float(row["learned_b2"])])


ckpt_biases = {mn: _bias_for(mn) for mn in
               ["default", "branched", "resnet", "SAVED_BEST_ENSEMBLE"]}
val_recs = {mn: aggregate_recordings(_st["oof_all"][mn]) for mn in CV_MODELS}
if _ens.get("val_rec"):
    val_recs["ENSEMBLE"] = _ens["val_rec"]

# Map rank-1 selected checkpoints to their cached per-checkpoint test recordings.
_sel = pl.read_csv("tables_3class/ensemble_selected_ckpts.csv").with_columns(
    pl.col("rank").cast(pl.Int64))
rank1: dict[str, tuple[int, str]] = {}
for mn in ["default", "branched", "resnet"]:
    sub = _sel.filter(pl.col("model") == mn).sort("rank")
    if len(sub):
        r = sub.row(0, named=True)
        rank1[mn] = (int(r["fold"]), str(r["ckpt"]))

_top = _ens.get("top_folds") or _best.get("top_folds") or []
_model_recs = _ens.get("test_model_recs") or []
recs: dict[str, dict] = {}
for i, c in enumerate(_top):
    mn = c.get("mn", c.get("model"))
    fi = int(c.get("fi", c.get("fold", -1)))
    cp = os.path.basename(str(c.get("path") or c.get("ckpt") or ""))
    if mn in rank1 and rank1[mn] == (fi, cp) and i < len(_model_recs):
        recs[mn] = _model_recs[i]
recs["SAVED_BEST_ENSEMBLE"] = _best["best_rec"]
for mn in ["default", "branched", "resnet"]:
    if mn not in recs:
        print(f"!! {mn}: rank-1 test probabilities are not cached; skipping its figure")
print("=> CORAL recs:", sorted(recs))

rows: list[dict] = []
results: dict[str, dict] = {}
coral_biases_used: dict[str, np.ndarray] = {}
print("\\n=== per-model CORAL thresholds (own VAL) + saved best ensemble ===")
for mn, rec in recs.items():
    if mn == "ENSEMBLE":
        continue
    is_saved = (mn == "SAVED_BEST_ENSEMBLE")
    if is_saved:
        _br = _best.get("best_row", {})
        a = float(_br.get("t1_used", 0.5))
        b = float(_br.get("t2_used", 0.5))
        ranked = []
        f1_v = float(_br.get("macro_f1", np.nan))
        rh_v = float(_br.get("recall_high", np.nan))
        acc_v = float(_br.get("acc_total", np.nan))
        print(f"[{mn}] LOADED soft-voting thresholds t1={a:.2f} t2={b:.2f}")
    else:
        ranked, best = _tune(val_recs[mn])
        f1_v, a, b, rh_v, acc_v, d1, d2 = best
        print(f"[{mn}] t1={a:.2f} t2={b:.2f} d1={d1:+.2f} d2={d2:+.2f} | "
              f"val F1={f1_v:.4f} high_rec={rh_v:.4f} acc={acc_v:.2f}%")

    tuned_pred = _predict(rec, a, b)
    tuned_m, tuned_cm = _full(rec, tuned_pred)
    base_b = ckpt_biases[mn]
    if is_saved:
        tuned_b = _coral_tune_biases(base_b, a, b)
        base_pred = _predict_coral(rec, tuned_b, base_b)
        coral_biases_used[mn] = tuned_b
        assert np.array_equal(base_pred, tuned_pred), \
            f"{mn}: tuned CORAL biases must reproduce the soft-voted thresholds"
    else:
        base_pred = _predict_learned(rec, base_b)
        coral_biases_used[mn] = base_b
    base_m, base_cm = _full(rec, base_pred)

    tuned_name = "saved_tuned" if is_saved else "tuned"
    results[mn] = {"base": (base_m, base_cm), "tuned": (tuned_m, tuned_cm),
                   "rec": rec, "t1": a, "t2": b,
                   "val": (f1_v, rh_v, acc_v), "ranked": ranked,
                   "preds": {"learned_coral": base_pred, tuned_name: tuned_pred}}
    for variant, metrics, t1_r, t2_r in [
        ("learned_coral", base_m, None, None),
        (tuned_name, tuned_m, a, b),
    ]:
        rows.append({"model": mn, "variant": variant, "t1": t1_r, "t2": t2_r,
                     "learned_b1": round(float(base_b[0]), 4),
                     "learned_b2": round(float(base_b[1]), 4),
                     "tune_source": "saved ensemble" if is_saved else f"{mn} VAL OOF",
                     "val_F1": f1_v, "val_high_recall": rh_v, "val_acc": acc_v,
                     "n_recordings": len(rec["true"]), **metrics})

print(pl.DataFrame(rows).select(["model", "variant", "t1", "t2", "acc_total", "macro_f1",
                                 "recall_low", "recall_med", "recall_high"]))

# ---- figures ----
_RC_KEYS = ("font.size", "axes.titlesize", "axes.labelsize",
            "xtick.labelsize", "ytick.labelsize", "legend.fontsize", "figure.titlesize")
_rc_old = {k: plt.rcParams[k] for k in _RC_KEYS}
plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
                     "xtick.labelsize": 10, "ytick.labelsize": 10,
                     "legend.fontsize": 9, "figure.titlesize": 15})

for mn, res in results.items():
    rec = res["rec"]
    base_m, base_cm = res["base"]
    tuned_m, tuned_cm = res["tuned"]
    t1, t2 = res["t1"], res["t2"]
    f1_v, rh_v, acc_v = res["val"]
    base_title = ("CORAL-tuned biases at saved t1/t2"
                  if mn == "SAVED_BEST_ENSEMBLE" else "learned CORAL biases")

    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(2, 4, hspace=0.45, wspace=0.32)
    for col, (cm, title) in enumerate([(base_cm, base_title),
                                       (tuned_cm, f"thresholds {t1:.2f}/{t2:.2f}")]):
        ax = fig.add_subplot(gs[0, col])
        ConfusionMatrixDisplay(cm, display_labels=RANK_NAMES).plot(
            ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{mn} - {title}\\nTEST CM")

    ax = fig.add_subplot(gs[0, 2])
    for r in range(3):
        ax.hist(rec["p2"][rec["true"] == r], bins=30, alpha=0.5, range=(0, 1),
                label=RANK_NAMES[r])
    ax.axvline(0.5, color="tab:red", ls="--", lw=2, label="default cutoff 0.50")
    ax.axvline(t2, color="tab:green", ls="--", lw=2, label=f"t2={t2:.2f}")
    ax.set_title("TEST P(rank>med)"); ax.set_xlabel("P(rank>med)")
    ax.set_ylabel("recordings"); ax.legend()

    ax = fig.add_subplot(gs[0, 3]); ax.axis("off")
    b = coral_biases_used[mn]
    _head = ("TUNED ON ITS OWN VAL OOF" if mn != "SAVED_BEST_ENSEMBLE"
             else "LOADED SAVED BEST ENSEMBLE")
    _bias_head = ("CORAL-TUNED BIASES" if mn == "SAVED_BEST_ENSEMBLE"
                  else "LEARNED BIASES")
    ax.text(
        0.02, 0.98,
        _head + "\\n"
        f"t1={t1:.2f}  t2={t2:.2f}   floor={HIGH_RECALL_FLOOR:.2f}\\n"
        f"val F1={f1_v:.4f}  high_rec={rh_v:.4f}  acc={acc_v:.2f}%\\n\\n"
        f"{_bias_head} ({mn})\\n"
        f"b1={b[0]:.3f}  b2={b[1]:.3f}\\n\\n"
        f"TUNED TEST STACK\\n"
        f"acc={tuned_m['acc_total']:.2f}%  macroF1={tuned_m['macro_f1']:.4f}\\n"
        f"rec L/M/H={tuned_m['recall_low']:.2f}/{tuned_m['recall_med']:.2f}/{tuned_m['recall_high']:.2f}\\n"
        f"prec L/M/H={tuned_m['precision_low']:.2f}/{tuned_m['precision_med']:.2f}/{tuned_m['precision_high']:.2f}\\n"
        f"AUC r1/r2={tuned_m['auc_r1']}/{tuned_m['auc_r2']}",
        va="top", fontsize=10, family="monospace")

    _w = 0.35
    panels = [
        (gs[1, 0], ["acc_total", "acc_low", "acc_med", "acc_high"], ["total", *RANK_NAMES], "accuracy (%)", (0, 105)),
        (gs[1, 1], ["macro_f1", "f1_low", "f1_med", "f1_high"], ["macro", *RANK_NAMES], "F1", (0, 1.02)),
        (gs[1, 2], ["precision_low", "precision_med", "precision_high"], RANK_NAMES, "precision", (0, 1.02)),
        (gs[1, 3], ["recall_low", "recall_med", "recall_high"], RANK_NAMES, "recall", (0, 1.02)),
    ]
    for sub, keys, labels, title, ylim in panels:
        ax = fig.add_subplot(sub)
        _xx = np.arange(len(keys))
        ax.bar(_xx - _w / 2, [base_m[k] for k in keys], _w, label="learned")
        ax.bar(_xx + _w / 2, [tuned_m[k] for k in keys], _w, label="tuned")
        for i, k in enumerate(keys):
            ax.text(i - _w / 2, base_m[k], f"{base_m[k]:.2f}", ha="center", va="bottom", fontsize=7)
            ax.text(i + _w / 2, tuned_m[k], f"{tuned_m[k]:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(_xx); ax.set_xticklabels(labels)
        ax.set_ylim(*ylim); ax.set_title(title)
        ax.grid(axis="y", alpha=0.3); ax.legend()

    fig.suptitle(f"{mn} - learned CORAL biases vs tuned/saved thresholds, TEST")
    fig.savefig(f"figures_3class/coral_threshold_{mn}_permodel.png", dpi=150, bbox_inches="tight")
    plt.show(fig); plt.close(fig)

plt.rcParams.update(_rc_old)
print("=> figures: figures_3class/coral_threshold_*_permodel.png")
'''

C29 = '''import sys

sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
'''

REPLACEMENTS = {1: C1, 2: C2, 3: C3, 9: C9, 10: C10, 18: C18,
                22: C22, 23: "", 25: C25, 27: C27, 29: C29}
KEEP = {0, 4, 5, 6, 7, 16, 17, 19, 20, 21, 24, 28}

README = """# Ordinal inference bundle

Inference/figure-regeneration only. No training, no augmentation experiment, no
checkpoint files. All predictions come from cached probabilities in
`run_state_3class/*.pt`.

## Setup

1. Extract this zip and run all cells from the extracted root (CWD matters).
2. The dataset is NOT included. Place the existing 1.66 GiB file at
   `data/final_processed_dataset.npz` (copy/hardlink/symlink from
   `project_notebooks/data/final_processed_dataset.npz` in the original repo).
3. Use the Python 3.13 venv/kernel from the project. Install `requirements.txt`.
   GPU is optional; allow several GB RAM for the NPZ.
4. Run cells top to bottom. `tables_3class/` and `figures_3class/` must be writable.

## Contents

- `ordinal_inference.ipynb` - setup, OOF eval, ensemble eval, CORAL tuning/figures.
- `modules/` - only the 9 local modules imported by that notebook.
- `figures_3class/*.png`, `tables_3class/*.csv` - all artifacts except `a2*`.
- `run_state_3class/*.pt` - cached OOF/ensemble/best predictions.

## Notes

- No network needed on the inference path. `ResNetModel.__init__` would need cached
  ImageNet weights, but no models are constructed in this notebook.
- `modules/dataset/make_dataset.py` imports `soundfile` at module load; it is in
  `requirements.txt`. It also calls `os.add_dll_directory()` only if an
  `ffmpeg_dlls/` folder exists; that folder is intentionally not bundled.
- Static-only artifacts (shipped but not regenerated because their per-fold test
  probabilities lived in excluded checkpoints):
  `best_cvfold_test_giant.png`, `best_f1_fold_test_giant.png`,
  `best_cvfold_test_eval.csv`, `best_f1_fold_test_eval.csv`.
- All `a2*` augmentation artifacts are intentionally excluded.
"""


def code_cell(src: str, cid: str, metadata: dict | None = None) -> dict:
    return {"cell_type": "code", "execution_count": None, "id": cid,
            "metadata": metadata or {}, "outputs": [],
            "source": src.splitlines(keepends=True)}


def build_notebook() -> bytes:
    nb = json.loads((SRC / "neural_network_3class.ipynb").read_text(encoding="utf-8"))
    cells = []
    for i, cell in enumerate(nb["cells"]):
        if i in REPLACEMENTS:
            src = REPLACEMENTS[i]
            if not src.strip():
                continue
            cells.append(code_cell(src, f"inf-{i}", cell.get("metadata")))
        elif i in KEEP:
            new = dict(cell)
            if new["cell_type"] == "code":
                new["outputs"] = []
                new["execution_count"] = None
            cells.append(new)
        elif i == 23:
            continue
    nb["cells"] = cells
    return (json.dumps(nb, ensure_ascii=False, indent=1) + "\n").encode("utf-8")


def build_requirements() -> bytes:
    lines = (SRC / "requirements.txt").read_text(encoding="utf-8").splitlines()
    if not any(line.strip().lower().startswith("soundfile") for line in lines):
        lines.append("soundfile")
    return ("\n".join(lines) + "\n").encode("utf-8")


def main() -> None:
    entries: dict[str, bytes] = {
        "ordinal_inference.ipynb": build_notebook(),
        "README.md": README.encode("utf-8"),
        "requirements.txt": build_requirements(),
    }
    files: list[tuple[str, Path]] = []
    for rel in MODULES:
        files.append((rel, SRC / rel))
    for folder, pattern in [("figures_3class", "*.png"), ("tables_3class", "*.csv")]:
        for p in sorted((SRC / folder).glob(pattern)):
            if "a2" in p.name.lower():
                continue
            files.append((f"{folder}/{p.name}", p))
    for rel in STATES:
        p = SRC / rel
        assert p.exists(), p
        files.append((rel, p))

    for rel, p in files:
        assert p.exists(), p
        assert rel not in entries, rel
        assert not os.path.isabs(rel) and "\\" not in rel and ".." not in rel, rel
        entries[rel] = p.read_bytes()

    assert len(entries) == 83, len(entries)
    assert all(".pth.tar" not in n for n in entries)
    assert all("a2" not in n.lower() for n in entries)

    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6, allowZip64=True) as zf:
        for name in sorted(entries):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, entries[name])

    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes, {len(entries)} entries)")
    print(f"modules={len(MODULES)} png={sum(1 for n in entries if n.endswith('.png'))} "
          f"csv={sum(1 for n in entries if n.endswith('.csv'))} "
          f"pt={sum(1 for n in entries if n.endswith('.pt'))}")


if __name__ == "__main__":
    main()
