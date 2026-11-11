"""One-shot notebook restructuring for the CORAL ordinal conversion (Tasks 5-7).
Rewrites cells in neural_network_3class.ipynb; deletes dead sweep cells; inserts
eval + figures cells after the CV cell. Run once from project_notebooks/.
"""
import json

NB = 'neural_network_3class.ipynb'
nb = json.load(open(NB, encoding='utf-8'))
by_id = {c['id']: c for c in nb['cells']}


def set_src(cid: str, src: str) -> None:
    by_id[cid]['source'] = src.splitlines(keepends=True)


CELL2 = '''import os

# ---- Crash-proof run state: per-epoch autosave + resume (train_ordinal) ----
RUN_STATE_DIR: str = "run_state_3class"
os.makedirs(RUN_STATE_DIR, exist_ok=True)

def _ckpt_path(name: str) -> str:
    return os.path.join(RUN_STATE_DIR, f"{name}_latest.pth.tar")

def _atomic_torch_save(obj: dict, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)  # crash mid-write never corrupts the good file
'''

CELL11 = '''# ==== Model factory setup + separate-stream pipelines for the branched model ====
from modules.nn import BranchedCoughCNN
from modules.nn.base_nn import ResNetModel

CV_MODELS: list[str] = ["default", "branched", "resnet"]  # each model gets its own 5-fold CV run

# separate spec/mfcc streams for the branched model (pipeline returns (spec, mfcc))
mel_train_plain_sep: SpectMFCCPipeline = SpectMFCCPipeline(combind=False).to(device)
mel_eval_sep: SpectMFCCPipeline = SpectMFCCPipeline(combind=False).to(device)
mel_eval_sep = torch.compile(mel_eval_sep) if device.type == "cuda" else mel_eval_sep
'''

CELL12 = '''# ---- Helpers shared by train_ordinal: model factory + best-checkpoint resume ----
def _make_model(model_name: str, num_classes: int = 1) -> nn.Module:
    """Model factory by string. resnet = ResNet50 ImageNet weights, head swapped, full finetune."""
    if model_name == "default":
        return DefaultCNN(num_classes)
    if model_name == "branched":
        return BranchedCoughCNN(num_classes)
    if model_name == "resnet":
        return ResNetModel(num_classes, freeze_backbone=False)
    raise ValueError(f"unknown model_name {model_name!r} (default|branched|resnet)")

def _ckpt_best_path(name: str) -> str:
    return os.path.join(RUN_STATE_DIR, f"{name}_best.pth.tar")

trained_models_all: dict[str, nn.Module] = {}  # registry _resume_with_best / train_ordinal write into

def _resume_with_best(name: str, model: nn.Module, optimizer: optim.Optimizer,
                      scheduler: optim.lr_scheduler.LRScheduler,
                      scaler: torch.amp.GradScaler, num_epochs: int) -> tuple[int, dict, float, int]:
    """Load _latest checkpoint into model/opt/sched/scaler; derive best val_loss
    and trailing plateau length from history. Returns (epochs_done, history, best_vl, plateau)."""
    best_vl, plate = float("inf"), 0
    path = _ckpt_path(name)
    fresh: dict[str, list[float]] = {"train_loss": [], "train_acc": [], "train_F1": [],
                                     "val_loss": [], "val_acc": [], "val_F1": [],
                                     "val_auc_r1": [], "val_auc_r2": []}
    if not os.path.exists(path):
        return 0, fresh, best_vl, plate
    ckpt = torch.load(path, weights_only=True, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    epochs_done: int = ckpt["epoch"] + 1  # 'epoch' is the 0-based index of the last COMPLETED epoch
    history: dict[str, list[float]] = ckpt["history"]
    for k in ("train_acc", "train_F1", "val_auc_r1", "val_auc_r2"):  # older checkpoints predate these keys
        history.setdefault(k, []).extend([float("nan")] * (len(history["val_loss"]) - len(history.get(k, []))))
    if history["val_loss"]:
        best_vl = min(history["val_loss"])
        for vl in reversed(history["val_loss"]):
            if vl <= best_vl + 1e-6:
                break
            plate += 1
    if epochs_done >= num_epochs:
        trained_models_all[name] = model
    print(f"=> [{name}] resumed at epoch {epochs_done + 1}/{num_epochs} (best val_loss {best_vl:.4f})" if epochs_done < num_epochs
          else f"=> [{name}] already complete ({num_epochs}/{num_epochs} epochs) - restored, no retrain")
    return epochs_done, history, best_vl, plate
'''

CELL16 = '''# ---- 5-fold CV: ordinal models, pooled OOF recording-level metrics ----
# Every pool recording is validated exactly once across the folds. Each fold's
# best-val-loss checkpoint feeds pooled out-of-fold segment probabilities, which
# are averaged per RECORDING before any metric is computed (segment-level
# numbers are never the headline). Logs append to logs_3class/ordinal_cv_experiment.txt.
import sys, time, atexit

LOG_PATH_CV = r"logs_3class/ordinal_cv_experiment.txt"
_logf_cv = open(LOG_PATH_CV, "a", encoding="utf-8")
sys.stdout = _logf_cv
sys.stderr = _logf_cv
atexit.register(_logf_cv.close)

HIST_COLS: list[str] = ["train_loss", "train_acc", "train_F1", "val_loss", "val_acc", "val_F1", "val_auc_r1", "val_auc_r2"]

oof_all: dict[str, dict[str, np.ndarray]] = {}  # model -> pooled per-segment OOF across folds
cv_rows: list[dict] = []
_orig = (train_subset, val_subset)
try:
    for mn in CV_MODELS:
        seg_acc: dict[str, list] = {"path": [], "true_rank": [], "p1": [], "p2": [], "fold": []}
        for fi, (tr_pos, va_pos) in enumerate(FOLDS):
            train_subset = Subset(dataset, [cv_pool[p] for p in tr_pos])
            val_subset = Subset(dataset, [cv_pool[p] for p in va_pos])
            key = f"cv__{mn}__f{fi}"
            print(f"\\n=== fold {fi + 1}/{len(FOLDS)}  {mn} ===")
            t0 = time.time()
            hist, oof = train_ordinal(key, model_name=mn)
            best_ep = int(np.argmin(hist["val_loss"]))
            cv_rows.append({
                "model": mn, "fold": fi, "epochs": len(hist["val_loss"]),
                "val_loss": round(hist["val_loss"][best_ep], 4),
                "val_acc": round(hist["val_acc"][best_ep], 4),
                "val_F1": round(hist["val_F1"][best_ep], 4),
                "val_auc_r1": round(hist["val_auc_r1"][best_ep], 4),
                "val_auc_r2": round(hist["val_auc_r2"][best_ep], 4),
            })
            print(f"=> {key}: best val_loss {hist['val_loss'][best_ep]:.4f} @ epoch {best_ep + 1} ({time.time() - t0:.0f}s)")
            seg_acc["path"].extend(oof["path"])
            seg_acc["true_rank"].extend(oof["true_rank"])
            seg_acc["p1"].extend(oof["p1"])
            seg_acc["p2"].extend(oof["p2"])
            seg_acc["fold"].extend([fi] * len(oof["p1"]))
            pl.DataFrame({k: hist[k] for k in HIST_COLS}).write_csv(f"cv__{mn}__f{fi}_experiment_stats.csv")
        oof_all[mn] = {k: np.asarray(v) for k, v in seg_acc.items()}
finally:
    train_subset, val_subset = _orig

# persist OOF state so eval/figures survive a fresh kernel
def _save_oof_state(obj: dict) -> None:
    tmp = "run_state_3class/ordinal_oof_state.pt.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, "run_state_3class/ordinal_oof_state.pt")

_save_oof_state({"oof_all": oof_all, "cv_rows": cv_rows})
print("=> OOF state saved to run_state_3class/ordinal_oof_state.pt")
pl.DataFrame(cv_rows).write_csv("cv_ordinal_folds.csv")
print("=> wrote cv_ordinal_folds.csv")
'''

CELL_EVAL = '''# ---- Recording-level OOF eval per model (the headline numbers) ----
# fresh-kernel recovery: load OOF state persisted by the CV cell
if "oof_all" not in globals() or not oof_all:
    _st = torch.load("run_state_3class/ordinal_oof_state.pt", weights_only=False)
    oof_all, cv_rows = _st["oof_all"], _st["cv_rows"]
    print("=> OOF state restored from run_state_3class/ordinal_oof_state.pt")

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

EVAL_COLS: list[str] = ["model", "n_recordings", "macro_f1", "recall_low", "recall_med", "recall_high",
                        "auc_r1", "auc_r2", "macro_auc", "rank_mae"]
eval_rows: list[dict] = []
rec_level: dict[str, dict[str, np.ndarray]] = {}
for mn in CV_MODELS:
    rec = aggregate_recordings(oof_all[mn])
    rec_level[mn] = rec
    m = ordinal_oof_metrics(rec["true"], rec["pred"], rec["p1"], rec["p2"])
    eval_rows.append({"model": mn, "n_recordings": len(rec["true"]),
                      **{k: (round(v, 4) if v is not None else None) for k, v in m.items()}})
    cm = confusion_matrix(rec["true"], rec["pred"], labels=[0, 1, 2])
    print(f"\\n=== OOF recording-level [{mn}] ({len(rec['true'])} recordings) ===")
    print(f"(rows=true ['low','med','high'], cols=pred)\\n{cm}")
    print({k: m[k] for k in m})
pl.DataFrame(eval_rows).write_csv("ordinal_oof_eval.csv")
print("=> wrote ordinal_oof_eval.csv")
'''

CELL_FIGS = '''# ---- Figures: per-fold histories, ordinal CMs, P(risk>med) distributions ----
os.makedirs("figures_3class", exist_ok=True)
RANK_NAMES: list[str] = ["low", "med", "high"]

for mn in CV_MODELS:
    # history: one val-loss line per fold
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted = False
    for fi in range(N_FOLDS):
        csv = f"cv__{mn}__f{fi}_experiment_stats.csv"
        if os.path.exists(csv):
            df = pl.read_csv(csv)
            ax.plot(df["val_loss"].to_list(), label=f"fold {fi}")
            plotted = True
    if plotted:
        ax.set_title(f"{mn}: val loss per CV fold"); ax.set_xlabel("epoch"); ax.legend()
        fig.tight_layout(); fig.savefig(f"figures_3class/ordinal_{mn}_history.png", dpi=200)
        plt.show(fig); plt.close(fig)

    # ordinal confusion matrix (recording level)
    rec = rec_level[mn]
    cm = confusion_matrix(rec["true"], rec["pred"], labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(cm, display_labels=RANK_NAMES)
    disp.plot(cmap="Blues")
    plt.title(f"{mn} - OOF recording-level confusion matrix")
    plt.savefig(f"figures_3class/ordinal_{mn}_cm.png", bbox_inches="tight", dpi=200)
    plt.show(); plt.close("all")

    # honest signal picture: per-recording P(rank>med) by true rank
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for r in range(3):
        ax.hist(rec["p2"][rec["true"] == r], bins=30, alpha=0.5, label=RANK_NAMES[r], range=(0, 1))
    ax.set_title(f"{mn}: per-recording P(rank>med) by true rank"); ax.set_xlabel("P(rank>med)"); ax.legend()
    fig.tight_layout(); fig.savefig(f"figures_3class/ordinal_{mn}_p2_hist.png", dpi=200)
    plt.show(fig); plt.close(fig)
print("figures written to figures_3class/")
'''

set_src('dea62985', CELL2)      # cell 2: drop dead resume_or_fresh + trained_models registry
set_src('27d2f6c3', CELL11)     # cell 11: model setup only
set_src('3a915f12', CELL12)     # cell 12: helpers only (factory, best ckpt, resume)
set_src('6347db5a', CELL16)     # cell 16: CV loop + OOF accumulation + persist

# cell 7: drop the trained_models registry line (resume_or_fresh is gone)
c7 = by_id['23e32579']
src7 = ''.join(c7['source'])
old7 = 'trained_models: dict[str, nn.Module] = {}  # registry resume_or_fresh writes into (kept alive)\n\n'
assert old7 in src7, 'cell 7 head line not found'
c7['source'] = src7.replace(old7, '').splitlines(keepends=True)

# delete dead cells: aug sweep + its eval + summary, multi-model loop + test eval + its plots
for dead in ('a8d74ae9', '96204cc1', 'd863b494', '30123159', 'ba554e5c', '7b62b338'):
    nb['cells'] = [c for c in nb['cells'] if c['id'] != dead]


def new_cell(src: str, cid: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "id": cid,
            "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)}


idx = next(i for i, c in enumerate(nb['cells']) if c['id'] == '6347db5a')
nb['cells'].insert(idx + 1, new_cell(CELL_EVAL, 'eval1oof'))
nb['cells'].insert(idx + 2, new_cell(CELL_FIGS, 'figs1oof'))

json.dump(nb, open(NB, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)

nb2 = json.load(open(NB, encoding='utf-8'))
for i, c in enumerate(nb2['cells']):
    compile(''.join(c['source']), f'cell{i}', 'exec')
print('OK:', len(nb2['cells']), 'cells;', [c['id'] for c in nb2['cells']])
