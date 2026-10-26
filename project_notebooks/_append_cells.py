import json, ast

path = 'neural_network.ipynb'
nb = json.load(open(path, encoding='utf-8'))
cells = nb['cells']

def newcell(cid, src):
    return {"cell_type": "code", "execution_count": None, "id": cid,
            "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)}

src_a = '''# ==== Multi-model training (DefaultCNN / BranchedCoughCNN / ResNet50 finetune) ====
# Appended block: config, trainer, training loop, eval, plots — new cells at the end.
# Existing cascade cells above are untouched.
from modules.nn import ResNetModel  # extra import for this block

# pick model + augmentation BY STRING before running:
# model: "default" | "branched" | "resnet"   aug: "both" | "wave_only" | "spec_only"
TRAINED_CONFIGS: dict[str, list[str]] = {
    "default":  ["both", "wave_only", "spec_only"],
    "branched": ["both", "wave_only", "spec_only"],
    "resnet":   ["both", "wave_only", "spec_only"],
}
AUG_MAP: dict[str, tuple[bool, bool]] = {"both": (True, True), "wave_only": (True, False), "spec_only": (False, True)}
MODEL_PATIENCE: int = 20  # early stop: val_loss plateau of this many epochs
TRAINED_RUNS: list[tuple[str, str]] = [(mn, cfg) for mn, cfgs in TRAINED_CONFIGS.items() for cfg in cfgs]
print("Runs scheduled:", [f"{mn}/{cfg}" for mn, cfg in TRAINED_RUNS])

# separate spec/mfcc streams for the branched model (forward returns (spec, mfcc))
mel_train_plain_sep: SpectMFCCPipeline = SpectMFCCPipeline(combind=False).to(device)
mel_train_spec_sep: SpectMFCCPipeline = SpectMFCCPipeline(transforms=SpecAugmentations(), combind=False).to(device)
mel_eval_sep: SpectMFCCPipeline = SpectMFCCPipeline(combind=False).to(device)
mel_eval_sep = torch.compile(mel_eval_sep) if device.type == "cuda" else mel_eval_sep'''

src_b = '''# ---- Multi-model trainer: same checkpointing/resume as train_binary, plus
# model-by-string, best-checkpoint saves, and early stop on val_loss plateaus ----
def _make_model(model_name: str, num_classes: int = 2) -> nn.Module:
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

def _resume_with_best(name: str, model: nn.Module, optimizer: optim.Optimizer,
                      scheduler: optim.lr_scheduler.LRScheduler,
                      scaler: torch.amp.GradScaler, num_epochs: int) -> tuple[int, dict, float, int]:
    """Load _latest checkpoint into model/opt/sched/scaler; derive best val_loss
    and trailing plateau length from history. Returns (epochs_done, history, best_vl, plateau)."""
    best_vl, plate = float("inf"), 0
    path = _ckpt_path(name)
    if not os.path.exists(path):
        return 0, {"train_loss": [], "val_loss": [], "val_acc": [], "val_F1": [], "val_auc": [], "val_pr_auc": []}, best_vl, plate
    ckpt = torch.load(path, weights_only=True, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    epochs_done: int = ckpt["epoch"] + 1
    history: dict[str, list[float]] = ckpt["history"]
    if history["val_loss"]:
        best_vl = min(history["val_loss"])
        for vl in reversed(history["val_loss"]):
            if vl <= best_vl + 1e-6:
                break
            plate += 1
    if epochs_done >= num_epochs:
        trained_models[name] = model
    print(f"=> [{name}] resumed at epoch {epochs_done + 1}/{num_epochs} (best val_loss {best_vl:.4f})" if epochs_done < num_epochs
          else f"=> [{name}] already complete ({num_epochs}/{num_epochs} epochs) — restored, no retrain")
    return epochs_done, history, best_vl, plate

def _collect_probs_mn(model: nn.Module, mn: str, indices: Sequence[int], positive: int,
                      task: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """collect_probs generalized per model: branched takes separate (spec, mfcc)
    streams; default/resnet take the combined 3-channel image."""
    use_sep = (mn == "branched")
    if task is not None:
        task_ds = TaskDataset(dataset, task)
        pos = task_ds.positions_for(list(indices))
        src = task_ds
    else:
        pos = list(indices)
        src = dataset
    loader = torch.utils.data.DataLoader(
        Subset(src, pos), BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    probs: list[float] = []
    targets: list[int] = []
    model.eval()
    with torch.inference_mode():
        for waveforms, labels in loader:
            waveforms = waveforms.to(device, non_blocking=True)
            raw = labels.long()
            if use_sep:
                spec, mfcc = mel_eval_sep(waveforms)
                if spec.ndim == 3:
                    spec, mfcc = spec.unsqueeze(0), mfcc.unsqueeze(0)
                outputs = model(spec, mfcc)
            else:
                images = mel_eval(waveforms)
                if images.ndim == 3:
                    images = images.unsqueeze(0)
                outputs = model(images)
            probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
            if task is not None:
                t = TASK_TARGETS[task](raw)
            else:
                t = binary_labels(raw, positive)
            targets.extend(t.numpy())
    return np.array(probs), np.array(targets)

def train_model(name: str, positive: int, use_wave: bool, use_spec: bool,
                model_name: str = "default",
                positives: frozenset[int] | None = None,
                task: str | None = None,
                patience: int = 20) -> dict[str, list[float]]:
    """train_binary generalized: model by string, best-checkpoint saves,
    early stop when val_loss stalls for `patience` epochs. Crash-proof identical."""
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_acc": [], "val_F1": [], "val_auc": [], "val_pr_auc": []}

    if task is not None:
        task_ds = TaskDataset(dataset, task)
        have = set(task_ds.indices)
        tr_base = [i for i in train_subset.indices if i in have]
        va_base = [i for i in val_subset.indices if i in have]
        tr_pos = task_ds.positions_for(tr_base)
        va_pos = task_ds.positions_for(va_base)
        tgt = TASK_TARGETS[task](dataset.labels[torch.tensor(tr_base)])
        n_pos = int(tgt.sum()); n_neg = len(tgt) - n_pos
        w = 1.0 / torch.tensor([n_neg, n_pos], dtype=torch.float32).clamp(min=1)
    else:
        tr_idx, va_idx = train_subset.indices, val_subset.indices
        if positives is not None:
            in_group = torch.isin(train_labels, torch.tensor(sorted(positives)))
            counts = torch.tensor([(~in_group).sum(), in_group.sum()]).float()
        else:
            counts = torch.bincount(train_labels, minlength=3).float()
            counts = torch.tensor([counts.sum() - counts[positive], counts[positive]])
        w = 1.0 / counts.clamp(min=1)
    w = w / w.sum() * 2.0  # normalize so mean weight is 1 (keeps loss scale comparable)
    w = w.to(device)

    criterion: nn.Module = FocalLoss(alpha=w, gamma=2.0)

    m = _make_model(model_name)
    if device.type == "cuda":
        model: nn.Module = torch.compile(m).to(device)
    else:
        model = m.to(device)

    optimizer: optim.AdamW = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    scaler: torch.amp.GradScaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    scheduler: optim.lr_scheduler.CosineAnnealingLR = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    epochs_done, history, best_val_loss, plateau = _resume_with_best(
        name, model, optimizer, scheduler, scaler, num_epochs)

    use_sep: bool = (model_name == "branched")
    mel_train = (mel_train_spec_sep if use_spec else mel_train_plain_sep) if use_sep else (mel_train_spec if use_spec else mel_train_plain)

    if task is not None:
        tr_src, va_src = task_ds, task_ds
    else:
        tr_src = va_src = dataset

    train_ds: AugmentedSubset | Subset = AugmentedSubset(
        tr_src, tr_pos if task is not None else tr_idx,
        WaveformAugmentor(16_000)
    ) if use_wave else Subset(tr_src, tr_pos if task is not None else tr_idx)

    train_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        train_ds, BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True,
        persistent_workers=True, prefetch_factor=4,
    )
    val_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        Subset(va_src, va_pos if task is not None else va_idx), BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True,
        persistent_workers=True, prefetch_factor=4,
    )

    def relabel(raw: torch.Tensor) -> torch.Tensor:
        if task is not None:
            return TASK_TARGETS[task](raw)
        if positives is not None:
            return binary_labels_grouped(raw, positives)
        return binary_labels(raw, positive)

    def forward_features(waveforms: torch.Tensor, raw: torch.Tensor, mel):
        """Combined pipelines return one (B,3,F,T) image; separated return (spec, mfcc)."""
        feats = mel(waveforms, raw)
        if use_sep:
            spec, mfcc = feats
            if spec.ndim == 3:
                spec, mfcc = spec.unsqueeze(0), mfcc.unsqueeze(0)
            return spec, mfcc
        if feats.ndim == 3:
            feats = feats.unsqueeze(0)
        return feats

    for epoch in range(epochs_done, num_epochs):
        model.train()
        running_loss = 0.0

        for waveforms, labels in tqdm(train_loader, desc=f"[{name}] train {epoch+1}/{num_epochs}", leave=False):
            waveforms = waveforms.to(device, non_blocking=True)
            raw = labels.long().to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            feats = forward_features(waveforms, raw, mel_train)  # fp32 outside autocast (see train_binary note)
            target = relabel(raw)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(*feats) if use_sep else model(feats)
                loss = criterion(outputs, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        scheduler.step()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds, all_targets, all_probs = [], [], []

        with torch.inference_mode():
            for waveforms, labels in tqdm(val_loader, desc=f"[{name}] val {epoch+1}/{num_epochs}", leave=False):
                waveforms = waveforms.to(device, non_blocking=True)
                raw = labels.long().to(device, non_blocking=True)
                feats = forward_features(waveforms, raw, mel_eval_sep if use_sep else mel_eval)
                target = relabel(raw)
                with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    outputs = model(*feats) if use_sep else model(feats)
                    loss = criterion(outputs, target)
                all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
                val_loss += loss.item()
                total += raw.size(0)
                correct += (torch.argmax(outputs, dim=1) == target).sum().item()

        val_macro_f1 = f1_score(all_targets, all_preds, average="macro")
        val_auc_score = roc_auc_score(all_targets, all_probs)
        pr_curve, pr_targets, _ = precision_recall_curve(all_targets, all_probs)
        val_pr = auc(pr_targets, pr_curve)
        train_epoch_loss = running_loss / len(train_loader)
        val_epoch_loss = val_loss / len(val_loader)
        val_accuracy = 100.0 * correct / total

        history["train_loss"].append(train_epoch_loss)
        history["val_loss"].append(val_epoch_loss)
        history["val_F1"].append(val_macro_f1)
        history["val_acc"].append(val_accuracy)
        history["val_auc"].append(val_auc_score)
        history["val_pr_auc"].append(val_pr)

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "history": history,
        }
        _atomic_torch_save(state, _ckpt_path(name))
        if val_epoch_loss < best_val_loss - 1e-6:
            best_val_loss = val_epoch_loss
            plateau = 0
            _atomic_torch_save(state, _ckpt_best_path(name))  # best checkpoint on val_loss improvement
            print(f"=> [{name}] new best val_loss {best_val_loss:.4f} (epoch {epoch+1})")
        else:
            plateau += 1

        print(f"[{name}] Epoch [{epoch+1}/{num_epochs}] -> "
              f"Train Loss: {train_epoch_loss:.4f} | "
              f"Val Loss: {val_epoch_loss:.4f} | "
              f"Val F1: {val_macro_f1:.4f} | "
              f"Val AUC: {val_auc_score:.4f} | "
              f"Val PR-AUC: {val_pr:.4f} | "
              f"Val Acc: {val_accuracy:.2f}%")

        if plateau >= patience:
            print(f"=> [{name}] early stop: val_loss plateaued for {patience} epochs (best {best_val_loss:.4f})")
            break

    trained_models[name] = model
    return history'''

src_c = '''# ---- Multi-model training loop: 2-stage cascade x models x aug configs ----
# All outputs labeled trained_*. Seeds from disk (never wipes/KeyErrors on re-run),
# resumes mid-run, short-circuits finished runs.
import time
LOG_PATH3 = r"logs/trained_models_experiment_2026-10-25.txt"
_logf3 = open(LOG_PATH3, "a", encoding="utf-8")
atexit.register(_logf3.close)

def seed_trained_state() -> tuple[dict, dict]:
    """Seed trained_results / trained_models_all from CSVs + checkpoints on disk
    (plus whatever is already in memory) so re-runs never wipe or KeyError."""
    results: dict[str, dict[str, list[float]]] = {}
    if "trained_results" in globals():
        results.update(trained_results)
    models_all: dict[str, dict] = {}
    if "trained_models_all" in globals():
        models_all.update(trained_models_all)
    for mn, cfg in TRAINED_RUNS:
        for stage, task in [("stage1", "covid_symp_vs_healthy"), ("stage2", "covid_vs_symp")]:
            key = f"{mn}__{task}__{cfg}"
            if key not in results and os.path.exists(f"trained_{key}_experiment_stats.csv"):
                df = pl.read_csv(f"trained_{key}_experiment_stats.csv")
                results[key] = {col: df[col].to_list() for col in CSV_COLS}
            if stage not in models_all.setdefault(mn, {}).setdefault(cfg, {}) and os.path.exists(_ckpt_path(key)):
                ckpt = torch.load(_ckpt_path(key), weights_only=True, map_location=device)
                m = _make_model(mn).to(device)
                # Lazy models need materialized params before load_state_dict
                with torch.inference_mode():
                    if mn == "branched":
                        s = torch.zeros(1, 3, 104, 32, device=device)
                        _ = m(s, s)
                    else:
                        _ = m(torch.zeros(1, 3, 104, 32, device=device))
                if any(k.startswith("_orig_mod.") for k in ckpt["model_state_dict"]):
                    m = torch.compile(m)
                m.load_state_dict(ckpt["model_state_dict"])
                m.eval()
                models_all[mn][cfg][stage] = m
    return results, models_all

trained_results, trained_models_all = seed_trained_state()
trained_stop_info: dict[str, str] = {}  # key -> "complete" | "early stop @ epoch N"

print("STARTING MULTI-MODEL EXPERIMENTS")
for mn, cfg in TRAINED_RUNS:
    uw, us = AUG_MAP[cfg]
    for stage_num, (stage, task, positive) in enumerate(
        [("stage1", "covid_symp_vs_healthy", -1), ("stage2", "covid_vs_symp", 1)], start=1
    ):
        key = f"{mn}__{task}__{cfg}"
        if key in trained_results and len(trained_results[key]["val_loss"]) >= num_epochs:
            print(f"\\n--- {key} --- already complete, skipping (restored from disk)")
            continue
        print(f"\\n--- {key} --- (stage {stage_num}: {task})")
        t0 = time.time()
        hist = train_model(key, positive=positive, use_wave=uw, use_spec=us,
                           model_name=mn, task=task, patience=MODEL_PATIENCE)
        trained_results[key] = hist
        pl.DataFrame({k: hist[k] for k in CSV_COLS}).write_csv(f"trained_{key}_experiment_stats.csv")
        trained_models_all.setdefault(mn, {}).setdefault(cfg, {})[stage] = trained_models[key]
        done = len(hist["val_loss"])
        trained_stop_info[key] = "complete" if done >= num_epochs else f"early stop @ epoch {done}"
        print(f"=> {key}: {done}/{num_epochs} epochs in {time.time() - t0:.0f}s — {trained_stop_info[key]}")

print("\\n=== Multi-model training done ===")
for (mn, cfg) in TRAINED_RUNS:
    for task in ("covid_symp_vs_healthy", "covid_vs_symp"):
        key = f"{mn}__{task}__{cfg}"
        if key in trained_results:
            h = trained_results[key]
            print(f"{key:<44} {len(h['val_loss']):>3}/60 epochs  final val_acc {h['val_acc'][-1]:.2f}%")'''

src_d = '''# ---- Multi-model cascade eval: thresholds + 3-class decision on TEST ----
# Mirrors the cascade eval cell for every trained_{model} run.
# Persists to run_state/trained_eval_state.pt so a fresh kernel skips the eval pass.
def _save_trained_eval(obj: dict) -> None:
    tmp = "run_state/trained_eval_state.pt.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, "run_state/trained_eval_state.pt")

def _load_trained_eval() -> dict | None:
    p = "run_state/trained_eval_state.pt"
    return torch.load(p, weights_only=False) if os.path.exists(p) else None

if "trained_preds" in globals() and trained_models_all:
    # fresh eval pass (models in memory from training)
    trained_preds: dict[str, np.ndarray] = {}
    trained_eval_rows: list[dict] = []
    trained_stage2_rows: list[dict] = []

    test_y3_t = np.array([int(dataset.labels[i]) for i in test_subset.indices])

    for mn, cfg in TRAINED_RUNS:
        key1, key2 = f"{mn}__covid_symp_vs_healthy__{cfg}", f"{mn}__covid_vs_symp__{cfg}"
        if key1 not in trained_results:
            continue
        m1 = trained_models_all[mn][cfg]["stage1"]
        m2 = trained_models_all[mn][cfg]["stage2"]

        val_probs1, val_y1 = _collect_probs_mn(m1, mn, val_subset.indices, positive=-1, task="covid_symp_vs_healthy")
        tau_sick = tune_threshold(val_y1, val_probs1)
        val_probs2, val_y2 = _collect_probs_mn(m2, mn, sick_val_idx, positive=1, task="covid_vs_symp")
        tau_covid = tune_threshold(val_y2, val_probs2)

        test_probs1, _ = _collect_probs_mn(m1, mn, test_subset.indices, positive=-1, task="covid_symp_vs_healthy")
        flagged = np.where(test_probs1 >= tau_sick)[0]
        flagged_idx = [test_subset.indices[i] for i in flagged]
        flagged_sick = [i for i in flagged_idx if int(dataset.labels[i]) in SICK]
        covid_probs_flagged, _ = _collect_probs_mn(m2, mn, flagged_sick, positive=1, task="covid_vs_symp")

        pred3 = np.zeros(len(test_y3_t), dtype=int)
        sick_rows = np.array([j for j, i in enumerate(flagged_idx) if int(dataset.labels[i]) in SICK])
        pred3[sick_rows] = np.where(covid_probs_flagged >= tau_covid, 1, 2)

        cm3 = confusion_matrix(test_y3_t, pred3, labels=[0, 1, 2])
        trained_eval_rows.append({
            "experiment": f"trained_{key1}", "tau_sick": round(tau_sick, 4), "tau_covid": round(tau_covid, 4),
            "healthy_recall": round(cm3[0, 0] / cm3[0].sum(), 4),
            "covid_recall": round(cm3[1, 1] / cm3[1].sum(), 4),
            "symp_recall": round(cm3[2, 2] / cm3[2].sum(), 4),
        })
        trained_preds[key1] = pred3
        print(f"\\n=== TRAINED CASCADE 3-class TEST [{mn}/{cfg}] === tau_sick={tau_sick:.3f} tau_covid={tau_covid:.3f}")
        for cls in range(3):
            denom = cm3[cls].sum()
            print(f"  {LABELS_INV[cls]:<12} recall: {cm3[cls, cls] / denom:.4f}" if denom else f"  {LABELS_INV[cls]:<12} recall: n/a")

        s2_probs, s2_y = _collect_probs_mn(m2, mn, sick_test_idx, positive=1, task="covid_vs_symp")
        s2_preds = (s2_probs >= tau_covid).astype(int)
        s2_pr, s2_t, _ = precision_recall_curve(s2_y, s2_probs)
        trained_stage2_rows.append({
            "experiment": f"trained_{key2}",
            "test_roc_auc": round(roc_auc_score(s2_y, s2_probs), 4),
            "test_pr_auc": round(auc(s2_t, s2_pr), 4),
            "covid_recall": round((s2_preds[s2_y == 1] == 1).mean(), 4),
            "symp_recall": round((s2_preds[s2_y == 0] == 0).mean(), 4),
        })

    pl.DataFrame(trained_eval_rows).write_csv("trained_cascade_eval_results.csv")
    pl.DataFrame(trained_stage2_rows).write_csv("trained_stage2_standalone_eval.csv")
    _save_trained_eval({"trained_preds": trained_preds, "test_y3": test_y3_t,
                        "trained_eval_rows": trained_eval_rows, "trained_stage2_rows": trained_stage2_rows})
    print("=> trained eval state saved")
else:
    # fresh kernel after a crash: restore instead of re-running the eval pass
    _tev = _load_trained_eval()
    if _tev is None:
        raise RuntimeError("trained_preds not in memory and no run_state/trained_eval_state.pt — run the trained-model training + eval cells first")
    trained_preds = _tev["trained_preds"]
    test_y3_t = _tev["test_y3"]
    trained_eval_rows = _tev["trained_eval_rows"]
    trained_stage2_rows = _tev["trained_stage2_rows"]
    print("=> trained eval state restored from run_state/trained_eval_state.pt (eval pass skipped)")'''

src_e = '''# ---- Multi-model plots + summary table (mirrors cascade plot cell) ----
# History plots, 3-class CMs, stage CMs, summary table — all labeled trained_*.
# crash recovery: results from CSVs, preds from eval state
if "trained_results" not in globals() or not trained_results:
    trained_results = {}
    for mn, cfg in TRAINED_RUNS:
        for task in ("covid_symp_vs_healthy", "covid_vs_symp"):
            key = f"{mn}__{task}__{cfg}"
            csv = f"trained_{key}_experiment_stats.csv"
            if os.path.exists(csv):
                df = pl.read_csv(csv)
                trained_results[key] = {col: df[col].to_list() for col in CSV_COLS}
    print("=> trained_results rebuilt from trained_*_experiment_stats.csv")

if "trained_preds" not in globals():
    _tep = "run_state/trained_eval_state.pt"
    assert os.path.exists(_tep), "run_state/trained_eval_state.pt missing — run the trained-eval cell first"
    _tev = torch.load(_tep, weights_only=False)
    trained_preds, test_y3_t = _tev["trained_preds"], _tev["test_y3"]
    trained_eval_rows, trained_stage2_rows = _tev["trained_eval_rows"], _tev["trained_stage2_rows"]
    print("=> trained preds/test_y3 restored from run_state/trained_eval_state.pt")

# history plots
for (mn, cfg) in TRAINED_RUNS:
    for task in ("covid_symp_vs_healthy", "covid_vs_symp"):
        key = f"{mn}__{task}__{cfg}"
        if key not in trained_results:
            continue
        hist = trained_results[key]
        fig, (ax_loss, ax_acc, ax_f1) = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(f"trained {mn} / {task} / {cfg}")
        epochs = range(1, len(hist["train_loss"]) + 1)
        ax_loss.plot(epochs, hist["train_loss"], label="train")
        ax_loss.plot(epochs, hist["val_loss"], label="val")
        ax_loss.set_title("Loss"); ax_loss.legend()
        ax_acc.plot(epochs, hist["val_acc"]); ax_acc.set_title("Val Accuracy (%)")
        ax_f1.plot(epochs, hist["val_F1"]); ax_f1.set_title("Val F1")
        fig.tight_layout()
        fig.savefig(f"figures/trained_{key}_history.png", bbox_inches="tight")
        plt.show(fig)
        plt.close(fig)

# 3-class cascade CMs
for key, pred3 in trained_preds.items():
    cm3 = confusion_matrix(test_y3_t, pred3, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(cm3, display_labels=list(LABELS_INV.values()))
    disp.plot(cmap="Blues")
    plt.title(f"Trained cascade 3-class TEST - {key}")
    plt.savefig(f"figures/trained_cascade_3class_{key}.png", bbox_inches="tight")
    plt.show()
    plt.close("all")

# stage-1/stage-2 binary CMs (need model-aware inference + splits in memory)
if "_collect_probs_mn" not in globals() or "test_subset" not in globals() or "trained_models_all" not in globals():
    print("=> skipping trained stage CMs: _collect_probs_mn/test_subset/trained_models_all not in memory")
else:
    for mn, cfg in TRAINED_RUNS:
        if cfg not in trained_models_all.get(mn, {}):
            continue
        for stage, task, idxs_val, idxs_test, labels_disp, positive in [
            ("stage1", "covid_symp_vs_healthy", val_subset.indices, test_subset.indices, ["healthy", "covid+symp"], -1),
            ("stage2", "covid_vs_symp", sick_val_idx, sick_test_idx, ["symptomatic", "COVID-19"], 1),
        ]:
            m = trained_models_all[mn][cfg].get(stage)
            if m is None:
                continue
            tp, ty = _collect_probs_mn(m, mn, idxs_test, positive=positive, task=task)
            _, vy = _collect_probs_mn(m, mn, idxs_val, positive=positive, task=task)
            tau = tune_threshold(vy, tp)
            p = (tp >= tau).astype(int)
            cm = confusion_matrix(ty, p, labels=[0, 1])
            print(f"trained {mn} {stage} TEST [{cfg}] tau={tau:.3f}")
            print(cm)
            disp = ConfusionMatrixDisplay(cm, display_labels=labels_disp)
            disp.plot(cmap="Blues")
            plt.title(f"Trained {mn} {stage} TEST - {cfg}")
            plt.savefig(f"figures/trained_{mn}_{stage}_{task}_{cfg}_cm.png", bbox_inches="tight")
            plt.show()
            plt.close("all")

# summary table: same stats as results_table, per trained run, + epoch counts
rows = []
for key, hist in trained_results.items():
    for metric in ("train_loss", "val_loss", "val_acc"):
        vals = hist[metric]
        rows.append({
            "experiment": f"trained_{key}", "metric": metric,
            "max": round(max(vals), 4), "min": round(min(vals), 4),
            "final": round(vals[-1], 4), "avg": round(sum(vals) / len(vals), 4),
            "std": round(float(np.std(vals)), 4), "epochs": len(vals),
        })
summary_df = pl.DataFrame(rows)
pl.Config.set_ascii_tables()  # ponytail: cp1252 console can't render polars box chars
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=120):
    print(summary_df)
summary_df.write_csv("trained_experiment_summary.csv")
# eval CSVs rewritten here too so a fresh-kernel restore refreshes them
pl.DataFrame(trained_eval_rows).write_csv("trained_cascade_eval_results.csv")
pl.DataFrame(trained_stage2_rows).write_csv("trained_stage2_standalone_eval.csv")
print("All trained-model figures + CSVs written to figures/ and notebook root.")'''

cells.append(newcell('mm-config', src_a))
cells.append(newcell('mm-trainer', src_b))
cells.append(newcell('mm-train', src_c))
cells.append(newcell('mm-eval', src_d))
cells.append(newcell('mm-plots', src_e))

json.dump(nb, open(path, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)

nb2 = json.load(open(path, encoding='utf-8'))
for c in nb2['cells']:
    if c['cell_type'] == 'code':
        ast.parse(''.join(c['source']))
print('appended:', [c.get('id') for c in nb2['cells'][-5:]])
print('ALL CELLS PARSE')
