"""One-shot install of the aug-config experiment + tables_3class/ retarget (run once)."""
import json

NB = 'neural_network_3class.ipynb'
nb = json.load(open(NB, encoding='utf-8'))
by_id = {c['id']: c for c in nb['cells']}


def set_src(cid: str, src: str) -> None:
    by_id[cid]['source'] = src.splitlines(keepends=True)


# ---- 1. train_ordinal: aug-config params ----
c7 = by_id['23e32579']
src = ''.join(c7['source'])

old = "    min_epoch: int = 15,\n) -> tuple[dict, dict]:"
assert old in src, 'sig anchor missing'
src = src.replace(old, "    min_epoch: int = 15,\n    wave_aug: bool = True,\n    spec_aug: bool = True,\n) -> tuple[dict, dict]:")

old = "    # per-run spec-aug pipeline; waveform aug always on (single \"both\" config)\n" \
      "    mel_train = SpectMFCCPipeline(\n" \
      "        transforms=SpecAugmentations(augment_probs=aug_probs).to(device), combind=not use_sep,\n" \
      "    ).to(device)"
assert old in src, 'mel_train anchor missing'
src = src.replace(old,
    "    # per-run aug pipelines: spec-aug only when spec_aug, waveform aug only when wave_aug\n"
    "    mel_train = SpectMFCCPipeline(\n"
    "        transforms=SpecAugmentations(augment_probs=aug_probs).to(device) if spec_aug else None,\n"
    "        combind=not use_sep,\n"
    "    ).to(device)\n"
    "    print(f\"[{name}] aug config: wave={'on' if wave_aug else 'OFF'} spec={'on' if spec_aug else 'OFF'}\")")

old = "WaveformAugmentor(16_000, augment_probs=aug_probs))"
assert old in src, 'waveform augmentor anchor missing'
src = src.replace(old,
    "WaveformAugmentor(16_000, augment_probs=aug_probs if wave_aug else {0: 0.0, 1: 0.0, 2: 0.0}))")
c7['source'] = src.splitlines(keepends=True)

# ---- 2. move aggregate_recordings into the helpers cell ----
c_eval = by_id['eval1oof']
src_eval = ''.join(c_eval['source'])
start = src_eval.index('def aggregate_recordings')
end = src_eval.index('\n\nEVAL_COLS')
agg_fn = src_eval[start:end] + '\n'
src_eval = src_eval[:start] + '# aggregate_recordings lives in the helpers cell (shared with the aug test cell)\n' + src_eval[end:].lstrip('\n')
c_eval['source'] = src_eval.splitlines(keepends=True)

c_helpers = by_id['3a915f12']
set_src('3a915f12', ''.join(c_helpers['source']) + '\n\n' + agg_fn)

# ---- 3. new cells between helpers and CV loop ----
CELL_A = '''# ---- Augmentation-config experiment (DefaultCNN only, fixed split - NOT CV) ----
# Isolates the augmentation axis on the from-scratch baseline BEFORE any CV fold
# training. Split is recording-level via dataset_split_by_recording over ALL paths
# (NOT cell 4's segment-random dataset_split: that holdout leaks sibling segments
# across pool/test, so it must not serve as a test set). The third output here is
# this experiment's own test set.
import sys, time, contextlib

os.makedirs("tables_3class", exist_ok=True)
os.makedirs("figures_3class", exist_ok=True)

AUG_CONFIGS: dict[str, tuple[bool, bool]] = {
    "wave_only": (True, False),   # waveform aug only
    "spec_only": (False, True),   # spec-aug (stretch + time/freq masks) only
    "both": (True, True),         # the config the CV runs use
}

aug_train_subset, aug_val_subset, aug_test_subset = dataset_split_by_recording(
    dataset, dataset.paths, 0.8, 0.1, seed=SEED
)
print(f"aug split (recording-level): train {len(aug_train_subset.indices)} | "
      f"val {len(aug_val_subset.indices)} | test {len(aug_test_subset.indices)} segments")
_aug_tr_ranks: dict[int, int] = {}
for i in aug_train_subset.indices:
    _aug_tr_ranks[RANK[int(dataset.labels[i])]] = _aug_tr_ranks.get(RANK[int(dataset.labels[i])], 0) + 1
print("aug train segment ranks:", _aug_tr_ranks)

@contextlib.contextmanager
def aug_log():
    """Append-mode stdout/stderr redirect to the aug experiment log; restores on exit."""
    f = open("logs_3class/aug_experiment.txt", "a", encoding="utf-8")
    out, err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = f
    try:
        yield f
    finally:
        sys.stdout, sys.stderr = out, err
        f.close()
'''

CELL_B = '''# ---- Train the 3 aug configs on the fixed split (cache-proof: name-keyed checkpoints) ----
AUG_HIST_COLS: list[str] = ["train_loss", "train_acc", "train_F1", "val_loss", "val_acc", "val_F1",
                            "val_auc_r1", "val_auc_r2"]

def _aug_epochs_done(name: str) -> int:
    """Epochs already cached for this run (0 if no checkpoint) - the cache-proof probe."""
    p = _ckpt_path(name)
    if not os.path.exists(p):
        return 0
    return int(torch.load(p, weights_only=True, map_location="cpu")["epoch"]) + 1

_aug_orig = (train_subset, val_subset)
_aug_cached = 0
with aug_log():
    try:
        for cfg, (w, s) in AUG_CONFIGS.items():
            name = f"aug__{cfg}__default"
            done_before = _aug_epochs_done(name)
            train_subset, val_subset = aug_train_subset, aug_val_subset
            t0 = time.time()
            hist, _ = train_ordinal(name, wave_aug=w, spec_aug=s)
            pl.DataFrame({k: hist[k] for k in AUG_HIST_COLS}).write_csv(
                f"tables_3class/{name}_experiment_stats.csv")
            print(f"=> {name}: {len(hist['val_loss'])} epochs, best val_loss "
                  f"{min(hist['val_loss']):.4f} ({time.time() - t0:.0f}s)")
            if done_before >= num_epochs:
                _aug_cached += 1
                print(f"=> cache proof: {name} served from checkpoint (no retrain)")
    finally:
        train_subset, val_subset = _aug_orig
print(f"cache: {_aug_cached}/{len(AUG_CONFIGS)} aug runs served from checkpoint")
'''

CELL_C = '''# ---- Test-set eval per aug config (recording-level, from best checkpoints) ----
_AUG_EVAL_COLS: list[str] = ["config", "n_recordings", "macro_f1", "recall_low", "recall_med",
                             "recall_high", "auc_r1", "auc_r2", "macro_auc", "rank_mae"]
aug_eval_rows: list[dict] = []
aug_test_rec: dict[str, dict[str, np.ndarray]] = {}
_test_loader = torch.utils.data.DataLoader(
    Subset(dataset, aug_test_subset.indices), BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True,
)
with aug_log():
    for cfg in AUG_CONFIGS:
        name = f"aug__{cfg}__default"
        aug_model = CoralOrdinal(_make_model("default", num_classes=1)).to(device)
        best = torch.load(_ckpt_best_path(name), weights_only=True, map_location=device)
        aug_model.load_state_dict(best["model_state_dict"])
        aug_model.eval()
        oof_true: list[int] = []
        oof_p1: list[float] = []
        oof_p2: list[float] = []
        with torch.inference_mode():
            for waveforms, labels in tqdm(_test_loader, desc=f"[{name}] test", leave=False):
                waveforms = trim_tile(waveforms).to(device, non_blocking=True)
                raw = labels.long().to(device)
                rank = torch.tensor([RANK[int(l)] for l in raw.tolist()], device=device)
                feats = mel_eval(waveforms, raw)
                if feats.ndim == 3:
                    feats = feats.unsqueeze(0)
                logits = aug_model(feats)
                probs = torch.sigmoid(logits.float())
                oof_true.extend(rank.cpu().tolist())
                oof_p1.extend(probs[:, 0].cpu().tolist())
                oof_p2.extend(probs[:, 1].cpu().tolist())
        aug_oof = {
            "path": np.asarray([dataset.get_path(int(i)) for i in aug_test_subset.indices], dtype=object),
            "true_rank": np.asarray(oof_true, dtype=int),
            "p1": np.asarray(oof_p1), "p2": np.asarray(oof_p2),
        }
        rec = aggregate_recordings(aug_oof)
        aug_test_rec[cfg] = rec
        m = ordinal_oof_metrics(rec["true"], rec["pred"], rec["p1"], rec["p2"])
        aug_eval_rows.append({"config": cfg, "n_recordings": len(rec["true"]),
                              **{k: (round(v, 4) if v is not None else None) for k, v in m.items()}})
        cm = confusion_matrix(rec["true"], rec["pred"], labels=[0, 1, 2])
        print(f"\\n=== aug test [{cfg}] ({len(rec['true'])} recordings) ===")
        print(f"(rows=true ['low','med','high'], cols=pred)\\n{cm}")
        print({k: m[k] for k in m})
pl.DataFrame(aug_eval_rows).write_csv("tables_3class/aug_test_eval.csv")
_tmp = "run_state_3class/aug_test_state.pt.tmp"
torch.save({"rec": aug_test_rec, "rows": aug_eval_rows}, _tmp)
os.replace(_tmp, "run_state_3class/aug_test_state.pt")
print("=> tables_3class/aug_test_eval.csv + run_state_3class/aug_test_state.pt written")
'''

CELL_D = '''# ---- Figures: aug experiment (training curves, test CMs, P(rank>med) by true rank) ----
# fresh-kernel recovery: load test state stashed by the test-eval cell
if "aug_test_rec" not in globals() or not aug_test_rec:
    _st = torch.load("run_state_3class/aug_test_state.pt", weights_only=False)
    aug_test_rec = _st["rec"]
    print("=> aug test state restored from run_state_3class/aug_test_state.pt")

_AUG_COLORS: dict[str, str] = {"wave_only": "tab:blue", "spec_only": "tab:orange", "both": "tab:green"}
_AUG_RANK_NAMES: list[str] = ["low", "med", "high"]

# training curves: 2x2 panels, one line per config (train loss dashed on the loss panel)
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
panels = [("val_loss", "val loss"), ("val_F1", "val macro F1"),
          ("val_auc_r1", "val AUC r1"), ("val_auc_r2", "val AUC r2")]
for cfg in AUG_CONFIGS:
    df = pl.read_csv(f"tables_3class/aug__{cfg}__default_experiment_stats.csv")
    for ax, (col, title) in zip(axes.flat, panels):
        ax.plot(df[col].to_list(), label=cfg, color=_AUG_COLORS[cfg])
    axes.flat[0].plot(df["train_loss"].to_list(), color=_AUG_COLORS[cfg], ls="--", alpha=0.45)
axes.flat[0].set_title("loss (dashed = train)")
for ax in axes.flat:
    ax.set_xlabel("epoch")
fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="upper right")
fig.tight_layout()
fig.savefig("figures_3class/aug_default_curves.png", dpi=200)
plt.show(fig); plt.close(fig)

# test confusion matrices (recording level, counts)
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, cfg in zip(axes, AUG_CONFIGS):
    rec = aug_test_rec[cfg]
    cm = confusion_matrix(rec["true"], rec["pred"], labels=[0, 1, 2])
    ConfusionMatrixDisplay(cm, display_labels=_AUG_RANK_NAMES).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{cfg} - test recordings")
fig.tight_layout()
fig.savefig("figures_3class/aug_default_test_cm.png", dpi=200)
plt.show(fig); plt.close("all")

# P(rank>med) per recording by true rank (the honest signal picture)
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, cfg in zip(axes, AUG_CONFIGS):
    rec = aug_test_rec[cfg]
    for r in range(3):
        ax.hist(rec["p2"][rec["true"] == r], bins=30, alpha=0.5, range=(0, 1), label=_AUG_RANK_NAMES[r])
    ax.set_title(cfg); ax.set_xlabel("P(rank>med)")
axes[0].set_ylabel("recordings"); axes[0].legend()
fig.tight_layout()
fig.savefig("figures_3class/aug_default_p2_hist.png", dpi=200)
plt.show(fig); plt.close(fig)
print("aug figures written to figures_3class/")
'''


def new_cell(src: str, cid: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "id": cid,
            "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)}


idx = next(i for i, c in enumerate(nb['cells']) if c['id'] == '3a915f12')
for off, (src, cid) in enumerate([(CELL_A, 'aug1split'), (CELL_B, 'aug2train'),
                                  (CELL_C, 'aug3test'), (CELL_D, 'aug4figs')]):
    nb['cells'].insert(idx + 1 + off, new_cell(src, cid))

# ---- 4. retarget all CSV writes/reads to tables_3class/ ----
c_cv = by_id['6347db5a']
src = ''.join(c_cv['source'])
a = 'pl.DataFrame({k: hist[k] for k in HIST_COLS}).write_csv(f"cv__{mn}__f{fi}_experiment_stats.csv")'
b = 'pl.DataFrame({k: hist[k] for k in HIST_COLS}).write_csv(f"tables_3class/cv__{mn}__f{fi}_experiment_stats.csv")'
assert a in src
src = src.replace(a, b)
a = 'pl.DataFrame(cv_rows).write_csv("cv_ordinal_folds.csv")'
b = 'pl.DataFrame(cv_rows).write_csv("tables_3class/cv_ordinal_folds.csv")'
assert a in src
c_cv['source'] = src.replace(a, b).splitlines(keepends=True)

a = 'pl.DataFrame(eval_rows).write_csv("ordinal_oof_eval.csv")'
b = 'pl.DataFrame(eval_rows).write_csv("tables_3class/ordinal_oof_eval.csv")'
assert a in ''.join(c_eval['source'])
c_eval['source'] = ''.join(c_eval['source']).replace(a, b).splitlines(keepends=True)

c_figs = by_id['figs1oof']
src = ''.join(c_figs['source'])
a = 'csv = f"cv__{mn}__f{fi}_experiment_stats.csv"'
b = 'csv = f"tables_3class/cv__{mn}__f{fi}_experiment_stats.csv"'
assert a in src
c_figs['source'] = src.replace(a, b).splitlines(keepends=True)

json.dump(nb, open(NB, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)

nb2 = json.load(open(NB, encoding='utf-8'))
for i, c in enumerate(nb2['cells']):
    compile(''.join(c['source']), f'cell{i}', 'exec')
print('OK:', len(nb2['cells']), 'cells;', [c['id'] for c in nb2['cells']])
