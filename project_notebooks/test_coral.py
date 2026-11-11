"""Self-checks for modules/nn/coral.py — plain asserts, mirrors test_split_stratified.py style."""
import sys
import numpy as np
import torch

sys.path.insert(0, ".")

from modules.nn import CoralOrdinal, coral_targets, coral_pos_weights, ordinal_predict, ordinal_oof_metrics

# --- CoralOrdinal forward: single tensor input ---
backbone = torch.nn.Linear(4, 1)
coral = CoralOrdinal(backbone, num_ranks=2)
x = torch.randn(5, 4)
out = coral(x)
assert out.shape == (5, 2), f"expected (5, 2), got {tuple(out.shape)}"
# zero-init biases -> both cumulative logits equal at init
assert torch.allclose(out[:, 0], out[:, 1]), "zero-init biases must make both cumulative logits equal"
print("OK: CoralOrdinal forward shape + zero-init bias equality")

# --- pair input (spec, mfcc) case ---
class _PairBackbone(torch.nn.Module):
    def forward(self, a, b):
        return torch.nn.functional.linear(a, torch.ones(1, 2))  # (B,1) from (B,2) input

pair_model = CoralOrdinal(_PairBackbone(), num_ranks=2)
out_p = pair_model((torch.randn(3, 2), torch.randn(3, 2)))
assert out_p.shape == (3, 2), f"pair input: expected (3, 2), got {tuple(out_p.shape)}"
print("OK: CoralOrdinal pair input forwards")

# --- coral_targets: rank (B,) -> cumulative (B, K-1) ---
rank = torch.tensor([0, 1, 2, 1, 0])
t = coral_targets(rank)
expected = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
assert torch.equal(t, expected), f"coral_targets mismatch: {t}"
print("OK: coral_targets cumulative encoding")

# --- coral_pos_weights: recording-level inverse-frequency ---
w = coral_pos_weights([9480, 2593, 623])
assert w.shape == (2,)
# pos_weight[k] = n(rank<=k) / n(rank>k)  (BCEWithLogitsLoss: upweights rare positives)
assert abs(w[0].item() - 9480 / 3216) < 1e-4, f"w[0]={w[0].item()} want {9480/3216}"
assert abs(w[1].item() - 12073 / 623) < 1e-4, f"w[1]={w[1].item()} want {12073/623}"
print("OK: coral_pos_weights cumulative pos weights")

# --- ordinal_predict: threshold crossings ---
logits = torch.tensor([[-2.0, -2.0], [3.0, -2.0], [3.0, 3.0]])
pred = ordinal_predict(logits)
assert torch.equal(pred, torch.tensor([0, 1, 2])), f"ordinal_predict mismatch: {pred}"
print("OK: ordinal_predict threshold crossings")

# --- ordinal_oof_metrics: hand-built 6-recording example ---
# true ranks: [0,0,1,1,2,2]; predictions: [0,1,1,1,2,0] (two mis-ranked, each off by 1)
true_r = np.array([0, 0, 1, 1, 2, 2])
pred_r = np.array([0, 1, 1, 1, 2, 0])
# hand-computed CM (rows=true): [[1,1,0],[0,2,0],[1,0,1]]
# recalls: low=1/2, med=2/2, high=1/2
# per-class F1 (zero_division=0): healthy: p=1/2, r=1/2 -> 0.5 ; med: p=2/3, r=1 -> 0.8 ; high: p=1/1, r=1/2 -> 2/3
# macro F1 = (0.5 + 0.8 + 2/3) / 3 = 2/3
# rank MAE = (1+1)/6 = 1/3  (rec 1: |1-0|=1, rec 5: |0-2|=2? no: pred_r[5]=0, |0-2|=2)
# recompute MAE: |0-0| + |1-0| + |1-1| + |1-1| + |2-2| + |0-2| = 0+1+0+0+0+2 = 3 -> 3/6 = 0.5
p1 = np.array([0.9, 0.6, 0.8, 0.9, 0.95, 0.2])   # vs (true>0): positives idx 2,3,4,5
p2 = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.3])    # vs (true>1): positives idx 4,5
m = ordinal_oof_metrics(true_r, pred_r, p1, p2)
MACRO_F1_EXPECTED = (0.5 + 0.8 + 2.0 / 3.0) / 3.0  # per-class F1s: 0.5, 0.8, 2/3
assert abs(m["macro_f1"] - MACRO_F1_EXPECTED) < 1e-9, f"macro_f1={m['macro_f1']} want {MACRO_F1_EXPECTED}"
assert abs(m["recall_low"] - 0.5) < 1e-9
assert abs(m["recall_med"] - 1.0) < 1e-9
assert abs(m["recall_high"] - 0.5) < 1e-9
assert abs(m["rank_mae"] - 0.5) < 1e-9, f"rank_mae={m['rank_mae']} want 0.5"
# AUC sanity: p1 ranks the (true>0) group well-ish; just check they're in (0,1)
for k in ("auc_r1", "auc_r2", "macro_auc"):
    assert m[k] is not None and 0.0 <= m[k] <= 1.0, f"{k}={m[k]}"
print("OK: ordinal_oof_metrics hand-computed values")

# --- degenerate single-rank true vector -> AUC None, no raise ---
true_d = np.zeros(5, dtype=int)
pred_d = np.zeros(5, dtype=int)
p1_d = np.full(5, 0.3)
p2_d = np.full(5, 0.2)
m_d = ordinal_oof_metrics(true_d, pred_d, p1_d, p2_d)
assert m_d["auc_r1"] is None and m_d["auc_r2"] is None and m_d["macro_auc"] is None
print("OK: ordinal_oof_metrics degenerate fold -> None AUCs without raising")

print("OK: all coral checks passed")
