"""CORAL-style ordinal regression utilities: cumulative-logit head, targets,
recording-level pos weights, rank prediction, and OOF metric computation."""
from typing import Sequence

import numpy as np
import torch
from torch import nn

from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score


class CoralOrdinal(nn.Module):
    """Wraps a backbone that outputs (B, 1) and produces cumulative logits (B, K-1).

    logit_k(x) = backbone(x) - bias_k, with shared backbone weights and per-rank
    biases (zero-initialized so all cumulative logits start equal).
    """

    def __init__(self, backbone: nn.Module, num_ranks: int = 2):
        super().__init__()
        self.backbone = backbone
        self.num_ranks = num_ranks
        self.biases = nn.Parameter(torch.zeros(num_ranks))

    def forward(self, x):
        score = self.backbone(*x) if isinstance(x, tuple) else self.backbone(x)
        score = score.reshape(-1)  # (B,)
        return score.unsqueeze(1) - self.biases.unsqueeze(0)  # (B, K-1)


def coral_targets(rank: torch.Tensor, num_ranks: int = 3) -> torch.Tensor:
    """rank (B,) ints in {0..K-1} -> cumulative binary targets (B, K-1) float, target_k = rank > k."""
    rank = rank.reshape(-1).long()
    return (rank.unsqueeze(1) > torch.arange(num_ranks - 1).unsqueeze(0)).float()


def coral_pos_weights(rec_counts: Sequence[int]) -> torch.Tensor:
    """Recording-level counts per rank (length K) -> pos_weight (K-1,) for BCEWithLogitsLoss.

    For cumulative task k (P(rank > k)): positives = ranks > k, negatives = ranks <= k.
    pos_weight[k] = n(pos) / n(neg) — i.e. n(rank > k) / n(rank <= k). Counts clamped min=1.
    """
    counts = torch.tensor([float(c) for c in rec_counts]).clamp(min=1.0)
    k = len(rec_counts) - 1
    below = torch.cumsum(counts, dim=0)[:k]          # n(rank <= k) for k = 0..K-2
    above = counts.sum() - below                     # n(rank > k)
    return below / above


def ordinal_predict(logits: torch.Tensor) -> torch.Tensor:
    """Predicted rank (B,) = sum of sigmoid cumulative logits > 0.5."""
    return (torch.sigmoid(logits) > 0.5).sum(dim=1)


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return None


def ordinal_oof_metrics(
    rec_true_rank: np.ndarray,
    rec_pred_rank: np.ndarray,
    rec_p1: np.ndarray,
    rec_p2: np.ndarray,
) -> dict:
    """Recording-level ordinal metrics. AUCs guarded: degenerate fold -> None."""
    cm = confusion_matrix(rec_true_rank, rec_pred_rank, labels=[0, 1, 2])
    recalls = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    f1s = f1_score(rec_true_rank, rec_pred_rank, average=None, labels=[0, 1, 2], zero_division=0)

    auc_r1 = _safe_auc((rec_true_rank > 0).astype(int), rec_p1)
    auc_r2 = _safe_auc((rec_true_rank > 1).astype(int), rec_p2)
    aucs = [a for a in (auc_r1, auc_r2) if a is not None]

    return {
        "macro_f1": float(f1s.mean()),
        "recall_low": float(recalls[0]),
        "recall_med": float(recalls[1]),
        "recall_high": float(recalls[2]),
        "auc_r1": auc_r1,
        "auc_r2": auc_r2,
        "macro_auc": float(np.mean(aucs)) if aucs else None,
        "rank_mae": float(np.abs(rec_pred_rank - rec_true_rank).mean()),
    }
