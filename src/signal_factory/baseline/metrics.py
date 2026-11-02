"""Pre-registered metrics (spec §6): recall at 1% FPR with a frozen threshold, PR/ROC AUC.

The threshold is chosen once, on all validation rows, as the **smallest** score such that
flagging ``score >= threshold`` keeps the realised FPR within budget. Under ties this is
the conservative side: a tied block that would breach the budget is left unflagged.
``recall_at_threshold`` then reports the realised FPR wherever the threshold is applied.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

FPR = 0.01


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    recall: float
    actual_fpr: float


def _split(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must have the same shape")
    positive = labels == 1
    return scores[positive], scores[~positive]


def _rate(flagged: int, total: int) -> float:
    return float(flagged) / total if total else float("nan")


def recall_at_threshold(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> ThresholdResult:
    """Recall and realised FPR when flagging ``score >= threshold``."""
    pos, neg = _split(scores, labels)
    return ThresholdResult(
        threshold=float(threshold),
        recall=_rate(int((pos >= threshold).sum()), len(pos)),
        actual_fpr=_rate(int((neg >= threshold).sum()), len(neg)),
    )


def recall_at_fpr(scores: np.ndarray, labels: np.ndarray, fpr: float = FPR) -> ThresholdResult:
    """The smallest threshold whose realised FPR is within ``fpr``, with its recall."""
    _, neg = _split(scores, labels)
    neg_sorted = np.sort(neg)
    candidates = np.unique(np.asarray(scores, dtype=np.float64))
    flagged_negatives = len(neg) - np.searchsorted(neg_sorted, candidates, side="left")
    within = flagged_negatives <= fpr * len(neg)
    if within.any():
        threshold = float(candidates[np.argmax(within)])  # candidates ascending: first valid
    else:
        threshold = float(np.nextafter(candidates[-1], np.inf))  # flag nothing
    return recall_at_threshold(scores, labels, threshold)


def pr_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(average_precision_score(labels, scores))


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores))


def seen_mask(eval_uids: pd.Series, train_uids: Iterable[str]) -> np.ndarray:
    """True where the row's ``uid`` appears among training uids; a NULL ``uid`` is unseen."""
    known = {u for u in train_uids if not pd.isna(u)}
    seen = eval_uids.isin(list(known)).to_numpy(dtype=bool)
    return seen & eval_uids.notna().to_numpy(dtype=bool)
