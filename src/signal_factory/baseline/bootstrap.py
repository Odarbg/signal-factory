"""Paired bootstrap for the difference in recall between two score vectors (spec §6).

Rows are resampled with replacement and the **same** resample is applied to both rows of
the ablation. Recall at a frozen threshold depends only on how many resampled positives
each row flags, so each resample is summarised by the counts of positives in the four
cells (flagged by neither, by A only, by B only, by both). Drawing those cell counts
directly (a binomial for the number of positives, then a multinomial over cells) has
exactly the distribution of row resampling and avoids an ``n_resamples × n_rows`` index
matrix on 100k+ final rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_RESAMPLES = 1000
PERCENTILES = (2.5, 97.5)


@dataclass(frozen=True)
class BootstrapResult:
    mean_diff: float
    lo: float
    hi: float
    n_resamples: int


def paired_recall_diff(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    labels: np.ndarray,
    threshold_a: float,
    threshold_b: float,
    n_resamples: int = N_RESAMPLES,
    seed: int = 0,
) -> BootstrapResult:
    """Percentile interval of ``recall(b) - recall(a)`` over paired row resamples."""
    scores_a, scores_b, labels = (np.asarray(v) for v in (scores_a, scores_b, labels))
    if not (scores_a.shape == scores_b.shape == labels.shape):
        raise ValueError("scores_a, scores_b and labels must have the same shape")
    n = len(labels)
    positive = labels == 1
    n_pos = int(positive.sum())
    if n == 0 or n_pos == 0:
        raise ValueError("need at least one row and one positive label")

    flag_a = scores_a[positive] >= threshold_a
    flag_b = scores_b[positive] >= threshold_b
    # cells: 0 = neither, 1 = A only, 2 = B only, 3 = both
    cell = flag_a.astype(np.int64) + 2 * flag_b.astype(np.int64)
    cell_probs = np.bincount(cell, minlength=4) / n_pos

    rng = np.random.default_rng(seed)
    k = rng.binomial(n, n_pos / n, size=n_resamples)  # positives per resample
    cells = rng.multinomial(k, cell_probs)  # shape (n_resamples, 4)
    with np.errstate(invalid="ignore", divide="ignore"):
        diff = (cells[:, 2] - cells[:, 1]) / k  # (B only − A only) / positives
    diff = diff[k > 0]

    lo, hi = np.percentile(diff, PERCENTILES)
    return BootstrapResult(
        mean_diff=float(diff.mean()), lo=float(lo), hi=float(hi), n_resamples=n_resamples
    )
