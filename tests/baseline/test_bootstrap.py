import numpy as np
import pytest

from signal_factory.baseline.bootstrap import N_RESAMPLES, BootstrapResult, paired_recall_diff
from signal_factory.baseline.metrics import recall_at_threshold


def _toy(seed: int = 0, n: int = 3000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < 0.1).astype(np.int8)
    scores_a = np.clip(rng.beta(2, 5, n) + 0.25 * labels, 0, 1)
    scores_b = np.clip(scores_a + 0.15 * labels * rng.random(n), 0, 1)  # b is better on positives
    return scores_a, scores_b, labels


def test_n_resamples_is_pre_registered() -> None:
    assert N_RESAMPLES == 1000


def test_identical_scores_give_zero_interval() -> None:
    s, _, y = _toy()
    r = paired_recall_diff(s, s, y, 0.5, 0.5)
    assert isinstance(r, BootstrapResult)
    assert r.lo <= 0 <= r.hi and abs(r.mean_diff) < 1e-12
    assert r.n_resamples == N_RESAMPLES


def test_interval_contains_known_difference_on_synthetic() -> None:
    a, b, y = _toy()
    t = 0.6
    point = recall_at_threshold(b, y, t).recall - recall_at_threshold(a, y, t).recall
    r = paired_recall_diff(a, b, y, t, t, n_resamples=500)
    assert r.lo <= point <= r.hi
    assert r.lo > 0  # b really is better; the interval excludes zero
    assert r.mean_diff == pytest.approx(point, abs=0.02)


def test_different_thresholds_per_row_are_applied() -> None:
    a, _, y = _toy(2)
    r_same = paired_recall_diff(a, a, y, 0.6, 0.6, n_resamples=200)
    r_shifted = paired_recall_diff(a, a, y, 0.6, 0.4, n_resamples=200)
    assert abs(r_same.mean_diff) < 1e-12
    assert r_shifted.mean_diff > 0  # lower threshold on b flags more positives


def test_is_seed_deterministic() -> None:
    a, b, y = _toy(1)
    r1 = paired_recall_diff(a, b, y, 0.6, 0.6, n_resamples=300, seed=7)
    r2 = paired_recall_diff(a, b, y, 0.6, 0.6, n_resamples=300, seed=7)
    r3 = paired_recall_diff(a, b, y, 0.6, 0.6, n_resamples=300, seed=8)
    assert r1 == r2
    assert r1 != r3


def test_resample_is_paired() -> None:
    # If the same resample is applied to both rows, identical score vectors differ by exactly 0
    # in every resample, so the interval collapses to [0, 0] rather than merely covering zero.
    s, _, y = _toy(3)
    r = paired_recall_diff(s, s, y, 0.5, 0.5, n_resamples=200)
    assert r.lo == 0.0 and r.hi == 0.0


def test_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        paired_recall_diff(np.zeros(3), np.zeros(4), np.zeros(3, dtype=np.int8), 0.5, 0.5)
