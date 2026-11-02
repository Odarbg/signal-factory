import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from signal_factory.baseline.metrics import (
    FPR,
    ThresholdResult,
    pr_auc,
    recall_at_fpr,
    recall_at_threshold,
    roc_auc,
    seen_mask,
)


def _toy(seed: int = 0, n: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < 0.05).astype(np.int8)
    scores = rng.beta(2, 5, n) + 0.3 * labels
    return np.clip(scores, 0, 1), labels


def test_fpr_is_pre_registered() -> None:
    assert FPR == 0.01


def test_recall_at_fpr_matches_sklearn_roc_curve() -> None:
    scores, labels = _toy()
    r = recall_at_fpr(scores, labels)
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    ok = fpr <= 0.01
    assert r.recall == pytest.approx(tpr[ok].max())
    assert r.actual_fpr <= 0.01
    # the chosen threshold really realises that recall and FPR
    check = recall_at_threshold(scores, labels, r.threshold)
    assert check.recall == pytest.approx(r.recall) and check.actual_fpr == pytest.approx(
        r.actual_fpr
    )


def test_recall_at_fpr_with_ties_never_exceeds_budget() -> None:
    # Review Focus 5: 50 rows tie at 0.9 of which 20 are negatives (2.1% of 950 negatives),
    # so flagging the tied block would breach 1%. The conservative threshold flags nothing.
    scores = np.array([0.9] * 50 + [0.1] * 950)
    labels = np.array([1] * 30 + [0] * 20 + [0] * 950, dtype=np.int8)
    r = recall_at_fpr(scores, labels, 0.01)
    assert r.actual_fpr <= 0.01
    assert r.recall == 0.0
    assert r.threshold > 0.9


def test_recall_at_fpr_picks_smallest_threshold_within_budget() -> None:
    # negatives: 1000 scores at 0.0..0.999; flagging score >= 0.99 is 10/1000 = exactly 1%.
    neg = np.arange(1000) / 1000.0
    pos = np.array([0.995, 0.5])
    scores = np.concatenate([neg, pos])
    labels = np.array([0] * 1000 + [1, 1], dtype=np.int8)
    r = recall_at_fpr(scores, labels, 0.01)
    assert r.actual_fpr == pytest.approx(0.01)
    assert r.threshold == pytest.approx(0.99)
    assert r.recall == pytest.approx(0.5)


def test_recall_at_threshold_reports_realised_fpr() -> None:
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    labels = np.array([1, 0, 1, 0, 0], dtype=np.int8)
    r = recall_at_threshold(scores, labels, 0.7)
    assert isinstance(r, ThresholdResult)
    assert (r.threshold, r.recall) == (0.7, 1.0) and r.actual_fpr == pytest.approx(1 / 3)
    r2 = recall_at_threshold(scores, labels, 0.85)
    assert r2.recall == 0.5 and r2.actual_fpr == 0.0


def test_recall_at_threshold_flags_score_equal_to_threshold() -> None:
    r = recall_at_threshold(np.array([0.5, 0.4]), np.array([1, 0], dtype=np.int8), 0.5)
    assert r.recall == 1.0 and r.actual_fpr == 0.0


def test_recall_with_no_positives_is_nan_not_crash() -> None:
    r = recall_at_threshold(np.array([0.5, 0.4]), np.array([0, 0], dtype=np.int8), 0.5)
    assert np.isnan(r.recall) and r.actual_fpr == 0.5


def test_seen_mask_null_uid_is_unseen() -> None:
    # Review Focus 4
    m = seen_mask(pd.Series(["a", None, "b"], dtype=pd.StringDtype("pyarrow")), ["a"])
    assert m.dtype == np.bool_ and m.tolist() == [True, False, False]


def test_seen_mask_accepts_series_with_nulls_as_train_uids() -> None:
    train = pd.Series(["a", None], dtype=pd.StringDtype("pyarrow"))
    m = seen_mask(pd.Series(["a", None], dtype=pd.StringDtype("pyarrow")), train)
    assert m.tolist() == [True, False]


def test_pr_auc_and_roc_auc_match_sklearn() -> None:
    scores, labels = _toy(1)
    assert pr_auc(scores, labels) == pytest.approx(average_precision_score(labels, scores))
    assert roc_auc(scores, labels) == pytest.approx(roc_auc_score(labels, scores))
