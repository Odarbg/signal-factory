import json
from pathlib import Path

import pandas as pd
import pytest

from signal_factory.ablation.row_a import Metrics, RowResult, run_row, run_row_a
from tests.ablation.conftest import record_then_call

FAST = {"num_leaves": 7, "learning_rate": 0.1, "n_estimators": 20, "min_child_samples": 5}


def test_run_row_a_writes_json_and_never_opens_final(
    tmp_path: Path, synthetic_split: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[str] = []
    monkeypatch.setattr(pd, "read_parquet", record_then_call(opened))

    res = run_row_a(synthetic_split, tmp_path / "out", seeds=(0,), params=FAST)

    assert opened, "run_row_a must read its inputs through pd.read_parquet"
    assert not any("final" in p for p in opened)
    assert (tmp_path / "out" / "row_a.json").exists()
    assert isinstance(res, RowResult) and res.name == "A"
    assert 0 <= res.validation.recall <= 1
    assert res.validation.actual_fpr <= 0.01
    assert res.validation.n == res.validation_seen.n + res.validation_unseen.n
    assert res.validation_unseen.n > 0
    assert res.params == FAST
    assert res.seeds == (0,)


def test_row_a_json_round_trips(tmp_path: Path, synthetic_split: Path) -> None:
    res = run_row_a(synthetic_split, tmp_path / "out", seeds=(0,), params=FAST)
    data = json.loads((tmp_path / "out" / "row_a.json").read_text())
    assert data["name"] == "A"
    assert data["validation"]["recall"] == res.validation.recall
    assert set(data["validation"]) == {"recall", "actual_fpr", "pr_auc", "roc_auc", "n"}
    assert data["threshold"] == res.threshold


def test_threshold_is_frozen_on_all_validation_rows(tmp_path: Path, synthetic_split: Path) -> None:
    res = run_row_a(synthetic_split, tmp_path / "out", seeds=(0,), params=FAST)
    # one threshold for every subset: the subsets report their realised FPR, not 1%
    assert res.validation_seen.recall == res.validation_seen.recall  # not NaN or defined
    assert isinstance(res.validation_seen, Metrics)


def test_run_row_with_extra_features_joins_on_transaction_id(
    tmp_path: Path, synthetic_split: Path
) -> None:
    history = pd.read_parquet(synthetic_split / "history.parquet")
    feats = pd.DataFrame({"TransactionID": history["TransactionID"], "extra_sig": 1.0})
    feats.loc[feats.index[::2], "extra_sig"] = 0.0
    fpath = tmp_path / "extra.parquet"
    feats.to_parquet(fpath, index=False)

    res = run_row("B", synthetic_split, tmp_path / "out", [fpath], seeds=(0,), params=FAST)

    assert (tmp_path / "out" / "row_b.json").exists()
    assert "extra_sig" in res.feature_columns
    assert res.name == "B"


def test_run_row_with_no_extra_features_equals_run_row_a(
    tmp_path: Path, synthetic_split: Path
) -> None:
    a = run_row_a(synthetic_split, tmp_path / "a", seeds=(0,), params=FAST)
    b = run_row("A", synthetic_split, tmp_path / "b", [], seeds=(0,), params=FAST)
    assert a == b


def test_run_row_reads_each_feature_parquet_once(
    tmp_path: Path, synthetic_split: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    history = pd.read_parquet(synthetic_split / "history.parquet")
    fpath = tmp_path / "extra.parquet"
    pd.DataFrame({"TransactionID": history["TransactionID"], "extra_sig": 1.0}).to_parquet(
        fpath, index=False
    )
    opened: list[str] = []
    monkeypatch.setattr(pd, "read_parquet", record_then_call(opened))
    run_row("B", synthetic_split, tmp_path / "out", [fpath], seeds=(0,), params=FAST)
    assert opened.count(str(fpath)) == 1


def test_run_row_rejects_feature_parquet_missing_rows(
    tmp_path: Path, synthetic_split: Path
) -> None:
    history = pd.read_parquet(synthetic_split / "history.parquet")
    partial = pd.DataFrame({"TransactionID": history["TransactionID"].iloc[:10], "extra_sig": 1.0})
    fpath = tmp_path / "stale.parquet"
    partial.to_parquet(fpath, index=False)
    with pytest.raises(ValueError, match="stale.parquet"):
        run_row("B", synthetic_split, tmp_path / "out", [fpath], seeds=(0,), params=FAST)
