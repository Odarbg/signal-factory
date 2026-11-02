from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from signal_factory.data.split import (
    DEV_END,
    FINAL_END,
    FINAL_START,
    VALIDATION_END,
    VALIDATION_START,
    Period,
    SplitCounts,
    assign_period,
    write_split,
)


def test_boundary_constants_are_pre_registered() -> None:
    assert (DEV_END, VALIDATION_START, VALIDATION_END, FINAL_START, FINAL_END) == (
        98,
        106,
        138,
        146,
        183,
    )


@pytest.mark.parametrize(
    "day,period",
    [
        (1, "dev"),
        (98, "dev"),
        (99, "gap"),
        (105, "gap"),
        (106, "validation"),
        (138, "validation"),
        (139, "gap"),
        (145, "gap"),
        (146, "final"),
        (183, "final"),
    ],
)
def test_assign_period_boundaries(day: int, period: Period) -> None:
    assert assign_period(day) == period


@pytest.mark.parametrize("day", [184, 0, -1])
def test_assign_period_rejects_days_outside_dataset(day: int) -> None:
    with pytest.raises(ValueError):
        assign_period(day)


def _synthetic_all(path: Path) -> pd.DataFrame:
    # one row per day 1..183, plus a duplicate of every 10th day
    days = list(range(1, 184)) + list(range(10, 184, 10))
    n = len(days)
    df = pd.DataFrame(
        {
            "TransactionID": np.arange(1, n + 1),
            "isFraud": np.array([d % 7 == 0 for d in days], dtype="int64"),
            "TransactionDT": np.array(days) * 86400 + np.arange(n),
            "TransactionAmt": np.linspace(1.0, 100.0, n, dtype="float32"),
            "day": np.array(days, dtype="int32"),
            "uid": pd.Series([f"u{d % 5}" for d in days], dtype=pd.StringDtype("pyarrow")),
        }
    )
    df.to_parquet(path, index=False)
    return df


def test_write_split_files_are_disjoint_and_history_has_no_label(tmp_path: Path) -> None:
    src = tmp_path / "all.parquet"
    _synthetic_all(src)
    out = tmp_path / "split"

    counts = write_split(src, out)

    dev, val, fin, hist = (
        pd.read_parquet(out / f)
        for f in ("dev.parquet", "validation.parquet", "final.parquet", "history.parquet")
    )
    assert set(dev.day) <= set(range(1, 99))
    assert set(val.day) <= set(range(106, 139))
    assert set(fin.day) <= set(range(146, 184))
    assert not (set(dev.TransactionID) & set(val.TransactionID))
    assert not (set(val.TransactionID) & set(fin.TransactionID))
    assert not (set(dev.TransactionID) & set(fin.TransactionID))
    assert "isFraud" not in hist.columns
    assert "isFraud" in dev.columns and "isFraud" in val.columns and "isFraud" in fin.columns
    assert len(hist) == counts.history == counts.dev + counts.gap + counts.validation + counts.final
    assert set(hist.day) >= {99, 105, 139, 145}  # Review Focus 3: gap rows are history
    gap_days = set(range(99, 106)) | set(range(139, 146))
    assert not (set(dev.day) | set(val.day) | set(fin.day)) & gap_days


def test_write_split_counts(tmp_path: Path) -> None:
    src = tmp_path / "all.parquet"
    df = _synthetic_all(src)
    counts = write_split(src, tmp_path / "split")
    expected = SplitCounts(
        dev=int((df.day <= 98).sum()),
        gap=int(((df.day >= 99) & (df.day <= 105)).sum() + ((df.day >= 139) & (df.day <= 145)).sum()),
        validation=int(((df.day >= 106) & (df.day <= 138)).sum()),
        final=int(((df.day >= 146) & (df.day <= 183)).sum()),
        history=len(df),
    )
    assert counts == expected
    assert counts.gap == 14 + 2  # days 99..105 and 139..145, plus duplicates of days 100 and 140
