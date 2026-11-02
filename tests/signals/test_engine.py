from pathlib import Path
from typing import Any

import pandas as pd

from signal_factory.signals.compile import build_query
from signal_factory.signals.engine import compute_signals, connect
from signal_factory.signals.spec import SignalSpec

BASE: dict[str, Any] = {
    "version": 1,
    "hypothesis": "A fixture signal used to pin the engine's behaviour.",
    "entity_key": "uid",
    "window": "all",
}


def spec(id_: str, column: str, aggregation: str, **overrides: Any) -> SignalSpec:
    used = {"uid"} | ({column} if column != "*" else set())
    return SignalSpec.model_validate(
        {
            **BASE,
            "id": id_,
            "column": column,
            "aggregation": aggregation,
            "inputs": [{"column": c, "available_at": "decision_time"} for c in sorted(used)],
            **overrides,
        }
    )


SPECS_X5 = [
    spec("s_count", "*", "count"),
    spec("s_sum", "TransactionAmt", "sum"),
    spec("s_mean", "TransactionAmt", "mean", window={"hours": 24}),
    spec("s_nunique", "ProductCD", "count_distinct"),
    spec("s_recency", "*", "time_since_last"),
]


def test_connect_registers_history_table(tiny_history: pd.DataFrame) -> None:
    con = connect(tiny_history)
    assert con.execute("SELECT count(*) FROM history").fetchone() == (12,)


def test_compute_signals_from_dataframe_matches_single_query(tiny_history: pd.DataFrame) -> None:
    out = compute_signals(tiny_history, SPECS_X5)
    con = connect(tiny_history)
    direct = con.execute(build_query(SPECS_X5)).df()
    direct = direct.sort_values("TransactionID").reset_index(drop=True)
    for s in SPECS_X5:
        expected = pd.to_numeric(direct[s.id]).astype("float64")
        pd.testing.assert_series_equal(out[s.id], expected, check_names=False)


def test_compute_signals_output_shape_and_dtypes(tiny_history: pd.DataFrame) -> None:
    out = compute_signals(tiny_history, SPECS_X5)
    assert list(out.columns) == ["TransactionID", *[s.id for s in SPECS_X5]]
    assert len(out) == 12
    assert out["TransactionID"].tolist() == list(range(1, 13))
    assert all(out[s.id].dtype == "float64" for s in SPECS_X5)
    assert out.loc[out.TransactionID == 1, "s_count"].item() == 0.0
    assert pd.isna(out.loc[out.TransactionID == 1, "s_sum"].item())
    assert pd.isna(out.loc[out.TransactionID == 10, "s_count"].item())


def test_compute_signals_batches_and_preserves_order(tiny_history: pd.DataFrame) -> None:
    out = compute_signals(tiny_history, SPECS_X5, batch_size=2)
    assert list(out.columns) == ["TransactionID", *[s.id for s in SPECS_X5]]
    pd.testing.assert_frame_equal(out, compute_signals(tiny_history, SPECS_X5))


def test_compute_signals_with_no_specs(tiny_history: pd.DataFrame) -> None:
    out = compute_signals(tiny_history, [])
    assert list(out.columns) == ["TransactionID"] and len(out) == 12


def test_compute_signals_from_parquet_path(tmp_path: Path, tiny_history: pd.DataFrame) -> None:
    path = tmp_path / "history.parquet"
    tiny_history.to_parquet(path, index=False)
    pd.testing.assert_frame_equal(
        compute_signals(path, SPECS_X5), compute_signals(tiny_history, SPECS_X5)
    )


def test_compute_signals_dtype_can_be_float32(tiny_history: pd.DataFrame) -> None:
    import numpy as np

    out = compute_signals(tiny_history, SPECS_X5, dtype=np.float32)
    assert all(out[s.id].dtype == "float32" for s in SPECS_X5)
    wide = compute_signals(tiny_history, SPECS_X5)
    for s in SPECS_X5:
        pd.testing.assert_series_equal(out[s.id].astype("float64"), wide[s.id], check_names=False)
