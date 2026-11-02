from pathlib import Path

import pandas as pd

from signal_factory.routine.compute import compute_row_b
from signal_factory.routine.grid import routine_grid
from signal_factory.signals.engine import compute_signals


def test_compute_row_b_writes_all_grid_columns(tmp_path: Path, grid_history: pd.DataFrame) -> None:
    src = tmp_path / "history.parquet"
    grid_history.to_parquet(src, index=False)
    out = tmp_path / "row_b.parquet"

    n = compute_row_b(src, out, batch_size=150)

    df = pd.read_parquet(out)
    grid = routine_grid()
    assert n == len(grid)
    assert list(df.columns) == ["TransactionID", *[s.id for s in grid]]
    assert len(df) == len(grid_history)


def test_compute_row_b_cells_match_engine(tmp_path: Path, grid_history: pd.DataFrame) -> None:
    src = tmp_path / "history.parquet"
    grid_history.to_parquet(src, index=False)
    out = tmp_path / "row_b.parquet"
    compute_row_b(src, out, batch_size=400)

    df = pd.read_parquet(out).set_index("TransactionID")
    by_id = {s.id: s for s in routine_grid()}
    chosen = [
        by_id["rb__uid__24h__count__star"],
        by_id["rb__DeviceInfo__7d__count_distinct__P_emaildomain"],
    ]
    expected = compute_signals(grid_history, chosen).set_index("TransactionID")
    for s in chosen:
        pd.testing.assert_series_equal(df[s.id], expected[s.id], check_names=False)
