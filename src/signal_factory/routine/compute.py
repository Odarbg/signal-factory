"""Compute the row-B grid over ``history.parquet`` with the signal engine (spec §7 fallback path).

The grid is 1,626 columns over ~590k rows. Each batch is computed as float32 and written to
its own Parquet file; DuckDB then joins the batch files on ``TransactionID`` into the final
file, so the full matrix is never held in pandas.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import duckdb
import numpy as np

from signal_factory.routine.grid import routine_grid
from signal_factory.signals.compile import ID_COLUMN, quote_identifier
from signal_factory.signals.engine import compute_signals
from signal_factory.signals.spec import SignalSpec


def _batches(specs: list[SignalSpec], batch_size: int) -> list[list[SignalSpec]]:
    return [specs[i : i + batch_size] for i in range(0, len(specs), batch_size)]


def compute_row_b(history: Path, out_path: Path, batch_size: int = 150) -> int:
    """Write ``TransactionID`` plus every grid column (float32) to ``out_path``; return the column count."""
    grid = routine_grid()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out_path.parent) as tmp:
        parts: list[Path] = []
        for i, batch in enumerate(_batches(grid, batch_size)):
            frame = compute_signals(history, batch, batch_size=batch_size, dtype=np.float32)
            part = Path(tmp) / f"part_{i:04d}.parquet"
            frame.to_parquet(part, index=False)
            parts.append(part)
        _join_parts(parts, out_path)
    return len(grid)


def _join_parts(parts: list[Path], out_path: Path) -> None:
    con = duckdb.connect()
    try:
        for i, part in enumerate(parts):
            con.execute(f"CREATE VIEW p{i} AS SELECT * FROM read_parquet('{part.as_posix()}')")
        joins = " ".join(
            f"JOIN p{i} USING ({quote_identifier(ID_COLUMN)})" for i in range(1, len(parts))
        )
        query = f"SELECT * FROM p0 {joins} ORDER BY {quote_identifier(ID_COLUMN)}"
        con.execute(f"COPY ({query}) TO '{out_path.as_posix()}' (FORMAT PARQUET)")
    finally:
        con.close()
