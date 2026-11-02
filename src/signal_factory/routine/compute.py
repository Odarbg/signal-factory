"""Compute the row-B grid over ``history.parquet`` with the signal engine (spec §7 fallback path)."""

from __future__ import annotations

from pathlib import Path

from signal_factory.routine.grid import routine_grid
from signal_factory.signals.engine import compute_signals


def compute_row_b(history: Path, out_path: Path, batch_size: int = 150) -> int:
    """Write ``TransactionID`` plus every grid column to ``out_path``; return the column count."""
    grid = routine_grid()
    frame = compute_signals(history, grid, batch_size=batch_size)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, index=False)
    return len(grid)
