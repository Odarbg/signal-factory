"""Evaluate signal specs against a history table with DuckDB (spec §8.2).

``compute_signals`` is the one entry point every row of the ablation uses. It batches
specs into queries of at most ``batch_size`` window expressions, joins the batches on
``TransactionID`` and returns one float64 column per spec, in spec order.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import duckdb
import pandas as pd

from signal_factory.signals.compile import ID_COLUMN, build_query
from signal_factory.signals.spec import SignalSpec

HISTORY_TABLE = "history"


def connect(history: Path | pd.DataFrame) -> duckdb.DuckDBPyConnection:
    """An in-memory connection with ``history`` registered as a table or Parquet view."""
    con = duckdb.connect()
    if isinstance(history, pd.DataFrame):
        con.register(HISTORY_TABLE, history)
    else:
        path = str(history).replace("'", "''")
        con.execute(f"CREATE VIEW {HISTORY_TABLE} AS SELECT * FROM read_parquet('{path}')")
    return con


def _batches(specs: Sequence[SignalSpec], batch_size: int) -> list[Sequence[SignalSpec]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    return [specs[i : i + batch_size] for i in range(0, len(specs), batch_size)]


def compute_signals(
    history: Path | pd.DataFrame,
    specs: Sequence[SignalSpec],
    batch_size: int = 200,
) -> pd.DataFrame:
    """``TransactionID`` plus one float64 column per spec id, sorted by ``TransactionID``."""
    con = connect(history)
    try:
        ids = con.execute(f"SELECT {ID_COLUMN} FROM {HISTORY_TABLE} ORDER BY {ID_COLUMN}").df()
        index = ids.set_index(ID_COLUMN).index
        parts: list[pd.DataFrame] = []
        for batch in _batches(specs, batch_size):
            part = con.execute(build_query(batch, HISTORY_TABLE)).df().set_index(ID_COLUMN)
            part = part.reindex(index)
            parts.append(
                pd.DataFrame(
                    {spec.id: pd.to_numeric(part[spec.id]).astype("float64") for spec in batch},
                    index=index,
                )
            )
    finally:
        con.close()
    out = pd.concat([ids.set_index(ID_COLUMN), *parts], axis=1)
    return out.reset_index()
