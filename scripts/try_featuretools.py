"""Time-boxed featuretools attempt at the row-B grid (spec §7, §8.4, §10).

Builds an EntitySet with ``TransactionDT`` as the time index and a cutoff time per row
one second before its own ``TransactionDT`` (strict past, ties excluded), runs DFS with
the grid's primitives and training windows, and compares a sample of cells against
``compute_row_b``. Prints wall time; exits non-zero on mismatch or when the 8-hour time
box is exceeded. Either outcome is acceptable and is recorded in
``docs/decisions/row-b-backend.md``.

    uv sync --group routine-ft
    uv run python scripts/try_featuretools.py data/split/history.parquet --sample 2000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from signal_factory.routine.grid import (
    GRID_COUNT_DISTINCT_COLUMNS,
    GRID_KEYS,
    GRID_NUMERIC_COLUMNS,
)
from signal_factory.signals.engine import compute_signals
from signal_factory.signals.spec import (
    Aggregation,
    SignalInput,
    SignalSpec,
    Window,
    WindowDays,
    WindowHours,
)

TIME_BOX_SECONDS = 8 * 3600
WINDOWS: dict[str, pd.Timedelta | None] = {
    "all": None,
    "24h": pd.Timedelta(hours=24),
    "7d": pd.Timedelta(days=7),
}
PRIMITIVE_FOR = {
    "count": "count",
    "time_since_last": "time_since_last",
    "sum": "sum",
    "mean": "mean",
    "max": "max",
    "min": "min",
    "count_distinct": "num_unique",
}


def _probe_specs() -> list[SignalSpec]:
    """A handful of grid cells spanning every primitive and window, for the comparison."""
    specs: list[SignalSpec] = []
    windows: list[tuple[str, Window]] = [
        ("all", "all"),
        ("24h", WindowHours(hours=24)),
        ("7d", WindowDays(days=7)),
    ]
    probes: list[tuple[Aggregation, str]] = [
        ("count", "*"),
        ("time_since_last", "*"),
        ("mean", GRID_NUMERIC_COLUMNS[0]),
        ("count_distinct", GRID_COUNT_DISTINCT_COLUMNS[0]),
    ]
    for window_name, window in windows:
        for key in GRID_KEYS[:2]:
            for aggregation, column in probes:
                used = {key} | ({column} if column != "*" else set())
                specs.append(
                    SignalSpec(
                        id=f"ft__{key}__{window_name}__{aggregation}__{'star' if column == '*' else column}",
                        version=1,
                        hypothesis="featuretools comparison probe",
                        entity_key=key,
                        column=column,
                        aggregation=aggregation,
                        window=window,
                        inputs=[
                            SignalInput(column=c, available_at="decision_time")
                            for c in sorted(used)
                        ],
                    )
                )
    return specs


def run_dfs(
    history: pd.DataFrame, key: str, window: pd.Timedelta | None, cutoffs: pd.DataFrame
) -> pd.DataFrame:
    import featuretools as ft

    rows = history.loc[
        history[key].notna(),
        [
            "TransactionID",
            "TransactionDT_ts",
            key,
            GRID_NUMERIC_COLUMNS[0],
            GRID_COUNT_DISTINCT_COLUMNS[0],
        ],
    ]
    es = ft.EntitySet(id="txn")
    es = es.add_dataframe(
        dataframe_name="transactions",
        dataframe=rows,
        index="TransactionID",
        time_index="TransactionDT_ts",
    )
    es = es.normalize_dataframe(
        base_dataframe_name="transactions", new_dataframe_name=key, index=key
    )
    matrix, _ = ft.dfs(
        entityset=es,
        target_dataframe_name="transactions",
        agg_primitives=["count", "time_since_last", "mean", "num_unique"],
        trans_primitives=[],
        cutoff_time=cutoffs[cutoffs["TransactionID"].isin(rows["TransactionID"])],
        training_window=window,
        max_depth=2,
        verbose=False,
    )
    return pd.DataFrame(matrix)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history_parquet", type=Path)
    parser.add_argument("--sample", type=int, default=2000)
    args = parser.parse_args()

    start = time.perf_counter()
    history = pd.read_parquet(args.history_parquet)
    history["TransactionDT_ts"] = pd.Timestamp("2017-12-01") + pd.to_timedelta(
        history["TransactionDT"], unit="s"
    )
    sample_ids = np.sort(
        np.random.default_rng(0).choice(
            history["TransactionID"].to_numpy(), args.sample, replace=False
        )
    )
    cutoffs = pd.DataFrame(
        {
            "TransactionID": history["TransactionID"],
            "time": history["TransactionDT_ts"] - pd.Timedelta(seconds=1),  # strict past
        }
    )
    cutoffs = cutoffs[cutoffs["TransactionID"].isin(sample_ids)]

    engine = compute_signals(history, _probe_specs()).set_index("TransactionID").loc[sample_ids]
    mismatches: list[str] = []
    for window_name, window in WINDOWS.items():
        for key in GRID_KEYS[:2]:
            if time.perf_counter() - start > TIME_BOX_SECONDS:
                print("TIMED OUT: exceeded the 8-hour time box")
                return 2
            matrix = run_dfs(history, key, window, cutoffs)
            print(
                f"{key} {window_name}: featuretools matrix {matrix.shape} after {time.perf_counter() - start:.0f}s"
            )
            ft_count = matrix[f"{key}.COUNT(transactions)"]
            eng_count = engine[f"ft__{key}__{window_name}__count__star"].reindex(ft_count.index)
            # featuretools counts include the row's own group members before the cutoff; engine excludes NULL keys
            diff: Any = (ft_count.fillna(0) - eng_count.fillna(0)).abs()
            if (diff > 0).any():
                mismatches.append(f"{key}/{window_name}: {(diff > 0).sum()} count cells differ")

    elapsed = time.perf_counter() - start
    print(f"wall time {elapsed:.0f}s; mismatches: {mismatches or 'none'}")
    if mismatches:
        print("LEAKAGE/SEMANTICS MISMATCH")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
