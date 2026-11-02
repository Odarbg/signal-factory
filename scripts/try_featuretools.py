"""Time-boxed featuretools attempt at the row-B grid (spec §7, §8.4, §10).

What this script can and cannot show. It builds an EntitySet with ``TransactionDT`` as
the time index and a cutoff time per row one second before its own ``TransactionDT``
(strict past, ties excluded), runs DFS for a **probe** of the grid (2 keys × 3 windows ×
4 primitives) on a sampled cutoff set, compares every probe cell with the signal engine,
and prints wall time. It does not compute the 1,626-column grid; a "finished" verdict is
an extrapolation from the probe's wall time and is recorded as such in
``docs/decisions/row-b-backend.md``. Exit codes: 0 match, 1 mismatch, 2 time box
exceeded, 3 featuretools/woodwork unusable in this environment.

Finding 2026-11-02: with pandas 3.1.0, woodwork 0.31.0 does not retain its schema after
``DataFrame.ww.init`` (``df.ww.schema`` is ``None``), so featuretools 1.31.0 cannot build
an EntitySet here. The preflight below detects that and exits 3.

    uv sync --all-groups          # includes the routine-ft group
    uv run python scripts/try_featuretools.py data/split/history.parquet --sample 2000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

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


def woodwork_preflight() -> str | None:
    """Return a reason featuretools cannot run here, or ``None`` when it can."""
    try:
        import featuretools as ft
        import woodwork as ww
    except ImportError as exc:  # pragma: no cover - depends on the optional group
        return f"import failed: {exc} (run `uv sync --all-groups`)"
    probe = pd.DataFrame({"id": [1, 2], "x": [1.0, 2.0]})
    probe.ww.init(index="id")
    if probe.ww.schema is None:
        return (
            f"woodwork {ww.__version__} does not retain a schema under pandas {pd.__version__}; "
            f"featuretools {ft.__version__} cannot build an EntitySet"
        )
    return None


FT_COLUMN = {
    "count": "{key}.COUNT(transactions)",
    "time_since_last": "{key}.TIME_SINCE_LAST(transactions.TransactionDT_ts)",
    "mean": "{key}.MEAN(transactions.{numeric})",
    "count_distinct": "{key}.NUM_UNIQUE(transactions.{nunique})",
}


def _compare(matrix: pd.DataFrame, engine: pd.DataFrame, key: str, window_name: str) -> list[str]:
    """One line per probe primitive whose featuretools cells differ from the engine's."""
    out: list[str] = []
    for aggregation, template in FT_COLUMN.items():
        ft_name = template.format(
            key=key, numeric=GRID_NUMERIC_COLUMNS[0], nunique=GRID_COUNT_DISTINCT_COLUMNS[0]
        )
        column = (
            "star"
            if aggregation in ("count", "time_since_last")
            else (
                GRID_NUMERIC_COLUMNS[0] if aggregation == "mean" else GRID_COUNT_DISTINCT_COLUMNS[0]
            )
        )
        eng_name = f"ft__{key}__{window_name}__{aggregation}__{column}"
        if ft_name not in matrix.columns:
            out.append(f"{key}/{window_name}/{aggregation}: featuretools produced no {ft_name}")
            continue
        ft_values = pd.to_numeric(matrix[ft_name], errors="coerce")
        eng_values = engine[eng_name].reindex(ft_values.index)
        if aggregation == "count":
            ft_values, eng_values = ft_values.fillna(0), eng_values.fillna(0)
        both_null = ft_values.isna() & eng_values.isna()
        differ = ~both_null & ~np.isclose(ft_values.fillna(-1), eng_values.fillna(-1), atol=1e-4)
        if differ.any():
            out.append(f"{key}/{window_name}/{aggregation}: {int(differ.sum())} cells differ")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history_parquet", type=Path)
    parser.add_argument("--sample", type=int, default=2000)
    args = parser.parse_args()

    reason = woodwork_preflight()
    if reason is not None:
        print(f"FEATURETOOLS UNAVAILABLE: {reason}")
        return 3

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
            mismatches.extend(_compare(matrix, engine, key, window_name))

    elapsed = time.perf_counter() - start
    print(f"wall time {elapsed:.0f}s; mismatches: {mismatches or 'none'}")
    if mismatches:
        print("LEAKAGE/SEMANTICS MISMATCH")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
