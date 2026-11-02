"""The fixed row-B grid (spec §7). Pre-registered; not tuned.

keys {uid, card1_addr1, P_emaildomain, DeviceInfo} × windows {all, 24h, 7d} ×
{ count of *; time_since_last; sum/mean/max/min of TransactionAmt, dist1, C1–C14, D1–D15;
  count_distinct of ProductCD, card2–card6, addr1, P_emaildomain, R_emaildomain, DeviceInfo,
  excluding the grouping key itself }.

Ids follow ``rb__<key>__<window>__<aggregation>__<column>`` with ``star`` for ``*``.
"""

from __future__ import annotations

from signal_factory.signals.spec import (
    Aggregation,
    SignalInput,
    SignalSpec,
    Window,
    WindowDays,
    WindowHours,
)

GRID_KEYS: tuple[str, ...] = ("uid", "card1_addr1", "P_emaildomain", "DeviceInfo")
GRID_WINDOWS: tuple[Window, ...] = ("all", WindowHours(hours=24), WindowDays(days=7))
GRID_WINDOW_NAMES: dict[str, Window] = {
    "all": "all",
    "24h": WindowHours(hours=24),
    "7d": WindowDays(days=7),
}
GRID_NUMERIC_COLUMNS: tuple[str, ...] = (
    "TransactionAmt",
    "dist1",
    *[f"C{i}" for i in range(1, 15)],
    *[f"D{i}" for i in range(1, 16)],
)
GRID_NUMERIC_AGGREGATIONS: tuple[Aggregation, ...] = ("sum", "mean", "max", "min")
GRID_COUNT_DISTINCT_COLUMNS: tuple[str, ...] = (
    "ProductCD",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceInfo",
)
ROUTINE_HYPOTHESIS = "routine aggregation (row B grid)"


def grid_id(key: str, window: str, aggregation: str, column: str) -> str:
    if window not in GRID_WINDOW_NAMES:
        raise ValueError(f"window {window!r} is not in the grid: {sorted(GRID_WINDOW_NAMES)}")
    return f"rb__{key}__{window}__{aggregation}__{column}"


def _spec(key: str, window_name: str, aggregation: Aggregation, column: str) -> SignalSpec:
    used = {key} | ({column} if column != "*" else set())
    return SignalSpec(
        id=grid_id(key, window_name, aggregation, "star" if column == "*" else column),
        version=1,
        hypothesis=ROUTINE_HYPOTHESIS,
        entity_key=key,
        column=column,
        aggregation=aggregation,
        window=GRID_WINDOW_NAMES[window_name],
        inputs=[SignalInput(column=c, available_at="decision_time") for c in sorted(used)],
    )


def routine_grid() -> list[SignalSpec]:
    """Every row-B signal, in a fixed order: window, then key, then aggregation, then column."""
    specs: list[SignalSpec] = []
    for window_name in GRID_WINDOW_NAMES:
        for key in GRID_KEYS:
            specs.append(_spec(key, window_name, "count", "*"))
            specs.append(_spec(key, window_name, "time_since_last", "*"))
            for aggregation in GRID_NUMERIC_AGGREGATIONS:
                for column in GRID_NUMERIC_COLUMNS:
                    specs.append(_spec(key, window_name, aggregation, column))
            for column in GRID_COUNT_DISTINCT_COLUMNS:
                if column != key:
                    specs.append(_spec(key, window_name, "count_distinct", column))
    return specs
