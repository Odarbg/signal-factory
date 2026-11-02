"""Compile a :class:`SignalSpec` to one DuckDB window expression (spec §8.2).

Semantics, all pinned by tests:

- Strict past. The frame is ``RANGE BETWEEN <n> PRECEDING AND 1 PRECEDING`` over an integer
  ``TransactionDT``, so the current row and every same-timestamp row are excluded.
- Half-open window ``[t - n, t)``: a row exactly ``n`` seconds earlier is inside.
- Filter: aggregate over ``CASE WHEN <filter> THEN <column> END`` so unfiltered rows
  contribute NULL and are ignored by the aggregate.
- ``count`` yields 0 on empty history; every other aggregation yields NULL.
- NULL entity key yields NULL: NULL keys never form a shared partition.
"""

from __future__ import annotations

from collections.abc import Sequence

from signal_factory.signals.spec import Filter, SignalSpec, window_seconds

TIME_COLUMN = "TransactionDT"
ID_COLUMN = "TransactionID"
TABLE_ALIAS = "h"

_SIMPLE_AGGREGATES = {"sum": "sum", "mean": "avg", "max": "max", "min": "min"}


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def quote_literal(value: float | str) -> str:
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return repr(float(value))


def _column(name: str) -> str:
    return f"{TABLE_ALIAS}.{quote_identifier(name)}"


def _frame(spec: SignalSpec) -> str:
    seconds = window_seconds(spec.window)
    lower = "UNBOUNDED PRECEDING" if seconds is None else f"{seconds} PRECEDING"
    return (
        f"OVER (PARTITION BY {_column(spec.entity_key)} ORDER BY {_column(TIME_COLUMN)} "
        f"RANGE BETWEEN {lower} AND 1 PRECEDING)"
    )


def _predicate(filter_: Filter) -> str:
    op = "=" if filter_.op == "==" else filter_.op
    return f"{_column(filter_.column)} {op} {quote_literal(filter_.value)}"


def _argument(spec: SignalSpec) -> str:
    """The aggregated expression: the column, ``1`` for count, the timestamp for recency."""
    if spec.aggregation == "count":
        inner = "1"
    elif spec.aggregation == "time_since_last":
        inner = _column(TIME_COLUMN)
    else:
        inner = _column(spec.column)
    if spec.filter is None:
        return inner
    return f"CASE WHEN {_predicate(spec.filter)} THEN {inner} END"


def _aggregate(spec: SignalSpec) -> str:
    arg = _argument(spec)
    frame = _frame(spec)
    if spec.aggregation == "count":
        return f"count({arg}) {frame}"
    if spec.aggregation == "count_distinct":
        return f"count(DISTINCT {arg}) {frame}"
    if spec.aggregation == "time_since_last":
        return f"({_column(TIME_COLUMN)} - max({arg}) {frame})"
    return f"{_SIMPLE_AGGREGATES[spec.aggregation]}({arg}) {frame}"


def compile_signal(spec: SignalSpec) -> str:
    """A select-list expression (no SELECT/FROM) for the signal on table alias ``h``."""
    return f"CASE WHEN {_column(spec.entity_key)} IS NULL THEN NULL ELSE {_aggregate(spec)} END"


def build_query(specs: Sequence[SignalSpec], table: str = "history") -> str:
    """``SELECT h.TransactionID, <expr_1> AS <id_1>, … FROM <table> h``."""
    parts = [f"{_column(ID_COLUMN)} AS {quote_identifier(ID_COLUMN)}"]
    parts.extend(f"{compile_signal(s)} AS {quote_identifier(s.id)}" for s in specs)
    return f"SELECT {', '.join(parts)} FROM {table} {TABLE_ALIAS}"
