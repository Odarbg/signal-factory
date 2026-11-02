from typing import Any

import duckdb
import pandas as pd
import pytest

from signal_factory.signals.compile import build_query, compile_signal
from signal_factory.signals.spec import SignalSpec

BASE: dict[str, Any] = {
    "id": "sig",
    "version": 1,
    "hypothesis": "A fixture signal used to pin the compiler's semantics.",
    "entity_key": "uid",
    "column": "*",
    "aggregation": "count",
    "window": "all",
    "inputs": [{"column": "uid", "available_at": "decision_time"}],
}


def spec(**overrides: Any) -> SignalSpec:
    data = {**BASE, **overrides}
    used = {data["entity_key"]}
    if data["column"] != "*":
        used.add(data["column"])
    if data.get("filter"):
        used.add(data["filter"]["column"])
    data["inputs"] = [{"column": c, "available_at": "decision_time"} for c in sorted(used)]
    return SignalSpec.model_validate(data)


def values(con: duckdb.DuckDBPyConnection, s: SignalSpec) -> dict[int, float | None]:
    df = con.execute(build_query([s])).df().sort_values("TransactionID")
    out: dict[int, float | None] = {}
    for tid, v in zip(df["TransactionID"], df[s.id], strict=True):
        out[int(tid)] = None if pd.isna(v) else float(v)
    return out


def test_count_all_history_excludes_self_and_ties(con: duckdb.DuckDBPyConnection) -> None:
    # Review Focus 2: rows 2/3 and 7/8 share a timestamp and must not see each other.
    got = values(con, spec())
    assert {k: got[k] for k in range(1, 10)} == {
        1: 0, 2: 1, 3: 1, 4: 3, 5: 4,
        6: 0, 7: 1, 8: 1, 9: 3,
    }


def test_sum_window_24h_is_half_open(con: duckdb.DuckDBPyConnection) -> None:
    # Window is [t - 86400, t): row 4 (87400) still sees row 1 (1000); row 5 (87401) does not.
    got = values(con, spec(column="TransactionAmt", aggregation="sum", window={"hours": 24}))
    assert got[4] == 60.0  # 10 + 20 + 30
    assert got[5] == 90.0  # 20 + 30 + 40
    assert got[2] == 10.0 and got[3] == 10.0
    assert got[9] is None  # nothing within 24h of 100000 → NULL


def test_count_distinct_with_filter(con: duckdb.DuckDBPyConnection) -> None:
    s = spec(
        column="P_emaildomain",
        aggregation="count_distinct",
        filter={"column": "ProductCD", "op": "==", "value": "W"},
    )
    got = values(con, s)
    assert {k: got[k] for k in range(1, 10)} == {
        1: 0, 2: 1, 3: 1, 4: 2, 5: 2,
        6: 0, 7: 1, 8: 1, 9: 1,
    }


def test_numeric_filter_literal(con: duckdb.DuckDBPyConnection) -> None:
    s = spec(aggregation="count", filter={"column": "TransactionAmt", "op": ">=", "value": 30})
    got = values(con, s)
    assert got[5] == 2  # rows 3 (30) and 4 (40)
    assert got[4] == 1  # row 3


def test_mean_max_min(con: duckdb.DuckDBPyConnection) -> None:
    assert values(con, spec(column="TransactionAmt", aggregation="mean"))[4] == 20.0
    assert values(con, spec(column="TransactionAmt", aggregation="max"))[5] == 40.0
    assert values(con, spec(column="TransactionAmt", aggregation="min"))[9] == 60.0


def test_time_since_last_is_null_when_no_history(con: duckdb.DuckDBPyConnection) -> None:
    got = values(con, spec(aggregation="time_since_last"))
    assert got[1] is None and got[6] is None
    assert got[2] == 1000 and got[3] == 1000
    assert got[4] == 85400 and got[5] == 1
    assert got[9] == 98500


def test_time_since_last_respects_window(con: duckdb.DuckDBPyConnection) -> None:
    got = values(con, spec(aggregation="time_since_last", window={"hours": 1}))
    assert got[4] is None  # last event 85400s earlier, outside 1h
    assert got[5] == 1


def test_count_is_zero_when_no_history(con: duckdb.DuckDBPyConnection) -> None:
    assert values(con, spec())[1] == 0
    assert values(con, spec(window={"hours": 24}))[9] == 0


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"aggregation": "time_since_last"},
        {"column": "TransactionAmt", "aggregation": "sum"},
        {"column": "ProductCD", "aggregation": "count_distinct"},
    ],
)
def test_null_key_yields_null_not_group(
    con: duckdb.DuckDBPyConnection, overrides: dict[str, Any]
) -> None:
    # Review Focus 1: rows 10–12 have a NULL uid and must not form one shared group.
    got = values(con, spec(**overrides))
    assert got[10] is None and got[11] is None and got[12] is None


def test_build_query_names_columns_by_spec_id(con: duckdb.DuckDBPyConnection) -> None:
    a = spec(id="first_sig")
    b = spec(id="second_sig", column="TransactionAmt", aggregation="sum")
    df = con.execute(build_query([a, b])).df()
    assert list(df.columns) == ["TransactionID", "first_sig", "second_sig"]
    assert len(df) == 12


def test_build_query_uses_table_name() -> None:
    q = build_query([spec()], table="other_table")
    assert "FROM other_table h" in q


def test_string_literal_is_escaped() -> None:
    s = spec(filter={"column": "ProductCD", "op": "==", "value": "O'Brien"})
    assert "'O''Brien'" in compile_signal(s)


def test_compile_is_deterministic() -> None:
    s = spec(column="TransactionAmt", aggregation="sum", window={"days": 7})
    assert compile_signal(s) == compile_signal(s)
    assert "SELECT" not in compile_signal(s).upper().split("CASE")[0]


def test_filter_only_aggregates_filtered_rows_for_count(con: duckdb.DuckDBPyConnection) -> None:
    s = spec(filter={"column": "ProductCD", "op": "!=", "value": "W"})
    got = values(con, s)
    assert got[5] == 1  # only row 2 is C among rows 1–4
    assert got[1] == 0
