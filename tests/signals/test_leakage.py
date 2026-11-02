"""Acceptance gate (spec §8.2): appending later rows never changes earlier signal values."""

from typing import Any

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from signal_factory.signals.engine import compute_signals
from signal_factory.signals.spec import SignalSpec

UIDS = ["a", "b", "c", None]
PRODUCTS = ["W", "C"]


def _frame(rows: list[tuple[str | None, int, float, str]], first_id: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": list(range(first_id, first_id + len(rows))),
            "uid": pd.Series([r[0] for r in rows], dtype=pd.StringDtype("pyarrow")),
            "TransactionDT": [r[1] for r in rows],
            "TransactionAmt": pd.Series([r[2] for r in rows], dtype="float32"),
            "ProductCD": pd.Series([r[3] for r in rows], dtype=pd.StringDtype("pyarrow")),
        }
    )


row = st.tuples(
    st.sampled_from(UIDS),
    st.integers(min_value=0, max_value=200_000),
    st.floats(min_value=0.0, max_value=5000.0, allow_nan=False, width=32),
    st.sampled_from(PRODUCTS),
)


@st.composite
def histories(draw: st.DrawFn) -> pd.DataFrame:
    rows = draw(st.lists(row, min_size=1, max_size=30))
    return _frame(rows, first_id=1)


@st.composite
def later_rows(draw: st.DrawFn, history: pd.DataFrame) -> pd.DataFrame:
    start = int(history["TransactionDT"].max()) + 1
    rows = draw(
        st.lists(
            st.tuples(
                st.sampled_from(UIDS),
                st.integers(min_value=start, max_value=start + 200_000),
                st.floats(min_value=0.0, max_value=5000.0, allow_nan=False, width=32),
                st.sampled_from(PRODUCTS),
            ),
            min_size=1,
            max_size=10,
        )
    )
    return _frame(rows, first_id=len(history) + 1)


@st.composite
def signal_specs(draw: st.DrawFn) -> SignalSpec:
    aggregation = draw(
        st.sampled_from(["count", "count_distinct", "sum", "mean", "max", "min", "time_since_last"])
    )
    if aggregation in {"count", "time_since_last"}:
        column = "*"
    elif aggregation == "count_distinct":
        column = draw(st.sampled_from(["ProductCD", "TransactionAmt"]))
    else:
        column = "TransactionAmt"
    window: Any = draw(
        st.one_of(
            st.just("all"),
            st.integers(min_value=1, max_value=100).map(lambda h: {"hours": h}),
            st.integers(min_value=1, max_value=3).map(lambda d: {"days": d}),
        )
    )
    filter_: dict[str, Any] | None = draw(
        st.one_of(
            st.none(),
            st.sampled_from(PRODUCTS).map(
                lambda p: {"column": "ProductCD", "op": "==", "value": p}
            ),
            st.floats(min_value=0.0, max_value=5000.0, allow_nan=False).map(
                lambda a: {"column": "TransactionAmt", "op": ">=", "value": a}
            ),
        )
    )
    used = {"uid"} | ({column} if column != "*" else set())
    if filter_ is not None:
        used.add(filter_["column"])
    return SignalSpec.model_validate(
        {
            "id": "leak_probe",
            "version": 1,
            "hypothesis": "Random signal drawn by Hypothesis for the leakage gate.",
            "entity_key": "uid",
            "column": column,
            "aggregation": aggregation,
            "window": window,
            "filter": filter_,
            "inputs": [{"column": c, "available_at": "decision_time"} for c in sorted(used)],
        }
    )


@st.composite
def history_and_later(draw: st.DrawFn) -> tuple[pd.DataFrame, pd.DataFrame]:
    history = draw(histories())
    return history, draw(later_rows(history))


@settings(max_examples=200, deadline=None)
@given(data=history_and_later(), spec=signal_specs())
def test_appending_later_rows_never_changes_earlier_values(
    data: tuple[pd.DataFrame, pd.DataFrame], spec: SignalSpec
) -> None:
    history, later = data
    before = compute_signals(history, [spec]).set_index("TransactionID")[spec.id]
    combined = pd.concat([history, later], ignore_index=True)
    after = compute_signals(combined, [spec]).set_index("TransactionID")[spec.id]
    pd.testing.assert_series_equal(after.loc[before.index], before, check_names=False)
