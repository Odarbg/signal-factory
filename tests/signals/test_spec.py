from typing import Any

import pytest
from pydantic import ValidationError

from signal_factory.signals.columns import ENTITY_KEYS, FORBIDDEN_COLUMNS
from signal_factory.signals.spec import (
    SignalSpec,
    Window,
    WindowDays,
    WindowHours,
    signal_json_schema,
    window_seconds,
)

VALID_COUNT: dict[str, Any] = {
    "id": "uid_txn_count_24h",
    "version": 1,
    "hypothesis": "Many transactions by one pseudo-identity within a day suggests testing.",
    "entity_key": "uid",
    "column": "*",
    "aggregation": "count",
    "window": {"hours": 24},
    "inputs": [{"column": "uid", "available_at": "decision_time"}],
}

VALID_SUM: dict[str, Any] = {
    "id": "card1_amt_sum_7d",
    "version": 1,
    "hypothesis": "Spend concentrated on one issuer bank id over a week.",
    "entity_key": "card1",
    "column": "TransactionAmt",
    "aggregation": "sum",
    "window": {"days": 7},
    "inputs": [
        {"column": "card1", "available_at": "decision_time"},
        {"column": "TransactionAmt", "available_at": "decision_time"},
    ],
}


def test_entity_keys_and_forbidden_columns_are_pinned() -> None:
    assert ENTITY_KEYS == frozenset(
        {
            "uid",
            "card1",
            "card1_addr1",
            "P_emaildomain",
            "R_emaildomain",
            "DeviceInfo",
            "id_30",
            "id_31",
            "addr1",
        }
    )
    assert FORBIDDEN_COLUMNS == frozenset(
        {"isFraud", "TransactionID", "TransactionDT", "uid", "day", "card1_addr1"}
    )


def test_valid_count_spec_parses() -> None:
    spec = SignalSpec.model_validate(VALID_COUNT)
    assert spec.aggregation == "count"
    assert spec.missing == "null"
    assert spec.filter is None


def test_valid_sum_spec_parses() -> None:
    spec = SignalSpec.model_validate(VALID_SUM)
    assert isinstance(spec.window, WindowDays)
    assert spec.window.seconds() == 7 * 86400


def test_window_variants() -> None:
    assert SignalSpec.model_validate({**VALID_COUNT, "window": "all"}).window == "all"
    hours: Window = WindowHours(hours=3)
    assert window_seconds(hours) == 3 * 3600
    assert window_seconds("all") is None
    with pytest.raises(ValidationError):
        SignalSpec.model_validate({**VALID_COUNT, "window": {"hours": 0}})
    with pytest.raises(ValidationError):
        SignalSpec.model_validate({**VALID_COUNT, "window": {"days": -1}})
    with pytest.raises(ValidationError):
        SignalSpec.model_validate({**VALID_COUNT, "window": {"hours": 1, "days": 1}})


def test_label_is_never_a_column() -> None:
    with pytest.raises(ValidationError, match="isFraud"):
        SignalSpec.model_validate(
            {
                **VALID_SUM,
                "column": "isFraud",
                "inputs": [
                    {"column": "card1", "available_at": "decision_time"},
                    {"column": "isFraud", "available_at": "decision_time"},
                ],
            }
        )


def test_label_is_never_a_filter_column() -> None:
    with pytest.raises(ValidationError, match="isFraud"):
        SignalSpec.model_validate(
            {
                **VALID_COUNT,
                "filter": {"column": "isFraud", "op": "==", "value": 1},
                "inputs": [
                    {"column": "uid", "available_at": "decision_time"},
                    {"column": "isFraud", "available_at": "decision_time"},
                ],
            }
        )


@pytest.mark.parametrize("column", sorted(FORBIDDEN_COLUMNS))
def test_every_forbidden_column_is_rejected(column: str) -> None:
    with pytest.raises(ValidationError, match=column):
        SignalSpec.model_validate(
            {
                **VALID_SUM,
                "column": column,
                "inputs": [
                    {"column": "card1", "available_at": "decision_time"},
                    {"column": column, "available_at": "decision_time"},
                ],
            }
        )


def test_unknown_entity_key_rejected() -> None:
    with pytest.raises(ValidationError, match="entity_key"):
        SignalSpec.model_validate(
            {
                **VALID_COUNT,
                "entity_key": "card2",
                "inputs": [{"column": "card2", "available_at": "decision_time"}],
            }
        )


def test_count_requires_star_and_star_requires_count() -> None:
    with pytest.raises(ValidationError, match=r"\*"):
        SignalSpec.model_validate(
            {**VALID_COUNT, "column": "TransactionAmt", "inputs": VALID_SUM["inputs"]}
        )
    with pytest.raises(ValidationError, match=r"\*"):
        SignalSpec.model_validate(
            {
                **VALID_SUM,
                "column": "*",
                "inputs": [{"column": "card1", "available_at": "decision_time"}],
            }
        )


def test_time_since_last_requires_star() -> None:
    SignalSpec.model_validate({**VALID_COUNT, "aggregation": "time_since_last"})
    with pytest.raises(ValidationError, match=r"\*"):
        SignalSpec.model_validate({**VALID_SUM, "aggregation": "time_since_last"})


def test_inputs_must_cover_used_columns() -> None:
    with pytest.raises(ValidationError, match="inputs"):
        SignalSpec.model_validate({**VALID_SUM, "inputs": [VALID_SUM["inputs"][0]]})
    with pytest.raises(ValidationError, match="inputs"):
        SignalSpec.model_validate({**VALID_SUM, "inputs": [VALID_SUM["inputs"][1]]})
    with pytest.raises(ValidationError, match="inputs"):
        SignalSpec.model_validate(
            {**VALID_COUNT, "filter": {"column": "ProductCD", "op": "==", "value": "W"}}
        )
    SignalSpec.model_validate(
        {
            **VALID_COUNT,
            "filter": {"column": "ProductCD", "op": "==", "value": "W"},
            "inputs": [
                *VALID_COUNT["inputs"],
                {"column": "ProductCD", "available_at": "decision_time"},
            ],
        }
    )


def test_inputs_only_allow_decision_time() -> None:
    with pytest.raises(ValidationError, match="decision_time"):
        SignalSpec.model_validate(
            {**VALID_COUNT, "inputs": [{"column": "uid", "available_at": "label_time"}]}
        )


def test_extra_fields_are_forbidden() -> None:
    with pytest.raises(ValidationError):
        SignalSpec.model_validate({**VALID_COUNT, "depth": 2})


@pytest.mark.parametrize("bad_id", ["Ab", "1abc", "ab", "a b", "a-b", "x" * 65])
def test_id_pattern(bad_id: str) -> None:
    with pytest.raises(ValidationError, match="id"):
        SignalSpec.model_validate({**VALID_COUNT, "id": bad_id})


def test_id_allows_mixed_case_after_first_char() -> None:
    SignalSpec.model_validate({**VALID_COUNT, "id": "rb__P_emaildomain__all__count__star"})


def test_hypothesis_length_bounds() -> None:
    with pytest.raises(ValidationError, match="hypothesis"):
        SignalSpec.model_validate({**VALID_COUNT, "hypothesis": "too short"})
    with pytest.raises(ValidationError, match="hypothesis"):
        SignalSpec.model_validate({**VALID_COUNT, "hypothesis": "x" * 401})


def test_validate_columns_names_unknown() -> None:
    spec = SignalSpec.model_validate(
        {
            **VALID_SUM,
            "column": "nope",
            "inputs": [
                {"column": "card1", "available_at": "decision_time"},
                {"column": "nope", "available_at": "decision_time"},
            ],
        }
    )
    with pytest.raises(ValueError, match="nope"):
        spec.validate_columns(["TransactionAmt", "card1"])
    SignalSpec.model_validate(VALID_SUM).validate_columns(["TransactionAmt", "card1"])


def test_validate_columns_checks_filter_column() -> None:
    spec = SignalSpec.model_validate(
        {
            **VALID_COUNT,
            "filter": {"column": "ProductCD", "op": "==", "value": "W"},
            "inputs": [
                *VALID_COUNT["inputs"],
                {"column": "ProductCD", "available_at": "decision_time"},
            ],
        }
    )
    with pytest.raises(ValueError, match="ProductCD"):
        spec.validate_columns(["uid", "TransactionAmt"])


def test_json_schema_has_required_fields() -> None:
    schema = signal_json_schema()
    assert set(schema["required"]) >= {
        "id",
        "version",
        "hypothesis",
        "entity_key",
        "column",
        "aggregation",
        "window",
        "inputs",
    }
    assert "isFraud" not in str(schema)
