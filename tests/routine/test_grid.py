import pytest

from signal_factory.routine.grid import (
    GRID_COUNT_DISTINCT_COLUMNS,
    GRID_KEYS,
    GRID_NUMERIC_COLUMNS,
    GRID_WINDOWS,
    ROUTINE_HYPOTHESIS,
    grid_id,
    routine_grid,
)
from signal_factory.signals.columns import FORBIDDEN_COLUMNS
from signal_factory.signals.spec import SignalSpec, WindowDays, WindowHours


def test_grid_is_the_pre_registered_one() -> None:
    assert GRID_KEYS == ("uid", "card1_addr1", "P_emaildomain", "DeviceInfo")
    assert GRID_WINDOWS == ("all", WindowHours(hours=24), WindowDays(days=7))
    assert GRID_NUMERIC_COLUMNS == (
        "TransactionAmt",
        "dist1",
        *[f"C{i}" for i in range(1, 15)],
        *[f"D{i}" for i in range(1, 16)],
    )
    assert GRID_COUNT_DISTINCT_COLUMNS == (
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


def test_grid_size_and_shape() -> None:
    g = routine_grid()
    assert 1500 <= len(g) <= 1700
    assert len({s.id for s in g}) == len(g)
    # 3 windows × (4 keys × (count + time_since_last + 4 × 31 numeric) + (10+10+9+9) nunique)
    assert len(g) == 3 * (4 * (2 + 4 * 31) + 38)


def test_grid_never_uses_label_or_grouping_key_as_column() -> None:
    for s in routine_grid():
        assert s.column not in FORBIDDEN_COLUMNS
        assert s.column != s.entity_key
        assert s.filter is None
        assert s.filter_column not in FORBIDDEN_COLUMNS


def test_grid_contains_expected_members() -> None:
    ids = {s.id for s in routine_grid()}
    assert "rb__uid__24h__count__star" in ids
    assert "rb__uid__all__time_since_last__star" in ids
    assert "rb__card1_addr1__7d__mean__TransactionAmt" in ids
    assert "rb__P_emaildomain__all__count_distinct__DeviceInfo" in ids
    assert "rb__card1_addr1__all__count_distinct__addr1" in ids
    assert "rb__DeviceInfo__all__count_distinct__DeviceInfo" not in ids
    assert "rb__P_emaildomain__7d__count_distinct__P_emaildomain" not in ids


def test_grid_ids_follow_the_naming_rule() -> None:
    g = routine_grid()
    for s in g:
        window = (
            "all" if s.window == "all" else ("24h" if isinstance(s.window, WindowHours) else "7d")
        )
        column = "star" if s.column == "*" else s.column
        assert s.id == grid_id(s.entity_key, window, s.aggregation, column)
        assert s.id.startswith("rb__")


def test_grid_specs_all_validate() -> None:
    g = routine_grid()
    for s in g:
        SignalSpec.model_validate(s.model_dump())
        assert s.hypothesis == ROUTINE_HYPOTHESIS
        assert s.version == 1


def test_grid_is_deterministic() -> None:
    assert [s.id for s in routine_grid()] == [s.id for s in routine_grid()]


def test_grid_rejects_nothing_silently() -> None:
    with pytest.raises(ValueError):
        grid_id("uid", "1h", "count", "star")
