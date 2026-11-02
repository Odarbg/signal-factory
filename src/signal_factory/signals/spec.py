"""The depth-1 signal spec (spec §8.2), as a pydantic model and JSON schema.

A signal aggregates one column of an entity's strictly earlier history rows, optionally
filtered by one predicate, over a time window. The label is never a legal column.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from signal_factory.signals.columns import ENTITY_KEYS, FORBIDDEN_COLUMNS

Aggregation = Literal["count", "count_distinct", "sum", "mean", "max", "min", "time_since_last"]
STAR_AGGREGATIONS: frozenset[str] = frozenset({"count", "time_since_last"})
"""Aggregations that take no column and therefore require ``column == "*"``."""

SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 86400


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class WindowHours(_Strict):
    hours: int = Field(gt=0)

    def seconds(self) -> int:
        return self.hours * SECONDS_PER_HOUR


class WindowDays(_Strict):
    days: int = Field(gt=0)

    def seconds(self) -> int:
        return self.days * SECONDS_PER_DAY


Window = Literal["all"] | WindowHours | WindowDays


def window_seconds(window: Window) -> int | None:
    """Window length in seconds, or ``None`` for all history."""
    return None if window == "all" else window.seconds()


class Filter(_Strict):
    column: str
    op: Literal["==", "!=", "<", "<=", ">", ">="]
    value: float | str


class SignalInput(_Strict):
    column: str
    available_at: Literal["decision_time"]


class SignalSpec(_Strict):
    id: Annotated[str, Field(pattern=r"^[a-z][A-Za-z0-9_]{2,63}$")]
    version: int = Field(ge=1)
    hypothesis: str = Field(min_length=10, max_length=400)
    entity_key: str
    column: str
    aggregation: Aggregation
    window: Window
    filter: Filter | None = None
    missing: Literal["null"] = "null"
    inputs: list[SignalInput]

    @model_validator(mode="after")
    def _check_roles(self) -> SignalSpec:
        if self.entity_key not in ENTITY_KEYS:
            raise ValueError(f"entity_key {self.entity_key!r} is not one of {sorted(ENTITY_KEYS)}")
        for role, name in (("column", self.column), ("filter.column", self.filter_column)):
            if name in FORBIDDEN_COLUMNS:
                raise ValueError(f"{role} {name!r} is forbidden: never an input to a signal")
        if self.aggregation in STAR_AGGREGATIONS and self.column != "*":
            raise ValueError(f"aggregation {self.aggregation!r} requires column '*'")
        if self.aggregation not in STAR_AGGREGATIONS and self.column == "*":
            raise ValueError(f"column '*' is only valid with {sorted(STAR_AGGREGATIONS)}")
        declared = {i.column for i in self.inputs}
        missing = sorted(self.used_columns() - declared)
        if missing:
            raise ValueError(f"inputs must list every column used; missing {missing}")
        return self

    @property
    def filter_column(self) -> str | None:
        return None if self.filter is None else self.filter.column

    def used_columns(self) -> set[str]:
        """Every history column this signal reads: key, aggregated column, filter column."""
        used = {self.entity_key}
        if self.column != "*":
            used.add(self.column)
        if self.filter is not None:
            used.add(self.filter.column)
        return used

    def validate_columns(self, known: Iterable[str]) -> None:
        """Raise ``ValueError`` naming any used column that is not in ``known``."""
        unknown = sorted(self.used_columns() - set(known))
        if unknown:
            raise ValueError(f"signal {self.id!r} uses unknown columns {unknown}")


def signal_json_schema() -> dict[str, Any]:
    return SignalSpec.model_json_schema()
