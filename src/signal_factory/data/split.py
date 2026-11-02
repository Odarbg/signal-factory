"""Chronological split with 7-day gaps (spec §5).

Dev training rows are days 1–98; validation rows 106–138; final rows 146–183. Days
99–105 and 139–145 are gaps: present in ``history.parquet`` as feature history, absent
from every training and evaluation file. ``history.parquet`` holds all rows with the
label dropped, so nothing that computes features can see ``isFraud``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

Period = Literal["dev", "gap", "validation", "final"]

DEV_END = 98
VALIDATION_START = 106
VALIDATION_END = 138
FINAL_START = 146
FINAL_END = 183

LABEL = "isFraud"
PERIOD_FILES: dict[Period, str] = {
    "dev": "dev.parquet",
    "validation": "validation.parquet",
    "final": "final.parquet",
}
HISTORY_FILE = "history.parquet"


@dataclass(frozen=True)
class SplitCounts:
    dev: int
    gap: int
    validation: int
    final: int
    history: int


def assign_period(day: int) -> Period:
    """Map a day number (1-based) to its period; days outside 1–183 are an error."""
    if day < 1 or day > FINAL_END:
        raise ValueError(f"day {day} is outside the dataset's 1–{FINAL_END} range")
    if day <= DEV_END:
        return "dev"
    if day < VALIDATION_START:
        return "gap"
    if day <= VALIDATION_END:
        return "validation"
    if day < FINAL_START:
        return "gap"
    return "final"


def write_split(all_parquet: Path, out_dir: Path) -> SplitCounts:
    """Write dev/validation/final period files and the label-free history file."""
    df = pd.read_parquet(all_parquet)
    period = df["day"].map(assign_period)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for name, filename in PERIOD_FILES.items():
        rows = df.loc[period == name]
        rows.to_parquet(out_dir / filename, index=False)
        counts[name] = len(rows)
    df.drop(columns=[LABEL]).to_parquet(out_dir / HISTORY_FILE, index=False)
    return SplitCounts(
        dev=counts["dev"],
        gap=int((period == "gap").sum()),
        validation=counts["validation"],
        final=counts["final"],
        history=len(df),
    )
