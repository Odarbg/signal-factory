"""Measured, not decided (spec §10): how often a pseudo-identifier spans several cards."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

CARD_COLUMNS = ("card2", "card3", "card4", "card5", "card6")


@dataclass(frozen=True)
class CollisionStats:
    uids: int
    colliding: int
    rate: float


def uid_collision_rate(df: pd.DataFrame) -> CollisionStats:
    """Share of non-NULL uids with more than one distinct ``card2..card6`` tuple."""
    rows = df.loc[df["uid"].notna(), ["uid", *CARD_COLUMNS]]
    distinct = rows.drop_duplicates().groupby("uid", dropna=False).size()
    uids = len(distinct)
    colliding = int((distinct > 1).sum())
    return CollisionStats(
        uids=uids, colliding=colliding, rate=colliding / uids if uids else float("nan")
    )
