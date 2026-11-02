import numpy as np
import pandas as pd

from signal_factory.data.stats import CollisionStats, uid_collision_rate


def test_uid_collision_rate_counts_multi_card_uids() -> None:
    df = pd.DataFrame(
        {
            "uid": pd.Series(
                ["a", "a", "a", "b", "b", "c", None, None], dtype=pd.StringDtype("pyarrow")
            ),
            "card2": [1, 1, 2, 5, 5, 7, 9, 10],
            "card3": [150, 150, 150, 150, 150, 150, 150, 150],
            "card4": ["visa", "visa", "visa", "mc", "mc", "visa", "visa", "mc"],
            "card5": [226, 226, 226, 102, 102, 226, 226, 102],
            "card6": ["debit", "debit", "debit", "credit", "credit", "debit", "debit", "credit"],
        }
    )
    stats = uid_collision_rate(df)
    # a: two distinct card tuples (card2 1 vs 2) → colliding; b, c: one each; NULL uids ignored.
    assert stats == CollisionStats(uids=3, colliding=1, rate=1 / 3)


def test_uid_collision_rate_treats_nan_as_a_value_not_a_wildcard() -> None:
    df = pd.DataFrame(
        {
            "uid": pd.Series(["a", "a"], dtype=pd.StringDtype("pyarrow")),
            "card2": [np.nan, 1.0],
            "card3": [1.0, 1.0],
            "card4": ["visa", "visa"],
            "card5": [1.0, 1.0],
            "card6": ["debit", "debit"],
        }
    )
    assert uid_collision_rate(df).colliding == 1


def test_uid_collision_rate_with_no_uids() -> None:
    df = pd.DataFrame(
        {
            "uid": pd.Series([None], dtype=pd.StringDtype("pyarrow")),
            "card2": [1],
            "card3": [1],
            "card4": ["v"],
            "card5": [1],
            "card6": ["d"],
        }
    )
    stats = uid_collision_rate(df)
    assert stats.uids == 0 and stats.colliding == 0 and np.isnan(stats.rate)
