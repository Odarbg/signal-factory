"""Convert the raw IEEE-CIS train CSVs to one Parquet file with derived keys.

Adds (spec §4, §5):

- ``day``: ``TransactionDT // 86400`` relative to the first day present, starting at 1.
- ``uid``: ``f"{card1}_{addr1}_{D1n}"`` with ``D1n = day - D1``, the pseudo-entity
  identifier. NULL if any component is NULL: a NULL key abstains and never forms a group.
- ``card1_addr1``: ``f"{card1}_{addr1}"``, NULL if either component is NULL.

Floats are stored as float32 and object columns as Arrow strings to fit on a laptop.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SECONDS_PER_DAY = 86400
TRANSACTION_FILE = "train_transaction.csv"
IDENTITY_FILE = "train_identity.csv"
STRING_DTYPE = pd.StringDtype("pyarrow")


def _join_key(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Underscore-join integer-valued columns; NULL where any component is NULL."""
    complete = df[columns].notna().all(axis=1)
    parts = [df[c].where(complete, 0).astype("int64").astype(str) for c in columns]
    joined = parts[0]
    for part in parts[1:]:
        joined = joined + "_" + part
    return joined.where(complete).astype(STRING_DTYPE)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with ``day``, ``uid`` and ``card1_addr1`` added."""
    out = df.copy()
    first_day = int(out["TransactionDT"].min()) // SECONDS_PER_DAY
    out["day"] = (out["TransactionDT"] // SECONDS_PER_DAY - first_day + 1).astype("int32")
    out["D1n"] = out["day"] - out["D1"]
    out["uid"] = _join_key(out, ["card1", "addr1", "D1n"])
    out["card1_addr1"] = _join_key(out, ["card1", "addr1"])
    return out.drop(columns=["D1n"])


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    for column in df.columns:
        dtype = df[column].dtype
        if dtype == "float64":
            df[column] = df[column].astype("float32")
        elif dtype == "object":
            df[column] = df[column].astype(STRING_DTYPE)
    return df


def convert(
    raw_dir: Path,
    out_path: Path,
    transaction_file: str = TRANSACTION_FILE,
    identity_file: str = IDENTITY_FILE,
) -> None:
    """Join identity onto transactions, add derived columns, write Parquet."""
    transactions = pd.read_csv(raw_dir / transaction_file)
    identity = pd.read_csv(raw_dir / identity_file)
    df = transactions.merge(identity, on="TransactionID", how="left")
    df = add_derived_columns(df)
    df = _downcast(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
