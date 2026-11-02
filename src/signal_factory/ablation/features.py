"""Row A feature frame (spec §4): every original column except identifiers, keys and label."""

from __future__ import annotations

import numpy as np
import pandas as pd

DROP_COLUMNS: tuple[str, ...] = (
    "TransactionID",
    "TransactionDT",
    "uid",
    "day",
    "card1_addr1",
    "isFraud",
)
LABEL = "isFraud"


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifiers, derived keys and the label; string columns become ``category``."""
    x = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    for column in x.columns:
        dtype = x[column].dtype
        if dtype == object or pd.api.types.is_string_dtype(dtype):
            x[column] = x[column].astype("category")
    return x


def labels(df: pd.DataFrame) -> np.ndarray:
    return df[LABEL].to_numpy(dtype=np.int8)
