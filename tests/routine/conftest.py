import numpy as np
import pandas as pd
import pytest

NUMERIC = [
    "TransactionAmt",
    "dist1",
    *[f"C{i}" for i in range(1, 15)],
    *[f"D{i}" for i in range(1, 16)],
]
CATEGORICAL = [
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
]


@pytest.fixture(scope="session")
def grid_history() -> pd.DataFrame:
    """60 rows with every column the row-B grid touches, some NULL keys, some ties."""
    rng = np.random.default_rng(42)
    n = 60
    df = pd.DataFrame({"TransactionID": np.arange(1, n + 1)})
    df["TransactionDT"] = np.sort(rng.integers(0, 10 * 86400, n))
    df.loc[df.index[5], "TransactionDT"] = df.loc[df.index[4], "TransactionDT"]  # a tie
    for col in NUMERIC:
        values = rng.gamma(2.0, 10.0, n).astype("float32")
        values[rng.random(n) < 0.1] = np.nan
        df[col] = values
    for col in CATEGORICAL:
        choices = [f"{col}_{k}" for k in range(3)]
        df[col] = pd.Series(rng.choice(choices, n), dtype=pd.StringDtype("pyarrow"))
    for key in ("uid", "card1_addr1"):
        keys = pd.Series(rng.choice([f"{key}_{k}" for k in range(4)], n), dtype="object")
        keys[rng.random(n) < 0.1] = None
        df[key] = keys.astype(pd.StringDtype("pyarrow"))
    return df
