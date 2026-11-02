from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from signal_factory.data.split import write_split


def make_synthetic_all(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """A small dataset over all 183 days with a few informative columns and a uid."""
    rng = np.random.default_rng(seed)
    day = rng.integers(1, 184, n)
    card1 = rng.integers(1000, 1010, n)
    addr1 = rng.integers(100, 103, n)
    amt = rng.gamma(2.0, 50.0, n).astype("float32")
    c1 = rng.poisson(3.0, n).astype("float32")
    product = rng.choice(["W", "C", "H"], n)
    logit = -3.5 + 0.01 * amt + 0.4 * c1 + np.where(product == "C", 1.2, 0.0)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype("int64")
    uid = pd.Series([f"{c}_{a}_{d % 3}" for c, a, d in zip(card1, addr1, day, strict=True)])
    uid[rng.random(n) < 0.05] = None
    df = pd.DataFrame(
        {
            "TransactionID": np.arange(1, n + 1),
            "isFraud": y,
            "TransactionDT": day * 86400 + rng.integers(0, 86400, n),
            "TransactionAmt": amt,
            "ProductCD": pd.Series(product, dtype=pd.StringDtype("pyarrow")),
            "card1": card1,
            "card2": rng.integers(100, 105, n).astype("float32"),
            "card3": np.full(n, 150.0, dtype="float32"),
            "card4": pd.Series(
                rng.choice(["visa", "mastercard"], n), dtype=pd.StringDtype("pyarrow")
            ),
            "card5": rng.integers(100, 103, n).astype("float32"),
            "card6": pd.Series(rng.choice(["debit", "credit"], n), dtype=pd.StringDtype("pyarrow")),
            "addr1": addr1.astype("float32"),
            "C1": c1,
            "D1": rng.integers(0, 100, n).astype("float32"),
            "day": day.astype("int32"),
            "uid": uid.astype(pd.StringDtype("pyarrow")),
            "card1_addr1": pd.Series(
                [f"{c}_{a}" for c, a in zip(card1, addr1, strict=True)],
                dtype=pd.StringDtype("pyarrow"),
            ),
        }
    )
    return df.sort_values("TransactionDT").reset_index(drop=True)


@pytest.fixture
def synthetic_split(tmp_path: Path) -> Path:
    src = tmp_path / "all.parquet"
    make_synthetic_all().to_parquet(src, index=False)
    out = tmp_path / "split"
    write_split(src, out)
    return out


def record_then_call(opened: list[str]) -> Callable[..., pd.DataFrame]:
    real = pd.read_parquet

    def wrapper(path: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
        opened.append(str(path))
        return real(path, *args, **kwargs)

    return wrapper
