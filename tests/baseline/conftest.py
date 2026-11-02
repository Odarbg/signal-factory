import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def synthetic_xy() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """400 train / 200 eval rows with two informative numerics and one categorical."""
    rng = np.random.default_rng(123)

    def make(n: int) -> tuple[pd.DataFrame, np.ndarray]:
        amt = rng.gamma(2.0, 50.0, n).astype("float32")
        c1 = rng.poisson(3.0, n).astype("float32")
        product = rng.choice(["W", "C", "H"], n)
        logit = -3.0 + 0.01 * amt + 0.3 * c1 + np.where(product == "C", 1.0, 0.0)
        y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype("int8")
        x = pd.DataFrame(
            {
                "TransactionAmt": amt,
                "C1": c1,
                "ProductCD": pd.Series(product, dtype=pd.StringDtype("pyarrow")),
                "noise": rng.normal(size=n).astype("float32"),
            }
        )
        x.loc[x.index[:5], "C1"] = np.nan
        return x, y

    x_train, y_train = make(400)
    x_eval, y_eval = make(200)
    return x_train, y_train, x_eval, y_eval
