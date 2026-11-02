"""A 12-row history over two pseudo-identities and one NULL key.

Hand-computed expectations live in the test files; the layout is:

    id  uid   TransactionDT  TransactionAmt  ProductCD  P_emaildomain
     1  a           1000           10.0        W        gmail.com
     2  a           2000           20.0        C        gmail.com
     3  a           2000           30.0        W        yahoo.com   (tie with 2)
     4  a          87400           40.0        W        gmail.com   (exactly 24h after 1)
     5  a          87401           50.0        C        yahoo.com   (24h + 1s after 1)
     6  b            500           60.0        W        gmail.com
     7  b           1500           70.0        C        gmail.com
     8  b           1500           80.0        C        yahoo.com   (tie with 7)
     9  b         100000           90.0        W        gmail.com
    10  NULL         100            5.0        W        gmail.com
    11  NULL         200            6.0        W        gmail.com
    12  NULL         300            7.0        C        yahoo.com
"""

from collections.abc import Iterator

import duckdb
import pandas as pd
import pytest


@pytest.fixture
def tiny_history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": list(range(1, 13)),
            "uid": pd.Series(
                ["a", "a", "a", "a", "a", "b", "b", "b", "b", None, None, None],
                dtype=pd.StringDtype("pyarrow"),
            ),
            "TransactionDT": [1000, 2000, 2000, 87400, 87401, 500, 1500, 1500, 100000, 100, 200, 300],
            "TransactionAmt": pd.Series(
                [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 5.0, 6.0, 7.0],
                dtype="float32",
            ),
            "ProductCD": pd.Series(
                ["W", "C", "W", "W", "C", "W", "C", "C", "W", "W", "W", "C"],
                dtype=pd.StringDtype("pyarrow"),
            ),
            "P_emaildomain": pd.Series(
                [
                    "gmail.com", "gmail.com", "yahoo.com", "gmail.com", "yahoo.com",
                    "gmail.com", "gmail.com", "yahoo.com", "gmail.com",
                    "gmail.com", "gmail.com", "yahoo.com",
                ],
                dtype=pd.StringDtype("pyarrow"),
            ),
        }
    )


@pytest.fixture
def con(tiny_history: pd.DataFrame) -> Iterator[duckdb.DuckDBPyConnection]:
    connection = duckdb.connect()
    connection.register("history", tiny_history)
    yield connection
    connection.close()
