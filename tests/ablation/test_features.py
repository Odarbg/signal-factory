import numpy as np
import pandas as pd

from signal_factory.ablation.features import DROP_COLUMNS, base_features, labels


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "isFraud": [0, 1, 0],
            "TransactionDT": [100, 200, 300],
            "TransactionAmt": pd.Series([1.0, 2.0, 3.0], dtype="float32"),
            "ProductCD": pd.Series(["W", "C", None], dtype=pd.StringDtype("pyarrow")),
            "card1": [10, 20, 30],
            "day": pd.Series([1, 1, 2], dtype="int32"),
            "uid": pd.Series(["a", None, "b"], dtype=pd.StringDtype("pyarrow")),
            "card1_addr1": pd.Series(["10_1", "20_2", None], dtype=pd.StringDtype("pyarrow")),
            "DeviceInfo": ["x", "y", "x"],
        }
    )


def test_drop_columns_are_the_identifiers_keys_and_label() -> None:
    assert set(DROP_COLUMNS) == {
        "TransactionID", "TransactionDT", "uid", "day", "card1_addr1", "isFraud",
    }


def test_base_features_drops_ids_and_label_and_categorises() -> None:
    x = base_features(_frame())
    assert list(x.columns) == ["TransactionAmt", "ProductCD", "card1", "DeviceInfo"]
    assert isinstance(x["ProductCD"].dtype, pd.CategoricalDtype)
    assert isinstance(x["DeviceInfo"].dtype, pd.CategoricalDtype)
    assert x["TransactionAmt"].dtype == "float32"
    assert x["ProductCD"].isna().sum() == 1


def test_base_features_tolerates_missing_drop_columns() -> None:
    df = _frame().drop(columns=["isFraud", "uid"])
    assert "isFraud" not in base_features(df).columns


def test_base_features_does_not_mutate_input() -> None:
    df = _frame()
    before = df.copy()
    base_features(df)
    pd.testing.assert_frame_equal(df, before)


def test_labels_are_int8() -> None:
    y = labels(_frame())
    assert y.dtype == np.int8 and y.tolist() == [0, 1, 0]
