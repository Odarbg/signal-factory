from pathlib import Path

import pandas as pd

from signal_factory.data.convert import SECONDS_PER_DAY, add_derived_columns, convert

FIXTURES = Path(__file__).parent / "fixtures"


def test_seconds_per_day() -> None:
    assert SECONDS_PER_DAY == 86400


def test_day_starts_at_one_and_uid_components() -> None:
    df = pd.read_csv(FIXTURES / "mini_transaction.csv")
    out = add_derived_columns(df)
    assert out["day"].min() == 1
    assert out["day"].dtype == "int32"
    row = out.loc[out.TransactionID == 1].iloc[0]
    assert row["uid"] == f"{int(row.card1)}_{int(row.addr1)}_{int(row.day - row.D1)}"
    assert row["card1_addr1"] == f"{int(row.card1)}_{int(row.addr1)}"


def test_day_is_relative_to_first_day_present() -> None:
    df = pd.read_csv(FIXTURES / "mini_transaction.csv")
    out = add_derived_columns(df).set_index("TransactionID")
    # TransactionDT 86400 → day 1; 172800 → day 2; 432000 → day 5
    assert out.loc[1, "day"] == 1
    assert out.loc[3, "day"] == 2
    assert out.loc[8, "day"] == 5


def test_uid_is_stable_across_days_for_same_card_start() -> None:
    # Rows 1 and 8 share card1/addr1; D1 grows with the day, so D1n (= day - D1) is equal.
    out = add_derived_columns(pd.read_csv(FIXTURES / "mini_transaction.csv"))
    out = out.set_index("TransactionID")
    assert str(out.loc[1, "uid"]) == str(out.loc[8, "uid"])


def test_null_component_gives_null_uid() -> None:
    out = add_derived_columns(pd.read_csv(FIXTURES / "mini_transaction.csv"))
    assert out.loc[out.addr1.isna(), "uid"].isna().all()
    assert out.loc[out.D1.isna(), "uid"].isna().all()
    assert out.loc[out.addr1.isna(), "card1_addr1"].isna().all()
    assert out.loc[out.D1.isna() & out.addr1.notna(), "card1_addr1"].notna().all()


def test_add_derived_columns_does_not_mutate_input() -> None:
    df = pd.read_csv(FIXTURES / "mini_transaction.csv")
    before = df.copy()
    add_derived_columns(df)
    pd.testing.assert_frame_equal(df, before)


def test_convert_joins_identity_and_downcasts(tmp_path: Path) -> None:
    convert(
        FIXTURES,
        tmp_path / "all.parquet",
        transaction_file="mini_transaction.csv",
        identity_file="mini_identity.csv",
    )
    df = pd.read_parquet(tmp_path / "all.parquet")
    assert len(df) == 8
    assert df["DeviceInfo"].notna().sum() == 3
    assert df["TransactionAmt"].dtype == "float32"
    assert pd.api.types.is_string_dtype(df["ProductCD"])
    assert df["ProductCD"].dtype != object
    assert {"day", "uid", "card1_addr1", "isFraud"} <= set(df.columns)
    assert df["uid"].isna().sum() == 2
