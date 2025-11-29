import pandas as pd
from pathlib import Path


# =====================================================================
# DATA DIRECTORY
# =====================================================================
# All data for the Streamlit app must be inside the local /data folder.
# The user must manually place the Kaggle CSV files into /data.
# =====================================================================

DATA_DIR = Path("data")


def load_raw_data():
    """
    Loads all CSV files from the local /data directory.

    Expected files in /data:
        - train.csv
        - test.csv
        - stores.csv
        - oil.csv
        - holidays_events.csv
        - transactions.csv

    Missing files return None, but merge_full_dataset() will handle this.
    """
    files = {
        "train": "train.csv",
        "test": "test.csv",
        "stores": "stores.csv",
        "oil": "oil.csv",
        "holidays": "holidays_events.csv",
        "transactions": "transactions.csv",
    }

    dfs = {}
    for key, fname in files.items():
        path = DATA_DIR / fname
        if path.exists():
            dfs[key] = pd.read_csv(path)
        else:
            dfs[key] = None
    return dfs


# =====================================================================
# OIL PRICE — INTERPOLATION HELPER
# =====================================================================
def build_oil_full(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    oil_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a complete daily oil price series from earliest train date
    to latest test date, interpolating missing values.

    Output columns:
        - date
        - dcoilwtico (continuous, float32)
    """
    train_dates = pd.to_datetime(train_df["date"])
    test_dates = pd.to_datetime(test_df["date"])

    oil = oil_df.copy()
    oil["date"] = pd.to_datetime(oil["date"])

    # full date index
    full_range = pd.date_range(train_dates.min(), test_dates.max())
    oil_full = pd.DataFrame({"date": full_range})

    oil_full = oil_full.merge(oil, on="date", how="left")

    if "dcoilwtico" in oil_full.columns:
        oil_full["dcoilwtico"] = (
            oil_full["dcoilwtico"]
            .interpolate(method="linear", limit_direction="both")
            .fillna(method="bfill")
            .astype("float32")
        )

    return oil_full


# =====================================================================
# HOLIDAY FLAG — CLEAN, NO DUPLICATES
# =====================================================================
def build_holidays_flag(holidays_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a boolean holiday indicator.

    Logic:
        - Keep only rows where type == 'Holiday' and transferred == False
        - Aggregate to one row per date
    """
    holidays = holidays_df.copy()
    holidays["date"] = pd.to_datetime(holidays["date"])

    # ensure boolean for 'transferred'
    if "transferred" in holidays.columns:
        holidays["transferred"] = (
            holidays["transferred"]
            .fillna(False)
            .astype(bool)
        )
    else:
        holidays["transferred"] = False

    real_holidays = holidays[
        (holidays["type"] == "Holiday") &
        (holidays["transferred"] == False)
    ]

    holidays_flag = (
        real_holidays
        .groupby("date")["type"]
        .any()
        .reset_index()
    )

    holidays_flag.rename(columns={"type": "is_holiday"}, inplace=True)
    holidays_flag["is_holiday"] = holidays_flag["is_holiday"].astype(bool)
    return holidays_flag


# =====================================================================
# MAIN MERGE FUNCTION — THE ONE USED BY STREAMLIT
# =====================================================================
def merge_full_dataset() -> pd.DataFrame:
    """
    Merge train.csv with stores, oil, holidays, transactions.

    Adds:
        - oil price (interpolated)
        - holiday flag
        - transactions (store & date)
        - calendar features

    Ensures:
        - continuous dates for oil
        - no duplicated (store_nbr, family, date)
    """
    dfs = load_raw_data()

    train = dfs["train"]
    stores = dfs["stores"]
    oil = dfs["oil"]
    holidays = dfs["holidays"]
    transactions = dfs["transactions"]
    test = dfs["test"]

    if train is None:
        raise FileNotFoundError("train.csv missing in /data")

    # copy & datetime conversion
    train = train.copy()
    train["date"] = pd.to_datetime(train["date"])

    if test is None:
        # if test missing, assume train only
        test = train.copy()

    # ------------------------------------------------------------------
    # Stores
    # ------------------------------------------------------------------
    if stores is not None:
        train = train.merge(stores, on="store_nbr", how="left")

    # ------------------------------------------------------------------
    # Oil (continuous series)
    # ------------------------------------------------------------------
    if oil is not None:
        oil_full = build_oil_full(train, test, oil)
        train = train.merge(oil_full, on="date", how="left")

    # ------------------------------------------------------------------
    # Transactions
    # ------------------------------------------------------------------
    if transactions is not None:
        transactions = transactions.copy()
        transactions["date"] = pd.to_datetime(transactions["date"])
        train = train.merge(
            transactions[["date", "store_nbr", "transactions"]],
            on=["store_nbr", "date"],
            how="left",
        )
        if "transactions" in train.columns:
            # avoid chained assignment & deprecated inplace usage
            train["transactions"] = train["transactions"].fillna(0)

    # ------------------------------------------------------------------
    # Holidays
    # ------------------------------------------------------------------
    if holidays is not None:
        holidays_flag = build_holidays_flag(holidays)
        train = train.merge(holidays_flag, on="date", how="left")
        # fill missing with False and assign back
        train["is_holiday"] = train["is_holiday"].fillna(False)
    else:
        train["is_holiday"] = False

    # ------------------------------------------------------------------
    # Calendar features
    # ------------------------------------------------------------------
    train["year"] = train["date"].dt.year
    train["month"] = train["date"].dt.month
    train["day"] = train["date"].dt.day
    train["dayofweek"] = train["date"].dt.dayofweek
    train["weekofyear"] = train["date"].dt.isocalendar().week.astype(int)

    # ------------------------------------------------------------------
    # Remove unexpected duplicates
    # ------------------------------------------------------------------
    if {"store_nbr", "family", "date"}.issubset(train.columns):
        train = train.sort_values(["store_nbr", "family", "date"])
        train = train.drop_duplicates(
            subset=["store_nbr", "family", "date"],
            keep="first",
        )

    return train


# =====================================================================
# SERIES EXTRACTOR — FOR ANY STORE + FAMILY
# =====================================================================
def get_series(train_df: pd.DataFrame, store_nbr: int, family: str):
    """
    Extract a clean time series for one store+family with a DateTimeIndex.
    """
    subset = train_df[
        (train_df["store_nbr"] == store_nbr) &
        (train_df["family"] == family)
    ].copy()

    if subset.empty:
        return None

    subset["date"] = pd.to_datetime(subset["date"])
    subset = subset.sort_values("date")
    subset.set_index("date", inplace=True)

    return subset["sales"]
