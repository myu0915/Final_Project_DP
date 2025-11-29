"""
Feature engineering utilities for retail demand forecasting.

This module takes a merged dataframe (one store or full dataset)
and produces a feature-enhanced dataframe with:

- Calendar features: dow, month, weekend
- Lag features: 1, 7, 14, 28, 56
- Rolling window means: 7, 14, 28
- Promo/holiday/transactions/oil signals

The goal is to build a supervised learning matrix (X, y) for ML/DL models.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler


def create_features(df_in: pd.DataFrame):
    """
    Add calendar, lag, and rolling window features to a dataframe.

    Parameters
    ----------
    df_in : pd.DataFrame
        Must contain:
        - store_nbr
        - family
        - date
        - sales
        - onpromotion
        - transactions
        - dcoilwtico
        - is_holiday

    Returns
    -------
    df : pd.DataFrame
        The processed dataframe with all features added.
        Rows with insufficient history are dropped.
    feature_cols : list
        List of base feature column names created here.
        (These will later be passed into prepare_feature_columns.)
    """

    df = df_in.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Sort properly
    df = df.sort_values(by=["store_nbr", "family", "date"])

    # ----------------------------------------
    # 1. CALENDAR FEATURES
    # ----------------------------------------
    df["dow"] = df["date"].dt.dayofweek.astype("int16")
    df["month"] = df["date"].dt.month.astype("int16")
    df["is_weekend"] = (df["dow"] >= 5).astype(bool)

    # ----------------------------------------
    # 2. LAG FEATURES
    # ----------------------------------------
    lags = [1, 7, 14, 28, 56]
    for lag in lags:
        df[f"sales_lag_{lag}"] = (
            df.groupby(["store_nbr", "family"])["sales"]
              .shift(lag)
              .astype("float32")
        )

    # ----------------------------------------
    # 3. ROLLING MEAN FEATURES
    # ----------------------------------------
    rolls = [7, 14, 28]
    for r in rolls:
        df[f"roll_mean_{r}"] = (
            df.groupby(["store_nbr", "family"])["sales"]
              .shift(1)                 # avoid leakage
              .rolling(r)
              .mean()
              .astype("float32")
        )

    # ----------------------------------------
    # 4. Base feature column list (before scaling selection)
    # ----------------------------------------
    feature_cols = (
        [
            "onpromotion",
            "transactions",
            "dcoilwtico",
            "is_holiday",
            "dow",
            "month",
            "is_weekend",
        ]
        + [f"sales_lag_{l}" for l in lags]
        + [f"roll_mean_{r}" for r in rolls]
    )

    # ----------------------------------------
    # 5. Drop early rows with NaN in any feature
    # ----------------------------------------
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    return df, feature_cols


def prepare_feature_columns(df_proc: pd.DataFrame, base_feature_cols: list):
    """
    From the feature-engineered dataframe, build:
    - numeric_cols (to scale)
    - bool_cols (as 0/1)
    - final ordered feature_cols
    - an unfitted StandardScaler

    This is the refactored version of Step 6 from the notebook.

    Parameters
    ----------
    df_proc : pd.DataFrame
        Output of create_features(), with columns like:
        - transactions, dcoilwtico, dow, month
        - is_weekend, is_holiday, onpromotion
        - sales_lag_*, roll_mean_*
    base_feature_cols : list
        Feature columns returned by create_features().

    Returns
    -------
    df_proc : pd.DataFrame
        Same dataframe, but with boolean cols cast to int8.
    feature_cols : list
        Final ordered feature column names (numeric first, then booleans).
    numeric_cols : list
        Columns to be scaled.
    bool_cols : list
        Boolean indicator columns now represented as integers.
    scaler : StandardScaler
        Unfitted StandardScaler instance to be used later inside
        cross-validation / train-validation split.
    """

    # 1. Identify numeric columns
    numeric_cols = [
        "transactions",
        "dcoilwtico",
        "dow",    # day of week (0–6)
        "month",  # month (1–12)
    ] + [f for f in base_feature_cols if "sales_lag" in f or "roll_mean" in f]

    # 2. Identify boolean columns
    bool_cols = [
        "onpromotion",
        "is_holiday",
        "is_weekend",
    ]

    # Make sure we only keep columns that actually exist
    numeric_cols = [c for c in numeric_cols if c in df_proc.columns]
    bool_cols = [c for c in bool_cols if c in df_proc.columns]

    # 3. Convert booleans to small integers (0/1)
    if bool_cols:
        df_proc[bool_cols] = df_proc[bool_cols].astype("int8")

    # 4. Final feature order
    feature_cols = numeric_cols + bool_cols
    scaler = StandardScaler()

    return df_proc, feature_cols, numeric_cols, bool_cols, scaler
