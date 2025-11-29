# app/yfinance.py
# -------------------------------------------------------------------
# Utilities for working with yfinance-style time series (OHLCV).
# Provides:
#   - yfinance-style CSV preparation
#   - flexible transforms (level / log / log-return / pct-change)
#   - ARIMA + Random Forest forecasting helpers
# -------------------------------------------------------------------

from __future__ import annotations

from typing import Literal, Tuple, Dict, Optional, List

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


TransformType = Literal["level", "log", "log_return", "pct_change"]


# ---------- Generic helpers (self-contained) ------------------------


def _mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape_val = _mape(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape_val}


def _make_lag_features(series: pd.Series, n_lags: int = 7) -> pd.DataFrame:
    df = pd.DataFrame({"y": series})
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df = df.dropna()
    return df


# ---------- yfinance-specific preparation ---------------------------


def detect_price_column(df: pd.DataFrame) -> Optional[str]:
    """
    Try to auto-detect the main price column in a yfinance CSV.
    Preference: 'Adj Close' > 'Close' > 'close' > 'Price' etc.
    """
    candidates_priority = [
        "Adj Close",
        "AdjClose",
        "adj_close",
        "Close",
        "close",
        "Price",
        "price",
    ]
    cols_lower = {c.lower(): c for c in df.columns}

    for cand in candidates_priority:
        if cand in df.columns:
            return cand
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]

    # fallback: first numeric-looking column (excluding Date/Volume/Open/High/Low)
    ignore = {"date", "datetime", "open", "high", "low", "volume", "adj close", "close"}
    for c in df.columns:
        name = str(c).lower()
        if any(k in name for k in ignore):
            continue
        sample = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False),
                               errors="coerce")
        if sample.notna().mean() > 0.8:
            return c

    return None


def prepare_yfinance_series(
    df: pd.DataFrame,
    date_col: str = "Date",
    price_col: Optional[str] = None,
    transform: TransformType = "log_return",
) -> pd.Series:
    """
    Clean a yfinance-style dataframe and return a 1D pd.Series ready for modeling.

    Steps:
      - parse dates
      - sort, drop duplicates
      - auto-select price column if not provided
      - apply transformation:
          'level'      -> raw prices
          'log'        -> log(price)
          'log_return' -> diff(log(price))
          'pct_change' -> price.pct_change()
    """
    if date_col not in df.columns:
        # try to guess a date column
        for cand in ["Date", "Datetime", "date", "datetime"]:
            if cand in df.columns:
                date_col = cand
                break

    parsed_dates = pd.to_datetime(df[date_col].astype(str), errors="coerce")
    df = df.loc[parsed_dates.notna()].copy()
    df[date_col] = parsed_dates[parsed_dates.notna()]
    df = df.drop_duplicates(subset=[date_col]).sort_values(date_col)

    if price_col is None:
        price_col = detect_price_column(df)
        if price_col is None:
            raise ValueError("Could not detect a price column in yfinance dataframe.")

    # numeric conversion
    s = (
        df[price_col]
        .astype(str)
        .str.replace(",", "", regex=False)
    )
    s = pd.to_numeric(s, errors="coerce")
    s = s.loc[s.notna()]

    series = pd.Series(s.values, index=df.loc[s.index, date_col], name=price_col)
    series = series[~series.index.duplicated(keep="first")]
    series.index.name = "Date"

    if transform == "level":
        return series

    if transform == "log":
        if (series <= 0).any():
            raise ValueError("Log transform requires strictly positive prices.")
        return np.log(series)

    if transform == "log_return":
        if (series <= 0).any():
            raise ValueError("Log returns require strictly positive prices.")
        return np.log(series).diff().dropna()

    if transform == "pct_change":
        return series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

    raise ValueError(f"Unknown transform: {transform}")


# ---------- Forecasting wrappers -----------------------------------


def yf_arima_forecast(
    series: pd.Series,
    horizon: int,
    order: Tuple[int, int, int] = (1, 0, 0),
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Fit ARIMA(order) on the provided series and forecast 'horizon' steps ahead.
    Returns: (forecast_series, metrics_dict) where metrics compare last horizon points.
    """
    if len(series) <= horizon + 5:
        raise ValueError("Not enough data for given horizon.")

    train = series.iloc[:-horizon]
    test = series.iloc[-horizon:]

    model = ARIMA(train, order=order)
    fitted = model.fit()
    fc = fitted.forecast(steps=horizon)
    fc.index = test.index

    metrics = compute_metrics(test.values, fc.values)
    return fc, metrics


def yf_rf_forecast(
    series: pd.Series,
    horizon: int,
    n_lags: int = 10,
    n_estimators: int = 300,
    max_depth: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Random Forest lag-based forecast for yfinance series.
    """
    if len(series) <= horizon + n_lags + 5:
        raise ValueError("Not enough data for given horizon / lags.")

    train = series.iloc[:-horizon]
    test = series.iloc[-horizon:]

    df_lag = _make_lag_features(train, n_lags=n_lags)
    X_train = df_lag.drop(columns=["y"])
    y_train = df_lag["y"]

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    history = list(train.values)
    preds: List[float] = []
    for _ in range(horizon):
        last_vals = history[-n_lags:]
        x = np.array(last_vals).reshape(1, -1)
        preds.append(float(model.predict(x)[0]))
        history.append(preds[-1])

    fc = pd.Series(preds, index=test.index, name="rf_forecast")
    metrics = compute_metrics(test.values, fc.values)
    return fc, metrics
