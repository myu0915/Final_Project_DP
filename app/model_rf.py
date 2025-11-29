# app/model_rf.py

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def _make_lag_supervised(series: pd.Series, window: int) -> tuple[pd.DataFrame, pd.Series]:
    """
    Turn a univariate series into a supervised dataset with lag features.
    y_t ~ [y_{t-1}, ..., y_{t-window}]
    """
    s = pd.Series(series).astype(float)
    df = pd.DataFrame({"y": s})
    for lag in range(1, window + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df = df.dropna()
    X = df.drop(columns=["y"])
    y = df["y"]
    return X, y


def rf_forecast_with_uncertainty(
    series: pd.Series,
    window: int,
    horizon: int,
    n_estimators: int = 200,
    random_state: int = 42,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Random Forest time-series forecast with simple Gaussian uncertainty.
    Returns: mean, lower, upper, std series.
    """

    series = pd.Series(series).astype(float)
    X, y = _make_lag_supervised(series, window)

    if len(y) < window + 5:
        raise ValueError("Not enough history for the chosen RF window size.")

    # small validation slice for residual std
    k = max(1, min(28, len(y) // 3))
    X_train, X_val = X.iloc[:-k], X.iloc[-k:]
    y_train, y_val = y.iloc[:-k], y.iloc[-k:]

    model = RandomForestRegressor(
        n_estimators=n_estimators, random_state=random_state, n_jobs=-1
    )
    model.fit(X_train, y_train)

    # estimate residual std from validation
    val_pred = model.predict(X_val)
    resid = y_val.values - val_pred
    resid_std = float(np.std(resid, ddof=1)) if len(resid) > 1 else float(
        np.std(y.values - model.predict(X), ddof=1)
    )

    # recursive forecast
    history = list(series.values)
    preds: list[float] = []

    for _ in range(horizon):
        # last `window` values -> [y_t, ..., y_{t-window+1}] (reverse for lag_1,...)
        last_vals = history[-1 : -window - 1 : -1]
        if len(last_vals) < window:
            last_vals = ([history[-1]] * (window - len(last_vals))) + last_vals
        x_new = np.array(last_vals).reshape(1, -1)
        y_hat = float(model.predict(x_new)[0])
        preds.append(y_hat)
        history.append(y_hat)

    # build index
    idx = series.index
    last_idx = idx[-1]
    if isinstance(idx, pd.DatetimeIndex):
        freq = pd.infer_freq(idx) or "D"
        future_index = pd.date_range(
            start=last_idx + pd.tseries.frequencies.to_offset(freq),
            periods=horizon,
            freq=freq,
        )
    else:
        future_index = pd.RangeIndex(start=idx[-1] + 1, stop=idx[-1] + 1 + horizon)

    mean = pd.Series(preds, index=future_index, name="rf_mean")
    std = pd.Series(resid_std, index=future_index, name="rf_std")
    lower = mean - 1.96 * resid_std
    upper = mean + 1.96 * resid_std

    return mean, lower, upper, std
