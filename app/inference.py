"""
Unified inference helpers for ARIMA, SARIMAX and deep-learning models.

Used by the Streamlit UI:

- Forecast page:
  - `run_arima_forecast(...)`   -> when Model = "ARIMA"
  - `run_sarimax_forecast(...)` -> when Model = "SARIMAX"
  - `run_dl_forecast_with_uncertainty(...)` -> when Model is LSTM / TCN / TRANSFORMER

Also includes alignment fixes for:
    "ValueError: The indices for endog and exog are not aligned"
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Try to import internal model wrappers
try:
    from .model_arima import (
        fit_arima,
        fit_sarimax,
        forecast_arima,  # alias of forecast_model(...)
    )
    from .uncertainty import mc_dropout_predict
except ImportError:
    # Allows docs / static analysis without full runtime deps
    pass

# Where we store Keras models
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------
# Helper: Data Preprocessing & Alignment
# ---------------------------------------------------------------------
def _prepare_series(series: pd.Series, freq: str = "D") -> pd.Series:
    """
    Ensure the series has a proper DatetimeIndex and consistent frequency.

    - Enforces DatetimeIndex.
    - Fills missing calendar days via forward-fill.
    """
    series = series.copy()

    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)

    # Infer / enforce frequency; fill gaps (e.g. missing Sundays)
    if series.index.freq is None:
        series = series.asfreq(freq)
        series = series.fillna(method="ffill")

    return series.astype("float64")


def _align_exog(
    exog: Optional[pd.DataFrame],
    target_index: pd.Index,
) -> Optional[pd.DataFrame]:
    """
    Align exogenous data to match the target series index exactly.

    - Reindexes exog to target_index.
    - Fills missing rows with 0 (safe for 0/1 promotion flags).
    - Drops extra rows not in target_index.
    """
    if exog is None:
        return None

    exog = exog.copy()

    if not isinstance(exog.index, pd.DatetimeIndex):
        exog.index = pd.to_datetime(exog.index)

    exog_aligned = exog.reindex(target_index)
    exog_aligned = exog_aligned.fillna(0)

    return exog_aligned


# ---------------------------------------------------------------------
# Helper: path handling for DL models
# ---------------------------------------------------------------------
def get_model_path(
    model_type: str,
    store_nbr: int,
    family: str,
    window: int,
) -> Path:
    """
    Build a cache path for a trained Keras model.

    UI mapping:
    - model_type  : "LSTM" / "TCN" / "TRANSFORMER"
    - store_nbr   : selected store from UI
    - family      : selected product family from UI
    - window      : "DL Context Window" input from UI
    """
    safe_family = str(family).replace(" ", "_").replace("/", "_")
    filename = f"{model_type.lower()}_st{store_nbr}_fam_{safe_family}_win{window}.keras"
    return MODELS_DIR / filename


# ---------------------------------------------------------------------
# ARIMA (non-seasonal)
# ---------------------------------------------------------------------
def run_arima_forecast(
    series: pd.Series,
    steps: int = 28,
    exog: Optional[pd.DataFrame] = None,
    exog_future: Optional[pd.DataFrame] = None,
    order: Tuple[int, int, int] = (1, 1, 1),
    **kwargs,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Fit a **non-seasonal** ARIMA model and return forecast & intervals.

    Parameters
    ----------
    series :
        Target time series (daily sales) for one store + family.
        The Streamlit app passes only the last `training_window` days.
    steps :
        Forecast horizon in days (UI: "Forecast Horizon").
    exog :
        Historical exogenous data aligned with `series` (e.g., onpromotion).
    exog_future :
        Future exogenous values (length = steps).
    order :
        (p, d, q) triple used for ARIMA. The UI currently passes (1,1,1)
        by default, but you can change it later if needed.
    kwargs :
        Ignored; allows us to safely swallow things like `seasonal=` if
        someone accidentally passes them in.

    Notes
    -----
    - This is intentionally **non-seasonal**. For weekly patterns, use SARIMAX.
    """

    if "seasonal" in kwargs:
        # Just a soft warning in console, nothing visible to user
        print(f"[run_arima_forecast] Ignoring seasonal parameter in ARIMA mode: {kwargs['seasonal']}")

    # 1. Prepare Target
    series = _prepare_series(series, freq="D")

    # 2. Align Exogenous
    exog = _align_exog(exog, series.index)

    # 3. Log-transform
    series_log = np.log1p(series)

    # 4. Fit Model (robust, with fallbacks)
    try:
        model = fit_arima(series=series_log, order=order, exog=exog)
    except Exception as e:
        print(f"[run_arima_forecast] ARIMA fit failed with order={order} and exog. Error: {e}")
        try:
            model = fit_arima(series=series_log, order=order, exog=None)
        except Exception as e2:
            print(f"[run_arima_forecast] ARIMA fit failed again without exog. Error: {e2}")
            # Last-resort default
            model = fit_arima(series=series_log, order=(1, 1, 1), exog=None)

    # 5. Forecast
    mean_log, conf_int_log = forecast_arima(
        model=model,
        steps=steps,
        exog_future=exog_future,
    )

    # 6. Back-transform
    mean = pd.Series(np.expm1(mean_log.values), index=mean_log.index)
    conf_int = pd.DataFrame(
        {
            "lower": np.expm1(conf_int_log.iloc[:, 0].values),
            "upper": np.expm1(conf_int_log.iloc[:, 1].values),
        },
        index=conf_int_log.index,
    )

    return mean, conf_int


# ---------------------------------------------------------------------
# SARIMAX (seasonal ARIMA)
# ---------------------------------------------------------------------
def run_sarimax_forecast(
    series: pd.Series,
    steps: int = 28,
    exog: Optional[pd.DataFrame] = None,
    exog_future: Optional[pd.DataFrame] = None,
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal: Tuple[int, int, int, int] = (1, 1, 1, 7),
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Fit a SARIMAX model and return forecast & intervals.

    Directly controlled by the **Advanced SARIMAX Orders** UI section.

    Parameters
    ----------
    series :
        Target series (daily sales).
    steps :
        Forecast horizon in days.
    exog :
        Historical exogenous variables aligned with `series`.
    exog_future :
        Future exogenous values for the forecast horizon.
    order :
        (p, d, q) as chosen in UI.
    seasonal :
        (P, D, Q, s) as chosen in UI (s=7 for weekly seasonality).
    """

    # 1. Prepare target
    series = _prepare_series(series, freq="D")

    # 2. Align exog
    exog = _align_exog(exog, series.index)

    # 3. Log-transform
    series_log = np.log1p(series)

    # 4. Fit with user-specified orders, fallback if too aggressive
    try:
        model = fit_sarimax(
            series=series_log,
            order=order,
            seasonal=seasonal,
            exog=exog,
        )
    except Exception as e:
        print(f"[run_sarimax_forecast] Strict SARIMAX fit failed: {e}. "
              "Falling back to (1,1,1)x(0,1,1,7).")
        model = fit_sarimax(
            series=series_log,
            order=(1, 1, 1),
            seasonal=(0, 1, 1, 7),
            exog=exog,
        )

    # 5. Forecast
    mean_log, conf_int_log = forecast_arima(
        model=model,
        steps=steps,
        exog_future=exog_future,
    )

    # 6. Back-transform
    mean = pd.Series(np.expm1(mean_log.values), index=mean_log.index)
    conf_int = pd.DataFrame(
        {
            "lower": np.expm1(conf_int_log.iloc[:, 0].values),
            "upper": np.expm1(conf_int_log.iloc[:, 1].values),
        },
        index=conf_int_log.index,
    )

    return mean, conf_int


# ---------------------------------------------------------------------
# Deep-learning models + MC Dropout
# ---------------------------------------------------------------------
def _train_dl_model(
    model_type: str,
    series: pd.Series,
    window: int = 60,
    epochs: int = 10,
):
    """
    Train a DL model from scratch for the given series.
    """
    model_type = model_type.upper()
    try:
        if model_type == "LSTM":
            from . import model_lstm as mdl
            model = mdl.train_lstm(series, window=window, epochs=epochs)
        elif model_type == "TCN":
            from . import model_tcn as mdl
            model = mdl.train_tcn(series, window=window, epochs=epochs)
        elif model_type in {"TRANSFORMER", "TINY_TRANSFORMER"}:
            from . import model_transformer as mdl
            model = mdl.train_transformer(series, window=window, epochs=epochs)
        else:
            raise ValueError(f"Unknown DL model_type: {model_type}")
    except ImportError as e:
        raise ImportError("TensorFlow required for DL models.") from e
    return model


def _ensure_trained_model(
    model_type: str,
    series: pd.Series,
    store_nbr: int,
    family: str,
    window: int = 60,
    epochs: int = 10,
):
    """
    Load a cached DL model if available, otherwise train & cache it.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = get_model_path(model_type, store_nbr, family, window)

    try:
        from tensorflow.keras.models import load_model
    except ImportError:
        raise ImportError("TensorFlow is required for DL models.")

    if path.exists():
        try:
            model = load_model(path)
            return model, path
        except Exception:
            print("[_ensure_trained_model] Cached model corrupted, retraining...")

    model = _train_dl_model(model_type, series, window=window, epochs=epochs)
    model.save(path)
    return model, path


def run_dl_forecast_with_uncertainty(
    model_type: str,
    series: pd.Series,
    store_nbr: int,
    family: str,
    window: int = 60,
    n_samples: int = 50,
    epochs: int = 10,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Run a DL forecast with Monte Carlo Dropout to estimate uncertainty.

    Returns
    -------
    mean_s : mean forecast
    lower_s : lower 95% bound
    upper_s : upper 95% bound
    std_s :   std dev per forecast step
    """

    # Enforce daily freq
    series = _prepare_series(series, freq="D")

    if len(series) < window:
        raise ValueError(
            f"Series length ({len(series)}) is shorter than window={window}. "
            "Increase history or reduce the window size."
        )

    model, _ = _ensure_trained_model(
        model_type=model_type,
        series=series,
        store_nbr=store_nbr,
        family=family,
        window=window,
        epochs=epochs,
    )

    # Prepare last `window` points as input sequence
    arr = np.asarray(series[-window:], dtype="float32")
    x_input = arr.reshape((1, window, 1))

    # MC Dropout predictions
    mean_pred, std_pred = mc_dropout_predict(model, x_input, n_samples=n_samples)
    mean_pred = mean_pred.flatten()
    std_pred = std_pred.flatten()

    z = 1.96  # for ~95% interval
    lower = mean_pred - z * std_pred
    upper = mean_pred + z * std_pred

    # Future dates, independent of horizon (UI slices to needed length)
    last_date = series.index[-1]
    forecast_index = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=len(mean_pred),
        freq=series.index.freq,
    )

    mean_s = pd.Series(mean_pred, index=forecast_index)
    lower_s = pd.Series(lower, index=forecast_index)
    upper_s = pd.Series(upper, index=forecast_index)
    std_s = pd.Series(std_pred, index=forecast_index)

    return mean_s, lower_s, upper_s, std_s
