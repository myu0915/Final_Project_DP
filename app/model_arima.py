"""
ARIMA / SARIMAX helper functions.

This file does four main things:
1. Fit an ARIMA (non-seasonal) model on a single time series.
2. Fit a SARIMAX (seasonal ARIMA with exogenous variables) model.
3. Save / load fitted models to/from disk.
4. Make forecasts with confidence intervals.

How it connects to the UI
-------------------------
- The Streamlit **Forecast** page calls:
    - `fit_arima` indirectly via `run_arima_forecast(...)` (non-seasonal baseline).
    - `fit_sarimax` via `run_sarimax_forecast(...)` with the user-provided
      (p, d, q) and (P, D, Q, s).
    - `forecast_model` (aliased as `forecast_arima`) to generate forecast
      means and confidence intervals.

Notes
-----
- ARIMA is useful as a simple, fast, non-seasonal baseline.
- SARIMAX is more powerful: it supports seasonality and exogenous variables
  (promotions, holidays, calendar features, etc.).
"""

import pickle
from typing import Optional, Tuple, Union

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults

# A union type: either ARIMAResults or SARIMAXResults.
ModelResults = Union[ARIMAResults, SARIMAXResults]


# =====================================================================
# 1. ARIMA (non-seasonal) fitting
# =====================================================================
def fit_arima(
    series: pd.Series,
    order: Tuple[int, int, int] = (1, 1, 1),
    exog: Optional[pd.DataFrame] = None,
) -> ARIMAResults:
    """
    Fit a non-seasonal ARIMA model.

    Parameters
    ----------
    series : pd.Series
        The time series we want to model. Example: daily sales for one store-family.
        The index should already be in time order (DatetimeIndex or similar).
        (The app takes care of preparation & frequency before calling this.)
    order : tuple of 3 ints (p, d, q), optional
        Non-seasonal ARIMA order:
            p = autoregressive order (number of AR lags)
            d = differencing order
            q = moving-average order
        In the current UI, ARIMA is kept simple and usually uses the default (1,1,1),
        but you can override this in notebook experiments.
    exog : pd.DataFrame or None, optional
        Extra features aligned with the series (for example: promotions, holidays).
        Can be None if we do not use exogenous variables.

    Returns
    -------
    ARIMAResults
        A fitted ARIMA model that we can later use to forecast.
    """
    model = ARIMA(
        series,
        order=order,
        exog=exog,
    )
    # disp=0 behaviour is now controlled via fit options internally; using defaults
    results: ARIMAResults = model.fit()
    return results


# =====================================================================
# 2. SARIMAX (seasonal ARIMA) fitting
# =====================================================================
def fit_sarimax(
    series: pd.Series,
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal: Tuple[int, int, int, int] = (1, 1, 1, 7),
    exog: Optional[pd.DataFrame] = None,
) -> SARIMAXResults:
    """
    Fit a seasonal ARIMA (SARIMAX) model.

    Parameters
    ----------
    series : pd.Series
        The time series we want to model. Example: daily sales for one store-family.
        The index should be in time order (DatetimeIndex or similar). The app
        enforces daily frequency and fills gaps before calling this.
    order : tuple of 3 ints (p, d, q), optional
        Non-seasonal ARIMA order:
            p = autoregressive order
            d = differencing order
            q = moving-average order
        In the UI, these are read directly from (p, d, q) inputs.
    seasonal : tuple of 4 ints (P, D, Q, s), optional
        Seasonal ARIMA order:
            P = seasonal autoregressive order
            D = seasonal differencing order
            Q = seasonal moving-average order
            s = length of the season (e.g., 7 for weekly seasonality)
        In the UI, these come from (P, D, Q, s) inputs. The default (1,1,1,7)
        matches the guide we show to the user.
    exog : pd.DataFrame or None, optional
        Extra features aligned with the series (for example: promotions, holidays).
        Can be None if we do not use exogenous variables.
        The app takes care of aligning this to the series index.

    Returns
    -------
    SARIMAXResults
        A fitted SARIMAX model that we can later use to forecast.
    """
    model = SARIMAX(
        series,
        exog=exog,
        order=order,
        seasonal_order=seasonal,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    results: SARIMAXResults = model.fit(disp=False)
    return results


# =====================================================================
# 3. Save / load fitted models
# =====================================================================
def save_model(model: ModelResults, path: str) -> None:
    """
    Save a fitted ARIMA/SARIMAX model to a file using pickle.

    Parameters
    ----------
    model : ARIMAResults or SARIMAXResults
        The fitted model returned by fit_arima or fit_sarimax.
    path : str
        File path where the model will be saved,
        for example: "models/arima_store1_AUTOMOTIVE.pkl".
    """
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str) -> ModelResults:
    """
    Load a fitted ARIMA/SARIMAX model from a file.

    Parameters
    ----------
    path : str
        File path where the model was previously saved.

    Returns
    -------
    model : ARIMAResults or SARIMAXResults
        The fitted model, ready to be used for forecasting.
    """
    with open(path, "rb") as f:
        model: ModelResults = pickle.load(f)
    return model


# Backwards-compatible aliases if other code still uses these names
save_arima = save_model
load_arima = load_model


# =====================================================================
# 4. Forecast helpers
# =====================================================================
def forecast_model(
    model: ModelResults,
    steps: int,
    exog_future: Optional[pd.DataFrame] = None,
):
    """
    Use a fitted ARIMA/SARIMAX model to forecast future values.

    Parameters
    ----------
    model : ARIMAResults or SARIMAXResults
        The fitted model from fit_arima / fit_sarimax (or loaded with load_model).
    steps : int
        How many time steps ahead to forecast. Example: 28 for 28 days.
        In the UI this is the "Forecast horizon (days)" slider.
    exog_future : pd.DataFrame or None, optional
        Future exogenous values (for the forecast period).
        - If the model was trained with exogenous variables, we must provide them
          here with the same columns and **length = steps**.
        - If the model was trained without exogenous variables, this should be None.

    Returns
    -------
    mean : pd.Series
        The mean (central) forecast for each future time step.
    conf_int : pd.DataFrame
        Confidence intervals for the forecast. Usually contains two columns:
        one for the lower bound and one for the upper bound of the forecast.

    Raises
    ------
    ValueError
        If the model expects exogenous variables but exog_future is missing,
        or if exog_future length does not match `steps`.
    """

    # Basic sanity checks for exogenous variables
    model_uses_exog = getattr(model.model, "k_exog", 0) not in (None, 0)

    if model_uses_exog:
        if exog_future is None:
            raise ValueError(
                "This model was fitted with exogenous variables, "
                "but exog_future=None was provided to forecast_model(). "
                "The UI should build an exog_future DataFrame of length = steps."
            )
        if len(exog_future) != steps:
            raise ValueError(
                f"Length of exog_future ({len(exog_future)}) does not match "
                f"steps ({steps}). They must be equal."
            )
    else:
        # If model has no exog but caller passes something, we ignore it with a warning.
        if exog_future is not None:
            # Light-weight warning only; not raising to keep things user-friendly.
            print(
                "Warning: exog_future was provided, but the model was fitted "
                "without exogenous variables. exog_future will be ignored."
            )
            exog_future = None

    prediction = model.get_forecast(steps=steps, exog=exog_future)
    mean = prediction.predicted_mean
    conf_int = prediction.conf_int()
    return mean, conf_int


# Backwards-compatible alias used by inference.py
forecast_arima = forecast_model
