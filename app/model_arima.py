"""
ARIMA / SARIMAX helper functions.

This file does four main things:
1. Fit an ARIMA/SARIMAX model on a single time series.
2. Save the fitted model to disk.
3. Load the fitted model from disk.
4. Make a forecast with confidence intervals.
"""

import pickle
from typing import Optional, Tuple

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults


def fit_arima(
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
        The index should be in time order (e.g., datetime index).
    order : tuple of 3 ints (p, d, q)
        Non-seasonal ARIMA order:
        p = autoregressive order
        d = differencing order
        q = moving-average order
    seasonal : tuple of 4 ints (P, D, Q, s)
        Seasonal ARIMA order:
        P = seasonal autoregressive order
        D = seasonal differencing order
        Q = seasonal moving-average order
        s = length of the season (e.g., 7 for weekly seasonality)
    exog : pd.DataFrame or None
        Extra features aligned with the series (for example: promotions, holidays).
        Can be None if we do not use exogenous variables.

    Returns
    -------
    SARIMAXResults
        A fitted SARIMAX (ARIMA) model that we can later use to forecast.
    """

    # Create the SARIMAX model object.
    # enforce_stationarity/enforce_invertibility=False gives the optimizer more freedom.
    model = SARIMAX(
        series,
        exog=exog,
        order=order,
        seasonal_order=seasonal,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    # Fit (train) the model on the provided series.
    # disp=False hides the optimization output, so the console is not noisy.
    results = model.fit(disp=False)

    # Return the fitted model.
    return results


def save_arima(model: SARIMAXResults, path: str) -> None:
    """
    Save a fitted ARIMA/SARIMAX model to a file using pickle.

    Parameters
    ----------
    model : SARIMAXResults
        The fitted model returned by fit_arima.
    path : str
        File path where the model will be saved, for example: "models/arima_1_foo.pkl".
    """
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_arima(path: str) -> SARIMAXResults:
    """
    Load a fitted ARIMA/SARIMAX model from a file.

    Parameters
    ----------
    path : str
        File path where the model was previously saved.

    Returns
    -------
    SARIMAXResults
        The fitted model, ready to be used for forecasting.
    """
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def forecast_arima(
    model: SARIMAXResults,
    steps: int,
    exog_future: Optional[pd.DataFrame] = None,
):
    """
    Use a fitted ARIMA/SARIMAX model to forecast future values.

    Parameters
    ----------
    model : SARIMAXResults
        The fitted model from fit_arima (or loaded with load_arima).
    steps : int
        How many time steps ahead to forecast. Example: 28 for 28 days.
    exog_future : pd.DataFrame or None
        Future exogenous values (for the forecast period).
        If the model was trained with exogenous variables, we must provide them here
        with the same columns and length = steps. If no exogenous variables were used,
        this can be None.

    Returns
    -------
    mean : pd.Series
        The mean (central) forecast for each future time step.
    conf_int : pd.DataFrame
        Confidence intervals for the forecast. Usually contains two columns:
        one for the lower bound and one for the upper bound of the forecast.
    """

    # get_forecast returns an object with both the mean forecast and the uncertainty.
    prediction = model.get_forecast(steps=steps, exog=exog_future)

    # Mean forecast values (what we usually plot as the forecast line).
    mean = prediction.predicted_mean

    # Confidence intervals (lower and upper bounds).
    conf_int = prediction.conf_int()

    return mean, conf_int
