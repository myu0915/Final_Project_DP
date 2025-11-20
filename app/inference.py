"""
Inference helpers.

This file answers one main question:
"Given a model type and some data, how do I get a forecast?"

Right now we support:
- ARIMA (real, using statsmodels)
- LSTM / TCN / Transformer (dummy placeholders for now)
"""

from typing import Optional, Tuple

import pandas as pd

from .model_arima import fit_arima, forecast_arima
from .model_dl import get_model


def run_arima_forecast(
    series: pd.Series,
    steps: int = 28,
    exog: Optional[pd.DataFrame] = None,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Fit an ARIMA model on the given time series and forecast into the future.

    This is a simple, "one-shot" helper:
    - It fits the model.
    - It immediately makes a forecast.
    - It returns the forecast and the confidence interval.

    Parameters
    ----------
    series : pd.Series
        The historical target time series (for example: daily sales of one store-family).
        The index should be sorted by time (e.g., by date).
    steps : int
        How many time steps ahead we want to forecast. Default is 28 days.
    exog : pd.DataFrame or None
        Optional exogenous variables (future known inputs, like promotions or holidays).
        If used, exog must be aligned with `series` for fitting and have `steps` rows
        for forecasting.

    Returns
    -------
    mean : pd.Series
        The mean forecast for each of the future steps.
    conf_int : pd.DataFrame
        The confidence intervals for the forecast (lower and upper bounds).
    """

    # Fit the ARIMA / SARIMAX model on the past data.
    model = fit_arima(series=series, exog=exog)

    # Use the fitted model to forecast into the future.
    mean, conf_int = forecast_arima(model=model, steps=steps, exog_future=None)

    return mean, conf_int


def run_dl_forecast(
    model_name: str,
    x_input,
):
    """
    Run a forecast using one of the Deep Learning placeholder models.

    At this stage:
    - We do NOT have real trained DL models yet.
    - The dummy models simply return the input.

    This function is mainly here so the rest of the code (and Streamlit app)
    has a clean place to call when we plug in real LSTM / TCN / Transformer later.

    Parameters
    ----------
    model_name : str
        "LSTM", "TCN", or "Transformer".
    x_input : Any
        Dummy input for now. Later this will be your prepared time-series features.

    Returns
    -------
    Any
        For now, this just returns whatever the dummy model returns (usually x_input).
    """

    # Get the correct dummy model object based on the name.
    model = get_model(model_name)

    # Call the model's predict method.
    prediction = model.predict(x_input)

    return prediction


def run_forecast(
    model_type: str,
    series: Optional[pd.Series] = None,
    steps: int = 28,
    exog: Optional[pd.DataFrame] = None,
    x_input=None,
):
    """
    High-level helper that chooses which forecasting path to use.

    This function is a single entry point for your app:
    - If model_type == "ARIMA": use the ARIMA logic.
    - If model_type == "LSTM"/"TCN"/"Transformer": use the DL placeholder.

    Later, when you add real DL models, you can keep this interface the same.

    Parameters
    ----------
    model_type : str
        "ARIMA", "LSTM", "TCN", or "Transformer".
    series : pd.Series or None
        Time series for ARIMA. Required if model_type == "ARIMA".
    steps : int
        Forecast horizon for ARIMA.
    exog : pd.DataFrame or None
        Optional exogenous variables for ARIMA.
    x_input : Any
        Input for DL models (later: windowed time-series data).

    Returns
    -------
    result
        For ARIMA: (mean, conf_int)
        For DL models: placeholder prediction
    """

    model_type = mod_
