"""
Evaluation helpers.

This file provides:
1. Metric functions: MAPE, RMSE, MASE
2. A simple function to build a leaderboard DataFrame
3. A simple function placeholder for ablation study results

Later you can:
- Call these functions from notebooks.
- Save the leaderboard and ablation results to CSV in the results/ folder.
- Load those CSVs in the Streamlit "Model Leaderboard" page.
"""

from typing import Dict, List

import numpy as np
import pandas as pd


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error (MAPE).

    Note: we avoid division by zero by ignoring points where y_true == 0.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error (RMSE).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    insample: np.ndarray,
    seasonality: int = 1,
) -> float:
    """
    Mean Absolute Scaled Error (MASE).

    MASE compares your model against a naive seasonal forecast.

    Parameters
    ----------
    y_true : array
        Forecast horizon actuals.
    y_pred : array
        Forecast horizon predictions.
    insample : array
        In-sample (historical) data used to compute the naive seasonal error.
        Example: full training series.
    seasonality : int
        Seasonal period (for example, 7 for weekly seasonality in daily data).

    Returns
    -------
    float
        MASE value. Values < 1 mean the model is better than the naive forecast.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    insample = np.array(insample)

    if len(insample) <= seasonality:
        return np.nan

    # Naive seasonal errors: |Y_t - Y_{t-m}|
    naive_errors = np.abs(insample[seasonality:] - insample[:-seasonality])
    scale = np.mean(naive_errors)

    if scale == 0:
        return np.nan

    errors = np.abs(y_true - y_pred)
    return float(np.mean(errors / scale))


def build_leaderboard(
    metrics_per_model: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """
    Build a leaderboard DataFrame from a nested dictionary.

    Example input:
    metrics_per_model = {
        "ARIMA": {"MAPE": 12.3, "RMSE": 100.5, "MASE": 0.8},
        "LSTM":  {"MAPE": 10.1, "RMSE":  95.0, "MASE": 0.7},
    }

    Returns a DataFrame:

        model   MAPE   RMSE   MASE
        ARIMA   12.3   100.5  0.8
        LSTM    10.1   95.0   0.7

    You can then:
    - Save it as CSV in results/model_metrics.csv
    - Load it in Streamlit and display as a table.
    """
    rows = []
    for model_name, metric_dict in metrics_per_model.items():
        row = {"model": model_name}
        row.update(metric_dict)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by MAPE ascending if present
    if "MAPE" in df.columns:
        df = df.sort_values("MAPE")

    return df.reset_index(drop=True)


def build_ablation_table(
    feature_sets: List[str],
    mape_values: List[float],
) -> pd.DataFrame:
    """
    Simple helper to build an ablation study table.

    Example:
    feature_sets = [
        "All features",
        "No promotions",
        "No oil prices",
        "No holidays",
    ]
    mape_values = [10.5, 12.0, 11.3, 10.9]

    Returns:

        feature_set       MAPE
        All features      10.5
        No promotions     12.0
        No oil prices     11.3
        No holidays       10.9

    You can compute other metrics as well if you like.
    """
    df = pd.DataFrame(
        {"feature_set": feature_sets, "MAPE": mape_values}
    )

    df = df.sort_values("MAPE")
    return df.reset_index(drop=True)
