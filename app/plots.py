"""
Plotting utilities for the retail demand forecasting project.

These functions are used by:
- Jupyter notebooks (EDA, model analysis)
- Streamlit app (forecast visualization)

Matplotlib is the base plotting library.
Seaborn is optional and used only for nicer bar/box plots in EDA.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - optional dependency
    sns = None


# -------------------------------------------------------------------------
# Global style helpers
# -------------------------------------------------------------------------
def _ensure_datetime_index(series: pd.Series) -> pd.Series:
    """Make sure a Series has a DatetimeIndex (used in several plots)."""
    if not isinstance(series.index, pd.DatetimeIndex):
        s = series.copy()
        s.index = pd.to_datetime(s.index)
        return s
    return series


def _apply_base_style() -> None:
    """
    Apply a clean, analytics-oriented style.

    Called inside plotting functions so notebooks or other code don't have to
    remember to set it manually.
    """
    plt.style.use("default")
    if sns is not None:
        sns.set_theme(style="whitegrid")


# -------------------------------------------------------------------------
# Forecast plot (used by Streamlit UI)
# -------------------------------------------------------------------------
def plot_forecast(
    actual: pd.Series,
    forecast: pd.Series,
    lower: Optional[pd.Series] | None = None,
    upper: Optional[pd.Series] | None = None,
    title: str = "Forecast vs Actual",
    *,
    show_rolling: bool = True,
    rolling_window: int = 7,
    show_split_line: bool = True,
    ci_label: str = "Prediction interval",
    highlight_last_actual: bool = True,
) -> plt.Figure:
    """
    Plot actual history and forecast with optional confidence/prediction bands.

    This is the main plot used in the Streamlit app.

    Parameters
    ----------
    actual : pd.Series
        Historical actual values with a DatetimeIndex (or convertible to one).
    forecast : pd.Series
        Forecasted values with a DatetimeIndex (future dates).
    lower : pd.Series, optional
        Lower bound of confidence interval for forecast.
    upper : pd.Series, optional
        Upper bound of confidence interval for forecast.
    title : str, default "Forecast vs Actual"
        Plot title.
    show_rolling : bool, default True
        If True, overlay a smoothed rolling mean of the actuals to make
        the trend easier to read on noisy series.
    rolling_window : int, default 7
        Window size (in days) for the rolling mean.
    show_split_line : bool, default True
        If True, draw a vertical dashed line at the first forecast date.
    ci_label : str, default "Prediction interval"
        Legend label for the shaded confidence band.
    highlight_last_actual : bool, default True
        If True, marks the last historical point (forecast anchor) with a dot.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    _apply_base_style()

    # Ensure datetime indexes so Matplotlib formats ticks correctly
    actual = _ensure_datetime_index(actual)
    forecast = _ensure_datetime_index(forecast)
    if lower is not None:
        lower = _ensure_datetime_index(lower)
    if upper is not None:
        upper = _ensure_datetime_index(upper)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Colours chosen to work on both light/dark backgrounds
    actual_color = "#60a5fa"         # blue-400
    actual_smooth_color = "#1d4ed8"  # blue-700
    forecast_color = "#f97316"       # orange-400
    ci_color = "#bfdbfe"             # blue-200
    last_point_color = "#0f172a"     # slate-900

    # Plot actuals (daily)
    ax.plot(
        actual.index,
        actual.values,
        label="Actual (daily)",
        color=actual_color,
        linewidth=1.4,
        alpha=0.9,
    )

    # Optional rolling mean for smoother trend
    if show_rolling and len(actual) >= max(rolling_window, 3):
        smooth = (
            actual.rolling(
                window=rolling_window,
                min_periods=max(2, rolling_window // 2),
            )
            .mean()
        )
        ax.plot(
            smooth.index,
            smooth.values,
            label=f"Actual (rolling {rolling_window}d)",
            color=actual_smooth_color,
            linewidth=2.0,
        )

    # If requested, highlight last historical point
    if highlight_last_actual and len(actual) > 0:
        last_idx = actual.index[-1]
        ax.scatter(
            [last_idx],
            [actual.iloc[-1]],
            color=last_point_color,
            s=40,
            zorder=5,
            label="Last observed",
        )

    # Plot forecast line
    ax.plot(
        forecast.index,
        forecast.values,
        label="Forecast",
        color=forecast_color,
        linewidth=2.0,
    )

    # Prediction interval shading
    if lower is not None and upper is not None:
        ax.fill_between(
            forecast.index,
            lower.values,
            upper.values,
            alpha=0.25,
            color=ci_color,
            label=ci_label,
        )

    # Vertical split between history and forecast
    if show_split_line and len(forecast) > 0:
        split_date = forecast.index.min()
        ax.axvline(
            split_date,
            linestyle="--",
            linewidth=1.2,
            color="#6b7280",  # gray-500
            alpha=0.9,
        )
        # Annotate once at top-right just beside the line
        y_top = ax.get_ylim()[1]
        ax.text(
            split_date,
            y_top,
            " forecast start",
            color="#6b7280",
            fontsize=9,
            ha="left",
            va="top",
        )

    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Sales (units)", fontsize=11)

    ax.grid(True, which="major", alpha=0.3)
    ax.legend(frameon=True, framealpha=0.9)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


# -------------------------------------------------------------------------
# Simple time-series plot for EDA
# -------------------------------------------------------------------------
def plot_sales_over_time(
    df: pd.DataFrame,
    date_col: str = "date",
    sales_col: str = "sales",
    title: str = "Sales Over Time",
    *,
    freq: Optional[str] = None,
    show_rolling: bool = True,
    rolling_window: int = 7,
) -> plt.Figure:
    """
    Simple time-series plot of sales over time.

    Used for EDA to inspect:
    - Trend
    - Seasonality
    - Spikes and anomalies

    Parameters
    ----------
    df : pd.DataFrame
        Input data frame.
    date_col : str, default "date"
        Column containing dates.
    sales_col : str, default "sales"
        Column containing sales volumes.
    title : str, default "Sales Over Time"
        Plot title.
    freq : str, optional
        If provided, resamples the series using this pandas frequency string
        (e.g. 'W' for weekly, 'M' for monthly).
    show_rolling : bool, default True
        Whether to overlay a rolling mean.
    rolling_window : int, default 7
        Window for rolling mean (in time steps after resampling).

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    _apply_base_style()

    df_sorted = df.copy()
    df_sorted[date_col] = pd.to_datetime(df_sorted[date_col])
    df_sorted = df_sorted.sort_values(date_col)

    ts = df_sorted.set_index(date_col)[sales_col]

    if freq is not None:
        ts = ts.resample(freq).sum()

    fig, ax = plt.subplots(figsize=(15, 5))

    ax.plot(ts.index, ts.values, linewidth=1.4, color="#4b5563", label="Sales")

    if show_rolling and len(ts) >= max(rolling_window, 3):
        smooth = ts.rolling(
            window=rolling_window,
            min_periods=max(2, rolling_window // 2),
        ).mean()
        ax.plot(
            smooth.index,
            smooth.values,
            linewidth=2.0,
            color="#1d4ed8",
            label=f"Rolling {rolling_window}d",
        )

    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Unit Sales")
    ax.grid(True, which="major", alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


# -------------------------------------------------------------------------
# Average sales by day of week
# -------------------------------------------------------------------------
def plot_avg_sales_by_dow(
    df: pd.DataFrame,
    date_col: str = "date",
    sales_col: str = "sales",
    title: str = "Average Sales by Day of Week",
) -> Optional[plt.Figure]:
    """
    Bar plot of average sales by day of week.

    If Seaborn is installed, uses `sns.barplot` with a nicer style.
    Otherwise falls back to a simple Matplotlib bar chart.
    """
    _apply_base_style()

    df_dow = df.copy()
    df_dow[date_col] = pd.to_datetime(df_dow[date_col])
    df_dow["day_name"] = df_dow[date_col].dt.day_name()

    dow_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    grouped = (
        df_dow.groupby("day_name")[sales_col]
        .mean()
        .reindex(dow_order)
        .dropna()
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    if sns is not None:
        sns.barplot(
            x=grouped.index,
            y=grouped.values,
            ax=ax,
            palette="Blues",
        )
    else:
        ax.bar(grouped.index, grouped.values, color="#60a5fa")

    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Average Unit Sales")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


# -------------------------------------------------------------------------
# Promotion effect plot
# -------------------------------------------------------------------------
def plot_promo_effect(
    df: pd.DataFrame,
    promo_col: str = "onpromotion",
    sales_col: str = "sales",
    title: str = "Sales vs. Promotion",
) -> Optional[plt.Figure]:
    """
    Boxplot showing distribution of sales on promotion vs not on promotion.

    If Seaborn is available, uses `sns.boxplot`.
    Otherwise falls back to Matplotlib's `ax.boxplot`.
    """
    _apply_base_style()

    df_plot = df.copy()
    if df_plot[promo_col].dtype != "O":
        df_plot["promo_label"] = np.where(
            df_plot[promo_col].astype(bool), "Promo", "No promo"
        )
    else:
        df_plot["promo_label"] = df_plot[promo_col].astype(str)

    fig, ax = plt.subplots(figsize=(10, 5))

    if sns is not None:
        sns.boxplot(
            data=df_plot,
            x="promo_label",
            y=sales_col,
            ax=ax,
            palette="Set2",
        )
    else:
        groups = [g[sales_col].values for _, g in df_plot.groupby("promo_label")]
        labels = list(df_plot.groupby("promo_label").groups.keys())
        ax.boxplot(groups, labels=labels)
        ax.set_xlabel("Promotion status")

    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_xlabel("Is on Promotion?")
    ax.set_ylabel("Unit Sales")
    fig.tight_layout()
    return fig


# -------------------------------------------------------------------------
# Forecast error / residual plots (expert diagnostics)
# -------------------------------------------------------------------------
def plot_forecast_error(
    actual: pd.Series,
    forecast: pd.Series,
    *,
    title: str = "Forecast error (Actual  Forecast)",
) -> plt.Figure:
    """
    Plot forecast errors over time (expert-level diagnostic).

    Creates a two-row figure:

    - Top: Actual vs Forecast, restricted to the index overlap.
    - Bottom: Error series e_t = y_t - ŷ_t with a zero reference line.

    Parameters
    ----------
    actual : pd.Series
        Actual values with a DatetimeIndex.
    forecast : pd.Series
        Forecast values with a DatetimeIndex.
        Only the overlapping index range is used.
    title : str, default "Forecast error (Actual - Forecast)"
        Main figure title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    _apply_base_style()

    actual = _ensure_datetime_index(actual)
    forecast = _ensure_datetime_index(forecast)

    # Align on common dates
    common_idx = actual.index.intersection(forecast.index)
    if len(common_idx) == 0:
        raise ValueError("No overlapping dates between actual and forecast series.")

    y = actual.loc[common_idx]
    y_hat = forecast.loc[common_idx]
    err = y - y_hat

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.3]},
    )

    # Top: levels
    ax_top.plot(
        y.index,
        y.values,
        label="Actual",
        color="#4b5563",
        linewidth=1.5,
    )
    ax_top.plot(
        y_hat.index,
        y_hat.values,
        label="Forecast",
        color="#f97316",
        linewidth=1.8,
    )
    ax_top.set_ylabel("Sales (units)")
    ax_top.set_title(title, fontsize=14, weight="bold")
    ax_top.legend(frameon=True, framealpha=0.9)
    ax_top.grid(True, which="major", alpha=0.3)

    # Bottom: error series
    ax_bottom.plot(
        err.index,
        err.values,
        label="Error (y - ŷ)",
        color="#dc2626",  # red-600
        linewidth=1.4,
    )
    ax_bottom.axhline(0, color="#6b7280", linestyle="--", linewidth=1)
    ax_bottom.set_ylabel("Error")
    ax_bottom.set_xlabel("Date")
    ax_bottom.grid(True, which="major", alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_residual_distribution(
    actual: pd.Series,
    forecast: pd.Series,
    *,
    title: str = "Residual distribution",
    show_percentage: bool = True,
) -> plt.Figure:
    """
    Plot histogram (and density, if Seaborn is available) of forecast residuals.

    Residuals are defined as e_t = y_t - ŷ_t, computed on the overlapping
    part of `actual` and `forecast`.

    Optionally also plots the distribution of percentage errors.

    Parameters
    ----------
    actual : pd.Series
        Actual values with a DatetimeIndex.
    forecast : pd.Series
        Forecast values with a DatetimeIndex.
    title : str, default "Residual distribution"
        Figure title.
    show_percentage : bool, default True
        If True, a second subplot with percentage errors is drawn.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    _apply_base_style()

    actual = _ensure_datetime_index(actual)
    forecast = _ensure_datetime_index(forecast)

    common_idx = actual.index.intersection(forecast.index)
    if len(common_idx) == 0:
        raise ValueError("No overlapping dates between actual and forecast series.")

    y = actual.loc[common_idx].astype(float)
    y_hat = forecast.loc[common_idx].astype(float)
    resid = y - y_hat

    # Avoid division by zero for percentage errors
    denom = y.replace(0, np.nan)
    pct_err = 100.0 * resid / denom

    if show_percentage:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(6.5, 5))
        ax2 = None

    # Residual distribution
    if sns is not None:
        sns.histplot(resid.dropna(), kde=True, ax=ax1, color="#60a5fa")
    else:
        ax1.hist(resid.dropna(), bins=30, color="#60a5fa", alpha=0.8)
    ax1.set_xlabel("Residual (y - ŷ)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Residuals")

    # Percentage errors
    if show_percentage and ax2 is not None:
        valid_pct = pct_err.replace([np.inf, -np.inf], np.nan).dropna()
        if sns is not None:
            sns.histplot(valid_pct, kde=True, ax=ax2, color="#f97316")
        else:
            ax2.hist(valid_pct, bins=30, color="#f97316", alpha=0.8)
        ax2.set_xlabel("Percentage error (%)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Percentage errors")

    fig.suptitle(title, fontsize=14, weight="bold")
    fig.tight_layout()
    return fig
