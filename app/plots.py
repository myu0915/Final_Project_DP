"""
Plotting helpers used in Streamlit.

This file provides simple functions for:
- Plotting actual vs forecast values
- Displaying confidence intervals (if available)

We keep this minimal so everything works nicely inside Streamlit.
"""

import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional


def plot_forecast(
    actual: Optional[pd.Series],
    forecast: pd.Series,
    lower: Optional[pd.Series] = None,
    upper: Optional[pd.Series] = None,
):
    """
    Create a line plot of actual values and forecasted values.

    Parameters
    ----------
    actual : pd.Series or None
        Historical data to plot. If None, we only show the forecast.
    forecast : pd.Series
        The forecasted future values.
    lower : pd.Series or None
        Lower bound of confidence interval. Optional.
    upper : pd.Series or None
        Upper bound of confidence interval. Optional.

    Returns
    -------
    fig : Matplotlib Figure
        Figure object that can be displayed in Streamlit using st.pyplot(fig).
    """

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 4))

    # Plot actual values (if provided)
    if actual is not None and len(actual) > 0:
        ax.plot(actual.index, actual.values, label="Actual", color="black")

    # Plot forecast values
    ax.plot(forecast.index, forecast.values, label="Forecast", color="blue")

    # Plot confidence interval if available
    if lower is not None and upper is not None:
        ax.fill_between(
            forecast.index,
            lower.values,
            upper.values,
            color="lightblue",
            alpha=0.4,
            label="Confidence Interval",
        )

    # Titles and labels
    ax.set_title("Forecast vs Actual")
    ax.set_xlabel("Time")
    ax.set_ylabel("Sales")

    ax.legend()
    fig.tight_layout()

    return fig
