"""
Inventory decision analytics:
- Safety Stock (SS)
- Reorder Point (ROP)

Works with:
- Forecast mean only  → deterministic mode
- Forecast mean + std → uncertainty-aware mode
"""

import numpy as np
from scipy.stats import norm


def calc_rop_ss(mean_forecast, std_forecast, lead_time_days, z_score):
    """
    Calculate Safety Stock (SS) and Reorder Point (ROP).

    Parameters
    ----------
    mean_forecast : array-like
        Daily forecast mean values for the horizon (e.g., length=28).
    std_forecast : array-like
        Daily standard deviations. If unavailable → use zeros.
    lead_time_days : int
        Supplier lead time in days.
    z_score : float
        Z-score for service level (e.g., 1.645 for 95%).

    Returns
    -------
    mu_L : float
        Expected demand during lead time.
    sigma_L : float
        Forecast std dev during lead time.
    safety_stock : float
        Z * sigma_L.
    reorder_point : float
        mu_L + safety_stock.
    """
    mean_forecast = np.asarray(mean_forecast)
    std_forecast = np.asarray(std_forecast)

    if np.any(np.isnan(mean_forecast)) or np.any(np.isnan(std_forecast)):
        return np.nan, np.nan, np.nan, np.nan

    # Demand during lead time
    mu_L = np.sum(mean_forecast[:lead_time_days])

    # Variance of sum of independent daily errors
    sigma_L = np.sqrt(np.sum(std_forecast[:lead_time_days] ** 2))

    # Safety stock
    safety_stock = z_score * sigma_L

    # Reorder point
    reorder_point = mu_L + safety_stock

    return mu_L, sigma_L, safety_stock, reorder_point


def z_from_service_level(service_level_pct):
    """
    Convert service level % → Z-score.
    """
    return norm.ppf(service_level_pct / 100.0)
