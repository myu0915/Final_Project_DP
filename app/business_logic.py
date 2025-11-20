"""
Business logic helpers.

This file focuses on the connection between forecasts and inventory decisions.

Right now we implement:
1. Safety stock calculation
2. Reorder point calculation

Later you can extend this with:
- Holding cost
- Stockout cost
- Service-level optimization
"""


def compute_safety_stock(std_demand: float, z_value: float) -> float:
    """
    Compute safety stock given the demand uncertainty and desired service level.

    Parameters
    ----------
    std_demand : float
        Standard deviation of demand during the lead time.
        This often comes from forecast uncertainty (for example, from ARIMA
        confidence intervals or Monte Carlo dropout).
    z_value : float
        Z-score for the desired service level.
        Examples (approximate):
        - 0.84  → 80% service level
        - 1.28  → 90% service level
        - 1.65  → 95% service level
        - 2.05  → 98% service level

    Returns
    -------
    float
        Safety stock quantity.
    """
    safety_stock = z_value * std_demand
    return safety_stock


def compute_reorder_point(
    mean_daily_demand: float,
    lead_time_days: float,
    safety_stock: float,
) -> float:
    """
    Compute the reorder point (ROP).

    Concept:
    - During the lead time (supplier delay), we expect to sell
      mean_daily_demand * lead_time_days units.
    - To protect against uncertainty, we also keep safety_stock.

    Formula:
        Reorder Point = (mean demand per day * lead time in days) + safety stock

    Parameters
    ----------
    mean_daily_demand : float
        Average demand per day for the item.
        This can come from your forecast (28-day prediction turned into a daily mean).
    lead_time_days : float
        Supplier lead time in days (how long it takes to receive new inventory).
    safety_stock : float
        Safety stock computed from `compute_safety_stock`.

    Returns
    -------
    float
        The reorder point level. When inventory drops below this, you place an order.
    """
    reorder_point = mean_daily_demand * lead_time_days + safety_stock
    return reorder_point
