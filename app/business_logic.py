"""
Business logic helpers.

This file focuses on the connection between forecasts and inventory decisions.

Right now we implement:
1. Safety stock calculation
2. Reorder point calculation
3. Simple financial impact calculations

Extendable with:
- Holding cost
- Stockout cost
- Service-level optimization
- More advanced financial models
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


# ---------------------------------------------------------------------------
# Financial helpers
# ---------------------------------------------------------------------------

def compute_inventory_financials(
    safety_stock: float,
    reorder_point: float,
    unit_cost: float,
    holding_cost_rate: float,
) -> dict:
    """
    Compute simple financial metrics for an inventory policy.

    Parameters
    ----------
    safety_stock : float
        Safety stock in units.
    reorder_point : float
        Reorder point in units.
    unit_cost : float
        Purchase cost per unit (in currency, e.g. dollars).
    holding_cost_rate : float
        Annual holding cost rate as a fraction of unit value.
        Example: 0.25 = 25% of inventory value per year.

    Returns
    -------
    dict
        Dictionary with:
        - 'safety_stock_value'
        - 'reorder_point_value'
        - 'annual_carrying_cost_safety_stock'
        - 'annual_carrying_cost_rop'
    """
    # Value of inventory positions (in currency)
    safety_stock_value = safety_stock * unit_cost
    reorder_point_value = reorder_point * unit_cost

    # Annual carrying cost (very simple approximation)
    annual_carrying_cost_safety_stock = safety_stock_value * holding_cost_rate
    annual_carrying_cost_rop = reorder_point_value * holding_cost_rate

    return {
        "safety_stock_value": safety_stock_value,
        "reorder_point_value": reorder_point_value,
        "annual_carrying_cost_safety_stock": annual_carrying_cost_safety_stock,
        "annual_carrying_cost_rop": annual_carrying_cost_rop,
    }


def compute_expected_stockout_cost(
    mean_daily_demand: float,
    lead_time_days: float,
    service_level: float,
    stockout_cost_per_unit: float,
) -> float:
    """
    Approximate expected stockout cost per replenishment cycle.

    Very simple, dashboard-oriented model:
    - Probability of a stockout during a cycle ~= (1 - service_level)
    - If a stockout happens, it affects roughly
      mean_daily_demand * lead_time_days units.
    - Each unit of unmet demand has a cost (lost margin, rush shipment, penalty).

    This is meant for illustrative purposes on the dashboard, not detailed
    operations planning.

    Parameters
    ----------
    mean_daily_demand : float
        Average demand per day.
    lead_time_days : float
        Lead time in days.
    service_level : float
        Target cycle service level between 0 and 1 (e.g. 0.95).
    stockout_cost_per_unit : float
        Monetary cost per unit of unsatisfied demand.

    Returns
    -------
    float
        Expected stockout cost per cycle (in currency).
    """
    expected_units_in_lead_time = mean_daily_demand * lead_time_days

    # Clamp service level into [0, 1] just to be safe
    service_level_clamped = max(0.0, min(1.0, service_level))
    prob_stockout = 1.0 - service_level_clamped

    expected_stockout_units = prob_stockout * expected_units_in_lead_time
    expected_stockout_cost = expected_stockout_units * stockout_cost_per_unit

    return expected_stockout_cost
