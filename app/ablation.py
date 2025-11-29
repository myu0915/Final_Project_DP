"""
Ablation Study Module
---------------------

This module compares:
- Full-feature LSTM (all covariates)
- Lags-only LSTM (only sales lag / rolling features)

It uses the same rolling CV engine (fit_cv) for a fair comparison.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd

from app.training import fit_cv
from app.model_lstm import build_lstm


def _select_lag_feature_indices(feature_cols: List[str]) -> Tuple[List[int], List[str]]:
    """
    Identify indices of *lag-based* features among all feature columns.

    We treat these as lag-based:
        - columns starting with "sales_lag_"
        - columns starting with "roll_mean_"

    Parameters
    ----------
    feature_cols : list of str
        Ordered list of all feature names used to build X (axis=2).

    Returns
    -------
    lag_indices : list of int
        Positions of lag features along the last axis of X.
    lag_feature_names : list of str
        Names of the selected lag-based features.
    """
    lag_feature_names = [
        c
        for c in feature_cols
        if c.startswith("sales_lag_") or c.startswith("roll_mean_")
    ]

    if not lag_feature_names:
        raise ValueError(
            "No lag / rolling features found in feature_cols. "
            "Expected columns like 'sales_lag_7' or 'roll_mean_28'."
        )

    lag_indices = [feature_cols.index(c) for c in lag_feature_names]
    return lag_indices, lag_feature_names


def run_lag_ablation(
    feature_cols: List[str],
    X_train: np.ndarray,
    Y_train: np.ndarray,
    cv_folds: List[Tuple[slice, slice]],
    lookback: int,
    horizon: int,
    epochs: int,
    batch_size: int,
    leader_full_lstm: pd.DataFrame,
) -> Tuple[pd.DataFrame, float]:
    """
    Run an ablation experiment using only lag-based features.

    Parameters
    ----------
    feature_cols : list
        All feature names (strings) corresponding to axis=2 of X_train.
    X_train : np.ndarray
        Full training windows (samples, lookback, features).
    Y_train : np.ndarray
        Forecast targets, shape (samples, horizon).
    cv_folds : list of (slice, slice)
        Expanding-window cross-validation fold indices.
    lookback : int
        Historical window size (for logging only).
    horizon : int
        Forecast horizon (for logging only).
    epochs : int
        CV training epochs.
    batch_size : int
        CV training batch size.
    leader_full_lstm : pd.DataFrame
        Leaderboard results from the full-feature LSTM
        (val_MAPE / val_RMSE / val_MASE per fold).

    Returns
    -------
    leader_ablation : pd.DataFrame
        Leaderboard for the lags-only LSTM (same columns as leader_full_lstm).
    mape_diff : float
        Difference in mean validation MAPE:
            mape_lags_only - mape_full
        Positive means lags-only is worse than the full model.
    """
    # ------------------------------------------------------------------
    # 1. Select only lag-based features from X_train
    # ------------------------------------------------------------------
    n_samples, lb, n_features = X_train.shape
    assert lb == lookback, f"lookback mismatch: X has {lb}, got {lookback}"

    lag_indices, lag_feature_names = _select_lag_feature_indices(feature_cols)

    print("\n[ABLATION] Using only lag-based features:")
    for name in lag_feature_names:
        print(f"  - {name}")
    print(
        f"[ABLATION] Selected {len(lag_feature_names)} lag features "
        f"out of {n_features} total."
    )

    # Subset X_train over feature dimension
    X_lags = X_train[:, :, lag_indices]

    # ------------------------------------------------------------------
    # 2. Run CV training with only lag-based features
    # ------------------------------------------------------------------
    print(
        f"\n[ABLATION] Training LSTM with lags only "
        f"(lookback={lookback}, horizon={horizon}, epochs={epochs})"
    )

    leader_ablation, _, _ = fit_cv(
        build_fn=build_lstm,
        X_tr=X_lags,
        Y_tr=Y_train,
        folds=cv_folds,
        epochs=epochs,
        batch=batch_size,
        model_name="LSTM (Lags Only)",
    )

    # ------------------------------------------------------------------
    # 3. Compare results vs full-feature LSTM
    # ------------------------------------------------------------------
    full_mape = leader_full_lstm["val_MAPE"].mean()
    lag_mape = leader_ablation["val_MAPE"].mean()
    mape_diff = lag_mape - full_mape

    print("\n--- Ablation Study Summary ---")
    print(f"Full LSTM MAPE:  {full_mape:.2f}%")
    print(f"Lags-only MAPE: {lag_mape:.2f}%")
    print(f"Difference:      {mape_diff:.2f} percentage points worse.")

    return leader_ablation, mape_diff
