"""
Backtesting and time-series cross-validation utilities.

This module provides:
- A time-based train/test split for windowed DL data
- An expanding-window rolling CV index generator
"""

import numpy as np


def time_based_train_test_split(X, Y, T, train_frac: float = 0.8):
    """
    Time-based train/test split for windowed data.

    Parameters
    ----------
    X : np.ndarray
        Shape: (samples, lookback, features)
    Y : np.ndarray
        Shape: (samples, horizon)
    T : np.ndarray
        Shape: (samples,) timestamps for each sample
    train_frac : float
        Fraction of samples to use for training (e.g. 0.8).

    Returns
    -------
    X_train, X_test, Y_train, Y_test, T_train, T_test
    """
    n_samples = len(X)
    split_idx = int(n_samples * train_frac)

    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]
    T_train, T_test = T[:split_idx], T[split_idx:]

    return X_train, X_test, Y_train, Y_test, T_train, T_test


def rolling_cv_indices(n_train: int, n_folds: int = 3, step: int = 28):
    """
    Build index ranges for expanding-window cross-validation.

    This mirrors the notebook logic:

        - Start with an initial train window
        - Validate on the next `step` samples (usually = HORIZON)
        - Expand the train window forward and repeat

    Parameters
    ----------
    n_train : int
        Total number of training samples (windows).
    n_folds : int
        Number of CV folds to create.
    step : int
        Size of each validation block (e.g., horizon length).

    Returns
    -------
    indices : list of (train_slice, val_slice)
        Each element is a pair of Python slice objects that can be
        applied to X_train / Y_train / T_train.
    """
    indices = []

    # Initial training size: leave room for all validation folds
    initial_train_size = n_train - (n_folds * step)
    if initial_train_size < step * 2:
        print("Warning: Small dataset. You may need fewer folds or a smaller step.")
        initial_train_size = max(step * 2, n_train - n_folds * step)

    print(f"Initial train size: {initial_train_size} samples")

    for i in range(n_folds):
        train_end = initial_train_size + i * step
        val_end = train_end + step

        if val_end > n_train:
            print(f"Fold {i+1} exceeds data length. Stopping.")
            break

        train_slice = slice(0, train_end)
        val_slice = slice(train_end, val_end)
        indices.append((train_slice, val_slice))

    return indices
