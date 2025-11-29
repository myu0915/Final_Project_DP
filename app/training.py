"""
Training utilities for deep-learning time-series models.

This module provides:
- fit_cv: generic expanding-window cross-validation trainer for any Keras model
  builder (LSTM, TCN, Transformer, etc.).
"""

from typing import Callable, List, Tuple

import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping

from .evaluation import safe_mape, rmse, mase


def fit_cv(
    build_fn: Callable,
    X_tr: np.ndarray,
    Y_tr: np.ndarray,
    folds: List[Tuple[slice, slice]],
    epochs: int = 25,
    batch: int = 256,
    model_name: str = "Model",
):
    """
    Generic expanding-window cross-validation trainer.

    Parameters
    ----------
    build_fn :
        Function that builds and returns an uncompiled Keras model.
        Signature must be: build_fn(input_shape=(lookback, n_features),
                                   horizon=horizon)
    X_tr : np.ndarray
        Training input windows, shape (n_samples, lookback, n_features).
    Y_tr : np.ndarray
        Training targets, shape (n_samples, horizon).
    folds : list of (train_slice, val_slice)
        Output of your rolling_cv_indices() function. Each element is a pair
        of Python slice objects that index X_tr / Y_tr along axis 0.
    epochs : int
        Maximum number of epochs per fold.
    batch : int
        Batch size for model.fit.
    model_name : str
        Name used only for logging.

    Returns
    -------
    leaderboard_df : pd.DataFrame
        One row per fold with val_MAPE, val_RMSE, val_MASE.
    final_model : tf.keras.Model
        Model from the last fold (can be re-trained on all data later if you
        wish, or used directly as a reasonable final model).
    final_scaler : StandardScaler
        Scaler fitted on the training data of the last fold.
    """

    leaderboard = []        # metrics per fold
    models: List[tf.keras.Model] = []
    scalers: List[StandardScaler] = []

    # Infer dimensions from data instead of relying on globals
    lookback = X_tr.shape[1]
    n_features = X_tr.shape[2]
    horizon = Y_tr.shape[1]

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
    )

    for i, (tr_slice, val_slice) in enumerate(folds):
        print(f"\n--- {model_name} - Fold {i + 1}/{len(folds)} ---")

        # 1) Split into train/val for this fold
        X_train_fold = X_tr[tr_slice]
        Y_train_fold = Y_tr[tr_slice]
        X_val_fold = X_tr[val_slice]
        Y_val_fold = Y_tr[val_slice]

        # 2) Fit scaler on TRAIN only
        scaler_fold = StandardScaler()

        X_train_flat = X_train_fold.reshape(-1, n_features)
        scaler_fold.fit(X_train_flat)

        X_train_scaled = scaler_fold.transform(X_train_flat).reshape(X_train_fold.shape)
        X_val_scaled = scaler_fold.transform(
            X_val_fold.reshape(-1, n_features)
        ).reshape(X_val_fold.shape)

        # 3) Build a fresh model for this fold
        tf.keras.backend.clear_session()
        model = build_fn(input_shape=(lookback, n_features), horizon=horizon)

        model.fit(
            X_train_scaled,
            Y_train_fold,
            validation_data=(X_val_scaled, Y_val_fold),
            epochs=epochs,
            batch_size=batch,
            callbacks=[early_stop],
            verbose=0,
        )

        # 4) Validation predictions + metrics
        y_pred_val = model.predict(X_val_scaled, verbose=0)

        mape_val = safe_mape(Y_val_fold, y_pred_val)
        rmse_val = rmse(Y_val_fold, y_pred_val)
        mase_val = mase(Y_val_fold, y_pred_val, Y_tr[tr_slice].flatten())

        print(
            f"Fold {i + 1} Val MAPE: {mape_val:.2f}%  "
            f"RMSE: {rmse_val:.2f}  MASE: {mase_val:.2f}"
        )

        leaderboard.append(
            {
                "fold": i + 1,
                "val_MAPE": mape_val,
                "val_RMSE": rmse_val,
                "val_MASE": mase_val,
            }
        )
        models.append(model)
        scalers.append(scaler_fold)

        # Clean memory aggressively between folds
        del (
            X_train_fold,
            Y_train_fold,
            X_val_fold,
            Y_val_fold,
            X_train_scaled,
            X_val_scaled,
            X_train_flat,
        )
        gc.collect()

    leaderboard_df = pd.DataFrame(leaderboard)

    # Last model and scaler are returned as "final" ones
    final_model = models[-1]
    final_scaler = scalers[-1]

    return leaderboard_df, final_model, final_scaler
