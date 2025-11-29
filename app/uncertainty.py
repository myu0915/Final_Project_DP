"""
MC Dropout utilities for deep learning models.

Used by:
- app.inference.run_dl_forecast_with_uncertainty

API:
    mean_pred, std_pred = mc_dropout_predict(model, x_input, n_samples=50)

where
    model    : Keras model with at least one Dropout layer
    x_input  : np.ndarray, shape (batch, timesteps, features)
    n_samples: number of stochastic forward passes

Returns
    mean_pred : np.ndarray, shape (batch, horizon)
    std_pred  : np.ndarray, shape (batch, horizon)
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf


@tf.function
def _mc_step(model, x_batch):
    """
    Single stochastic forward pass with dropout *enabled*.

    We force training=True so that Dropout stays ON and produces
    different outputs across calls.
    """
    return model(x_batch, training=True)


def mc_dropout_predict(model, x_input, n_samples: int = 50):
    """
    Run Monte Carlo Dropout for a trained Keras model.

    Parameters
    ----------
    model : tf.keras.Model
        Trained model with Dropout layers.
    x_input : np.ndarray
        Input batch of shape (batch_size, timesteps, features).
    n_samples : int, default 50
        Number of stochastic forward passes.

    Returns
    -------
    mean_pred : np.ndarray
        Mean prediction across MC samples, shape (batch, horizon).
    std_pred : np.ndarray
        Standard deviation across MC samples, shape (batch, horizon).
    """
    # Ensure numpy array
    x_input = np.asarray(x_input, dtype=np.float32)

    preds = []

    for _ in range(int(n_samples)):
        y = _mc_step(model, x_input)
        # Convert tensor → numpy, shape (batch, horizon)
        preds.append(y.numpy())

    # Stack along MC dimension: (n_samples, batch, horizon)
    preds = np.stack(preds, axis=0)

    # Mean & std across MC samples
    mean_pred = preds.mean(axis=0)
    std_pred = preds.std(axis=0)

    return mean_pred, std_pred
