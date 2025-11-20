"""
Uncertainty estimation helpers.

Goal:
- For Deep Learning models (LSTM, TCN, Transformer), we want to estimate
  prediction uncertainty using Monte Carlo Dropout.

Idea of Monte Carlo Dropout:
- Dropout is usually active only during training.
- For MC Dropout, we force dropout to stay ON at prediction time.
- We run N forward passes (for example, 50 or 100 times).
- Each pass gives a slightly different prediction.
- We then compute:
  - mean prediction  → "best guess"
  - standard deviation / percentiles → uncertainty
"""

import numpy as np


def mc_dropout_predict(
    model,
    x_input,
    n_samples: int = 50,
):
    """
    Run Monte Carlo Dropout for a given model and input.

    IMPORTANT:
    - This function assumes that the model has dropout layers.
    - It also assumes that there is some way to tell the model:
      "Please run in training mode so dropout is active."

    Right now, our LSTM/TCN/Transformer are only dummy placeholders,
    so this function is more of a TEMPLATE for later.

    Parameters
    ----------
    model :
        A Deep Learning model with dropout (Keras or PyTorch) that supports
        a "training=True" or similar option when calling it.
    x_input :
        Input batch or sequence for which we want predictions.
    n_samples : int
        How many stochastic forward passes to run.

    Returns
    -------
    mean_pred : np.ndarray
        The mean prediction across all MC samples.
    std_pred : np.ndarray
        The standard deviation of predictions (measure of uncertainty).
    all_preds : np.ndarray
        Array of shape (n_samples, ...) with all raw predictions.
    """

    # Placeholder behavior for now:
    # - We just call model.predict once and "fake" uncertainty.
    # - This keeps the pipeline from breaking.
    # - Later, you will replace this with real MC dropout loops.

    # Single prediction (dummy, since our models are placeholder).
    single_pred = model.predict(x_input)

    # Convert to numpy array for consistency.
    single_pred = np.array(single_pred)

    # Fake multiple samples by repeating the same prediction.
    all_preds = np.repeat(single_pred[None, ...], n_samples, axis=0)

    # Mean and std across the MC axis (axis=0).
    mean_pred = all_preds.mean(axis=0)
    std_pred = all_preds.std(axis=0)

    return mean_pred, std_pred, all_preds
