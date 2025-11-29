"""
Environment & plotting utilities for the project.

This module is mainly intended for use in Jupyter notebooks, for example:

    from app.utils_env import print_env_report, set_plot_style, free_memory

It:
- Sets random seeds for reproducibility
- Configures TensorFlow & logging
- Provides helper functions for:
    * printing environment info
    * setting a consistent plot style
    * cleaning up memory between model runs
"""

import os
import sys
import gc
import json
import random
import warnings
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Conv1D, BatchNormalization,
    GlobalAveragePooling1D, MultiHeadAttention, LayerNormalization,
    Add, Embedding, Activation
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Optional ipywidgets (for notebooks only)
try:
    import ipywidgets
    from ipywidgets import interact, FloatSlider, IntSlider
    WIDGETS = True
except ImportError:
    WIDGETS = False


# =====================================================================
# REPRODUCIBILITY — MAKING RESULTS AS REPEATABLE AS POSSIBLE
# =====================================================================

SEED = 123

# 1) Make Python's own hashing deterministic.
os.environ["PYTHONHASHSEED"] = str(SEED)

# 2) Python’s built-in random numbers.
random.seed(SEED)

# 3) NumPy's RNG.
np.random.seed(SEED)

# 4) TensorFlow RNG.
tf.random.set_seed(SEED)

# 5) Tell TensorFlow to be more deterministic.
os.environ["TF_DETERMINISTIC_OPS"] = "1"

# Reduce TensorFlow's log spam.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# =====================================================================
# WARNINGS — CLEANING UP NOISY OUTPUT
# =====================================================================

warnings.filterwarnings("ignore")
warnings.filterwarnings(
    "ignore",
    message=r"The name tf\.losses\.sparse_softmax_cross_entropy is deprecated"
)

pd.options.display.float_format = "{:,.2f}".format


# =====================================================================
# ENVIRONMENT REPORT
# =====================================================================

def print_env_report():
    """Print basic environment info (Python, Pandas, NumPy, TensorFlow, GPU)."""
    print(f"Python: {sys.version.split(' ')[0]}")
    print(f"Pandas: {pd.__version__}")
    print(f"Numpy: {np.__version__}")
    print(f"TensorFlow: {tf.__version__}")

    gpu_devices = tf.config.list_physical_devices("GPU")
    if gpu_devices:
        for g in gpu_devices:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass
        print(f"GPU detected: {gpu_devices[0].name}")
    else:
        print("No GPU detected. Running on CPU.")


# =====================================================================
# PLOTTING STYLE
# =====================================================================

def set_plot_style():
    """
    Configure Matplotlib + Seaborn so that all plots look clean and consistent.
    """
    try:
        sns.set_theme(context="notebook", style="whitegrid")
    except Exception:
        pass

    plt.style.use("fivethirtyeight")

    plt.rcParams["figure.figsize"] = (12, 7)
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["figure.dpi"] = 110


# =====================================================================
# MEMORY CLEANUP
# =====================================================================

def free_memory():
    """
    Manually free memory.
    Useful if you build many models one after another in a notebook.
    """
    gc.collect()
    try:
        tf.keras.backend.clear_session()
    except Exception:
        pass
