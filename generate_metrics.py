"""
Offline script to generate model performance metrics for the leaderboard.

Models:
- Naive_last_value  (simple baseline, single-series)
- ARIMA(5,1,0)      (classical baseline, single-series)
- LSTM_global_family (DL model trained on ALL stores for one family)

Output:
    results/model_metrics.csv

This CSV is read by the Streamlit "Model Leaderboard" page.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from app.data_utils import merge_full_dataset
from app.evaluation import safe_mape, rmse, mase, eval_arima_baseline
from app.features import create_features, prepare_feature_columns
from app.windows import make_windows
from app.backtesting import rolling_cv_indices
from app.training import fit_cv
from app.model_lstm import build_lstm
from app.config import MAX_GLOBAL_DL_WINDOWS


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
# For the naive & ARIMA single-series baselines
BASELINE_STORE = 1
BASELINE_FAMILY = "GROCERY I"

# Shared horizon
HORIZON = 28

# LSTM GLOBAL FAMILY TRAINING SETTINGS
LSTM_LOOKBACK = 60      # days of history per window
LSTM_EPOCHS = 15        # more conservative epochs for CPU
LSTM_BATCH = 256        # OK because we have fewer windows now
LSTM_FOLDS = 2          # 2-fold expanding CV to keep it fast


# =====================================================================
# Helpers for single-series baselines
# =====================================================================
def _get_store_family_series(
    df: pd.DataFrame, store_nbr: int, family: str
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Extract sales series and subset df for a specific store/family.
    Returns:
        series (pd.Series) sorted by date (index=date)
        subset_df (pd.DataFrame) sorted by date
    """
    subset = df[(df["store_nbr"] == store_nbr) & (df["family"] == family)].copy()

    if subset.empty:
        raise ValueError(f"No data found for store={store_nbr}, family={family}")

    subset["date"] = pd.to_datetime(subset["date"])
    subset = subset.sort_values("date")

    series = subset["sales"].astype("float32")
    series.index = subset["date"]

    return series, subset


def eval_naive_last_value_series(series: pd.Series, horizon: int) -> Dict[str, float]:
    """
    Naive baseline: forecast the last observed value from the training
    series for all future horizon days.
    """
    if len(series) <= horizon + 1:
        raise ValueError("Series too short for the requested horizon.")

    train = series.iloc[:-horizon]
    test = series.iloc[-horizon:]

    last_val = float(train.iloc[-1])
    y_pred = np.full_like(test.values, fill_value=last_val, dtype="float32")

    mape_val = float(safe_mape(test.values, y_pred))
    rmse_val = float(rmse(test.values, y_pred))
    mase_val = float(mase(test.values, y_pred, train.values))

    return {"MAPE": mape_val, "RMSE": rmse_val, "MASE": mase_val}


def eval_arima_series(series: pd.Series, horizon: int) -> Dict[str, float]:
    """
    Evaluate ARIMA(5,1,0) on the last `horizon` days of the series.
    """
    if len(series) <= horizon + 1:
        raise ValueError("Series too short for the requested horizon.")

    train = series.iloc[:-horizon]
    test = series.iloc[-horizon:]

    arima_metrics = eval_arima_baseline(train.values, test.values, order=(5, 1, 0))

    if arima_metrics is None:
        raise RuntimeError("ARIMA evaluation failed.")

    return {
        "MAPE": float(arima_metrics["MAPE"]),
        "RMSE": float(arima_metrics["RMSE"]),
        "MASE": float(arima_metrics["MASE"]),
    }


# =====================================================================
# GLOBAL FAMILY LSTM METRICS
# =====================================================================
def eval_lstm_family_global(
    df_full: pd.DataFrame,
    target_family: str,
    lookback: int,
    horizon: int,
    epochs: int,
    batch_size: int,
    n_folds: int,
) -> Dict[str, float]:
    """
    Train and evaluate a LSTM on ALL stores, but only for one product family.

    Steps:
      - Filter df_full to target_family
      - create_features()            → lag/rolling/calendar/holiday/etc.
      - prepare_feature_columns()    → feature_cols, scaler
      - make_windows()               → X, Y, T (all stores, one family)
      - (OPTIONAL) subsample windows to MAX_GLOBAL_DL_WINDOWS
      - rolling_cv_indices()         → folds
      - fit_cv(build_lstm, ...)      → per-fold val_MAPE/RMSE/MASE

    Metrics are averaged across folds.
    """
    df_fam = df_full[df_full["family"] == target_family].copy()
    if df_fam.empty:
        raise ValueError(f"No rows found for target_family='{target_family}'.")

    print(f"Creating features for GLOBAL LSTM on family='{target_family}' ...")
    df_feat, base_feature_cols = create_features(df_fam)

    df_proc, feature_cols, numeric_cols, bool_cols, scaler = prepare_feature_columns(
        df_feat, base_feature_cols
    )

    # Drop rows with NaNs in features/target (from lags/rolls, etc.)
    cols_to_check = feature_cols + ["sales"]
    cols_to_check = [c for c in cols_to_check if c in df_proc.columns]
    df_proc = df_proc.dropna(subset=cols_to_check).reset_index(drop=True)

    if len(df_proc) <= lookback + horizon:
        raise ValueError("Not enough rows after feature engineering for LSTM.")

    print(f"Feature-engineered rows available for {target_family}: {len(df_proc)}")

    print("Building windows for GLOBAL FAMILY LSTM ...")
    X, Y, T = make_windows(
        df_proc,
        lookback=lookback,
        horizon=horizon,
        feature_cols=feature_cols,
        target_col="sales",
    )

    n_samples = X.shape[0]
    print(f"LSTM window samples (all stores, family='{target_family}'): {n_samples}")

    # Optional CPU/RAM-friendly cap on number of windows
    if MAX_GLOBAL_DL_WINDOWS is not None and n_samples > MAX_GLOBAL_DL_WINDOWS:
        print(
            f"Subsampling from {n_samples} to {MAX_GLOBAL_DL_WINDOWS} windows "
            "for GLOBAL FAMILY LSTM to respect resource limits."
        )
        rng = np.random.default_rng(42)
        idx = rng.choice(n_samples, size=MAX_GLOBAL_DL_WINDOWS, replace=False)
        X = X[idx]
        Y = Y[idx]
        T = T[idx]
        n_samples = X.shape[0]
        print(
            f"LSTM window samples after subsampling: {n_samples} "
            f"(cap = {MAX_GLOBAL_DL_WINDOWS})"
        )

    if n_samples < n_folds * 2:
        # Reduce folds if data is small after subsampling
        n_folds = max(1, n_samples // (2 * horizon))
        print(f"Adjusting number of LSTM folds to {n_folds} due to limited data.")

    print("Creating rolling CV folds for GLOBAL FAMILY LSTM ...")
    folds = rolling_cv_indices(n_train=n_samples, n_folds=n_folds, step=horizon)
    if not folds:
        raise ValueError("No CV folds could be created for LSTM evaluation.")

    print(
        f"\nTraining GLOBAL FAMILY LSTM (all stores, family='{target_family}') "
        "for leaderboard ..."
    )
    leaderboard_df, _, _ = fit_cv(
        build_fn=build_lstm,
        X_tr=X,
        Y_tr=Y,
        folds=folds,
        epochs=epochs,
        batch=batch_size,
        model_name=f"LSTM_global_{target_family}",
    )

    # Aggregate across folds
    mape_mean = float(leaderboard_df["val_MAPE"].mean())
    rmse_mean = float(leaderboard_df["val_RMSE"].mean())
    mase_mean = float(leaderboard_df["val_MASE"].mean())

    print(
        f"\n[LSTM_global_{target_family}] Mean CV metrics → "
        f"MAPE={mape_mean:.2f}%, RMSE={rmse_mean:.2f}, MASE={mase_mean:.3f}"
    )

    return {"MAPE": mape_mean, "RMSE": rmse_mean, "MASE": mase_mean}


# =====================================================================
# MAIN
# =====================================================================
def main():
    # -----------------------------------------------------------------
    # 1. Load merged dataset (ALL stores & families)
    # -----------------------------------------------------------------
    print("Loading merged dataset ...")
    df = merge_full_dataset()

    # Ensure proper date type & sorting
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    # -----------------------------------------------------------------
    # 2. Single-series baselines for a reference store/family
    # -----------------------------------------------------------------
    series, subset_df = _get_store_family_series(
        df, store_nbr=BASELINE_STORE, family=BASELINE_FAMILY
    )

    rows = []

    print(
        f"\nEvaluating baselines for store={BASELINE_STORE}, "
        f"family='{BASELINE_FAMILY}', horizon={HORIZON}"
    )

    # Naive baseline
    print("\n[Baseline] Naive_last_value ...")
    naive_metrics = eval_naive_last_value_series(series, HORIZON)
    rows.append(
        {
            "store_nbr": BASELINE_STORE,
            "family": BASELINE_FAMILY,
            "horizon": HORIZON,
            "model": "Naive_last_value",
            **naive_metrics,
        }
    )

    # ARIMA baseline
    print("\n[Baseline] ARIMA(5,1,0) ...")
    arima_metrics = eval_arima_series(series, HORIZON)
    rows.append(
        {
            "store_nbr": BASELINE_STORE,
            "family": BASELINE_FAMILY,
            "horizon": HORIZON,
            "model": "ARIMA(5,1,0)",
            **arima_metrics,
        }
    )

    # -----------------------------------------------------------------
    # 3. GLOBAL FAMILY LSTM (all stores, one family)
    # -----------------------------------------------------------------
    print("\n[DL] LSTM_global_family (all stores, one family) ...")
    lstm_metrics = eval_lstm_family_global(
        df_full=df,
        target_family=BASELINE_FAMILY,
        lookback=LSTM_LOOKBACK,
        horizon=HORIZON,
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH,
        n_folds=LSTM_FOLDS,
    )

    # For clarity: this row is global across stores, but only 1 family
    rows.append(
        {
            "store_nbr": 0,                       # 0 => all stores
            "family": f"{BASELINE_FAMILY}_GLOBAL",
            "horizon": HORIZON,
            "model": "LSTM_global_family",
            **lstm_metrics,
        }
    )

    # -----------------------------------------------------------------
    # 4. Save leaderboard CSV
    # -----------------------------------------------------------------
    out_path = Path("results/model_metrics.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_path, index=False)

    print("\nSaved leaderboard metrics to:", out_path.resolve())
    print(df_out)


if __name__ == "__main__":
    main()
