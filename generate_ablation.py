"""
Offline script to generate a *real* LSTM ablation study.

It will:
- Load and merge the Kaggle data.
- Feature-engineer a single (store, family) series.
- Build windows for (lookback, horizon).
- Run expanding-window CV for:
    1) Full-feature LSTM
    2) Lags-only LSTM (via app.ablation.run_lag_ablation)
- Save results to results/ablation_study.csv, which the Streamlit
  app reads on the "Ablation & Feature Importance" page.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from app.data_utils import merge_full_dataset
from app.features import create_features, prepare_feature_columns
from app.windows import make_windows
from app.backtesting import rolling_cv_indices
from app.training import fit_cv
from app.model_lstm import build_lstm
from app.ablation import run_lag_ablation
from app.config import MAX_GLOBAL_DL_WINDOWS


def run_full_lstm_with_ablation(
    store_nbr: int = 1,
    family: str = "GROCERY I",
    lookback: int = 60,
    horizon: int = 28,
    epochs: int = 10,
    batch_size: int = 256,
    n_folds: int = 3,
):
    # ------------------------------------------------------------------
    # 1. Load & filter data
    # ------------------------------------------------------------------
    print(f"Loading merged dataset for store={store_nbr}, family={family} ...")
    df = merge_full_dataset()

    subset = df[(df["store_nbr"] == store_nbr) & (df["family"] == family)].copy()
    if subset.empty:
        raise ValueError(f"No data for store {store_nbr}, family {family}")

    subset = subset.sort_values("date")

    # ------------------------------------------------------------------
    # 2. Feature engineering
    # ------------------------------------------------------------------
    print("Creating features ...")
    df_feat, base_feature_cols = create_features(subset)

    df_proc, feature_cols, numeric_cols, bool_cols, scaler = prepare_feature_columns(
        df_feat, base_feature_cols
    )

    # Drop rows with NaN in features or target (from early lags)
    cols_to_check = feature_cols + ["sales"]
    cols_to_check = [c for c in cols_to_check if c in df_proc.columns]
    df_proc = df_proc.dropna(subset=cols_to_check)

    # ------------------------------------------------------------------
    # 3. Build windows
    # ------------------------------------------------------------------
    print("Building supervision windows ...")
    X, Y, T = make_windows(
        df_proc,
        lookback=lookback,
        horizon=horizon,
        feature_cols=feature_cols,
        target_col="sales",
    )

    n_samples = X.shape[0]
    print(f"Total window samples: {n_samples}")

    # Optional CPU/RAM-friendly cap on number of windows
    if MAX_GLOBAL_DL_WINDOWS is not None and n_samples > MAX_GLOBAL_DL_WINDOWS:
        print(
            f"Subsampling from {n_samples} to {MAX_GLOBAL_DL_WINDOWS} windows "
            "for LSTM ablation to respect resource limits."
        )
        rng = np.random.default_rng(42)
        idx = rng.choice(n_samples, size=MAX_GLOBAL_DL_WINDOWS, replace=False)
        X = X[idx]
        Y = Y[idx]
        T = T[idx]
        n_samples = X.shape[0]
        print(f"After subsampling: {n_samples} windows")

    if n_samples < n_folds * 2:
        raise ValueError(
            f"Not enough window samples ({n_samples}) for {n_folds} folds. "
            "Consider using fewer folds or a smaller horizon."
        )

    # ------------------------------------------------------------------
    # 4. CV folds (expanding window)
    # ------------------------------------------------------------------
    print("Creating rolling CV folds ...")
    folds = rolling_cv_indices(n_train=n_samples, n_folds=n_folds, step=horizon)

    # ------------------------------------------------------------------
    # 5. Full-feature LSTM CV
    # ------------------------------------------------------------------
    print("\nTraining FULL LSTM (all features) ...")
    leader_full, final_model, final_scaler = fit_cv(
        build_fn=build_lstm,
        X_tr=X,
        Y_tr=Y,
        folds=folds,
        epochs=epochs,
        batch=batch_size,
        model_name="Full LSTM (all features)",
    )

    # ------------------------------------------------------------------
    # 6. Lags-only ablation (uses same folds)
    # ------------------------------------------------------------------
    leader_ablation, mape_diff = run_lag_ablation(
        feature_cols=feature_cols,
        X_train=X,
        Y_train=Y,
        cv_folds=folds,
        lookback=lookback,
        horizon=horizon,
        epochs=epochs,
        batch_size=batch_size,
        leader_full_lstm=leader_full,
    )

    return leader_full, leader_ablation, mape_diff


def main(output_path: Path | None = None):
    if output_path is None:
        output_path = Path("results/ablation_study.csv")

    # You can tweak these defaults later if you want
    store_nbr = 1
    family = "GROCERY I"

    # Stronger settings for LSTM ablation
    lookback = 90       # see more history
    horizon = 28
    epochs = 30         # more training epochs
    batch_size = 128    # smaller batch for better convergence
    n_folds = 3         # keep 3 folds for now

    leader_full, leader_ablation, mape_diff = run_full_lstm_with_ablation(
        store_nbr=store_nbr,
        family=family,
        lookback=lookback,
        horizon=horizon,
        epochs=epochs,
        batch_size=batch_size,
        n_folds=n_folds,
    )

    # Label experiments for the CSV that Streamlit will show
    full_df = leader_full.copy()
    full_df.insert(0, "experiment", "Full LSTM (all features)")
    full_df.insert(1, "removed_feature_group", "None")
    full_df["description"] = "Baseline LSTM with all covariates and lag features."

    abl_df = leader_ablation.copy()
    abl_df.insert(0, "experiment", "LSTM (lags only)")
    abl_df.insert(1, "removed_feature_group", "All exogenous features")
    abl_df["description"] = "LSTM trained only on lagged sales features."

    combined = pd.concat([full_df, abl_df], axis=0, ignore_index=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    print(f"\nSaved ablation results to: {output_path.resolve()}")
    print(combined)
    print(f"\nMAPE difference (lags-only - full): {mape_diff:.2f} percentage points")


if __name__ == "__main__":
    main()
