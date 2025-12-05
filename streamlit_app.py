# This app was built with ❤️ in VS Code using Streamlit for Deep Forecast,
# for final project and belongs to Deep Forecast course.
# All Kaggle data belongs to its respective owners. Please see LICENSE for details.
# In this project Kaggle data is used for educational purposes only.

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# RETAIL DEMAND FORECASTING DASHBOARD
# ---------------------------------------------------------------------
# Streamlit web app for time-series forecasting and inventory decisions
# using ARIMA, SARIMAX, Random Forest, and Deep Learning models.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import streamlit as st  # main web app framework
import pandas as pd  # data manipulation and analysis (Excel-like)
import numpy as np  # numerical computing like MATLAB
from pathlib import Path  # filesystem path manipulations
import shutil  # file operations like copying/moving files
from typing import Optional, List  # type hints

import altair as alt  # nicer charts on Core CSV page

# Extra imports for Core Project page (Track 1)
try:
    from sklearn.ensemble import RandomForestRegressor  # Random Forest model benchmark
    from sklearn.metrics import mean_squared_error, mean_absolute_error  # error metrics
    from statsmodels.tsa.arima.model import ARIMA  # ARIMA model for time series forecasting
    from statsmodels.tsa.statespace.sarimax import SARIMAX  # SARIMAX model for seasonal TS
except ImportError:
    st.error("Missing required libraries. Please install: scikit-learn, statsmodels")
    st.stop()

# -------------------------------------------------------------------------
# Global config + DESIGN
# -------------------------------------------------------------------------
st.set_page_config(page_title="RETAIL DEMAND FORECASTING", layout="wide")

# Local utility modules
# Ensure these files exist in your 'app/' directory.
# If running locally without them, some functionality will be limited.
try:
    # import local app modules and handle missing files gracefully
    from app.data_utils import load_raw_data, merge_full_dataset
    from app.plots import plot_forecast
    from app.inference import (
        run_arima_forecast,
        run_sarimax_forecast,
        run_dl_forecast_with_uncertainty,
        get_model_path,
    )
    from app.business_logic import (
        compute_safety_stock,
        compute_reorder_point,
        compute_inventory_financials,
        compute_expected_stockout_cost,
    )
    # RF helper for Kaggle page
    from app.model_rf import rf_forecast_with_uncertainty
except ImportError:
    st.error("Local app modules not found. Ensure the 'app/' directory exists and contains all required files.")
    st.stop()

# Constants used across the app
MODEL_METRICS_FILE = Path("results/model_metrics.csv")  # leaderboard data from backtesting
ABLATION_FILE = Path("results/ablation_study.csv")  # ablation study results
MODELS_DIR = Path("models")  # directory where trained models is are saved
HOME_IMAGE_PATH = Path("assets/home_banner.png")  # banner image for the home page
LOGO_PATH = Path("assets/logo.png")  # Path to the logo file

# Global CSS for Sidebar and Typography
st.markdown(
    """
    <style>
    /* Global app background */
    .main {
        background-color: #233662;
    }
    .stApp {
        background-color: #233662;
    }

    /* Sidebar background */
    section[data-testid="stSidebar"] {
        background: #0f172a;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] label {
        color: #01f757 !important;
    }
    section[data-testid="stSidebar"] input:checked + div {
        background-color: #022c22 !important;
        color: #0bf25c !important;
        border-radius: 999px;
        font-weight: 600;
    }
    section[data-testid="stSidebar"] label {
        color: #022c22 !important;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        transition: background-color 0.2s;
    }
    section[data-testid="stSidebar"] label:hover {
        background-color: rgba(2, 44, 34, 0.1);
    }

    /* Title font styling */
    h1 {
        font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: #f3f4f6;
        text-transform: uppercase;
        font-weight: 900;
        letter-spacing: 0.1em;
        font-size: 2.5rem;
        text-shadow: 0 0 5px rgba(34, 197, 94, 0.4);
        margin-bottom: 0.6rem;
    }

    /* Global font */
    html, body, [class*="css"] {
        font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    /* Utility classes */
    .highlight-pill {
        display: inline-block;
        background: #14532d;
        color: #0bf25c;
        padding: 0.1rem 0.55rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 1.05rem;
        margin-bottom: 0.3rem;
    }
    .app-subtitle {
        color: #9ca3af;
        font-size: 1.0rem;
        max-width: 900px;
        margin-bottom: 1rem;
        line-height: 1.6;
    }

    .main > div { padding-top: 0.25rem; }

    div[data-baseweb="tooltip"] {
        width: 300px !important;
    }

    /* Run Forecast Button Style */
    div.stButton > button {
        border: 1px solid #ef4444;
        background-color: #ef4444;
        color: white;
        font-weight: bold;
        transition: 0.3s;
        height: 46px;
        font-size: 15px;
    }
    div.stButton > button:hover {
        border-color: #b91c1c;
        background-color: #b91c1c;
        color: #ffffff;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1),
                    0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }

    /* Compact inputs */
    div[data-testid="stVerticalBlock"] > div {
        gap: 0.5rem;
    }
    
    /* Make input columns more compact on Core Project page */
    .stSelectbox, .stNumberInput, .stRadio > label, .stSlider {
        margin-bottom: -15px; /* Reduce spacing below elements */
        padding-top: 0;
    }
    
    /* Info Box Style */
    .info-box {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 8px;
        border-left: 5px solid #3b82f6;
        margin-bottom: 20px;
    }

    /* ----- CARD GRID SYSTEM (Home & Leaderboard) ----- */
    .card-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.2rem;
        margin-bottom: 2rem;
    }

    .card-window {
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: #0f172a;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease-in-out;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
    }

    .card-window:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 15px rgba(0, 0, 0, 0.3);
        border-color: rgba(255, 255, 255, 0.25);
    }

    .card-window h4 {
        margin: 0.5rem 0 0.8rem 0;
        font-size: 1.25rem;
        font-weight: 700;
        color: #f3f4f6;
    }

    .card-tag {
        display: inline-block;
        font-size: 0.75rem;
        padding: 0.2rem 0.6rem;
        border-radius: 99px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        align-self: flex-start;
        margin-bottom: 0.5rem;
    }

    .card-window p {
        margin: 0;
        font-size: 0.95rem;
        color: #cbd5f5;
        line-height: 1.6;
    }

    .card-window small {
        display: block;
        margin-top: 1rem;
        font-size: 0.85rem;
        color: #94a3b8;
        border-top: 1px solid rgba(255,255,255,0.1);
        padding-top: 0.75rem;
    }

    .theme-blue   { border-left: 5px solid #3b82f6; background: linear-gradient(145deg, #0f172a, #1e293b); }
    .theme-green  { border-left: 5px solid #22c55e; background: linear-gradient(145deg, #0f172a, #14532d); }
    .theme-purple { border-left: 5px solid #a855f7; background: linear-gradient(145deg, #0f172a, #3b0764); }
    .theme-orange { border-left: 5px solid #f97316; background: linear-gradient(145deg, #0f172a, #431407); }
    .theme-red    { border-left: 5px solid #ef4444; background: linear-gradient(145deg, #0f172a, #450a0a); }

    .tag-blue   { background-color: rgba(59, 130, 246, 0.2); color: #93c5fd; }
    .tag-green  { background-color: rgba(34, 197, 94, 0.2);  color: #86efac; }
    .tag-purple { background-color: rgba(168, 85, 247, 0.2); color: #d8b4fe; }
    .tag-orange { background-color: rgba(249, 115, 22, 0.2); color: #fdba74; }
    .tag-red    { background-color: rgba(239, 68, 68, 0.2);  color: #fca5a5; }

    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------------
# Sidebar navigation
# -------------------------------------------------------------------------

# Display Logo in the sidebar
st.sidebar.image(str(LOGO_PATH), use_container_width=False, output_format="PNG")

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    [
        "Home",
        "Core Project: CSV Forecast",
        "Data",
        "Kaggle-Test Forecast",
        "Inventory",
        "Model Leaderboard",
        "Ablation & Feature Importance",
    ],
)

st.title("RETAIL DEMAND FORECASTING DASHBOARD")

# -------------------------------------------------------------------------
# Helper: store last forecast in session_state
# -------------------------------------------------------------------------
def save_last_forecast(
    model_type: str,
    store_nbr: int,
    family: str,
    actual: pd.Series,
    mean: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
    std: Optional[pd.Series] = None,
):
    st.session_state["last_forecast"] = {
        "model_type": model_type,
        "store_nbr": store_nbr,
        "family": family,
        "actual": actual,
        "mean": mean,
        "lower": lower,
        "upper": upper,
        "std": std,
    }

# -------------------------------------------------------------------------
# Helpers for Core Project Page (Track 1)
# -------------------------------------------------------------------------
def mape(y_true, y_pred) -> float:
    """Mean Absolute Percentage Error (in %)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def compute_metrics(y_true, y_pred) -> dict:
    """Compute RMSE, MAE, MAPE for a pair of arrays/series."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape_val = mape(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape_val}

def make_lag_features(series: pd.Series, n_lags: int = 7) -> pd.DataFrame:
    """Create a supervised learning dataframe with lag features for ML models."""
    df = pd.DataFrame({"y": series})
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df = df.dropna()
    return df

def rf_recursive_forecast(
    train_series: pd.Series,
    horizon: int,
    n_lags: int = 7,
    n_estimators: int = 200,
    max_depth: Optional[int] = None,
    random_state: int = 42,
) -> pd.Series:
    """
    Random Forest recursive forecast with lag features.
    Works for any univariate time series.
    """
    supervised = make_lag_features(train_series, n_lags=n_lags)
    X_train = supervised.drop(columns=["y"])
    y_train = supervised["y"]

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    history = list(train_series.values)
    preds: List[float] = []
    for _ in range(horizon):
        last_vals = history[-n_lags:]
        x = np.array(last_vals).reshape(1, -1)
        preds.append(float(model.predict(x)[0]))
        history.append(preds[-1])

    future_index = pd.RangeIndex(start=0, stop=horizon)
    return pd.Series(preds, index=future_index, name="rf_forecast")

def find_numeric_candidate_columns(df: pd.DataFrame, exclude: str) -> List[str]:
    """
    Tolerant numeric-column detection:
    - ignores the chosen date column
    - tries to coerce a sample to numeric
    - if at least 60% of values are numeric, treat as numeric column.
    """
    numeric_cols: List[str] = []
    for col in df.columns:
        if col == exclude:
            continue
        sample = (
            df[col]
            .dropna()
            .astype(str)
            .str.replace(",", "", regex=False)
            .iloc[:200]
        )
        if sample.empty:
            continue

        coerced = pd.to_numeric(sample, errors="coerce")
        ratio = coerced.notna().mean()
        if ratio >= 0.6:
            numeric_cols.append(col)
    return numeric_cols

def detect_datetime_columns(df: pd.DataFrame, sample_size: int = 200) -> List[str]:
    """
    Heuristically detect columns that look like datetimes.
    """
    candidates: List[str] = []
    for col in df.columns:
        series = df[col].dropna().astype(str).iloc[:sample_size]
        if series.empty:
            continue
        parsed = pd.to_datetime(series, errors="coerce")
        if parsed.notna().mean() >= 0.7:
            candidates.append(col)
    return candidates

def infer_frequency(idx: pd.DatetimeIndex) -> tuple[str, str]:
    """
    Roughly infer the calendar frequency from a DatetimeIndex.
    Returns (code, human_label).
    """
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 3:
        return "UNKNOWN", "unknown"

    diffs = idx.to_series().diff().dropna().dt.days
    if diffs.empty:
        return "UNKNOWN", "unknown"

    gap = float(diffs.mode().iloc[0])

    if 0.5 <= gap <= 1.5:
        return "D", "daily (~1 day gaps)"
    if 6 <= gap <= 8:
        return "W", "weekly (~7 day gaps)"
    if 27 <= gap <= 31:
        return "M", "monthly (~30 day gaps)"
    if 80 <= gap <= 100:
        return "Q", "quarterly (~90 day gaps)"

    return "OTHER", f"~{int(round(gap))}-day gaps"

# -------------------------------------------------------------------------
# Home
# -------------------------------------------------------------------------
def render_home():
    st.markdown(
        """
    ### Deep Learning & Statistical Forecasting System

    Welcome to the Retail Demand Forecasting Dashboard. This application serves as a bridge between raw sales data and actionable business decisions.
    It allows you to train and compare classical statistical methods (like ARIMA) against modern deep learning architectures (like LSTMs and Transformers)
    to predict future demand for the Corporacion Favorita dataset.

    Use the modules below to navigate the workflow.
    """,
        unsafe_allow_html=True,
    )

    if HOME_IMAGE_PATH.exists():
        st.image(str(HOME_IMAGE_PATH), use_container_width=True)

    st.markdown("---")
    st.subheader("Application Modules")

    st.markdown(
        """
<div class="card-grid">
<div class="card-window theme-blue">
<span class="card-tag tag-blue">Step 1</span>
<h4>Data Exploration</h4>
<p>Before modeling, you must understand the the data. This module lets you view the raw training records and visualize aggregated sales trends over time. Look for seasonality, missing days, or zero-sales events.</p>
</div>

<div class="card-window theme-green">
<span class="card-tag tag-green">Step 2</span>
<h4>Forecasting Engine</h4>
<p>The core of the application. Select a store and product family, configure your model parameters (such as lookback window or seasonality settings), and generate a future forecast with confidence intervals.</p>
</div>

<div class="card-window theme-purple">
<span class="card-tag tag-purple">Step 3</span>
<h4>Inventory Optimization</h4>
<p>Turn predictions into business value. Use the forecast output to calculate optimal Safety Stock and Reorder Points (ROP) based on your target service level and supplier lead time.</p>
</div>

<div class="card-window theme-orange">
<span class="card-tag tag-orange">Analytics</span>
<h4>Model Leaderboard</h4>
<p>How do we know which model is best? This module displays the performance metrics (MAPE, RMSE, MASE) calculated from backtesting. Use this to objectively select the winner.</p>
</div>

<div class="card-window theme-red">
<span class="card-tag tag-red">Research</span>
<h4>Ablation Study</h4>
<p>Understanding the "Why". This module shows the results of removing specific features (like Oil Price or Holidays) to see how much they actually contribute to the model's accuracy.</p>
</div>
</div>
    """,
        unsafe_allow_html=True,
    )

    # Core Project description
    st.markdown("---")
    st.subheader("Core Project: CSV Forecast (Track 1)")

    left_col, right_col = st.columns([2.1, 1.4])

    with left_col:
        st.markdown(
            """
<div class="info-box">
    <strong>Objective.</strong> Provide a flexible <em>time-series laboratory</em> where the instructor can drop in
    different datasets (retail sales, airline passengers, housing prices, yfinance series, etc.) and compare
    <span style="color:#ef7777; font-weight:bold;">ARIMA/SARIMA</span> against
    <span style="color:#ef7777; font-weight:bold;">RANDOM FOREST</span> under the same interface.
    <br><br>
    <strong>Workflow on the Core Project page:</strong>
    <ol>
      <li><strong>Choose data source</strong> - either the built-in <code>train.csv</code> from Corporación Favorita
          or an uploaded CSV.</li>
      <li><strong>Select the time index</strong> - the app auto-detects candidate date / time columns,
          but the user can override this manually.</li>
      <li><strong>Select the numeric target</strong> - robust numeric-column detection tolerates messy CSVs
          (commas, header quirks, yfinance-style exports).</li>
      <li><strong>Prepare the series</strong> - handle missing values (drop, forward fill, interpolate) and
          choose how many of the latest observations form the modeling window.</li>
      <li><strong>Configure models</strong> - manually set ARIMA / SARIMA orders and Random Forest lag window
          / number of trees, matching the requirements of the assignment.</li>
      <li><strong>Evaluate</strong> - the last <em>H</em> points (forecast horizon) are held out as a test set.
          The app reports RMSE, MAE and MAPE and plots:
          <ul>
            <li>combined Actual vs Forecasts plot,</li>
            <li>per-model detail charts for ARIMA/SARIMA and Random Forest.</li>
          </ul>
      </li>
    </ol>
    <strong>Pedagogical goal.</strong> Show how different statistical and ML models behave when the data
    has clear seasonality (AirPassengers), trend with structural breaks (Logan housing) or near-random-walk
    behaviour (yfinance stock prices).
</div>
            """,
            unsafe_allow_html=True,
        )

    with right_col:
        st.markdown("#### Quick recipes for common teaching datasets")
        st.markdown(
            """
**1. Airline Passengers (monthly)** - Use SARIMA with yearly seasonality:  
 `(p,d,q) = (1,1,1)`, `(P,D,Q,s) = (1,1,1,12)`  
- Horizon: 12-24 months  
- Nice example of strong seasonality.

**2. Logan Housing (monthly prices)** - No clear seasonality → use non-seasonal ARIMA:  
 `(p,d,q) ≈ (2,1,2)`, seasonality `"off"`  
- Horizon: 12 months  
- Illustrates trend + shocks.

**3. yfinance stock prices (daily)** - Treat as finance series (near random walk).  
- Option: transform to returns and use a small ARIMA (e.g. `(1,0,1)`)  
  vs Random Forest with lagged returns.  
- Good example of models on noisy, weakly predictable data.
            """
        )
        st.caption(
            "The same Core Project interface is used for all of these by changing only "
            "the date column, target column and model parameters."
        )

# -------------------------------------------------------------------------
# Core Project Page (Track 1) - CSV upload + ARIMA/SARIMA + Random Forest
# -------------------------------------------------------------------------
def render_core_project_page() -> None:
    st.header("Core Project: CSV Time Series Forecasting")

    st.markdown(
        "Upload **any univariate time series** (yfinance, airline passengers, housing, "
        "retail, etc.), pick a date column and target, then compare **ARIMA / SARIMA** "
        "and **Random Forest** using RMSE / MAE / MAPE."
    )

    # Data source
    source = st.radio(
        "Data source",
        ["Sample dataset (data/train.csv)", "Upload your own CSV"],
        horizontal=True,
    )

    df: Optional[pd.DataFrame] = None

    if source == "Sample dataset (data/train.csv)":
        dfs = load_raw_data()
        df = dfs.get("train")
        if df is None:
            st.error("Could not load `train.csv` from `data/`.")
            return
    else:
        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded is None:
            st.info("Upload a CSV to begin.")
            return

        try:
            df = pd.read_csv(uploaded)

            # tidy “Unnamed: 0” and multi-row header (yfinance style)
            df.rename(
                columns=lambda x: "Date" if "Unnamed: 0" in str(x) else x,
                inplace=True,
            )
            if len(df) > 0 and df.iloc[0].isnull().all():
                df = df.iloc[1:]
            if (
                len(df) > 0
                and df.iloc[0].astype(str).str.lower().str.contains("date").any()
            ):
                df = df.iloc[1:]
        except Exception as e:
            st.error(f"Could not read CSV file: {e}")
            return

    if df is None or df.empty:
        st.error("No data available.")
        return

    st.subheader("Preview")
    st.dataframe(df.head(), use_container_width=True)

    # Column selection - Compacted using 2 columns
    st.markdown("##### Data / Target Selection")
    col_date, col_target = st.columns(2)
    with col_date:
        date_candidates = detect_datetime_columns(df)
        date_col = st.selectbox(
            "Date / time column",
            date_candidates if date_candidates else list(df.columns),
            index=0,
        )

    with col_target:
        numeric_cols = find_numeric_candidate_columns(df, exclude=date_col)
        if not numeric_cols:
            st.error(
                "No numeric candidate columns found. Make sure the date column is correct."
            )
            return

        target_col = st.selectbox("Target column (numeric)", numeric_cols)

    if date_col == target_col:
        st.error("Date column and target column must be different.")
        return

    # Parse index + numeric target
    parsed_dates = pd.to_datetime(df[date_col].astype(str), errors="coerce")
    valid_date_mask = parsed_dates.notna()
    if valid_date_mask.sum() == 0:
        st.error(f"Column `{date_col}` has no valid dates.")
        return

    df = df.loc[valid_date_mask].copy()
    df[date_col] = parsed_dates[valid_date_mask]
    df = df.drop_duplicates(subset=[date_col], keep="first").sort_values(date_col)

    target_clean = df[target_col].astype(str).str.replace(",", "", regex=False)
    target_values = pd.to_numeric(target_clean, errors="coerce")
    valid_num_mask = target_values.notna()

    if valid_num_mask.sum() == 0:
        st.error(f"Column `{target_col}` has no usable numeric values.")
        return

    df = df.loc[valid_num_mask].copy()
    target_values = target_values[valid_num_mask]

    base_series = pd.Series(target_values.values, index=df[date_col], name=target_col)
    base_series = base_series[~base_series.index.duplicated(keep="first")].astype(float)
    base_series.index.name = "Date"

    # Layout: left = data prep (3 columns), right = model settings (compacted)
    st.subheader("Configure & Run Models")
    prep_chart_col, settings_col = st.columns([3, 1])  # Adjusted column ratio

    arima_order = (1, 1, 1)
    seasonal_order: Optional[tuple] = None
    rf_n_lags = 7
    rf_n_estimators = 200
    
    # LEFT: series transform, history selection, and plot
    with prep_chart_col:
        st.markdown("##### Target transform & history")

        # Use 3 columns for data prep
        t_col, m_col, h_col = st.columns(3)
        
        with t_col:
            transform = st.selectbox(
                "Transform",
                ["Raw values", "Log values", "Log returns (Δ log)", "Percent change"],
                help=(
                    "• Airline / housing / demand → Raw or Log\n"
                    "• Stocks (yfinance) → Log returns or Percent change"
                ),
            )

        series = base_series.copy()
        try:
            if transform == "Log values":
                if (series <= 0).any():
                    st.error("Log transform requires strictly positive values.")
                    return
                series = np.log(series)
            elif transform == "Log returns (Δ log)":
                if (series <= 0).any():
                    st.error("Log returns require strictly positive values.")
                    return
                series = np.log(series).diff().dropna()
            elif transform == "Percent change":
                series = (
                    series.pct_change()
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                )
        except Exception as e:
            st.error(f"Error applying transform: {e}")
            return

        with m_col:
            strategy = st.radio(
                "Missing values",
                ["Drop", "Forward fill", "Linear interpolate"],
                horizontal=True,
            )
        
        if strategy == "Drop":
            series = series.dropna()
        elif strategy == "Forward fill":
            series = series.ffill()
        else:
            series = series.interpolate()

        if len(series) < 20:
            st.warning("Very short series – metrics may be unstable.")

        with h_col:
            history_len = st.slider(
                "History window (last N points)",
                min_value=min(20, len(series)),
                max_value=len(series),
                value=min(144, len(series)),
                step=1,
            )
        series = series.tail(history_len)

        st.line_chart(series, use_container_width=True)
        st.caption(
            f"Usable points: {len(series)}. "
            "The last H points form the test set (H = forecast horizon)."
        )

    # RIGHT: model selection + parameter menus
    with settings_col:
        st.markdown("##### Models & parameters")
        
        smart_hints = st.checkbox(
            "🔉 Smart hints (no auto changes)",
            value=True,
            help=(
                "Shows gentle suggestions based on the data and your settings. "
                "It never changes parameters automatically."
            ),
        )
        
        max_horizon_val = max(1, min(60, len(series) - 5))
        horizon = st.number_input(
            "Forecast horizon (steps)",
            min_value=1,
            max_value=max_horizon_val,
            value=min(24, max_horizon_val),
            step=1,
        )

        use_arima = st.checkbox("ARIMA / SARIMA (statsmodels)", value=True)
        use_rf = st.checkbox("Random Forest (sklearn)", value=True)

        # ARIMA / SARIMA parameters
        if use_arima:
            with st.expander("ARIMA / SARIMA parameters", expanded=False):
                st.caption("Non-Seasonal (p, d, q):")
                c1, c2, c3 = st.columns(3)
                with c1:
                    p = st.number_input("p (AR)", 0, 5, 1, key="arima_p")
                with c2:
                    d = st.number_input("d (diff)", 0, 2, 1, key="arima_d")
                with c3:
                    q = st.number_input("q (MA)", 0, 5, 1, key="arima_q")
                arima_order = (int(p), int(d), int(q))

                st.markdown("---")
                use_seasonal = st.checkbox(
                    "Enable seasonal SARIMA",
                    value=False,
                    help="For monthly airline data use s=12, for weekly data use s=7, etc.",
                )
                if use_seasonal:
                    st.caption("Seasonal (P, D, Q, s):")
                    cP, cD, cQ, cS = st.columns(4)
                    with cP:
                        P = st.number_input("P", 0, 5, 1, key="sea_P")
                    with cD:
                        D = st.number_input("D", 0, 2, 1, key="sea_D")
                    with cQ:
                        Q = st.number_input("Q", 0, 5, 1, key="sea_Q")
                    with cS:
                        s = st.number_input(
                            "s",
                            1,
                            365,
                            12,
                            key="sea_s",
                        )
                    seasonal_order = (int(P), int(D), int(Q), int(s))
                else:
                    seasonal_order = None

        # Smart hints
        if smart_hints and use_arima:
            freq_code, freq_label = infer_frequency(base_series.index)

            if freq_code == "M" and seasonal_order is None:
                st.info(
                    f"📎 Detected **{freq_label}**. For monthly airline / housing data, "
                    "try enabling seasonal SARIMA with season length **s = 12**."
                )
            elif freq_code == "W" and seasonal_order is None:
                st.info(
                    f"📎 Detected **{freq_label}**. For weekly retail data, "
                    "seasonal SARIMA with **s = 7** is often useful."
                )

            min_val = float(base_series.min())
            max_val = float(base_series.max())
            if transform == "Raw values" and min_val > 0 and max_val > 3 * min_val:
                st.caption(
                    "💡 Large upward trend detected with strictly positive values. "
                    "You may also experiment with **Log values** to stabilise variance."
                )

            if transform in ["Log returns (Δ log)", "Percent change"] and freq_code in (
                "M",
                "Q",
            ):
                st.caption(
                    "💡 You're using returns on relatively low-frequency data "
                    f"({freq_label}). That's fine for experiments, but for classic "
                    "AirPassengers-style examples, **Raw** or **Log values** are more common."
                )

        # Random Forest parameters
        if use_rf:
            with st.expander("Random Forest parameters", expanded=False):
                max_lags_allowed = max(1, min(60, len(series) - 5))
                rf_n_lags = st.slider(
                    "Lag features",
                    1,
                    max_lags_allowed,
                    7,
                    step=1,
                    key="rf_lags",
                )
                rf_n_estimators = st.slider(
                    "Trees",
                    50,
                    500,
                    200,
                    step=50,
                    key="rf_n_estimators",
                )

        run_core = st.button("RUN", use_container_width=True)

    if not (use_arima or use_rf):
        st.info("Select at least one model to run.")
        return
    if not run_core:
        return

    # Train / test split
    horizon = int(horizon)
    if len(series) <= horizon + 5:
        st.error("Not enough history for this horizon. Reduce H or increase N.")
        return

    train = series.iloc[:-horizon]
    test = series.iloc[-horizon:]

    st.write(
        f"Train size: **{len(train)}**, Test size: **{len(test)}** "
        f"(predicting the last {horizon} points)"
    )

    forecasts: dict[str, pd.Series] = {}
    metrics_list: List[dict] = []

    # ARIMA / SARIMA
    if use_arima:
        try:
            label = (
                f"ARIMA{arima_order}"
                if seasonal_order is None
                else f"SARIMA{arima_order}x{seasonal_order}"
            )
            with st.spinner(f"Fitting {label}..."):
                if seasonal_order is not None:
                    model = SARIMAX(
                        train,
                        order=arima_order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                else:
                    model = ARIMA(train, order=arima_order)

                fitted = model.fit()
                arima_fc = fitted.forecast(steps=horizon)
                arima_fc.index = test.index

            forecasts[label] = arima_fc
            m = compute_metrics(test.values, arima_fc.values)
            m["Model"] = label
            metrics_list.append(m)
        except Exception as e:
            st.error(f"ARIMA / SARIMA failed: {e}")

    # Random Forest
    if use_rf:
        try:
            with st.spinner("Training Random Forest..."):
                rf_fc = rf_recursive_forecast(
                    train,
                    horizon=horizon,
                    n_lags=int(rf_n_lags),
                    n_estimators=int(rf_n_estimators),
                )
                rf_fc.index = test.index
            forecasts["Random Forest"] = rf_fc
            m = compute_metrics(test.values, rf_fc.values)
            m["Model"] = "Random Forest"
            metrics_list.append(m)
        except Exception as e:
            st.error(f"Random Forest failed: {e}")

    if not forecasts:
        st.error("No forecasts were generated.")
        return

    # --- UPDATED LAYOUT START ---

    # 1. Metrics table (Full width)
    st.subheader("Model Performance (Test Set)")
    metrics_df = pd.DataFrame(metrics_list).set_index("Model")
    st.dataframe(metrics_df.style.format("{:.2f}"), use_container_width=True)

    # 2. Combined Forecast Plot + Forecast Data (Two columns)
    st.subheader("Forecast Results")

    hist_window = min(len(series), max(60, horizon * 2))
    history_series = series.tail(hist_window)
    combined_df = pd.DataFrame({"Actual": history_series})
    for name, pred in forecasts.items():
        combined_df[name] = pred.reindex(history_series.index)

    combined_df = combined_df.reset_index().rename(columns={"Date": "Date"})
    combined_long = combined_df.melt(
        id_vars="Date", var_name="Type", value_name="Value"
    )

    color_domain = list(combined_df.columns[1:])
    color_range = ["#22d3ee", "#f97316", "#22c55e", "#eab308"][: len(color_domain)]
    neon_colors = alt.Scale(domain=color_domain, range=color_range)

    combined_chart = (
        alt.Chart(combined_long)
        .mark_line(strokeWidth=2.4)
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Value:Q", title="Value"),
            color=alt.Color("Type:N", scale=neon_colors, legend=alt.Legend(title="Series")),
            tooltip=["Date:T", "Type:N", "Value:Q"],
        )
        .properties(height=350)
    )

    plot_col, table_col = st.columns([3, 2])
    with plot_col:
        st.markdown("##### Combined Actual vs. Forecasts")
        st.altair_chart(combined_chart, use_container_width=True)
    
    with table_col:
        mean_forecast = next(iter(forecasts.values()))  # first available forecast
        table = pd.DataFrame({"Forecast": mean_forecast}).rename_axis('Date')
        
        st.markdown("##### Forecast Values (Next H Steps)")
        st.dataframe(
            table.style.format("{:.2f}"),
            height=400,
            use_container_width=True,
        )

    # 3. Detail charts (Two columns)
    st.subheader("Model Details")

    detail_col1, detail_col2 = st.columns(2)

    def model_detail_chart(model_name: str, color_hex: str):
        if model_name not in forecasts:
            return None
        y_hat = forecasts[model_name]
        history_detail = series.tail(hist_window)
        df_plot = pd.DataFrame(
            {"Actual": history_detail, "Forecast": y_hat.reindex(history_detail.index)}
        ).reset_index()
        df_long = df_plot.melt(id_vars="Date", var_name="Type", value_name="Value")
        scale = alt.Scale(domain=["Actual", "Forecast"], range=["#a5b4fc", color_hex])
        return (
            alt.Chart(df_long)
            .mark_line(strokeWidth=2.2)
            .encode(
                x="Date:T",
                y="Value:Q",
                color=alt.Color("Type:N", scale=scale, legend=None),
                tooltip=["Date:T", "Type:N", "Value:Q"],
            )
            .properties(height=230)
        )

    with detail_col1:
        arima_name = next(
            (
                m
                for m in forecasts.keys()
                if m.startswith("ARIMA") or m.startswith("SARIMA")
            ),
            None,
        )
        if arima_name:
            st.caption(arima_name)
            chart = model_detail_chart(arima_name, "#f97316")
            if chart is not None:
                st.altair_chart(chart, use_container_width=True)

    with detail_col2:
        if "Random Forest" in forecasts:
            st.caption("Random Forest")
            chart = model_detail_chart("Random Forest", "#22c55e")
            if chart is not None:
                st.altair_chart(chart, use_container_width=True)

    with st.expander("Run configuration (for reporting)"):
        st.markdown(
            f"""
            **Source:** `{source}`  
            **Date column:** `{date_col}` **Target:** `{target_col}`  
            **Transform:** `{transform}`  

            **ARIMA order:** `{arima_order}`  
            **Seasonal order:** `{seasonal_order}`  

            **RF lags:** `{rf_n_lags}`  **RF trees:** `{rf_n_estimators}`
            """
        )

# -------------------------------------------------------------------------
# Data page
# -------------------------------------------------------------------------
def render_data_page():
    st.header("Data Exploration")
    dfs = load_raw_data()
    train = dfs.get("train")

    if train is None:
        st.error("Could not find `train.csv` in the `data/` folder.")
        return

    top_col1, top_col2 = st.columns([2, 1])
    with top_col1:
        st.subheader("Raw `train.csv` sample")
        st.dataframe(train.head(10))
    with top_col2:
        st.subheader("Quick info")
        st.write(f"- Number of rows: **{len(train):,}**")
        st.write(f"- Columns: **{', '.join(train.columns)}**")

    if "date" in train.columns:
        train = train.copy()
        train["date"] = pd.to_datetime(train["date"])
        train = train.sort_values("date")
        st.subheader("Aggregated Daily Sales")
        chart_col, text_col = st.columns([2, 1])
        with chart_col:
            daily = train.groupby("date")["sales"].sum().sort_index()
            st.line_chart(daily, use_container_width=True)
        with text_col:
            st.write(
                "This plot shows total sales per day across all stores. "
                "Note the annual seasonality and New Year closures (zeros)."
            )

# -------------------------------------------------------------------------
# Kaggle Forecast page (advanced)
# -------------------------------------------------------------------------
def render_forecast_page():
    st.header("Kaggle Store Sales: Advanced Forecast")

    smart_assist = st.checkbox(
        "🔉 SMART MODEL ASSISTANT (seasonality-aware hints)",
        value=True,
        help="When enabled, the dashboard analyses weekly seasonality and suggests appropriate models.",
    )

    with st.expander(
        "📘 Analyst Guide: Parameter Selection (Click to expand)", expanded=False
    ):
        st.markdown("##### 1. Data & Model Selection Strategy")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.info(
                "**ARIMA (Non-seasonal)**\n\nBest for items with simple trends or no obvious weekly patterns (e.g., Automotive, Hardware)."
            )
        with m2:
            st.success(
                "**SARIMAX (Seasonal)**\n\nEssential for **Grocery** items. Use `s=7` to capture the weekly sales heartbeat (Saturday peaks)."
            )
        with m3:
            st.warning(
                "**Deep Learning & ML**\n\nBest for non-linear effects and rich feature sets. Requires more data history to train effectively."
            )

        st.divider()

        st.markdown("##### 2. Tuning Tips")
        t1, t2 = st.columns(2)
        with t1:
            st.markdown("**Training Window**")
            st.caption(
                "**60-90 days:** Responsive to recent changes.\n\n"
                "**365 days:** Conservative, stable seasonality."
            )
        with t2:
            st.markdown("**Promo Scenarios**")
            st.caption(
                "**Historical:** Assumes future promotions match the past.\n\n"
                "**All days promoted:** Simulates 'Maximum Potential' capacity."
            )

    train = merge_full_dataset()
    train["date"] = pd.to_datetime(train["date"])
    train = train.sort_values("date")

    store_options = sorted(train["store_nbr"].unique())
    family_options = sorted(train["family"].unique())

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])

    with c1:
        selected_store = st.selectbox("Store", store_options, index=0)
    with c2:
        selected_family = st.selectbox("Product family", family_options, index=0)
    with c3:
        model_type = st.selectbox(
            "Model Architecture",
            [
                "ARIMA",
                "SARIMAX",
                "LSTM",
                "TCN",
                "TRANSFORMER",
                "Random Forest (ML)",
            ],
            index=0,
        )
    with c4:
        st.write("")
        st.write("")
        run_clicked = st.button("GENERATE FORECAST", use_container_width=True)

    p1, p2, p3, p4 = st.columns(4)

    with p1:
        horizon = st.slider("Forecast Horizon (days)", 7, 60, 28, step=7)
    with p2:
        history_window = st.number_input(
            "Display History (days)", 60, 365, 365, step=30
        )

    training_window = 365
    promo_scenario = "Historical promotions"
    window = 60
    epochs = 10
    n_samples = 50
    arima_order = (1, 1, 1)
    seasonal_order = None
    n_trees = 200

    if model_type in ["ARIMA", "SARIMAX"]:
        with p3:
            training_window = st.number_input(
                "Training Window (days)", 90, 730, 365, step=30
            )
        with p4:
            promo_scenario = st.selectbox(
                "Promo Scenario",
                ["Historical promotions", "No future promotions", "All days promoted"],
            )

        if model_type == "SARIMAX":
            with st.expander("⚙️ Advanced SARIMAX Orders (Optional)", expanded=False):
                st.caption(
                    "Manually tune (p,d,q) x (P,D,Q,s). Default is (1,1,1) x (1,1,1,7)."
                )
                s_c1, s_c2, s_c3, s_c4, s_c5, s_c6 = st.columns(6)
                with s_c1:
                    p_ar = st.number_input("p", 0, 5, 1)
                with s_c2:
                    d_diff = st.number_input("d", 0, 2, 1)
                with s_c3:
                    q_ma = st.number_input("q", 0, 5, 1)
                with s_c4:
                    P_sea = st.number_input("P", 0, 3, 1)
                with s_c5:
                    D_sea = st.number_input("D", 0, 2, 1)
                with s_c6:
                    Q_sea = st.number_input("Q", 0, 3, 1)

                use_seasonality = st.checkbox(
                    "Enable Weekly Seasonality (s=7)", value=True
                )
                arima_order = (p_ar, d_diff, q_ma)
                seasonal_order = (
                    (P_sea, D_sea, Q_sea, 7) if use_seasonality else (0, 0, 0, 0)
                )
        else:
            arima_order = (1, 1, 1)
            seasonal_order = None

    elif model_type in ["LSTM", "TCN", "TRANSFORMER"]:
        with p3:
            window = st.number_input("DL Context Window", 30, 120, 60, step=10)
        with p4:
            dl_sub1, dl_sub2 = st.columns(2)
            with dl_sub1:
                epochs = st.number_input("Epochs", 3, 50, 10)
            with dl_sub2:
                n_samples = st.number_input("MC Samples", 10, 200, 50)

    elif model_type == "Random Forest (ML)":
        with p3:
            window = st.number_input(
                "RF Lag Window (days of history)", 7, 60, 30, step=1
            )
        with p4:
            n_trees = st.number_input("Number of Trees", 50, 500, 200, step=50)

    st.divider()

    mask = (train["store_nbr"] == selected_store) & (
        train["family"] == selected_family
    )
    subset = train.loc[mask].copy()

    if subset.empty:
        st.warning(f"No data found for Store {selected_store} - {selected_family}")
        return

    subset = subset.sort_values("date")
    subset.set_index("date", inplace=True)
    series = subset["sales"]

    st.write(
        f"Selected store **{selected_store}**, family **{selected_family}** — {len(series)} historical observations."
    )

    if smart_assist:
        with st.container():
            hint_col, metric_col = st.columns([4, 1])
            weekly_autocorr = np.nan
            try:
                weekly_series = series.asfreq("D").fillna(method="ffill")
                if len(weekly_series) >= 14:
                    weekly_autocorr = float(weekly_series.autocorr(lag=7))
            except Exception:
                pass

            with metric_col:
                val_str = (
                    f"{weekly_autocorr:.2f}"
                    if not np.isnan(weekly_autocorr)
                    else "n/a"
                )
                st.metric("Weekly autocorr (lag 7)", val_str)

            with hint_col:
                if not np.isnan(weekly_autocorr):
                    strength = abs(weekly_autocorr)
                    msg = f"Weekly seasonality strength (|lag-7 autocorr| ≈ {strength:.2f}). "

                    if strength < 0.30:
                        if model_type == "ARIMA":
                            st.success(
                                "🔉 **SMART ASSISTANT:** "
                                + msg
                                + "Your choice of **ARIMA** is well aligned (weak seasonality)."
                            )
                        elif model_type == "SARIMAX":
                            st.info(
                                "🔉 **SMART ASSISTANT:** "
                                + msg
                                + "Weekly pattern is weak. **ARIMA** might be simpler/more stable than SARIMAX."
                            )
                        else:
                            st.info(
                                "🔉 **SMART ASSISTANT:** "
                                + msg
                                + "DL/ML is fine, but consider a simple ARIMA baseline."
                            )
                    else:
                        if model_type == "SARIMAX":
                            st.success(
                                "🔉 **SMART ASSISTANT:** "
                                + msg
                                + "Strong weekly pattern detected. **SARIMAX (s=7)** is an excellent choice."
                            )
                        else:
                            st.warning(
                                "🔉 **SMART ASSISTANT:** "
                                + msg
                                + "Strong weekly pattern. Consider switching to **SARIMAX** to capture seasonality."
                            )

    if run_clicked:
        with st.spinner(f"Running {model_type} model..."):

            if model_type in ["ARIMA", "SARIMAX"]:
                train_series = series.tail(int(training_window))

                exog_train, exog_future = None, None
                if "onpromotion" in subset.columns:
                    exog_full = subset["onpromotion"].astype(float)
                    exog_full = exog_full[~exog_full.index.duplicated(keep="first")]
                    exog_train = exog_full.loc[train_series.index]

                    if promo_scenario == "Historical promotions":
                        last_flag = exog_full.iloc[-1]
                        exog_future = pd.DataFrame(
                            {"onpromotion": np.full(horizon, last_flag)}
                        )
                    elif promo_scenario == "No future promotions":
                        exog_future = pd.DataFrame({"onpromotion": np.zeros(horizon)})
                    else:
                        exog_future = pd.DataFrame({"onpromotion": np.ones(horizon)})

                if model_type == "ARIMA":
                    mean_forecast, conf_int = run_arima_forecast(
                        series=train_series,
                        steps=horizon,
                        exog=exog_train,
                        exog_future=exog_future,
                        order=arima_order,
                    )
                else:
                    mean_forecast, conf_int = run_sarimax_forecast(
                        series=train_series,
                        steps=horizon,
                        exog=exog_train,
                        exog_future=exog_future,
                        order=arima_order,
                        seasonal=seasonal_order,
                    )

                if not isinstance(mean_forecast.index, pd.DatetimeIndex):
                    last_date = series.index[-1]
                    future_index = pd.date_range(
                        start=last_date + pd.Timedelta(days=1),
                        periods=horizon,
                        freq="D",
                    )
                    mean_forecast.index = future_index
                    conf_int.index = future_index

                lower, upper = conf_int.iloc[:, 0], conf_int.iloc[:, 1]
                std_est = (upper - lower) / (2 * 1.96)

            elif model_type in ["LSTM", "TCN", "TRANSFORMER"]:
                if len(series) < window + horizon + 10:
                    st.error("Not enough data for this window size.")
                    return

                mean_s, lower_s, upper_s, std_s = run_dl_forecast_with_uncertainty(
                    model_type=model_type,
                    series=series,
                    store_nbr=selected_store,
                    family=selected_family,
                    window=int(window),
                    n_samples=int(n_samples),
                    epochs=int(epochs),
                )

                last_date = series.index[-1]
                future_index = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=horizon,
                    freq="D",
                )
                mean_forecast = mean_s.iloc[:horizon].set_axis(future_index)
                lower = lower_s.iloc[:horizon].set_axis(future_index)
                upper = upper_s.iloc[:horizon].set_axis(future_index)
                std_est = std_s.iloc[:horizon].set_axis(future_index)

            elif model_type == "Random Forest (ML)":
                if len(series) < window + horizon + 10:
                    st.error("Not enough data for this RF window size.")
                    return

                mean_forecast, lower, upper, std_est = rf_forecast_with_uncertainty(
                    series=series,
                    window=int(window),
                    horizon=int(horizon),
                    n_estimators=int(n_trees),
                )

            recent_actual = series.tail(int(history_window))
            plot_col, table_col = st.columns([2.5, 1])

            with plot_col:
                st.subheader("Forecast vs Actuals")
                fig = plot_forecast(
                    actual=recent_actual,
                    forecast=mean_forecast,
                    lower=lower,
                    upper=upper,
                )
                st.pyplot(fig, use_container_width=True)

            with table_col:
                st.subheader("Forecast Data")
                table = pd.DataFrame(
                    {"Forecast": mean_forecast, "Lower": lower, "Upper": upper}
                )
                st.dataframe(
                    table.style.format("{:.2f}"),
                    height=400,
                    use_container_width=True,
                )

                csv = table.to_csv(index=True).encode("utf-8")
                st.download_button(
                    "📥 Download CSV",
                    csv,
                    "forecast.csv",
                    "text/csv",
                    key="download-csv",
                    use_container_width=True,
                )

            save_last_forecast(
                model_type=model_type,
                store_nbr=selected_store,
                family=selected_family,
                actual=recent_actual,
                mean=mean_forecast,
                lower=lower,
                upper=upper,
                std=std_est,
            )

            if model_type not in ["ARIMA", "SARIMAX"]:
                cached_path = get_model_path(
                    model_type, selected_store, selected_family, int(window)
                )
                st.toast(f"✅ Model cached: {cached_path.name}")

    st.write("")
    with st.expander("ℹ️ System Administration (Cache)", expanded=False):
        st.markdown(
            "Use this to clear stored Deep Learning models if you want to force retraining."
        )
        if st.button("Clear DL Model Cache"):
            model_dir = MODELS_DIR
            if model_dir.exists():
                shutil.rmtree(model_dir)
                model_dir.mkdir()
                st.success("Cache cleared!")

# -------------------------------------------------------------------------
# Inventory page - REDESIGNED
# -------------------------------------------------------------------------
def render_inventory_page():
    st.header("Inventory Decisions (Safety Stock & ROP) 📈")

    if "last_forecast" not in st.session_state:
        st.info(
            "Run a forecast first (on the Forecast page) to enable inventory analysis."
        )
        return

    lf = st.session_state["last_forecast"]
    st.markdown(
        f'<p class="app-subtitle" style="color: #cbd5e1;">Based on forecast for: <strong style="color: #fca5a5;">{lf["family"]}</strong> at Store <strong style="color: #fca5a5;">{lf["store_nbr"]}</strong> using <strong style="color: #fca5a5;">{lf["model_type"]}</strong>.</p>',
        unsafe_allow_html=True,
    )

    mean = lf["mean"]
    std = lf["std"]

    if std is None:
        st.warning(
            "⚠️ No uncertainty information available. Safety stock will be approximate based on historical standard deviation of the forecast."
        )
        # Use a sensible fallback: standard deviation of the last 60 days of actuals + forecast error
        fallback_std = float(lf["actual"].std() * 0.7 + mean.std() * 0.3)
        std_val = fallback_std if not np.isnan(fallback_std) and fallback_std > 0 else 1.0
    else:
        # Use the mean of the forecast standard deviation
        std_val = float(std.mean())

    mean_daily = float(mean.mean())

    # --- INPUT CONTROLS ---
    st.markdown("---")
    st.subheader("Configuration")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        lead_time_days = st.number_input(
            "🚚 Lead time (days)", 
            min_value=1.0, 
            value=14.0, 
            step=1.0,
            help="Time taken from order placement to stock arrival.",
            key="lt_days"
        )
    with col2:
        service_level = st.slider(
            "📅 Target service level", 
            0.80, 
            0.99, 
            0.95, 
            step=0.01,
            format="%.2f",
            help="The probability of avoiding a stockout during the lead time.",
            key="serv_lvl"
        )

    # Map service level to nearest z-score
    z_map = {
        0.80: 0.84, 0.85: 1.04, 0.90: 1.28, 
        0.95: 1.65, 0.98: 2.05, 0.99: 2.33,
    }
    z = min(z_map.items(), key=lambda kv: abs(kv[0] - service_level))[1]
    
    # Financial inputs in the third column
    with col3:
        unit_cost = st.number_input(
            "💵 Unit cost ($/unit)", min_value=0.0, value=2.50, step=0.10, key="u_cost"
        )
        
    # Recalculate based on inputs
    safety_stock = compute_safety_stock(std_val, z)
    reorder_point = compute_reorder_point(mean_daily, lead_time_days, safety_stock)

    # --- CORE METRICS ---
    st.markdown("---")
    st.subheader("Inventory Recommendations & Key Inputs")
    
    col_ss, col_rop, col_demand, col_std = st.columns(4)

    # Use custom HTML/CSS for enhanced metric cards
    st.markdown("""
        <style>
        .metric-card {
            background-color: #0f172a;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            border-top: 5px solid;
            margin-bottom: 20px;
        }
        .metric-title {
            font-size: 0.9rem;
            color: #94a3b8;
            margin-bottom: 5px;
            display: flex;
            align-items: center;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
        }
        .ss-border { border-top-color: #22c55e; } /* Green */
        .rop-border { border-top-color: #3b82f6; } /* Blue */
        .demand-border { border-top-color: #f97316; } /* Orange */
        .std-border { border-top-color: #a855f7; } /* Purple */
        </style>
    """, unsafe_allow_html=True)


    with col_ss:
        st.markdown(f"""
            <div class="metric-card ss-border">
                <div class="metric-title">🛡️ Safety Stock (units)</div>
                <div class="metric-value" style="color: #86efac;">{safety_stock:,.0f}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col_rop:
        st.markdown(f"""
            <div class="metric-card rop-border">
                <div class="metric-title">🔴Reorder Point (units)</div>
                <div class="metric-value" style="color: #93c5fd;">{reorder_point:,.0f}</div>
            </div>
        """, unsafe_allow_html=True)

    with col_demand:
        st.markdown(f"""
            <div class="metric-card demand-border">
                <div class="metric-title">🟡Avg Daily Demand</div>
                <div class="metric-value" style="color: #fdba74;">{mean_daily:,.2f}</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col_std:
        st.markdown(f"""
            <div class="metric-card std-border">
                <div class="metric-title">🟠emand Stdev (Lead Time)</div>
                <div class="metric-value" style="color: #d8b4fe;">{std_val:,.2f}</div>
                <small style="color: #94a3b8;">(Z-score: {z:.2f})</small>
            </div>
        """, unsafe_allow_html=True)


    # --- EXPLANATORY NOTES ---
    st.markdown("##### What do these numbers mean?")
    note_col1, note_col2 = st.columns(2)
    
    with note_col1:
        st.markdown("""
<div class="card-window theme-green">
<span class="card-tag tag-green">Safety Stock</span>
<h4>Your Insurance Buffer</h4>
<p>This is the extra amount of inventory you hold to prevent stockouts if demand is higher than expected or if the supplier lead time is longer than anticipated. It's directly tied to your <strong>Target Service Level</strong>.</p>
</div>
        """, unsafe_allow_html=True)
        
    with note_col2:
        st.markdown("""
<div class="card-window theme-blue">
<span class="card-tag tag-blue">Reorder Point (ROP)</span>
<h4>The Order Trigger</h4>
<p>When your stock level drops to this number, you must place a new order. The ROP ensures you have enough stock to meet demand during the <strong>Lead Time</strong> (ROP = Demand during Lead Time + Safety Stock).</p>
</div>
        """, unsafe_allow_html=True)

    # --- FINANCIAL IMPACT ---
    st.markdown("---")
    st.subheader("Financial Impact of Inventory Policy 𝄃𝄃𝄂𝄂𝄀𝄁𝄃𝄂𝄂𝄃")

    # Financial Inputs (now consolidated to the second line of inputs)
    fin_in1, fin_in2, fin_in3 = st.columns(3)
    with fin_in1:
        # Unit cost is already defined in the main input section (col3)
        st.write("Unit cost is **$**", f"{unit_cost:,.2f}") 
    with fin_in2:
        holding_rate = st.number_input(
            "Annual holding cost rate",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.01,
            format="%.2f",
            help="Fraction of inventory value per year (e.g. 0.25 = 25%).",
            key="h_rate"
        )
    with fin_in3:
        stockout_cost_per_unit = st.number_input(
            "Stockout cost per unit ($)",
            min_value=0.0,
            value=5.00,
            step=0.50,
            key="so_cost",
            help="Lost margin / rush shipping / penalty per unit not served.",
        )

    fin = compute_inventory_financials(
        safety_stock=safety_stock,
        reorder_point=reorder_point,
        unit_cost=unit_cost,
        holding_cost_rate=holding_rate,
    )

    expected_stockout_cost = compute_expected_stockout_cost(
        mean_daily_demand=mean_daily,
        lead_time_days=lead_time_days,
        service_level=service_level,
        stockout_cost_per_unit=stockout_cost_per_unit,
    )

    # Financial Metrics (Cards)
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)

    with f_col1:
        st.markdown(f"""
            <div class="metric-card ss-border">
                <div class="metric-title">📦 Safety Stock Value</div>
                <div class="metric-value" style="color: #86efac;">${fin['safety_stock_value']:,.0f}</div>
            </div>
        """, unsafe_allow_html=True)

    with f_col2:
        st.markdown(f"""
            <div class="metric-card rop-border">
                <div class="metric-title">🏦 Inventory Value at ROP</div>
                <div class="metric-value" style="color: #93c5fd;">${fin['reorder_point_value']:,.0f}</div>
            </div>
        """, unsafe_allow_html=True)
        
    with f_col3:
        # Use Red/Orange for Cost
        cost_color = "#fca5a5" 
        st.markdown(f"""
            <div class="metric-card theme-red">
                <div class="metric-title" style="color: #fca5a5;">💸 Annual Carrying Cost (SS)</div>
                <div class="metric-value" style="color: {cost_color};">${fin['annual_carrying_cost_safety_stock']:,.0f}/year</div>
            </div>
        """, unsafe_allow_html=True)

    with f_col4:
        # Use Red for Cost
        st.markdown(f"""
            <div class="metric-card theme-red">
                <div class="metric-title" style="color: #fca5a5;">💔 Expected Stockout Cost (Cycle)</div>
                <div class="metric-value" style="color: {cost_color};">${expected_stockout_cost:,.0f}</div>
            </div>
        """, unsafe_allow_html=True) 
# -------------------------------------------------------------------------
# Leaderboard page
# -------------------------------------------------------------------------
def render_leaderboard_page():
    st.header("Model Leaderboard")

    st.markdown(
        """
<div class="card-grid">
<div class="card-window theme-blue">
<span class="card-tag tag-blue">The Baseline</span>
<h4>Naive Forecast</h4>
<p>The "Do Nothing" model. It assumes tomorrow's sales will be exactly the same as today's. If a complex AI model can't beat this score, the AI isn't working.</p>
</div>

<div class="card-window theme-green">
<span class="card-tag tag-green">The Scorecard</span>
<h4>MASE (Scaled Error)</h4>
<p>This is the most important number. <br>
<strong>1.0</strong> = Same accuracy as a naive guess.<br>
<strong>0.5</strong> = Twice as good as naive.<br>
<strong>We want this number to be LOW (below 1.0).</strong></p>
</div>

<div class="card-window theme-purple">
<span class="card-tag tag-purple">Cost of Mistakes</span>
<h4>RMSE (Root Mean Sq. Error)</h4>
<p>This measures the average "size" of the error in units sold. It punishes big mistakes heavily. If this number is high, the model is making some very bad large guesses.</p>
</div>
</div>
    """,
        unsafe_allow_html=True,
    )

    path = MODEL_METRICS_FILE
    if path.exists():
        df = pd.read_csv(path)

        col_config = {
            "MASE": st.column_config.NumberColumn(
                "MASE",
                help="Mean Absolute Scaled Error. < 1 means better than a naive forecast.",
            ),
            "MAPE": st.column_config.NumberColumn(
                "MAPE", help="Mean Absolute Percentage Error. Lower is better."
            ),
            "RMSE": st.column_config.NumberColumn(
                "RMSE", help="Root Mean Squared Error. Lower is better."
            ),
        }

        st_df = df.style.format(
            {
                "MAPE": "{:.2f}%",
                "RMSE": "{:.2f}",
                "MASE": "{:.3f}",
            }
        ).highlight_min(subset=["MASE"], color="rgba(34, 197, 94, 0.2)")

        st.dataframe(
            st_df, use_container_width=True, height=400, column_config=col_config
        )
    else:
        st.info(f"No leaderboard data found in `{path}`.")

# -------------------------------------------------------------------------
# Ablation page
# -------------------------------------------------------------------------
def render_ablation_page():
    st.header("Ablation Study & Feature Importance")

    st.markdown(
        """
    <div class="info-box">
        <strong>🧬 What is this?</strong> This experiment removes specific feature groups (like holiday flags or oil prices)
        and retrains the model to see how performance changes.
        <ul>
            <li>If error <strong>increases</strong>, that feature was important.</li>
            <li>If error <strong>stays same</strong> (or drops), that feature might be noise.</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )

    path = ABLATION_FILE
    if path.exists():
        df = pd.read_csv(path)

        c1, c2 = st.columns([1.5, 1])

        with c1:
            st.subheader("Experiment Results")
            st.dataframe(
                df.style.format(
                    {
                        "val_MAPE": "{:.2f}%",
                        "val_RMSE": "{:.2f}",
                        "val_MASE": "{:.3f}",
                    }
                ).background_gradient(subset=["val_MAPE"], cmap="RdYlGn_r"),
                use_container_width=True,
                height=400,
            )

        with c2:
            st.subheader("Model Error Comparison (MAPE)")
            st.caption("📉 **Lower is better**. Bars represent forecast error percentage.")

            if "val_MAPE" in df.columns and "experiment" in df.columns:
                chart_data = (
                    df.set_index("experiment")["val_MAPE"].sort_values(ascending=True)
                )
                st.bar_chart(chart_data, color="#F87171")
    else:
        st.info(
            f"No ablation data found. Run the notebooks or backtesting script to generate `{path}`."
        )

# -------------------------------------------------------------------------
# Route pages
# -------------------------------------------------------------------------
if page == "Home":
    render_home()
elif page == "Core Project: CSV Forecast":
    render_core_project_page()
elif page == "Data":
    render_data_page()
elif page == "Kaggle-Test Forecast":
    render_forecast_page()
elif page == "Inventory":
    render_inventory_page()
elif page == "Model Leaderboard":
    render_leaderboard_page()
elif page == "Ablation & Feature Importance":
    render_ablation_page()
