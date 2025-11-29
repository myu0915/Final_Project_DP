
<p align="center">
  <img src="assets/usu_logo.png" alt="Utah State University Logo" width="420">
</p>


# Retail Demand Forecasting Dashboard

Interactive Streamlit application for **univariate time-series forecasting** and **inventory decision support**.
Built as a course project to demonstrate an end-to-end forecasting workflow:
from CSV upload → model configuration → forecast evaluation → inventory metrics.

---

## 1. Project Goals

> **Let a user **upload any time-series CSV**, select a date column and numeric target.**
> ***Provide a flexible **“model lab”** for comparing:***

- Classical econometric models (**ARIMA / SARIMA**).
- Machine learning models (**Random Forest Regressor** with lag features).

>> **Visualize and interpret:**
>>

- Actual vs forecast.
- Error metrics (RMSE, MAE, MAPE).

> ***(Extra) Connect forecasts to **inventory decisions** (Safety Stock & Reorder Point).***
> ***(Extra) Provide advanced modules for the **Corporación Favorita “Store Sales”** dataset.***

This app implements all requirements for **Track 1 – Core App**, plus several advanced, portfolio-style features.

---

## 2. App Structure

The Streamlit sidebar lets you navigate between pages:

1. **Home**

   - Overview of the system and the workflow.
   - “Core Project: CSV Forecast (Track 1)” section that explains the exam requirements,
     datasets, and example model configurations (AirPassengers, housing, yfinance).
2. **Core Project: CSV Forecast**

   > This is the *Track 1 core page*.
   >

   - Upload CSV or use built-in `data/train.csv`.
   - Select:
     > **Date / time column** (auto-detected, but fully manual).
     > **Target variable (numeric)** (robust detection tolerant to messy CSVs).
     >
   - Handle missing values: `Drop`, `Forward fill`, or `Linear interpolate`.
   - Choose **history window** (last N observations) and **forecast horizon H**.
   - Configure models:
     > **ARIMA / SARIMA**: manual `(p,d,q)` and optional seasonal `(P,D,Q,s)`.
     > **Random Forest Regressor**: number of lag features + number of trees.
     >
   - The last **H** points are held out as **test set**; rest is **train**.

   **Models used**

   - Econometric:
     > `statsmodels.tsa.arima.model.ARIMA` (supports ARIMA and SARIMA).
     >
   - Machine Learning:
     > `sklearn.ensemble.RandomForestRegressor` with custom lag feature engineering.
     >

   **Metrics**

   - RMSE, MAE, MAPE (implemented manually using NumPy / sklearn metrics).

   **Visualizations**

   - Combined Actual vs Forecast (Altair).
   - Per-model detail plots.
   - Metrics table comparing ARIMA/SARIMA vs Random Forest.
3. **Data**

   - Exploratory page for the Corporación Favorita `train.csv`.
   - Shows head of the data, basic row/column counts, and aggregated daily sales plot.
4. **Kaggle-Test Forecast**

   - Advanced forecasting engine specialized for the Store Sales dataset.
   - Supports multiple model types (`ARIMA`, `SARIMAX`, deep learning, RF).
   - “Smart assistant” hints about weekly seasonality and model choice.
   - Produces forecast + prediction intervals and allows CSV download.
   - (Depends on local helper modules in `app/`: `data_utils`, `inference`, `model_rf`, etc.)
5. **Inventory**

   - Uses the **last generated forecast** (any model) to compute:
     > Safety Stock.
     > Reorder Point (ROP).
     >
   - Inputs:
     > Lead time (days).
     > Target service level.
     >
   - Outputs are based on mean forecast and forecast uncertainty.
6. **Model Leaderboard**

   - Reads pre-computed model metrics from `results/model_metrics.csv`.
   - Shows a leaderboard of models (e.g., Naive, ARIMA, SARIMAX, DL models) by:
     > RMSE, MAPE, and MASE.
     >
   - Uses highlighting to emphasize the best MASE.
7. **Ablation & Feature Importance**

   - Reads ablation results from `results/ablation_study.csv`.
   - Shows how removing feature groups (e.g., holidays, oil price) affects validation error.
   - Includes a table and a bar chart of validation MAPE.

## 3. Data & Example Datasets

The app is designed to work with many time-series CSVs, for example:

- **Corporación Favorita “Store Sales”** (Kaggle) – daily sales per store/family.
- **AirPassengers** – classic monthly international airline passengers.
- **Logan housing prices** – monthly real-estate series with trend and structural breaks.
- **yfinance stock prices** – daily OHLCV data (e.g., AAPL, MSFT).

For the **Core Project page**, the only requirements for a dataset are:

- One column that can be parsed as a date/time.
- One numeric target column.
- A univariate time-indexed series (after filtering).

---

## 4. Modeling Details (Core Project)

### Feature Engineering for Random Forest

For a target series `y_t`, the app constructs lag features:

`lag_k(t) = y_{t-k},  k = 1, ..., L`

- **L** (number of lags) is chosen by the user.
- Rows with incomplete lags are dropped.
- Forecasting uses a **recursive strategy**: each predicted point is fed back as a new lag.
- User picks forecast horizon `H`.
- The last `H` observations are used as the **test set**.
- All earlier observations are used for **training**.
- Metrics are always reported on the test set only.

### Metrics

- **RMSE** – penalizes large errors strongly.
- **MAE** – average absolute error.
- **MAPE** – average absolute percentage error (ignoring zero targets).

Lower is better for all three.

---

## 5. How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# 2. Create environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run streamlit_app.py
```

---

# Folder Tree

```
📁 Final Project Deep Forecasting
├── 📝 LICENSE
├── 📄 streamlit_app.py                  # Main Streamlit interface (all pages)
├── 📄 requirements.txt                  # Python dependencies for deployment
├── 📄 README.md                         # Project documentation
├── 📄 .gitignore                        # Git ignore rules

├── 📁 models/
│   └── 📄 lstm_st1_fam_AUTOMOTIVE_win60.keras   # Cached LSTM model (advanced pages)

├── 📁 notebooks/
│   └── 📄 04_backtest.ipynb             # Dev notebook for backtesting & experiments

├── 📁 results/
│   ├── 📄 ablation_study.csv            # Feature ablation results
│   └── 📄 model_metrics.csv             # Leaderboard model metrics (MASE, RMSE, etc.)

├── 📁 artifacts/
│   ├── 📄 decision_pack.json            # Exported decisions for production-style workflow
│   ├── 📄 forecast_next_28d.csv         # Example saved forecast
│   └── 📄 metrics.json                  # Cached metric summaries (advanced)

├── 📁 app/                              # Modular backend — production-style structure
│   ├── 📄 __init__.py
│   ├── 📄 ablation.py                   # Ablation study computation
│   ├── 📄 backtesting.py                # Rolling CV / backtesting utilities
│   ├── 📄 business_logic.py             # Safety stock & inventory calculations
│   ├── 📄 config.py
│   ├── 📄 constants.py
│   ├── 📄 data_utils.py                 # Load & merge datasets (train/test, external sources)
│   ├── 📄 evaluation.py                 # RMSE, MAPE, MAE & evaluation helpers
│   ├── 📄 features.py                   # Feature engineering (lags, rolling stats)
│   ├── 📄 inference.py                  # ARIMA / SARIMAX / DL forecasting orchestration
│   ├── 📄 inventory.py                  # Inventory module logic
│   ├── 📄 model_arima.py                # Classical ARIMA wrapper
│   ├── 📄 model_dl.py                   # Deep learning model manager
│   ├── 📄 model_lstm.py                 # LSTM forecasting model
│   ├── 📄 model_rf.py                   # Random Forest forecasting + uncertainty
│   ├── 📄 model_tcn.py                  # Temporal Convolutional Network model
│   ├── 📄 model_transformer.py          # Transformer forecasting model
│   ├── 📄 plots.py                      # Plotting utilities (matplotlib)
│   ├── 📄 training.py                   # Training pipelines
│   ├── 📄 uncertainty.py                # Variance estimation & MC dropout helpers
│   ├── 📄 utils_env.py                  # Config helpers for cloud/local environments
│   ├── 📄 windows.py                    # Windowing utilities for backtests
│   └── 📄 yfinance.py                   # Custom yfinance preprocessing & helpers

├── 📁 data/
│   ├── 📄 holidays_events.csv
│   ├── 📄 oil.csv
│   ├── 📄 README_data.md
│   ├── 📄 sample_submission.csv
│   ├── 📄 stores.csv
│   ├── 📄 test.csv
│   ├── 📄 train.csv                     # Main Corporación Favorita dataset
│   └── 📄 transactions.csv



```

```
<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/13/Utah_State_University_logo.svg/2560px-Utah_State_University_logo.svg.png" 
       alt="Utah State University Logo" width="420">
</p>

<h2 align="center">
  <strong>Retail Demand Forecasting Dashboard</strong><br>
  <em>Utah State University — DATA-5630-001 XL — Fall 2025</em>
</h2>

<p align="center">
  Time-series forecasting laboratory built for academic and educational purposes.<br>
  Supports CSV upload, ARIMA/SARIMA, Random Forest, and forecast comparison tools.
</p>

<hr>
```
