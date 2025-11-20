
# Deep Learning for Retail Demand Forecasting

Using the Kaggle *Store Sales – Time Series Forecasting* Dataset

This project predicts 28-day ahead sales for Corporación Favorita retail stores using both classical and deep learning models. It integrates forecasting, uncertainty estimation, and inventory decision support into an interactive Streamlit application.

## Forecasting Models

The project implements and compares:

- **ARIMA / SARIMAX** (classical statistical model)
- **LSTM** (Deep Learning baseline)
- **TCN – Temporal Convolutional Network**
- **Tiny Transformer – Attention-based model**

Each model is evaluated using time-series cross-validation and compared using multiple error metrics.

## Streamlit Web Application

The project includes a full Streamlit dashboard that allows users to:

- Visualize forecasts vs. actual sales
- Explore prediction uncertainty (confidence intervals or MC Dropout)
- Compare model performance in a leaderboard
- Run promotion-based "what-if" scenarios
- Convert forecasts into inventory recommendations
- Review ablation study results (impact of removing certain features)

---

# Repository Structure

```text
notebooks/         → Jupyter notebooks (EDA, ARIMA, DL, evaluation)
app/               → Python modules (data loading, models, metrics, plots, business logic)
models/            → Saved model weights & scalers (local only, ignored by Git)
results/           → Evaluation metrics & ablation study CSVs
data/              → Dataset folder (CSV files stored locally only)
streamlit_app.py   → Main Streamlit app entry point
requirements.txt   → Python dependencies
.gitignore         → Git ignore rules
```
