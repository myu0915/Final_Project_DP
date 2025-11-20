import streamlit as st

st.set_page_config(page_title="Retail Demand Forecasting", layout="wide")

st.title("Retail Demand Forecasting Dashboard")

st.write("""
This app demonstrates ARIMA and Deep Learning models (LSTM, TCN, Transformer)
for the Kaggle **Store Sales Time Series Forecasting** dataset.
""")

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Home", "Data", "Forecast", "Inventory", "Model Leaderboard"]
)

if page == "Home":
    st.header("Overview")
    st.write("Dashboard skeleton models and plots will be wired in as we go.")

elif page == "Data":
    st.header("Data Exploration")
    st.write("Coming soon: merged dataset view, feature summary, time-series plots.")

elif page == "Forecast":
    st.header("Forecasting Interface")
    st.write("Coming soon: model selector (ARIMA / LSTM / TCN / Transformer) and forecasts.")

elif page == "Inventory":
    st.header("Inventory Decisions")
    st.write("Coming soon: safety stock and reorder point based on forecast uncertainty.")

elif page == "Model Leaderboard":
    st.header("Model Comparison")
    st.write("Coming soon: MAPE / RMSE / MASE leaderboard and ablation study.")
