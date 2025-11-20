import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")


def load_raw_data():
    """
    Load individual CSV files from the Kaggle dataset if they exist.
    Returns a dict with keys: train, test, stores, oil, holidays, transactions.
    """
    paths = {
        "train": DATA_DIR / "train.csv",
        "test": DATA_DIR / "test.csv",
        "stores": DATA_DIR / "stores.csv",
        "oil": DATA_DIR / "oil.csv",
        "holidays": DATA_DIR / "holidays_events.csv",
        "transactions": DATA_DIR / "transactions.csv",
    }

    dfs = {}
    for name, path in paths.items():
        if path.exists():
            dfs[name] = pd.read_csv(path)
        else:
            dfs[name] = None
    return dfs


def load_merged_data() -> pd.DataFrame:
    """
    Placeholder for full merge logic of train + stores + oil + holidays + transactions.

    For now, returns an empty DataFrame so other parts of the app can import this
    module without crashing. We will implement the real merge later.
    """
    return pd.DataFrame()
