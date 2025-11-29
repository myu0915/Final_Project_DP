# app/config.py
from pathlib import Path
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Project root = one level above app/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Max number of windows for GLOBAL DL models (ablation + leaderboard)
# Set to None to use ALL windows.
MAX_GLOBAL_DL_WINDOWS = 500_000  # or 1_000_000 if your CPU/RAM can handle it



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Project root = one level above app/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Directories ---
DATA_DIR      = PROJECT_ROOT / "data"
RESULTS_DIR   = PROJECT_ROOT / "results"
MODELS_DIR    = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ASSETS_DIR    = PROJECT_ROOT / "assets"

# --- Files ---
HOME_IMAGE_PATH     = ASSETS_DIR / "home_banner.png"
MODEL_METRICS_FILE  = RESULTS_DIR / "model_metrics.csv"
ABLATION_FILE       = RESULTS_DIR / "ablation_study.csv"

# --- UI defaults (optional, but nice) ---
DEFAULT_FORECAST_HORIZON = 28
DEFAULT_HISTORY_WINDOW   = 365
DEFAULT_TRAINING_WINDOW  = 365
