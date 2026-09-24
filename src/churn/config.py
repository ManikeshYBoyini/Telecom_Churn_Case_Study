"""Project-wide paths, the random seed, and every threshold/grid the
original notebook used, so no later module hardcodes a magic number."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
SUBMISSIONS_DIR = OUTPUTS_DIR / "submissions"

TRAIN_CSV = DATA_RAW_DIR / "train.csv"
TEST_CSV = DATA_RAW_DIR / "test.csv"
DATA_DICTIONARY_CSV = DATA_RAW_DIR / "data_dictionary.csv"

RANDOM_SEED = 42

ID_COLUMN = "id"
TARGET_COLUMN = "churn_probability"
MISSING_THRESHOLD_PCT = 60.0

OUTLIER_COLUMNS = [
    f"{prefix}_{month}"
    for prefix in ("total_ic_mou", "total_og_mou", "onnet_mou", "offnet_mou", "total_rech_amt")
    for month in (6, 7, 8)
]
OUTLIER_MAX_TO_P99_RATIO = 6.0
OUTLIER_PERCENTILE = 0.99

HVC_QUANTILE = 0.65
LVC_QUANTILE = 0.20

TRAIN_SIZE = 0.7
N_PCA_COMPONENTS = 70

LR_PARAM_GRID = {
    "C": [0.1, 1, 5, 10],
    "class_weight": [{0: 0.1, 1: 0.9}, {0: 0.2, 1: 0.8}],
}
LR_CV_FOLDS = 5

RF_PARAM_GRID = {
    "n_estimators": [100, 200],
    "min_samples_leaf": [30, 40],
}
RF_CLASS_WEIGHT = {0: 0.2, 1: 0.8}
RF_CV_FOLDS = 3

XGB_PARAM_GRID = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05],
    "min_child_weight": [3, 5],
    "gamma": [0.1],
    "subsample": [0.8],
    "colsample_bytree": [0.8, 1.0],
}
XGB_N_ITER = 8
XGB_CV_FOLDS = 3
