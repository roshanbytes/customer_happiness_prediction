from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ACME-HappinessSurvey2020.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"
FEATURE_MIN = 1
FEATURE_MAX = 5

TARGET_COL = "Y"
FEATURE_COLS = ["X1", "X2", "X3", "X4", "X5", "X6"]

FEATURE_DESCRIPTIONS = {
    "X1": "my order was delivered on time",
    "X2": "contents of my order was as I expected",
    "X3": "I ordered everything I wanted to order",
    "X4": "I paid a good price for my order",
    "X5": "I am satisfied with my courier",
    "X6": "the app makes ordering easy for me",
}

RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
SCORING_METRIC = "accuracy"

TARGET_ACCURACY = 0.73
