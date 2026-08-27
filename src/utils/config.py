from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'ACME-HappinessSurvey2020.csv'

TARGET_COL = 'Y'
FEATURE_COLS = ['X1', "X2", "X3", "X4", "X5", "X6"]

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5