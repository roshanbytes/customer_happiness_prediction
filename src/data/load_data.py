import pandas as pd

from src.utils.config import FEATURE_COLS, RAW_DATA_PATH, TARGET_COL


def load_raw_data(path=RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    expected_cols = [TARGET_COL] + FEATURE_COLS
    missing = set(expected_cols) - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {missing}')

    if df.isnull().values.any():
        raise ValueError('Raw data contains missing values.')

    return df


def split_features_target(df: pd.DataFrame):
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    return X, y