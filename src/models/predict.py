import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.utils.config import FEATURE_COLS, FEATURE_MAX, FEATURE_MIN, MODEL_PATH

LABELS = {0: "unhappy", 1: "happy"}


def load_model():
    """
    Load the saved model.

    Returns
    -------
    fitted estimator
        The model object saved at MODEL_PATH, ready to call .predict() on.
    """
    return joblib.load(MODEL_PATH)


def predict_batch(csv_path):
    """
    Predict for every row in a CSV file.

    Parameters
    ----------
    csv_path : str or Path
        Path to a CSV containing the columns X1-X6.

    Returns
    -------
    pandas.DataFrame
        The input data with an added "prediction" column (0 or 1).

    Raises
    ------
    ValueError
        If any of the columns X1-X6 are missing from the CSV.
    """
    df = pd.read_csv(csv_path)
    missing = set(FEATURE_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {missing}")
    model = load_model()
    df["prediction"] = model.predict(df[FEATURE_COLS])
    return df


def predict_single(values):
    """
    Predict for one data point.

    Parameters
    ----------
    values : dict
        Mapping of each feature code to its rating.
        Every value must be within FEATURE_MIN-FEATURE_MAX (1-5).

    Returns
    -------
    int
        The predicted label: 0 (unhappy) or 1 (happy).

    Raises
    ------
    ValueError
        If any value in "values" falls outside the 1-5 rating scale.
    """
    for col in FEATURE_COLS:
        if not (FEATURE_MIN <= values[col] <= FEATURE_MAX):
            raise ValueError(f"{col}={values[col]} is out of range ({FEATURE_MIN}-{FEATURE_MAX})")
    df = pd.DataFrame([values], columns=FEATURE_COLS)
    model = load_model()
    return model.predict(df)[0]


def main():
    """
    Parse CLI arguments and run batch or single prediction mode.

    In batch mode, expects a CSV path and an optional --output path.
    In single mode, expects one --x1 through --x6 flag per feature.
    Prints the result to the console, and saves to --output
    in batch mode if given.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(description="Predict customer happiness (0 = unhappy, 1 = happy).")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    batch_parser = subparsers.add_parser("batch", help="Predict for a CSV of rows.")
    batch_parser.add_argument("csv_path", help="Path to a CSV with columns X1-X6.")
    batch_parser.add_argument("--output", help="Optional path to save predictions as a CSV.")

    single_parser = subparsers.add_parser("single", help="Predict for one data point.")
    for col in FEATURE_COLS:
        single_parser.add_argument(f"--{col.lower()}", type=int, required=True)

    args = parser.parse_args()

    if args.mode == "batch":
        result = predict_batch(args.csv_path)
        print(result)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(output_path, index=False)
            print(f"saved to {output_path}")
    elif args.mode == "single":
        values = {col: getattr(args, col.lower()) for col in FEATURE_COLS}
        prediction = predict_single(values)
        print(f"prediction: {prediction} ({LABELS[prediction]})")


if __name__ == "__main__":
    main()