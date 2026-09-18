import numpy as np
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

from src.utils.config import RANDOM_STATE, SCORING_METRIC, VALIDATION_SIZE


MODELS = {
    "AdaBoost": {
        "estimator": AdaBoostClassifier(random_state=RANDOM_STATE),
        "param_grid": {
            "n_estimators": [10, 20, 30, 40, 50, 60, 100],
            "learning_rate": [0.001, 0.003, 0.005, 0.009, 0.01, 0.03, 0.05, 0.07, 0.1, 0.5],
        },
    },

    "GradientBoosting": {
        "estimator": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "param_grid": {
            "n_estimators": [10, 20, 30, 40, 50, 60, 100],
            "learning_rate": [0.001, 0.003, 0.005, 0.009, 0.01, 0.03, 0.05, 0.07, 0.1, 0.5],
        },
    },

    "RandomForest": {
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
        "param_grid": {
            "n_estimators": [10, 20, 30, 40, 50, 60, 100],
            "max_depth": [None, 2, 3],
        },
    },
}


def get_split(y):
    """
    Build a single stratified train/validation split for tuning.

    Parameters
    ----------
    y : array-like
        Target labels, used to keep the same class balance in both
        halves of the split.

    Returns
    -------
    list of tuple
        A single (train_idx, val_idx) pair of row-position arrays,
        wrapped in a list so it can be passed as GridSearchCV's cv
        argument.
    """
    train_idx, val_idx = train_test_split(
        np.arange(len(y)), test_size=VALIDATION_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return [(train_idx, val_idx)]


def tune_model(name, X, y, cv=None):
    """
    Run a grid search over one model's hyperparameters.

    Parameters
    ----------
    name : str
        Key into MODELS identifying which model and parameter grid
        to search over.
    X : array-like
        Feature matrix to fit and score candidates on.
    y : array-like
        Target labels matching X.
    cv : iterable, optional
        Train/validation split to use. Defaults to a single
        stratified split from get_split(y) if not provided.

    Returns
    -------
    sklearn.model_selection.GridSearchCV
        The fitted search object. Use .best_score_, .best_params_,
        and .best_estimator_ to see the result.
    """
    if cv is None:
        cv = get_split(y)
    spec = MODELS[name]
    search = GridSearchCV(spec["estimator"], spec["param_grid"], cv=cv, scoring=SCORING_METRIC)
    search.fit(X, y)
    return search
