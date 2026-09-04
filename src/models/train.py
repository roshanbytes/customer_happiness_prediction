import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

from src.utils.config import RANDOM_STATE, SCORING_METRIC, VALIDATION_SIZE


MODELS = {
    "AdaBoost": {
        "estimator": AdaBoostClassifier(random_state=RANDOM_STATE),
        "param_grid": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 1.0],
        },
    },

    "RandomForest": {
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
        "param_grid": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 3, 5, 10],
        },
    },
}


def get_split(y):
    train_idx, val_idx = train_test_split(
        np.arange(len(y)), test_size=VALIDATION_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return [(train_idx, val_idx)]


def tune_model(name, X, y, cv=None):
    if cv is None:
        cv = get_split(y)
    spec = MODELS[name]
    search = GridSearchCV(spec["estimator"], spec["param_grid"], cv=cv, scoring=SCORING_METRIC)
    search.fit(X, y)
    return search
