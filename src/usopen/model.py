"""Training and evaluation of the match-outcome model."""

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, log_loss,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Features are computed over the whole history from 2000, but training uses
# only the modern game: before 2018 the pace and the surfaces were different.
# Feature window and training window are deliberately not the same window —
# ratings need a burn-in before they carry information.
TRAIN_START = "2018-01-01"
VAL_START = "2025-01-01"
TEST_START = "2026-01-01"


def split_by_time(
    features: pd.DataFrame,
    train_start: str = TRAIN_START,
    val_start: str = VAL_START,
    test_start: str = TEST_START,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train / validation / test split.

    Bounds are half-open [start, end): the lower one is included, the upper
    one is not. Adjacent intervals then meet without overlap and without
    gaps, so no match can land in two sets at once — a small leak that is
    easy to introduce and invisible afterwards.
    """
    date = features["Date"]

    train = features.loc[(date >= train_start) & (date < val_start)]
    val = features.loc[(date >= val_start) & (date < test_start)]
    test = features.loc[date >= test_start]

    return train, val, test


def train_model(train: pd.DataFrame, feature_set: list[str]) -> Pipeline:
    """Fit logistic regression on the given feature set.

    `StandardScaler(with_mean=False)` and `fit_intercept=False` are both there
    for antisymmetry: p(A,B) == 1 - p(B,A) holds identically only if the model
    adds no constant to the differences, and centring subtracts a constant
    just as an intercept does. This matters for the Monte Carlo — otherwise a
    player's title odds would depend on which side of the draw they were
    printed, which is an artefact of the code rather than of tennis.

    Scaling itself is not optional: pts_diff has a standard deviation ten
    thousand times larger than form_diff, and the L2 penalty treats all
    coefficients alike.
    """
    model = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(fit_intercept=False, max_iter=1000)),
    ])
    model.fit(train[feature_set], train["target"])
    return model


def predict_proba(model: Pipeline, data: pd.DataFrame, feature_set: list[str]):
    """Probability that Player_1 wins. predict_proba returns (n, 2); take column 1."""
    return model.predict_proba(data[feature_set])[:, 1]


def evaluate(y_true, p_1) -> dict[str, float]:
    """Four metrics. Accuracy takes classes, the other three take probabilities."""
    return {
        "accuracy": accuracy_score(y_true, (p_1 >= 0.5).astype(int)),
        "log_loss": log_loss(y_true, p_1),
        "brier": brier_score_loss(y_true, p_1),
        "roc_auc": roc_auc_score(y_true, p_1),
    }


def market_proba(data: pd.DataFrame) -> pd.Series:
    """Player_1's win probability implied by the bookmakers' odds.

    1/odd is the implied probability, but the two sum to more than one: the
    excess is the bookmaker's margin. Normalising removes it and leaves a
    genuine probability, which is what makes the market usable as a benchmark.
    """
    inv_1 = 1.0 / data["Odd_1"]
    inv_2 = 1.0 / data["Odd_2"]
    return inv_1 / (inv_1 + inv_2)


def usable_odds(data: pd.DataFrame) -> pd.Series:
    """Mask of rows whose odds can be used.

    Missing values are encoded two different ways in this column: -1, which
    load_matches turns into NaN, and 0.0, which it does not. A decimal odd is
    also strictly greater than one by definition — 1.0 would mean a certainty.
    """
    return (data["Odd_1"] > 1.0) & (data["Odd_2"] > 1.0)


def save(model: Pipeline, state: dict, path, data_cutoff=None) -> None:
    """Write model, player state and data cut-off to one file.

    All three travel together on purpose. The model alone cannot build
    features for a pair of names, and neither can say how stale it is: state
    is frozen at training time, so a forecast run after fresher data arrives
    would silently use yesterday's ratings.
    """
    joblib.dump({"model": model, "state": state, "data_cutoff": data_cutoff}, path)


def load(path) -> tuple[Pipeline, dict, object]:
    """Read back what save() wrote: (model, state, data_cutoff)."""
    blob = joblib.load(path)
    return blob["model"], blob["state"], blob.get("data_cutoff")
