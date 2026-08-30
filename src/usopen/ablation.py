"""Comparing feature sets and algorithms — the project's central experiment.

The question is "which information genuinely improves predictive power", not
"which model is best". Differences are checked with a paired bootstrap: on
2000 matches a gap of 0.005 log loss is indistinguishable from noise, and
without intervals it is easy to read improvements out of the table that are
not there.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from usopen.features import FEATURE_SETS
from usopen.model import evaluate, market_proba, predict_proba, train_model, usable_odds

N_BOOT = 5000


def nll(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Per-match log loss. Needed elementwise so that pairs can be resampled."""
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def bootstrap_diff(y, p_a, p_b, n_boot: int = N_BOOT, seed: int = 0):
    """Paired bootstrap of the log-loss difference. Positive means B is better."""
    d = nll(y, p_a) - nll(y, p_b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def run_ablation(train: pd.DataFrame, data: pd.DataFrame,
                 baseline: str = "ranking") -> pd.DataFrame:
    """Table of feature set -> metrics, with a CI on the gap to the baseline set."""
    y = data["target"].values
    preds = {name: predict_proba(train_model(train, feat), data, feat)
             for name, feat in FEATURE_SETS.items()}

    rows = []
    for name, p in preds.items():
        r = evaluate(y, p)
        d, lo, hi = bootstrap_diff(y, preds[baseline], p)
        rows.append({"set": name, "n": len(data), **r,
                     "Δ log loss": d, "ci_lo": lo, "ci_hi": hi,
                     "significant": (lo > 0) == (hi > 0) and name != baseline})

    ok = data[usable_odds(data)]
    r = evaluate(ok["target"].values, market_proba(ok).values)
    rows.append({"set": "MARKET", "n": len(ok), **r,
                 "Δ log loss": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "significant": True})
    return pd.DataFrame(rows).set_index("set")


def _tree(kind: str, train: pd.DataFrame, feat: list[str]):
    cls = {"RandomForest": RandomForestClassifier(n_estimators=300, min_samples_leaf=20,
                                                  random_state=0, n_jobs=-1),
           "HistGB": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                    random_state=0)}[kind]
    return cls.fit(train[feat], train["target"])


def compare_models(train: pd.DataFrame, data: pd.DataFrame,
                   feature_set: str = "+h2h_bo5") -> pd.DataFrame:
    """Logistic regression against trees, on one feature set.

    Trees are not antisymmetric: p(A,B) + p(B,A) != 1, so in the Monte Carlo a
    player's odds would start to depend on which side of the draw they sit.
    Both the raw and the mirror-averaged variant are reported; averaging is
    worth about -0.003 log loss, more than the H2H feature gained.
    """
    feat = FEATURE_SETS[feature_set]
    y = data["target"].values
    mirror = data.copy()
    mirror[feat] = -mirror[feat]        # swapping players negates every difference

    rows = []
    for name in ["LogReg", "RandomForest", "HistGB"]:
        m = train_model(train, feat) if name == "LogReg" else _tree(name, train, feat)
        p = m.predict_proba(data[feat])[:, 1]
        p_mirror = m.predict_proba(mirror[feat])[:, 1]
        p_sym = (p + 1 - p_mirror) / 2
        rows.append({"model": name,
                     "log loss": nll(y, p).mean(),
                     "log loss (symmetrised)": nll(y, p_sym).mean(),
                     "asymmetry": float(np.abs(p + p_mirror - 1).mean()),
                     **{k: v for k, v in evaluate(y, p_sym).items() if k != "log_loss"}})
    return pd.DataFrame(rows).set_index("model")
