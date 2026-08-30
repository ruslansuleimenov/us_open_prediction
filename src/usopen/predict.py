"""The bridge from a draw to the model: two names in, a probability out."""

from dataclasses import dataclass

import pandas as pd
from sklearn.pipeline import Pipeline

FEATURE_SET = ["rank_diff", "pts_diff", "elo_diff", "outdoor_hard_elo_diff",
               "h2h_edge", "h2h_bo5"]

# The US Open is best-of-five, so the H2H/format interaction is always on.
IS_BO5 = 1.0

# Fallbacks for players with no tour-level history: qualifiers and wildcards
# appearing for the first time.
DEFAULT_ELO = 1500.0
DEFAULT_FORM = 0.5


@dataclass
class Predictor:
    """A trained model together with player state at the end of the history.

    Both halves are required. The model alone cannot answer "Alcaraz versus
    Djokovic" — the draw supplies only names, and the features have to be
    reconstructed from the ratings held in `state`.
    """

    model: Pipeline
    state: dict[str, dict]
    default_rank: float
    default_pts: float

    def _player(self, name: str) -> dict[str, float]:
        s = self.state
        return {
            "rank": s["rank"].get(name, self.default_rank),
            "pts": s["pts"].get(name, self.default_pts),
            "elo": s["elo"].get(name, DEFAULT_ELO),
            "outdoor_hard_elo": s["outdoor_hard_elo"].get(name, DEFAULT_ELO),
        }

    def features(self, player_a: str, player_b: str) -> pd.DataFrame:
        """Build the feature row, A minus B — the same order used in training.

        Training reads these differences off a historical row; prediction
        rebuilds them from `state`. Keeping one implementation of the
        subtraction is what stops the two paths from drifting apart, which
        would flip the favourite without raising anything.
        """
        from usopen.features import h2h_edge

        a, b = self._player(player_a), self._player(player_b)
        h2h = h2h_edge(self.state["h2h_record"], player_a, player_b)
        return pd.DataFrame([{
            "rank_diff": a["rank"] - b["rank"],
            "pts_diff": a["pts"] - b["pts"],
            "elo_diff": a["elo"] - b["elo"],
            "outdoor_hard_elo_diff": a["outdoor_hard_elo"] - b["outdoor_hard_elo"],
            "h2h_edge": h2h,
            "h2h_bo5": h2h * IS_BO5,
        }])

    def predict_match(self, player_a: str, player_b: str) -> float:
        """Probability that `player_a` beats `player_b`."""
        return float(self.model.predict_proba(self.features(player_a, player_b))[0, 1])
