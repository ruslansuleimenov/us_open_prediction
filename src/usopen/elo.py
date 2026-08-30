"""Elo ratings: online logistic regression with one parameter per player."""

from collections import defaultdict
from typing import Callable, Any
import pandas as pd


K_BASE = 32.0
INITIAL_RATING = 1500.0

LEVEL_WEIGHT = {
    "Grand Slam": 1.25,
    "Masters Cup": 1.15,
    "Masters 1000": 1.10,
    "ATP500": 1.00,
    "ATP250": 0.90,
}

COURT_WEIGHT_OUTDOOR = {"Outdoor": 1.0, "Indoor": 0.6}

# For the hard-court rating: other surfaces still count, but less.
SURFACE_WEIGHT_HARD = {"Hard": 1.0, "Clay": 0.4, "Grass": 0.4, "Carpet": 0.4}


def compute_elo(
    matches: pd.DataFrame,
    weight_fn: Callable[[Any], float],
    k_base: float = K_BASE,
    initial: float = INITIAL_RATING,
) -> tuple[pd.Series, pd.Series, dict[str, float]]:
    """Ratings before each match, plus the final state.

    Returns (Player_1 rating before the match, Player_2 rating before the
    match, {player: final rating}). The first two feed training, the third
    feeds prediction on a draw where no historical row exists.

    `weight_fn` receives the whole match row, so the K factor can depend on
    anything in it — tournament tier, surface, indoor/outdoor. Passing
    `lambda r: 1.0` recovers textbook Elo, which is how the flat-K variant
    is A/B tested.
    """
    ratings = defaultdict(lambda: initial)
    rating_a = []
    rating_b = []
    for row in matches.itertuples():
        ra = ratings[row.Player_1]
        rb = ratings[row.Player_2]
        rating_a.append(ra)
        rating_b.append(rb)
        # Read, record, and only then update: reversing this would leak the
        # result of the match into its own feature.
        ea = 1 / (1 + 10 ** ((rb - ra)/ 400))
        sa = 1.0 if row.Winner == row.Player_1 else 0.0
        k = k_base * weight_fn(row)
        delta = k * (sa - ea)
        ratings[row.Player_1] += delta
        ratings[row.Player_2] -= delta
    return pd.Series(rating_a, index=matches.index), pd.Series(rating_b, index=matches.index), dict(ratings)

def general_weight(row) -> float:
    """Match weight for the general rating: tournament tier only."""
    return LEVEL_WEIGHT[row.Series]


def outdoor_hard_weight(row) -> float:
    """Match weight for the outdoor-hard rating: tier x surface x court."""
    return (
        LEVEL_WEIGHT[row.Series]
        * SURFACE_WEIGHT_HARD[row.Surface]
        * COURT_WEIGHT_OUTDOOR[row.Court]
    )
