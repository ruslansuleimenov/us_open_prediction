"""Monte Carlo simulation of the tournament bracket."""

import numpy as np
import pandas as pd

from usopen.predict import FEATURE_SET, Predictor

ROUND_NAMES = ["R64", "R32", "R16", "QF", "SF", "F", "Win"]


def build_probability_matrix(players: list[str], predictor: Predictor) -> np.ndarray:
    """Matrix of probs[i][j] = P(player i beats player j).

    Built ONCE with a single batched sklearn call. The simulation itself never
    touches the model: 127 matches x 10,000 tournaments would otherwise mean
    over a million predict_proba calls and twenty minutes of waiting.
    """
    n = len(players)
    rows = [predictor.features(a, b).iloc[0] for a in players for b in players]
    grid = pd.DataFrame(rows)[FEATURE_SET]
    probs = predictor.model.predict_proba(grid)[:, 1].reshape(n, n)
    np.fill_diagonal(probs, 0.5)          # nobody plays themselves
    return probs


def build_logit_matrix(players: list[str], predictor: Predictor) -> np.ndarray:
    """The same as build_probability_matrix, but in logits.

    In logit space a shift in a player's rating becomes an ordinary added
    term rather than a full recomputation of the matrix. That is what makes
    the per-tournament shock cheap enough to run inside the simulation loop.
    """
    p = build_probability_matrix(players, predictor)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def elo_shock_coefficient(model) -> float:
    """How far the logit moves when a player's rating changes by one point.

    The model computes z = sum(w_k * x_k / scale_k). Shifting a player's
    strength by d points moves both elo_diff and outdoor_hard_elo_diff by the
    same d, so the two contributions add.
    """
    coef = dict(zip(FEATURE_SET, model.named_steps["clf"].coef_[0]))
    scale = dict(zip(FEATURE_SET, model.named_steps["scaler"].scale_))
    return sum(coef[f] / scale[f] for f in ("elo_diff", "outdoor_hard_elo_diff"))


def simulate_once(
    n_players: int,
    logits: np.ndarray,
    rng: np.random.Generator,
    shock_sigma: float = 0.0,
    shock_coef: float = 0.0,
) -> list[np.ndarray]:
    """One run of the bracket.

    Returns a list of arrays: the indices of the players who survived each
    round. For a 128-player draw that is 7 entries: 64, 32, 16, 8, 4, 2, 1.

    Players meet in draw order — (0,1), (2,3), (4,5), ... — and each pair is
    decided by rng.random() against the pairwise probability.
    """
    # One strength offset per player FOR THE WHOLE TOURNAMENT, not per match.
    # Without it the simulator treats seven matches as independent draws,
    # although a player who arrives injured plays worse all fortnight. The
    # correlation fattens the tails and lowers the favourite's odds — though
    # measured, the effect turns out small: 51.1% -> 49.5% at sigma 50.
    shift = (rng.normal(0.0, shock_sigma, n_players) if shock_sigma > 0
             else np.zeros(n_players))

    alive = np.arange(n_players)
    rounds: list[np.ndarray] = []

    while len(alive) > 1:
        first, second = alive[0::2], alive[1::2]
        z = logits[first, second] + shock_coef * (shift[first] - shift[second])
        p = 1.0 / (1.0 + np.exp(-z))
        # the whole round is one vectorised draw, not a loop over pairs
        first_wins = rng.random(len(first)) < p
        alive = np.where(first_wins, first, second)
        rounds.append(alive.copy())      # a copy: alive is rebound below

    return rounds


def run_simulations(
    players: list[str],
    logits: np.ndarray,
    n_sims: int = 10_000,
    seed: int = 0,
    shock_sigma: float = 0.0,
    shock_coef: float = 0.0,
) -> pd.DataFrame:
    """n_sims runs -> a player-by-round table of survival shares.

    Rows are players, columns are ROUND_NAMES, and each value is the fraction
    of simulations in which that player got through that round.
    """
    rng = np.random.default_rng(seed)
    n = len(players)
    counts = np.zeros((n, len(ROUND_NAMES)))

    for _ in range(n_sims):
        sim = simulate_once(n, logits, rng, shock_sigma, shock_coef)
        for round_idx, alive in enumerate(sim):
            counts[alive, round_idx] += 1

    table = pd.DataFrame(counts / n_sims, index=players, columns=ROUND_NAMES)
    return table.sort_values("Win", ascending=False)
