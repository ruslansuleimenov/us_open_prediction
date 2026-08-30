"""Features: player form, head-to-head, and assembly of the training table."""

from collections import defaultdict, deque

import pandas as pd

from usopen.elo import compute_elo, general_weight, outdoor_hard_weight

FORM_WINDOWS = (10, 20)

# Metadata: not features, but needed to slice the split and inspect errors.
META_COLUMNS = ["Date", "Series", "Surface", "Court", "Round",
                "Player_1", "Player_2", "Winner", "Odd_1", "Odd_2"]

# Sets for the ablation: what each portion of information actually adds.
FEATURE_SETS = {
    "ranking": ["rank_diff", "pts_diff"],
    "+form":   ["rank_diff", "pts_diff", "form_10_diff", "form_20_diff"],
    "+elo":    ["rank_diff", "pts_diff", "elo_diff", "outdoor_hard_elo_diff"],
    "full":    ["rank_diff", "pts_diff", "elo_diff", "outdoor_hard_elo_diff",
                "form_10_diff", "form_20_diff"],
    "+h2h":    ["rank_diff", "pts_diff", "elo_diff", "outdoor_hard_elo_diff",
                "h2h_edge"],
    "+h2h_bo5": ["rank_diff", "pts_diff", "elo_diff", "outdoor_hard_elo_diff",
                 "h2h_edge", "h2h_bo5"],
}


def _rate(history: deque, cold_start: float) -> float:
    """Win share over the history. An empty history yields cold_start, not ZeroDivisionError."""
    if not history:
        return cold_start
    return sum(history) / len(history)


def add_form(
    matches: pd.DataFrame,
    window: int,
    cold_start: float = 0.5,
) -> tuple[pd.Series, pd.Series, dict[str, float]]:
    """Win share over each player's last `window` matches BEFORE the current one.

    Returns Player_1 form, Player_2 form, and {player: form at the end of the
    history}. The first two feed training, the third feeds prediction.
    """
    history: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
    form_a: list[float] = []
    form_b: list[float] = []

    for row in matches.itertuples():
        # READ a number, not the deque itself: storing the object would put a
        # reference in the list, and every row would end up showing the final state.
        form_a.append(_rate(history[row.Player_1], cold_start))
        form_b.append(_rate(history[row.Player_2], cold_start))

        # UPDATE only after the features have been recorded.
        won_1 = row.Winner == row.Player_1
        history[row.Player_1].append(int(won_1))
        history[row.Player_2].append(int(not won_1))

    final = {p: _rate(d, cold_start) for p, d in history.items()}

    return (
        pd.Series(form_a, index=matches.index),
        pd.Series(form_b, index=matches.index),
        final,
    )


def add_h2h(
    matches: pd.DataFrame,
    prior: float = 2.0,
) -> tuple[pd.Series, dict[tuple[str, str], list[int]]]:
    """Head-to-head edge before the current match, from Player_1's point of view.

    Shrunk towards zero: (wins - losses) / (meetings + prior). 55% of matches
    have no prior meeting at all, and the prior turns that into an honest zero
    with no special case — the same trick as form's cold start.

    H2H is the one feature genuinely orthogonal to Elo. A rating is one number
    per player and therefore forces transitivity; head-to-head is a property
    of a pair, and captures the matchup effects a rating cannot express.
    """
    record: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    edges: list[float] = []

    for row in matches.itertuples():
        pair = tuple(sorted((row.Player_1, row.Player_2)))
        first, second = record[pair]
        w_1, w_2 = (first, second) if pair[0] == row.Player_1 else (second, first)

        edges.append((w_1 - w_2) / (w_1 + w_2 + prior))

        record[pair][0 if row.Winner == pair[0] else 1] += 1

    return pd.Series(edges, index=matches.index), dict(record)


def h2h_edge(record: dict, player_a: str, player_b: str, prior: float = 2.0) -> float:
    """The same quantity for a pair of names, for prediction on a draw."""
    pair = tuple(sorted((player_a, player_b)))
    first, second = record.get(pair, [0, 0])
    w_a, w_b = (first, second) if pair[0] == player_a else (second, first)
    return (w_a - w_b) / (w_a + w_b + prior)


def _latest_rankings(matches: pd.DataFrame) -> dict[str, dict]:
    """Each player's rank and points as of their most recent match."""
    rank: dict[str, float] = {}
    pts: dict[str, float] = {}
    for row in matches.itertuples():        # already sorted by date
        rank[row.Player_1], pts[row.Player_1] = row.Rank_1, row.Pts_1
        rank[row.Player_2], pts[row.Player_2] = row.Rank_2, row.Pts_2
    return {"rank": rank, "pts": pts}


def build_features(matches: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Match history -> (feature table, player state at the end of the history).

    The left branch of the pipeline (training) takes the table, the right one
    (prediction on a draw) takes the state. Both come out of the same pass, so
    they cannot disagree about what a feature means.
    """
    elo_1, elo_2, elo_state = compute_elo(matches, general_weight)
    oh_1, oh_2, oh_state = compute_elo(matches, outdoor_hard_weight)

    out = matches[META_COLUMNS].copy()

    # Project-wide convention: always Player_1 minus Player_2.
    # For ranks that means LOWER is better, so the sign runs against intuition.
    out["rank_diff"] = matches["Rank_1"] - matches["Rank_2"]
    out["pts_diff"] = matches["Pts_1"] - matches["Pts_2"]
    out["elo_diff"] = elo_1 - elo_2
    out["outdoor_hard_elo_diff"] = oh_1 - oh_2

    form_states: dict[str, dict] = {}
    for w in FORM_WINDOWS:
        f_1, f_2, f_state = add_form(matches, window=w)
        out[f"form_{w}_diff"] = f_1 - f_2
        form_states[f"form_{w}"] = f_state

    out["h2h_edge"], h2h_record = add_h2h(matches)
    out["is_bo5"] = (matches["Best of"] == 5).astype(float)
    # Does the effect strengthen over five sets? The fitted weights say yes:
    # the interaction is as large as the base term.
    out["h2h_bo5"] = out["h2h_edge"] * out["is_bo5"]

    out["target"] = (matches["Winner"] == matches["Player_1"]).astype(int)

    state = {"elo": elo_state, "outdoor_hard_elo": oh_state,
             **form_states, **_latest_rankings(matches),
             "h2h_record": h2h_record}
    return out, state
