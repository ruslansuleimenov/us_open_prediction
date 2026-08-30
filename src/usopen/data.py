"""Loading and normalising the raw match CSV."""

from pathlib import Path

import numpy as np
import pandas as pd

ROUND_ORDER = {
    "1st Round": 1,
    "2nd Round": 2,
    "3rd Round": 3,
    "4th Round": 4,
    "Quarterfinals": 5,
    "Semifinals": 6,
    "The Final": 7,
    "Round Robin": 0
}
# Missing values in these columns are encoded as -1, not as blanks.
SENTINEL_COLS = ["Rank_1", "Rank_2", "Pts_1", "Pts_2", "Odd_1", "Odd_2"]

# The ATP renamed its tiers in 2009. Identity entries are written out on
# purpose: .map() then yields NaN for anything unknown, which the assert
# below turns into a loud failure instead of a silent gap.
SERIES_CANONICAL = {
    "ATP250": "ATP250",
    "International": "ATP250",
    "International Gold": "ATP500",
    "Masters": "Masters 1000",
    "Grand Slam": "Grand Slam",
    "Masters 1000": "Masters 1000",
    "Masters Cup": "Masters Cup",
    "ATP500": "ATP500"
}


def load_matches(path: str | Path) -> pd.DataFrame:
    """Read the match CSV and put it into the shape the rest of the code expects.

    Repairs the -1 sentinels, canonicalises tournament tiers, and sorts
    chronologically. Rounds are ordered within a day as well: 17% of
    tournament-days span more than one round, and Elo must see the earlier
    round first.
    """
    df = pd.read_csv(path, parse_dates=["Date"])
    df[SENTINEL_COLS] = df[SENTINEL_COLS].replace(-1, np.nan)
    df["round_order"] = df["Round"].map(ROUND_ORDER)
    df["Series"] = df["Series"].map(SERIES_CANONICAL)
    assert df["Series"].notna().all()
    assert df["round_order"].notna().all()
    df = df.sort_values(["Date", "round_order"], kind="stable").reset_index(drop=True)
    return df
