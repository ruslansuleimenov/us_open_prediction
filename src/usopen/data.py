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
SENTINEL_COLS = ["Rank_1", "Rank_2", "Pts_1", "Pts_2", "Odd_1", "Odd_2"]


def load_matches(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df[SENTINEL_COLS] = df[SENTINEL_COLS].replace(-1, np.nan)
    df["round_order"] = df["Round"].map(ROUND_ORDER)
    assert df["round_order"].notna().all()
    df = df.sort_values(["Date", "round_order"], kind="stable").reset_index(drop=True)
    return df
