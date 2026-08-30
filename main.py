"""The whole pipeline: data -> model -> draw -> champion forecast."""

import pandas as pd

from usopen.data import load_matches
from usopen.draw import parse_draw, resolve_names
from usopen.features import build_features
from usopen.model import save, split_by_time, train_model
from usopen.paths import DATA_RAW, MATCHES_CSV, MODELS, OUTPUTS
from usopen.predict import FEATURE_SET, Predictor
from usopen.tournament import (build_logit_matrix, elo_shock_coefficient,
                                run_simulations)

N_SIMS = 10_000


def main() -> None:
    print("loading matches...")
    matches = load_matches(MATCHES_CSV)
    features, state = build_features(matches)

    print("training the model...")
    train, _val, _test = split_by_time(features)
    model = train_model(train, FEATURE_SET)
    save(model, state, MODELS / "usopen_2026.joblib",
         data_cutoff=matches["Date"].max())

    print("parsing the draw...")
    known = set(pd.concat([matches.Player_1, matches.Player_2]).unique())
    draw = resolve_names(parse_draw(DATA_RAW / "draw_2026_ms.pdf"), known)

    seeded = draw[draw.entry.isin(["Q", "W", "L"]) & draw.known]
    predictor = Predictor(
        model, state,
        default_rank=float(pd.Series([state["rank"][n] for n in seeded.name]).median()),
        default_pts=float(pd.Series([state["pts"][n] for n in seeded.name]).median()),
    )

    players = list(draw.sort_values("slot")["name"])
    print(f"building the {len(players)}x{len(players)} probability matrix...")
    logits = build_logit_matrix(players, predictor)

    print(f"simulating {N_SIMS} tournaments...")
    result = run_simulations(players, logits, n_sims=N_SIMS)

    out = OUTPUTS / "championship_probabilities.csv"
    result.to_csv(out)
    print(f"\nsaved: {out}\n")
    print(result.head(15).to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
