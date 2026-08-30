# us-open-prediction

A match-level probability model and Monte Carlo bracket simulator for the
2026 US Open men's singles. scikit-learn only — no gradient-boosting
libraries, no neural networks.

The machine learning solves exactly one problem: `P(A beats B)`. The
championship probability is not predicted, it is *simulated* — 10,000
runs of the real 128-player draw.

```
historical matches → player state (Elo, form, H2H) → match model
                                                          ↓
                                    official draw PDF → Monte Carlo → P(champion)
```

## Forecast

Generated 2026-08-30, before the first ball, from data through 2026-08-23.

| Player | R64 | R32 | R16 | QF | SF | F | **Win** |
|---|---|---|---|---|---|---|---|
| Alcaraz C. | 94.0% | 89.0% | 85.4% | 77.8% | 69.5% | 60.5% | **51.1%** |
| Zverev A. | 90.1% | 79.0% | 68.3% | 54.5% | 42.8% | 31.6% | **13.4%** |
| Djokovic N. | 80.7% | 66.3% | 57.7% | 44.1% | 32.1% | 11.3% | **7.2%** |
| Fritz T. | 93.0% | 77.4% | 54.6% | 37.1% | 23.9% | 12.0% | **3.3%** |
| Fils A. | 80.2% | 70.1% | 49.7% | 34.6% | 9.2% | 5.4% | **2.9%** |
| Auger-Aliassime F. | 78.3% | 61.5% | 47.3% | 31.6% | 18.2% | 9.2% | **2.8%** |

Full 128 rows in [`outputs/championship_probabilities.csv`](outputs/championship_probabilities.csv).

Jannik Sinner and João Fonseca withdrew and are absent from the draw.

Note that Zverev (Elo 1920) outranks Djokovic (Elo 1980) for the title
despite the lower rating: Zverev is in the top half and meets Alcaraz only
in the final, while Djokovic shares the bottom half and meets him in the
semifinal. Quarter 4 holds Alcaraz, Fils and Shelton together — Fils has the
draw's fourth-best Elo and still only 2.9%.

## The actual experiment

The point of the project is not "I trained a classifier." It is: **which
information genuinely improves predictive power?**

Evaluated on 2025 + 2026 (n = 4398), never seen in training. Δ is the
improvement in log loss over ATP-ranking-only, with a 95% interval from a
paired bootstrap over 5000 resamples.

| Feature set | accuracy | log loss | Brier | Δ log loss | 95% CI |
|---|---|---|---|---|---|
| ranking | 0.6478 | 0.6262 | 0.2188 | — | — |
| + form | 0.6471 | 0.6232 | 0.2175 | +0.0030 | [+0.0001, +0.0059] |
| + Elo | 0.6523 | 0.6185 | 0.2156 | +0.0077 | [+0.0030, +0.0122] |
| + H2H | 0.6528 | 0.6177 | 0.2152 | +0.0085 | [+0.0038, +0.0131] |
| + H2H×bo5 | 0.6526 | 0.6176 | 0.2152 | +0.0086 | [+0.0040, +0.0133] |
| **betting market** | 0.6811 | 0.5941 | 0.2048 | **+0.0328** | [+0.0255, +0.0403] |

Reproduce with `usopen compare`.

**Elo beats the ATP ranking, but only just.** +0.0077 log loss. Significant,
and consistent in direction across validation, test, and the hard-court
subset — but on the 1911-match test set alone it is indistinguishable from
noise. An effect this small needs four thousand matches to see.

**Form is real but redundant.** It beats ranking on its own (+0.0030), and
adds *nothing* once Elo is present: `+elo → full` is +0.0007, CI
[−0.0003, +0.0017]. Elo updates after every match, so it has already
absorbed recent form. Two different statements, easily conflated.

**Head-to-head works, and it is the only feature that is genuinely
orthogonal to Elo.** Elo assigns one number per player, which forces
transitivity. H2H is a property of a *pair* — it captures the matchup
effects a rating cannot express by construction. That is precisely why it
adds signal where form does not.

**H2H counts roughly double in five sets.** Fitting an interaction with
match format gives weights of +0.0363 on H2H and +0.0384 on H2H×bo5 — the
interaction is as large as the base effect. Restricting H2H to
bo5-outdoor-hard directly is not possible: 89% of such pairs have no prior
meeting.

**The market beats every model, everywhere, significantly** — 0.5941 against
our best 0.6176, a gap of +0.0243 with CI [+0.0177, +0.0306] on the 4390
matches carrying odds. That gap is the price of the information absent from public
historical data: injuries, motivation, travel, who is quietly carrying
something. It is the honest ceiling for this data source.

## Algorithms

| Model | log loss | log loss (symmetrised) | asymmetry |
|---|---|---|---|
| LogisticRegression | **0.6176** | 0.6176 | 0.0000 |
| HistGradientBoosting | 0.6244 | 0.6208 | 0.0471 |
| RandomForest | 0.6267 | 0.6239 | 0.0597 |

Logistic regression wins, which is the expected outcome rather than a
disappointing one: six smooth features, and `elo_diff` is already a logit by
construction, so there is little for a tree to carve up.

*asymmetry* is the mean of `|p(A,B) + p(B,A) − 1|`. It should be zero — the
answer must not depend on which player you list first. Tree models are off by
6 percentage points on average. Symmetrising them (averaging against the
mirrored input) recovers −0.003 log loss, which is **more than the entire H2H
feature gained**.

## Design decisions that matter

**Antisymmetry is enforced, not hoped for.** All features are differences,
`fit_intercept=False`, and `StandardScaler(with_mean=False)` — centring
subtracts a constant, which acts exactly like an intercept. With both,
`p(A,B) + p(B,A) = 1` holds to 1e-10 across all 16,384 pairs in the draw.
This is not cosmetic: bracket position is decided by the draw, so an
asymmetric model would make a player's title odds depend on which side of a
line they were printed.

**Feature window and training window are different windows.** Elo and form
are computed over the full history from 2000; the model is trained only on
2018 onward. Ratings need a burn-in — a player starting at 1500 on the first
day of the training set carries no information.

**Leakage control is a loop invariant.** Every state function reads the
player's state, writes it as the feature, and only then applies the match
result. Getting this backwards would leak the outcome into its own
prediction, and the symptom would be suspiciously good metrics rather than a
crash.

**Temporal split, never shuffled.** Train ≤ 2024 (16,481), validate 2025
(2,487), test 2026 (1,911). Half-open intervals `[start, end)` so no match
lands in two sets.

**Calibrated, and checked.** Across ten probability bins the model is within
±2% of observed frequencies — comparable to the bookmakers' own calibration.
On the 258 matches where it claims >85% confidence, it promised 90.5% and
delivered 91.1%. This matters because the simulator consumes probabilities,
not rankings: garbage in, garbage out.

## What did not work

Kept here deliberately. Negative results were the majority of the findings.

- **Rolling form, given Elo.** +0.0007, CI [−0.0003, +0.0017].
- **Form where it disagrees with Elo.** The hypothesis was that form matters
  specifically for players in decline. Stratifying by |z(form) − z(Elo)|, the
  top-divergence quartile gives +0.0004, CI [−0.0016, +0.0024]. The direct
  bias check runs the other way: where form favours a player, the model
  already slightly overshoots.
- **Elo trend** (rating now minus rating 20 matches ago). −0.0001, CI
  [−0.0006, +0.0004]. Djokovic, the motivating case, turns out to be only
  −25 over his last 20 matches; Alcaraz is −23. Three vivid early exits sat
  either side of an Australian Open final and a Wimbledon semifinal.
- **Correlated within-tournament shocks.** The simulator treats a player's
  seven matches as independent draws, which should overstate the favourite.
  Adding a per-tournament strength offset — Gaussian, and also one-sided to
  model injury — moves Alcaraz from 51.1% to 49.5% (σ=50) or 48.6% (20% of
  players lose 250 Elo). Calibrated against reality: a top-10 player exits in
  the first two rounds of a slam 11.0% of the time. The forecast is robust to
  this assumption, which was not the expected answer. Available as
  `--shock`, default off.

## Usage

```bash
uv pip install -e .
usopen train                              # fit, report metrics, save model + player state
usopen                                    # the forecast — title probability per player
```

`usopen` with no subcommand runs the forecast, which is the point of the
project. The rest:

```bash
usopen predict --sims 10000 --seed 0      # same thing, with knobs
usopen compare                            # the ablation and algorithm tables
usopen match "Alcaraz C." "Djokovic N."   # a single match, for a spot check
```

`usopen predict --shock 50` enables the correlated shock. `--seed` makes runs
reproducible; at the default 10,000 simulations a 30% probability carries a
standard error of about 0.5 points, so the second decimal is noise.

The saved model carries the data cut-off it was trained on. `predict` prints
that date in the table header and warns if the CSV has since moved ahead —
player state is frozen at training time, so a forecast run against fresher
data would otherwise silently use yesterday's ratings.

## Layout

```
src/usopen/
  data.py        loading, sentinel repair, tier canonicalisation, ordering
  elo.py         compute_elo(matches, weight_fn) → (before_1, before_2, final_state)
  features.py    form, head-to-head, feature assembly, FEATURE_SETS
  model.py       temporal split, training, metrics, market baseline, persistence
  ablation.py    the experiment: feature sets, algorithms, paired bootstrap
  draw.py        official draw PDF → 128 slots, name resolution
  predict.py     two names → probability
  tournament.py  probability/logit matrices, Monte Carlo
  cli.py         thin adapter; the engine does not depend on it
tests/unit/      loader contract, mutation-verified
```

Every state function returns `(series_1, series_2, final_state)`. The series
feed training; the final state feeds prediction. Both come from one pass, so
the two paths cannot drift apart.

## Data

ATP match results 2000–2026 from the Kaggle dataset
[`dissfya/atp-tennis-2000-2023daily-pull`](https://www.kaggle.com/datasets/dissfya/atp-tennis-2000-2023daily-pull),
which repackages tennis-data.co.uk. 68,591 matches. **Cut-off 2026-08-23**
(Cincinnati final); the Winston-Salem 250 played the following week is not
included.


The draw is parsed from the official
[2026 men's singles PDF](https://www.usopen.org/en_US/scores/draws/2026_MS_draw.pdf).
125 of 128 names match the dataset automatically, 6 through an explicit alias
table, and 3 debutants have no tour-level history and receive default ratings.

## Limitations

- **No serve statistics.** The source has no aces, service points, or break
  points. Serve profile is a style dimension Elo cannot see, and on hard
  courts it should matter. The Sackmann mirror has these columns through June
  2026, and ace rate is a stable player trait, so a snapshot would serve —
  this is the most promising unexplored feature.
- **No age.** A 39-year-old playing best-of-five on consecutive days is a
  different proposition from a 23-year-old, and nothing in these features
  captures it.
- **Split player identities.** 45 groups in the source CSV
  (`Mcdonald M.` / `McDonald M.`, `Herbert P.H.` / `Herbert P.` /
  `Herbert P-H.`), so some players' Elo is computed from partial history.
  Cannot be merged automatically — `Kuznetsov An.` and `Kuznetsov Al.` are
  probably different people.
- **Binary Elo.** A 6-0 6-0 win and a 7-6 7-6 win update the rating
  identically. `Score` parses cleanly for all 20,879 matches since 2018, so
  margin-of-victory Elo is available and untried.
- **The market is better, and we know by how much.**

## License

Code MIT. The underlying match data originates with tennis-data.co.uk;
Sackmann-format data is CC BY-NC-SA 4.0 and is not redistributed here.
