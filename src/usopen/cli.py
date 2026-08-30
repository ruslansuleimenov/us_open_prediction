"""Command line interface. A thin adapter — all the logic lives in the modules."""

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from usopen.ablation import compare_models, run_ablation
from usopen.data import load_matches
from usopen.draw import parse_draw, resolve_names
from usopen.features import build_features
from usopen.model import (evaluate, load, market_proba, predict_proba, save,
                          split_by_time, train_model, usable_odds)
from usopen.paths import DATA_RAW, MATCHES_CSV, MODELS, OUTPUTS
from usopen.predict import FEATURE_SET, Predictor
from usopen.tournament import (build_logit_matrix, elo_shock_coefficient,
                               run_simulations)

app = typer.Typer(
    add_completion=False,
    invoke_without_command=True,
    help="US Open forecast. With no subcommand, prints title probabilities.",
)
console = Console()

DEFAULT_MODEL = MODELS / "usopen_2026.joblib"
DEFAULT_DRAW = DATA_RAW / "draw_2026_ms.pdf"
DEFAULT_OUT = OUTPUTS / "championship_probabilities.csv"
DEFAULT_SIMS = 10_000
DEFAULT_TOP = 15


def _predictor(model, state, draw: pd.DataFrame) -> Predictor:
    """Defaults for debutants: the median of this draw's actual Q/W/L entrants."""
    seeded = draw[draw.entry.isin(["Q", "W", "L"]) & draw.known]
    return Predictor(
        model, state,
        default_rank=float(pd.Series([state["rank"][n] for n in seeded.name]).median()),
        default_pts=float(pd.Series([state["pts"][n] for n in seeded.name]).median()),
    )


@app.callback()
def _default(ctx: typer.Context) -> None:
    """No subcommand runs the forecast — it is what the project is for."""
    if ctx.invoked_subcommand is None:
        _run_predict(DEFAULT_DRAW, DEFAULT_MODEL, DEFAULT_SIMS, 0.0, 0,
                     DEFAULT_TOP, DEFAULT_OUT)


@app.command()
def train(
    matches: Path = typer.Option(MATCHES_CSV, help="Match history CSV."),
    out: Path = typer.Option(DEFAULT_MODEL, help="Where to write model and state."),
) -> None:
    """Fit the model on the history and save it alongside the player state."""
    with console.status("building features..."):
        df = load_matches(matches)
        features, state = build_features(df)
    cutoff = df["Date"].max()
    train_set, val, test = split_by_time(features)
    model = train_model(train_set, FEATURE_SET)

    table = Table(title=f"Quality — trained on {len(train_set)} matches, "
                        f"data through {cutoff:%Y-%m-%d}")
    for col in ["set", "n", "accuracy", "log loss", "brier", "roc auc"]:
        table.add_column(col, justify="left" if col == "set" else "right")
    for name, data in [("validation", val), ("test", test)]:
        r = evaluate(data["target"], predict_proba(model, data, FEATURE_SET))
        table.add_row(name, str(len(data)), f"{r['accuracy']:.4f}",
                      f"{r['log_loss']:.4f}", f"{r['brier']:.4f}", f"{r['roc_auc']:.4f}")
    ok = test[usable_odds(test)]
    r = evaluate(ok["target"], market_proba(ok).values)
    table.add_row("market (test)", str(len(ok)), f"{r['accuracy']:.4f}",
                  f"{r['log_loss']:.4f}", f"{r['brier']:.4f}", f"{r['roc_auc']:.4f}")
    console.print(table)

    out.parent.mkdir(parents=True, exist_ok=True)
    save(model, state, out, data_cutoff=cutoff)
    console.print(f"[green]saved:[/green] {out}")


def _run_predict(draw_pdf: Path, model_path: Path, sims: int, shock: float,
                 seed: int, top: int, out: Path) -> None:
    if not model_path.exists():
        console.print(f"[red]no model file:[/red] {model_path}\n"
                      f"run [bold]usopen train[/bold] first")
        raise typer.Exit(1)

    model, state, cutoff = load(model_path)
    matches = load_matches(MATCHES_CSV)
    known = set(pd.concat([matches.Player_1, matches.Player_2]).unique())
    draw = resolve_names(parse_draw(draw_pdf), known).sort_values("slot")

    latest = matches["Date"].max()
    if cutoff is not None and latest > cutoff:
        console.print(f"[yellow]model is stale[/yellow] — trained on data through "
                      f"{cutoff:%Y-%m-%d}, the CSV now runs to {latest:%Y-%m-%d}. "
                      f"Re-run [bold]usopen train[/bold] to use the newer matches.")

    missing = (~draw.known).sum()
    if missing:
        console.print(f"[yellow]{missing} players have no history[/yellow] — "
                      f"they are given Elo 1500")

    players = list(draw["name"])
    with console.status(f"{sims} tournament runs..."):
        logits = build_logit_matrix(players, _predictor(model, state, draw))
        result = run_simulations(players, logits, n_sims=sims, seed=seed,
                                 shock_sigma=shock,
                                 shock_coef=elo_shock_coefficient(model))

    stamp = "unknown cut-off" if cutoff is None else f"data through {cutoff:%Y-%m-%d}"
    title = f"US Open 2026 — {sims} simulations, {stamp}"
    if shock:
        title += f", shock σ={shock:g}"
    table = Table(title=title)
    table.add_column("player")
    for col in result.columns:
        table.add_column(col, justify="right")
    for name, row in result.head(top).iterrows():
        cells = [f"{v:.1%}" if v >= 0.001 else "<0.1%" for v in row]
        table.add_row(name, *cells)
    console.print(table)

    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out)
    console.print(f"[green]saved:[/green] {out}")


@app.command()
def predict(
    draw_pdf: Path = typer.Option(DEFAULT_DRAW, "--draw", help="Official draw PDF."),
    model_path: Path = typer.Option(DEFAULT_MODEL, "--model", help="Trained model file."),
    sims: int = typer.Option(DEFAULT_SIMS, "--sims", help="Number of tournament runs."),
    shock: float = typer.Option(0.0, "--shock",
        help="Std-dev in Elo points of a per-tournament strength offset, drawn "
             "once per player per run. Models a player who arrives off form "
             "playing worse all fortnight. Measured: the forecast is robust to it."),
    seed: int = typer.Option(0, help="Random seed, for reproducible runs."),
    top: int = typer.Option(DEFAULT_TOP, help="How many rows to print."),
    out: Path = typer.Option(DEFAULT_OUT, help="Where to write the CSV."),
) -> None:
    """Title probability for every player in the draw.

    Computed by Monte Carlo: the model gives the probability of each
    individual match, and the whole bracket is replayed `--sims` times.
    Running `usopen` with no subcommand does the same thing.
    """
    _run_predict(draw_pdf, model_path, sims, shock, seed, top, out)


@app.command()
def compare(
    matches: Path = typer.Option(MATCHES_CSV, help="Match history CSV."),
    on: str = typer.Option("both", help="Evaluate on: val, test or both."),
    save_csv: bool = typer.Option(True, help="Write the tables to outputs/."),
) -> None:
    """The experiment: which information actually improves predictive power."""
    features, _ = build_features(load_matches(matches))
    train_set, val, test = split_by_time(features)
    data = {"val": val, "test": test, "both": pd.concat([val, test])}[on]

    abl = run_ablation(train_set, data)
    t = Table(title=f"Feature sets ({on}, n={len(data)}) — Δ measured against 'ranking'")
    for c in ["set", "accuracy", "log loss", "brier", "roc auc", "Δ log loss", "95% CI"]:
        t.add_column(c, justify="left" if c == "set" else "right")
    for name, r in abl.iterrows():
        ci = "—" if pd.isna(r.ci_lo) else f"[{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]"
        delta = "—" if pd.isna(r["Δ log loss"]) else f"{r['Δ log loss']:+.4f}"
        style = "bold green" if r["significant"] and name != "ranking" else None
        t.add_row(name, f"{r.accuracy:.4f}", f"{r.log_loss:.4f}", f"{r.brier:.4f}",
                  f"{r.roc_auc:.4f}", delta, ci, style=style)
    console.print(t)

    cmp = compare_models(train_set, data)
    t2 = Table(title="Algorithms on the best feature set")
    for c in ["model", "log loss", "log loss (symmetrised)", "asymmetry", "accuracy", "brier"]:
        t2.add_column(c, justify="left" if c == "model" else "right")
    for name, r in cmp.iterrows():
        t2.add_row(name, f"{r['log loss']:.4f}", f"{r['log loss (symmetrised)']:.4f}",
                   f"{r['asymmetry']:.4f}", f"{r.accuracy:.4f}", f"{r.brier:.4f}")
    console.print(t2)
    console.print("[dim]asymmetry = mean |p(A,B) + p(B,A) − 1|; "
                  "zero for logistic regression by construction[/dim]")

    if save_csv:
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        abl.to_csv(OUTPUTS / "ablation.csv")
        cmp.to_csv(OUTPUTS / "model_comparison.csv")
        console.print(f"[green]saved:[/green] {OUTPUTS}/ablation.csv, model_comparison.csv")


@app.command()
def match(player_a: str, player_b: str,
          model_path: Path = typer.Option(DEFAULT_MODEL, "--model")) -> None:
    """One match: usopen match "Alcaraz C." "Djokovic N." """
    if not model_path.exists():
        console.print(f"[red]no model file:[/red] {model_path}\n"
                      f"run [bold]usopen train[/bold] first")
        raise typer.Exit(1)
    model, state, _ = load(model_path)
    p = Predictor(model, state, default_rank=145.0, default_pts=427.0)
    pa = p.predict_match(player_a, player_b)
    console.print(f"  [bold]{player_a}[/bold]  {pa:.1%}")
    console.print(f"  [bold]{player_b}[/bold]  {1 - pa:.1%}")


if __name__ == "__main__":
    app()
