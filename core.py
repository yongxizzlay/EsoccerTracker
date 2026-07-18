"""Core data + prediction logic for the esports soccer tracker.

Kept free of Streamlit imports so it can be unit-tested standalone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "date", "stage", "player_a", "player_b", "goals_a", "goals_b", "source",
]

# Shrinkage weight: how many "phantom" league-average matches to blend into a
# player's rates. Guards against tiny samples producing extreme lambdas.
SHRINK_K = 3.0

MAX_GOALS_GRID = 45  # per-player grid size for the Poisson joint distribution


def load_matches(path_or_buffer) -> pd.DataFrame:
    """Load and validate a matches CSV."""
    df = pd.read_csv(path_or_buffer)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["goals_a"] = pd.to_numeric(df["goals_a"], errors="coerce")
    df["goals_b"] = pd.to_numeric(df["goals_b"], errors="coerce")
    df = df.dropna(subset=["date", "player_a", "player_b", "goals_a", "goals_b"])
    df["goals_a"] = df["goals_a"].astype(int)
    df["goals_b"] = df["goals_b"].astype(int)
    df["total_goals"] = df["goals_a"] + df["goals_b"]
    return df.sort_values("date").reset_index(drop=True)


def long_form(df: pd.DataFrame) -> pd.DataFrame:
    """One row per player per match (perspective table)."""
    a = df.rename(columns={
        "player_a": "player", "player_b": "opponent",
        "goals_a": "gf", "goals_b": "ga",
    })[["date", "stage", "player", "opponent", "gf", "ga", "source"]]
    b = df.rename(columns={
        "player_b": "player", "player_a": "opponent",
        "goals_b": "gf", "goals_a": "ga",
    })[["date", "stage", "player", "opponent", "gf", "ga", "source"]]
    out = pd.concat([a, b], ignore_index=True)
    out["result"] = np.select(
        [out.gf > out.ga, out.gf == out.ga], ["W", "D"], default="L"
    )
    out["gd"] = out["gf"] - out["ga"]
    return out.sort_values("date").reset_index(drop=True)


def player_table(lf: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-player stats from the long-form table."""
    g = lf.groupby("player")
    t = pd.DataFrame({
        "matches": g.size(),
        "wins": g["result"].apply(lambda s: (s == "W").sum()),
        "draws": g["result"].apply(lambda s: (s == "D").sum()),
        "losses": g["result"].apply(lambda s: (s == "L").sum()),
        "gf": g["gf"].sum(),
        "ga": g["ga"].sum(),
    })
    t["win_rate"] = t["wins"] / t["matches"]
    t["gf_pm"] = t["gf"] / t["matches"]
    t["ga_pm"] = t["ga"] / t["matches"]
    t["gd_pm"] = t["gf_pm"] - t["ga_pm"]
    return t.sort_values(["win_rate", "gd_pm"], ascending=False)


@dataclass
class Prediction:
    player_a: str
    player_b: str
    lambda_a: float
    lambda_b: float
    p_win_a: float
    p_draw: float
    p_win_b: float
    expected_total: float
    line: float
    p_over: float
    p_under: float
    n_a: int
    n_b: int

    def fair_odds(self, p: float) -> float:
        return float("inf") if p <= 0 else 1.0 / p


def _shrunk_rate(total: float, n: int, league_rate: float) -> float:
    """Blend a player's observed per-match rate toward the league average."""
    return (total + SHRINK_K * league_rate) / (n + SHRINK_K)


def predict(df: pd.DataFrame, player_a: str, player_b: str,
            line: float = 16.5) -> Prediction:
    """Poisson model: attack/defense rates -> joint score distribution.

    lambda_A = shrunk_attack_A * (shrunk_defense_B / league_avg)
    Win/draw probs come from the joint independent-Poisson grid; over/under
    from the distribution of the total (sum of two Poissons is Poisson).
    """
    lf = long_form(df)
    league_avg = lf["gf"].mean()  # avg goals per player per match
    if not np.isfinite(league_avg) or league_avg <= 0:
        raise ValueError("Not enough data to build a model.")

    rates = {}
    for p in (player_a, player_b):
        rows = lf[lf.player == p]
        n = len(rows)
        if n == 0:
            raise ValueError(f"No matches on record for {p}.")
        rates[p] = {
            "n": n,
            "attack": _shrunk_rate(rows["gf"].sum(), n, league_avg),
            "defense": _shrunk_rate(rows["ga"].sum(), n, league_avg),
        }

    lam_a = rates[player_a]["attack"] * rates[player_b]["defense"] / league_avg
    lam_b = rates[player_b]["attack"] * rates[player_a]["defense"] / league_avg
    lam_a = float(np.clip(lam_a, 0.05, MAX_GOALS_GRID / 2))
    lam_b = float(np.clip(lam_b, 0.05, MAX_GOALS_GRID / 2))

    k = np.arange(MAX_GOALS_GRID + 1)
    log_fact = np.array([math.lgamma(i + 1) for i in k])
    pmf_a = np.exp(k * math.log(lam_a) - lam_a - log_fact)
    pmf_b = np.exp(k * math.log(lam_b) - lam_b - log_fact)
    joint = np.outer(pmf_a, pmf_b)  # joint[i, j] = P(A scores i, B scores j)

    p_win_a = float(np.tril(joint, -1).sum())  # i > j
    p_draw = float(np.trace(joint))
    p_win_b = float(np.triu(joint, 1).sum())  # j > i

    # Total goals: Poisson(lam_a + lam_b)
    lam_t = lam_a + lam_b
    kt = np.arange(2 * MAX_GOALS_GRID + 1)
    log_fact_t = np.array([math.lgamma(i + 1) for i in kt])
    pmf_t = np.exp(kt * math.log(lam_t) - lam_t - log_fact_t)
    p_under = float(pmf_t[kt <= math.floor(line)].sum())
    p_over = 1.0 - p_under

    return Prediction(
        player_a=player_a, player_b=player_b,
        lambda_a=lam_a, lambda_b=lam_b,
        p_win_a=p_win_a, p_draw=p_draw, p_win_b=p_win_b,
        expected_total=lam_t, line=line, p_over=p_over, p_under=p_under,
        n_a=rates[player_a]["n"], n_b=rates[player_b]["n"],
    )


def total_goals_pmf(pred: Prediction, upto: int | None = None) -> pd.DataFrame:
    """Distribution of total goals for a prediction (for charting)."""
    lam_t = pred.lambda_a + pred.lambda_b
    hi = upto or int(min(2 * MAX_GOALS_GRID, math.ceil(lam_t + 4 * math.sqrt(lam_t))))
    kt = np.arange(hi + 1)
    pmf = np.exp(kt * math.log(lam_t) - lam_t
                 - np.array([math.lgamma(i + 1) for i in kt]))
    return pd.DataFrame({"total_goals": kt, "probability": pmf})
