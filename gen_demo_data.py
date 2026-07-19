"""Generate the bundled dataset: real EWC 2026 play-in results plus a
clearly-labeled SIMULATED match history for demoing charts and the model.

Run: python gen_demo_data.py  (rewrites data/matches.csv deterministically)
"""

import numpy as np
import pandas as pd

# Real results (sources: egamersworld.com event page; ea.com FC Pro Open 26
# Finals review — July 2026). Play-in scores are two-leg aggregates.
REAL = [
    ("2026-07-12", "EWC 2026 Play-Ins", "Caccia98", "Mark11", 9, 8),
    ("2026-07-12", "EWC 2026 Play-Ins", "KSofuoglu", "Brice", 13, 7),
    ("2026-07-12", "EWC 2026 Play-Ins", "Gueric", "levyfinn", 14, 7),
    ("2026-07-12", "EWC 2026 Play-Ins", "Obrun2002", "GuiBarros", 15, 2),
    ("2026-07-12", "EWC 2026 Play-Ins", "Neat", "Tekkz", 17, 6),
    ("2026-06-08", "FC Pro Open 26 Finals QF", "Vejrgang", "ManuBachoore", 6, 5),
    ("2026-06-08", "FC Pro Open 26 Finals QF", "EmreYilmaz", "HHezerS", 8, 7),
    ("2026-06-08", "FC Pro Open 26 Finals QF", "Danipitbull", "Nassada", 7, 6),
    ("2026-06-08", "FC Pro Open 26 Finals SF", "Vejrgang", "Levi de Weerd", 6, 5),
    ("2026-06-08", "FC Pro Open 26 Finals SF", "EmreYilmaz", "Danipitbull", 9, 5),
    ("2026-06-08", "FC Pro Open 26 Finals", "Vejrgang", "EmreYilmaz", 4, 1),
]

# Rough skill tiers for the simulated history (attack, defense) in goals per
# two-leg tie. Purely illustrative.
PLAYERS = {
    "Obrun2002": (11.5, 6.0), "Neat": (11.0, 6.5), "Gueric": (10.5, 7.0),
    "KSofuoglu": (10.0, 7.0), "Caccia98": (9.5, 7.5), "Mark11": (9.0, 8.0),
    "Tekkz": (10.0, 7.5), "levyfinn": (8.5, 8.5), "Brice": (8.5, 8.5),
    "GuiBarros": (7.5, 9.5), "Jonny": (9.5, 7.5),
}

STAGES = ["FC Pro Open (sim)", "FC Pro Leagues (sim)", "Showmatch (sim)"]


def main() -> None:
    rng = np.random.default_rng(26)
    names = list(PLAYERS)
    league_def = float(np.mean([d for _, d in PLAYERS.values()]))

    rows = []
    dates = pd.date_range("2026-03-02", "2026-07-10", freq="D")
    for _ in range(130):
        a, b = rng.choice(names, size=2, replace=False)
        atk_a, def_a = PLAYERS[a]
        atk_b, def_b = PLAYERS[b]
        lam_a = atk_a * def_b / league_def
        lam_b = atk_b * def_a / league_def
        rows.append((
            str(pd.Timestamp(rng.choice(dates)).date()),
            str(rng.choice(STAGES)),
            a, b,
            int(rng.poisson(lam_a)), int(rng.poisson(lam_b)),
            "Simulated demo",
        ))

    real_rows = [(*r, "EWC 2026 Play-Ins (real)") for r in REAL]
    df = pd.DataFrame(
        rows + real_rows,
        columns=["date", "stage", "player_a", "player_b",
                 "goals_a", "goals_b", "source"],
    ).sort_values("date")
    df.to_csv("data/matches.csv", index=False)
    print(f"Wrote data/matches.csv with {len(df)} matches "
          f"({len(real_rows)} real, {len(rows)} simulated).")


if __name__ == "__main__":
    main()
