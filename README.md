# ⚽ Esports Soccer Tracker

A Streamlit dashboard for tracking the pro esports soccer scene (EA SPORTS FC /
FC Pro) with stats, charts, and Poisson-model predictions for match winners and
over/under total goals.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

## What's inside

- **Overview** — headline stats, total-goals distribution with your O/U line
  marked, weekly scoring trend.
- **Players** — win-rate leaderboard, attack vs defense per match, cumulative
  goal-difference form chart, full stats table.
- **Matches** — browsable match log, a form to log new results, CSV
  import/export from the sidebar.
- **Predictions** — pick two players and a total-goals line; get win/draw/win
  probabilities, over/under probabilities, fair (break-even) decimal odds, and
  the modelled total-goals distribution.

## Data

`data/matches.csv` ships with **5 real results** from the FC Pro 26 World
Championship Play-Ins at Esports World Cup 2026, plus **130 simulated matches**
(clearly labeled `Simulated demo` in the `source` column) so the charts and
model have enough history to be interesting. Untick *"Include simulated demo
matches"* in the sidebar to work from real results only, or import your own CSV
with columns:

```
date, stage, player_a, player_b, goals_a, goals_b, source
```

Regenerate the bundled dataset any time with `python gen_demo_data.py`.

## The model (and a warning)

Attack/defense rates per player, shrunk toward the league average, feed
independent Poisson score distributions; win/draw probabilities come from the
joint grid and total goals from `Poisson(λ_A + λ_B)`. Fair odds are
`1 ÷ probability`.

**This is not betting advice.** The sample is small, esports form is volatile,
and the model knows nothing about patches, formats, lag, or motivation. If you
bet, treat the output as one rough input, expect it to be wrong often, and only
stake what you can afford to lose.
