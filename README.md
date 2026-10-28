# Sportsanalytic
A way to analyze and bet on games

NBA game and player-points model, as pure Python logic (no network, no email, no betting). Feed it the data;
it returns picks, projections and edges against sportsbook lines.

## Package (`nba_picks/`)

| Module | What it does |
|---|---|
| `fitted.py` | **Data-backed models.** `predict_margin` / `fitted_pick`: expected margin and win probability from schedule-adjusted rating, back-to-backs, missing regulars and active-roster usage. `project_points`: player points from season average plus opponent defense and pace, defense vs position, the efficient/top-option x weak-position matchup, usage regression and vacated usage. |
| `market.py` | American odds to implied/no-vig probability, expected value, moneyline and points-prop edges, `compare_all`. |
| `calibrate.py` | Early-season shrinkage toward last season, log-odds blending of model and market. |
| `scoring.py` | The original 5-factor heuristic model (kept for comparison) and pick-log grading. |
| `config.py` | Every weight and threshold, with where each value came from. |
| `models.py` | Dataclasses for games, team context, players, lines and edges. |

```bash
python -m unittest discover -s tests -t .
```

## Research (`research/`)

How each idea was tested, out of sample (fit on 2023-24 + 2024-25, scored on 2025-26), against both the
actual margin and the closing spread. Needs `numpy pandas scipy scikit-learn statsmodels`.

```bash
python research/build_dataset.py DATA_DIR       # ESPN box scores + closing lines, 3 seasons
python research/features.py DATA_DIR            # pre-game features, no leakage
python research/test_hypotheses.py DATA_DIR     # -> research/results.txt
python research/test_compound.py DATA_DIR       # -> research/results_compound.txt
python research/fit_models.py DATA_DIR          # coefficients used in nba_picks/fitted.py
```

### Findings

| Idea | Improves the model? | Beats the closing line? |
|---|---|---|
| Strength of schedule | Slightly (schedule-adjusted rating) | No |
| Back-to-back | Yes, about -2.2 pts | No |
| 3-in-4, 5-in-7, rest days, 3-in-4 B2B vs plain B2B | No extra effect | No |
| Bench net, bench TS%, bench minutes, depth | No | No |
| Usage / missing regulars (team) | Yes, strongest team factor | No |
| Positional matchup, top option vs weak position (players) | Yes, +0.3 to +0.7 pts | Not testable: no prop-line history |
| Usage regression and vacated usage (players) | Yes, strongest player factor | Not testable |
| Stacking factors | Adds up linearly, no compounding | No |

The closing line already prices every team factor tested. Player-prop edges are measured against season
averages, not real lines, so forward-test before trusting them.
