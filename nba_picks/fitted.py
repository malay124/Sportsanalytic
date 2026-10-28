"""Models fitted on 2023-24 to 2025-26 box scores (research/fit_models.py), using only the features that
passed out-of-sample tests in research/test_hypotheses.py. Pure functions, no network.

Kept: schedule-adjusted rating, back-to-backs, missing regulars and active-roster usage (team margin);
opponent defense and pace, positional defense and the efficient/top-option x weak-position interaction,
usage regression and vacated usage (player points).
Dropped, no measurable effect: bench net rating, bench TS%, bench minutes share, rotation depth, rest days
beyond the back-to-back, 3-in-4, 5-in-7, strength of schedule faced as a separate term, player back-to-backs.
"""
import math

from .models import Game, TeamContext, TeamPick

# --- team margin: home margin = HOME_COURT + sum(coef * (home - away)) ---
HOME_COURT = 1.8504
TEAM_COEF = {"srs": 0.9415, "back_to_back": -2.2149, "missing_ppg": -0.0896, "usage_edge": 0.0697}
MARGIN_SD = 13.988             # residual sd of the margin, sets the win probability

# --- player points above the shrunk season average ---
PLAYER_CONST = -1.1918         # with no teammates missing; base_ppg shrunk with k = SHRINK_K_PLAYER
PLAYER_COEF = {"opp_def": 0.0677, "opp_pace": 0.1267, "dvp": 0.2832, "eff": -0.3372, "eff_x_dvp": 0.1510,
               "top_at_pos": 0.6125, "top_x_dvp": 0.3859, "usg_c": -7.2807, "vac_min": 0.0211,
               "vac_x_usg": 6.4808}
LEAGUE_USG = 0.1959            # league mean usage share among 15+ mpg players
LEAGUE_PPU, PPU_SD = 1.0183, 0.1147   # points per used possession: mean and sd


def _phi(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2))


def predict_margin(home: TeamContext, vis: TeamContext, neutral: bool = False) -> tuple[float, float, dict]:
    """Expected home margin, home win probability, and each factor's contribution in points."""
    parts = {"home court": 0.0 if neutral else HOME_COURT}
    for k, c in TEAM_COEF.items():
        parts[k] = c * (float(getattr(home, k)) - float(getattr(vis, k)))
    margin = sum(parts.values())
    return margin, _phi(margin / MARGIN_SD), parts


def fitted_pick(game: Game, home: TeamContext, vis: TeamContext, neutral: bool = False) -> TeamPick:
    """A TeamPick whose confidence is a fitted win probability and whose score is the expected margin."""
    margin, p_home, parts = predict_margin(home, vis, neutral)
    labels = {"home court": "home court", "srs": "schedule-adjusted rating", "back_to_back": "back-to-back",
              "missing_ppg": "missing regulars", "usage_edge": "active-roster scoring"}
    reasons = [f"{labels[k]} {v:+.1f}" for k, v in parts.items() if abs(v) >= 0.1]
    reasons.append(f"expected margin {game.home} {margin:+.1f}")
    if abs(p_home - 0.5) < 0.02:
        pick, conf = "Toss Up", 50
    elif p_home > 0.5:
        pick, conf = game.home, round(100 * p_home)
    else:
        pick, conf = game.visitor, round(100 * (1 - p_home))
    return TeamPick(game, margin, pick, conf, reasons)


def project_points(base_ppg: float, *, opp_def: float = 0.0, opp_pace: float = 0.0, dvp: float = 0.0,
                   ppu: float = LEAGUE_PPU, usg: float = LEAGUE_USG, top_at_pos: bool = False,
                   vac_min: float = 0.0, vac_usg: float = 0.0) -> tuple[float, dict]:
    """Fitted points projection: shrunk season average plus matchup, usage and availability adjustments.

    opp_def   opponent's defensive rating minus league average (+ = worse defense)
    opp_pace  opponent's pace minus league average, possessions per 48
    dvp       points above their own average that players at this position (G/F/C) score vs this opponent
    ppu       this player's points per used possession (FGA + 0.44 FTA + TOV)
    usg       this player's usage share of team possessions while on court (0.25 = 25%)
    top_at_pos  this player is his team's highest-usage player at the position
    vac_min / vac_usg  season minutes and usage of teammates (20+ mpg) not playing
    """
    eff = (ppu - LEAGUE_PPU) / PPU_SD
    x = {"opp_def": opp_def, "opp_pace": opp_pace, "dvp": dvp, "eff": eff, "eff_x_dvp": eff * dvp,
         "top_at_pos": float(top_at_pos), "top_x_dvp": float(top_at_pos) * dvp, "usg_c": usg - LEAGUE_USG,
         "vac_min": vac_min, "vac_x_usg": vac_usg * usg}
    parts = {k: PLAYER_COEF[k] * v for k, v in x.items()}
    parts["intercept"] = PLAYER_CONST
    return round(base_ppg + sum(parts.values()), 1), parts
