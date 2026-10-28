"""Calibration: pure functions, no network.

1. Shrinkage pulls small-sample stats toward a prior before they reach the model,
   so three games of +22.5 margin don't read as a dominant team.
2. Blending mixes the model's probability with the market's in log-odds, so an
   edge is a modest nudge off the line instead of a 30-point disagreement.
"""
import dataclasses
import math

from . import config as C
from .models import Player, Split


def shrink(value: float, n: float, prior: float, k: float) -> float:
    """Weighted average of an observed value (worth n games) and a prior (worth k games)."""
    if n <= 0:
        return prior
    return (n * value + k * prior) / (n + k)


def shrink_split(s: Split, games: int, k: float = C.SHRINK_K_TEAM, default_prior: float = C.LEAGUE_MARGIN) -> Split:
    """Shrink a team stat toward last season's overall value.

    Home and away splits each cover about half the games played. Last season's value is
    overall only, so it is the prior for both splits.
    """
    prior = s.prev if s.prev is not None else default_prior
    return Split(overall=shrink(s.overall, games, prior, k),
                 home=shrink(s.home, games / 2, prior, k),
                 away=shrink(s.away, games / 2, prior, k),
                 prev=s.prev)


def shrink_ats(cover_pct: float, games_at_location: float) -> float:
    """Cover % has almost no carry-over between seasons, so the prior is always the league's 50%."""
    return shrink(cover_pct, games_at_location, C.LEAGUE_ATS_PCT, C.SHRINK_K_ATS)


def shrink_player(p: Player, prior_ppg: float | None) -> Player:
    """Shrink a player's points per game toward a prior.

    The prior should be last season's average. Without one (rookies, players who
    weren't top scorers), pass the current points line: it is the market's estimate
    of the player's role, which is better than no prior at all.
    """
    if prior_ppg is None:
        return p
    return dataclasses.replace(p, ppg=round(shrink(p.ppg, p.gp, prior_ppg, C.SHRINK_K_PLAYER), 1))


def logit(p: float) -> float:
    p = min(max(p, C.PROB_CLIP), 1 - C.PROB_CLIP)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def blend(model_p: float, market_p: float, model_weight: float) -> float:
    """Log-odds blend of the model and the no-vig market probability."""
    return sigmoid(model_weight * logit(model_p) + (1 - model_weight) * logit(market_p))
