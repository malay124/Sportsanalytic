"""Compare model output to sportsbook prices: pure functions, no network.

Odds are American (-150, +130). "Market prob" always means the no-vig probability:
the book's two implied probabilities rescaled to sum to 1.

The model's raw probability is blended with the market's (see calibrate.blend) before
edge and EV are computed. Team confidence stands in for the raw win probability; it is a
heuristic score until the pick log shows each tier hits at its stated rate.
"""
import math

from . import config as C
from .calibrate import blend
from .models import MarketEdge, MoneyLine, PlayerProp, PropLine, TeamPick


def implied_prob(odds: int) -> float:
    """Break-even probability of American odds, vig included."""
    if odds < 0:
        return -odds / (-odds + 100)
    return 100 / (odds + 100)


def no_vig(odds_a: int, odds_b: int) -> tuple[float, float]:
    a, b = implied_prob(odds_a), implied_prob(odds_b)
    return a / (a + b), b / (a + b)


def hold(odds_a: int, odds_b: int) -> float:
    """The book's margin on a two-way market, e.g. 0.045 for -110/-110."""
    return implied_prob(odds_a) + implied_prob(odds_b) - 1


def profit_per_unit(odds: int) -> float:
    return 100 / -odds if odds < 0 else odds / 100


def expected_value(prob: float, odds: int) -> float:
    """Expected profit per 1 unit staked at these odds, if prob is the true win chance."""
    return prob * profit_per_unit(odds) - (1 - prob)


def _best_side(sides, flaggable: bool, model_weight: float) -> MarketEdge:
    """sides: (label, side, odds, raw_model_prob, market_prob, detail). Returns the higher-EV side.

    A side is a bet when, after blending, it has positive EV and at least MIN_EDGE, and the raw
    model is within MAX_DISAGREEMENT of the market (a bigger gap means bad inputs, not value).
    """
    edges = []
    for label, side, odds, raw, kp, detail in sides:
        mp = blend(raw, kp, model_weight)
        ev = expected_value(mp, odds)
        sane = abs(raw - kp) <= C.MAX_DISAGREEMENT
        if not sane:
            detail = f"{detail}; " if detail else ""
            detail += f"model {raw:.0%} vs market {kp:.0%}: too far apart, check inputs"
        edges.append(MarketEdge(label, side, odds, round(mp, 4), round(kp, 4), round(mp - kp, 4), round(ev, 4),
                                flaggable and sane and ev > 0 and mp - kp >= C.MIN_EDGE, detail, round(raw, 4)))
    return max(edges, key=lambda e: e.ev)


def moneyline_edge(pick: TeamPick, ml: MoneyLine) -> MarketEdge:
    """Model vs moneyline. Toss ups are reported but never flagged as bets."""
    p = pick.confidence / 100
    home_p = p if pick.pick == ml.home else (1 - p if pick.pick == ml.visitor else 0.5)
    mk_home, mk_vis = no_vig(ml.home_odds, ml.visitor_odds)
    return _best_side([
        (f"{ml.home} ML", ml.home, ml.home_odds, home_p, mk_home, ""),
        (f"{ml.visitor} ML", ml.visitor, ml.visitor_odds, 1 - home_p, mk_vis, ""),
    ], flaggable=not pick.is_toss_up, model_weight=C.TEAM_MODEL_WEIGHT)


def prob_over(projection: float, line: float) -> float:
    """P(points > line) under the Normal assumption in config. Integer lines ignore pushes."""
    sd = C.PROP_SD_A + C.PROP_SD_B * math.sqrt(max(projection, 0.0))
    return 0.5 * math.erfc((line - projection) / (sd * math.sqrt(2)))


def prop_edge(prop: PlayerProp, pl: PropLine) -> MarketEdge:
    over = prob_over(prop.proj_pts, pl.line)
    mk_over, mk_under = no_vig(pl.over_odds, pl.under_odds)
    gap = f"proj {prop.proj_pts} vs line {pl.line} ({prop.proj_pts - pl.line:+.1f})"
    return _best_side([
        (f"{pl.player} over {pl.line}", "over", pl.over_odds, over, mk_over, gap),
        (f"{pl.player} under {pl.line}", "under", pl.under_odds, 1 - over, mk_under, gap),
    ], flaggable=True, model_weight=C.PROP_MODEL_WEIGHT)


def compare_all(picks, moneylines, props, prop_lines) -> list[MarketEdge]:
    """Every market that has both a model view and a line, best EV first.

    Moneylines match picks by home team; prop lines match projections by player name.
    Pass player_props(..., min_conf=0) so every projection can be priced, not just the 70%+ ones.
    """
    by_home = {ml.home: ml for ml in moneylines}
    by_player = {pl.player: pl for pl in prop_lines}
    edges = [moneyline_edge(tp, by_home[tp.game.home]) for tp in picks if tp.game.home in by_home]
    edges += [prop_edge(p, by_player[p.player]) for p in props if p.player in by_player]
    return sorted(edges, key=lambda e: -e.ev)
