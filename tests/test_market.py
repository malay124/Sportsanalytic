import unittest

from nba_picks.calibrate import blend
from nba_picks.config import MIN_EDGE, TEAM_MODEL_WEIGHT
from nba_picks.market import (compare_all, expected_value, hold, implied_prob, moneyline_edge, no_vig, prob_over,
                              prop_edge)
from nba_picks.models import Game, MoneyLine, PlayerProp, PropLine, TeamPick

GAME = Game("Boston Celtics", "Brooklyn Nets", "BOS", "BKN")


def pick(team, conf):
    return TeamPick(GAME, 0.0, team, conf, [])


def prop(proj):
    return PlayerProp("Jayson Tatum", "Boston Celtics", "Brooklyn Nets", 24.0, proj, 80, [])


class OddsMath(unittest.TestCase):
    def test_implied_prob(self):
        self.assertAlmostEqual(implied_prob(-150), 0.6)
        self.assertAlmostEqual(implied_prob(130), 100 / 230)
        self.assertAlmostEqual(implied_prob(100), 0.5)

    def test_no_vig_and_hold(self):
        self.assertEqual(no_vig(-110, -110), (0.5, 0.5))
        self.assertAlmostEqual(hold(-110, -110), 220 / 210 - 1)
        self.assertAlmostEqual(sum(no_vig(-150, 130)), 1.0)

    def test_expected_value(self):
        self.assertAlmostEqual(expected_value(0.6, 100), 0.2)
        self.assertAlmostEqual(expected_value(0.5, -110), 0.5 * 100 / 110 - 0.5)
        self.assertAlmostEqual(expected_value(implied_prob(-150), -150), 0.0)

    def test_prob_over(self):
        self.assertAlmostEqual(prob_over(22.5, 22.5), 0.5)
        self.assertAlmostEqual(prob_over(24.0, 20.0), 0.6857, places=3)   # sd = 1.373 + 1.408 * sqrt(24)
        self.assertLess(prob_over(10.0, 14.0), 0.5)


class Moneyline(unittest.TestCase):
    ML = MoneyLine("Boston Celtics", "Brooklyn Nets", -150, 130)

    def test_value_on_model_pick(self):
        e = moneyline_edge(pick("Boston Celtics", 71), self.ML)
        mkt = 0.6 / (0.6 + 100 / 230)
        p = blend(0.71, mkt, TEAM_MODEL_WEIGHT)
        self.assertEqual(e.side, "Boston Celtics")
        self.assertAlmostEqual(e.market_prob, round(mkt, 4))
        self.assertEqual(e.raw_model_prob, 0.71)
        self.assertAlmostEqual(e.model_prob, round(p, 4))
        self.assertAlmostEqual(e.ev, round(p * 100 / 150 - (1 - p), 4))
        self.assertLess(e.edge, 0.71 - mkt)   # blending shrinks the edge
        self.assertEqual(e.is_bet, e.edge >= MIN_EDGE and e.ev > 0)

    def test_value_against_model_pick_when_market_overprices_it(self):
        heavy = MoneyLine("Boston Celtics", "Brooklyn Nets", -400, 300)
        e = moneyline_edge(pick("Boston Celtics", 55), heavy)
        self.assertEqual(e.side, "Brooklyn Nets")
        self.assertGreater(e.edge, 0)

    def test_huge_disagreement_is_not_a_bet(self):
        longshot = MoneyLine("Boston Celtics", "Brooklyn Nets", -1800, 1000)   # market: BKN ~9%
        e = moneyline_edge(pick("Boston Celtics", 65), longshot)               # raw model: BKN 35%
        self.assertEqual(e.side, "Brooklyn Nets")
        self.assertFalse(e.is_bet)
        self.assertIn("too far apart", e.detail)

    def test_toss_up_never_flagged(self):
        e = moneyline_edge(pick("Toss Up", 50), MoneyLine("Boston Celtics", "Brooklyn Nets", 200, -240))
        self.assertGreater(e.ev, 0)
        self.assertFalse(e.is_bet)

    def test_small_edge_not_flagged(self):
        e = moneyline_edge(pick("Boston Celtics", 59), self.ML)   # market ~58%
        self.assertFalse(e.is_bet)


class Props(unittest.TestCase):
    def test_over_when_projection_above_line(self):
        e = prop_edge(prop(27.0), PropLine("Jayson Tatum", 23.5, -115, -105))
        self.assertEqual((e.side, e.is_bet), ("over", True))
        self.assertIn("+3.5", e.detail)

    def test_under_when_projection_below_line(self):
        e = prop_edge(prop(20.0), PropLine("Jayson Tatum", 23.5, -115, -105))
        self.assertEqual(e.side, "under")

    def test_compare_all_matches_and_sorts(self):
        edges = compare_all([pick("Boston Celtics", 71)], [Moneyline.ML],
                            [prop(27.0), PlayerProp("Nobody", "", "", 0, 10, 0, [])],
                            [PropLine("Jayson Tatum", 23.5, -115, -105)])
        self.assertEqual(len(edges), 2)
        self.assertGreaterEqual(edges[0].ev, edges[1].ev)


if __name__ == "__main__":
    unittest.main()
