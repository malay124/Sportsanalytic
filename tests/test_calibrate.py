import unittest

from nba_picks.calibrate import blend, logit, shrink, shrink_ats, shrink_player, shrink_split
from nba_picks.models import Player, Split


class Shrinkage(unittest.TestCase):
    def test_shrink_weights(self):
        self.assertEqual(shrink(20.0, 10, 10.0, 10), 15.0)    # n == k: halfway
        self.assertEqual(shrink(20.0, 0, 10.0, 10), 10.0)     # no games: all prior
        self.assertAlmostEqual(shrink(20.0, 1000, 10.0, 10), 19.9, places=1)

    def test_early_season_margin_pulled_to_last_season(self):
        s = shrink_split(Split(overall=22.5, home=30.0, away=15.0, prev=2.0), games=3)
        self.assertAlmostEqual(s.overall, (3 * 22.5 + 10 * 2.0) / 13)
        self.assertAlmostEqual(s.home, (1.5 * 30.0 + 10 * 2.0) / 11.5)
        self.assertEqual(s.prev, 2.0)

    def test_no_prior_falls_back_to_league_default(self):
        s = shrink_split(Split(overall=10.0, home=10.0, away=10.0), games=10)
        self.assertAlmostEqual(s.overall, 5.0)

    def test_ats_regresses_to_fifty(self):
        self.assertAlmostEqual(shrink_ats(100.0, 2), (200 + 25 * 50) / 27)

    def test_player(self):
        p = shrink_player(Player("Duncan Robinson", "NY", 18.9, 12.0, gp=3), 12.5)
        self.assertAlmostEqual(p.ppg, round((3 * 18.9 + 3 * 12.5) / 6, 1))
        self.assertEqual(p.fga, 12.0)
        self.assertEqual(shrink_player(p, None), p)


class Blending(unittest.TestCase):
    def test_endpoints(self):
        self.assertAlmostEqual(blend(0.7, 0.4, 1.0), 0.7)
        self.assertAlmostEqual(blend(0.7, 0.4, 0.0), 0.4)

    def test_between_and_closer_to_market(self):
        p = blend(0.71, 0.40, 0.25)
        self.assertTrue(0.40 < p < 0.555)   # below the midpoint: market gets the bigger say

    def test_clipping(self):
        self.assertAlmostEqual(logit(1.0), logit(0.99))


if __name__ == "__main__":
    unittest.main()
