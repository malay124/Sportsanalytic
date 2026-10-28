import unittest

from nba_picks.fitted import (HOME_COURT, LEAGUE_PPU, LEAGUE_USG, MARGIN_SD, PLAYER_CONST, PLAYER_COEF, TEAM_COEF,
                              fitted_pick, predict_margin, project_points)
from nba_picks.models import Game, TeamContext

GAME = Game("Boston Celtics", "Brooklyn Nets", "BOS", "BKN")


class Margin(unittest.TestCase):
    def test_even_teams_get_home_court_only(self):
        m, p, _ = predict_margin(TeamContext(), TeamContext())
        self.assertAlmostEqual(m, HOME_COURT)
        self.assertGreater(p, 0.5)
        m, p, _ = predict_margin(TeamContext(), TeamContext(), neutral=True)
        self.assertEqual((m, p), (0.0, 0.5))

    def test_factors_add_up(self):
        home = TeamContext(srs=6.0, back_to_back=True, missing_ppg=20.0)
        vis = TeamContext(srs=-2.0, usage_edge=-3.0)
        m, _, parts = predict_margin(home, vis)
        expect = HOME_COURT + TEAM_COEF["srs"] * 8 + TEAM_COEF["back_to_back"] + TEAM_COEF["missing_ppg"] * 20 \
            + TEAM_COEF["usage_edge"] * 3
        self.assertAlmostEqual(m, expect)
        self.assertAlmostEqual(sum(parts.values()), expect)

    def test_pick_uses_win_probability(self):
        tp = fitted_pick(GAME, TeamContext(), TeamContext(srs=10.0))   # visitor much better
        self.assertEqual(tp.pick, "Brooklyn Nets")
        self.assertLess(tp.score, 0)
        self.assertGreater(tp.confidence, 70)
        self.assertEqual(fitted_pick(GAME, TeamContext(), TeamContext(), neutral=True).pick, "Toss Up")
        self.assertGreater(MARGIN_SD, 10)


class Points(unittest.TestCase):
    def test_average_player_in_neutral_spot(self):
        proj, _ = project_points(20.0)
        self.assertAlmostEqual(proj, round(20.0 + PLAYER_CONST, 1))

    def test_top_option_vs_weak_position_gets_boost(self):
        neutral, _ = project_points(20.0, ppu=LEAGUE_PPU + 0.1147, usg=LEAGUE_USG, top_at_pos=True)
        soft, _ = project_points(20.0, ppu=LEAGUE_PPU + 0.1147, usg=LEAGUE_USG, top_at_pos=True, dvp=1.0)
        expect = PLAYER_COEF["dvp"] + PLAYER_COEF["eff_x_dvp"] + PLAYER_COEF["top_x_dvp"]
        self.assertAlmostEqual(soft - neutral, expect, places=1)

    def test_vacated_usage_scales_with_own_usage(self):
        lo = project_points(20.0, usg=0.15, vac_usg=0.3)[0] - project_points(20.0, usg=0.15)[0]
        hi = project_points(20.0, usg=0.30, vac_usg=0.3)[0] - project_points(20.0, usg=0.30)[0]
        self.assertGreater(hi, lo)


if __name__ == "__main__":
    unittest.main()
