import unittest

from nba_picks.models import Game, Injury, Player, TeamContext
from nba_picks.scoring import (confidence_from_score, grade_pick, player_props, project_player, quarter_score,
                               score_game, tier_stats)

GAME = Game("Boston Celtics", "Brooklyn Nets", "BOS", "BKN", "BOS -12.5")
STAR = Player("Jaylen Brown", "BOS", 27.0, 20.0)
KEY = Player("Derrick White", "BOS", 16.0, 13.0)
PLAYERS = {p.name: p for p in (STAR, KEY)}


class ScoreGame(unittest.TestCase):
    def test_all_factors_neutral_is_toss_up(self):
        tp = score_game(GAME, TeamContext(), TeamContext(), {})
        self.assertEqual((tp.pick, tp.confidence, tp.score), ("Toss Up", 50, 0.0))

    def test_factor_sum_matches_original_weights(self):
        # visitor B2B +3, home star out -5, margin +4 -> +1.5, ATS -6 -> -1, quarters even
        home = TeamContext(margin=6.0, ats_cover_pct=44.0, injuries=[Injury("Jaylen Brown", "Out")])
        vis = TeamContext(back_to_back=True, margin=2.0, ats_cover_pct=50.0)
        tp = score_game(GAME, home, vis, PLAYERS)
        self.assertAlmostEqual(tp.score, -1.5)
        self.assertEqual(tp.pick, "Brooklyn Nets")
        self.assertEqual(tp.confidence, 59)   # 50 + 1.5/8*49
        self.assertIn("STAR OUT: Jaylen Brown", " ".join(tp.reasons))

    def test_key_player_and_questionable_status(self):
        home = TeamContext(injuries=[Injury("Derrick White", "Doubtful"), Injury("Jaylen Brown", "Day-To-Day")])
        self.assertAlmostEqual(score_game(GAME, home, TeamContext(), PLAYERS).score, -3.0)

    def test_both_on_back_to_back_cancels(self):
        tp = score_game(GAME, TeamContext(back_to_back=True), TeamContext(back_to_back=True), {})
        self.assertEqual(tp.score, 0.0)

    def test_quarter_weights(self):
        self.assertAlmostEqual(quarter_score({"q1": 1, "q2": 1, "q3": 1, "q4": 1}), 1.0)
        self.assertAlmostEqual(quarter_score({"q3": 5}), 2.0)
        home = TeamContext(quarters={"q1": 0, "q2": 0, "q3": 5, "q4": 0})
        self.assertAlmostEqual(score_game(GAME, home, TeamContext(), {}).score, 1.0)

    def test_confidence_band_and_cap(self):
        self.assertEqual(confidence_from_score(0.5), 50)
        self.assertEqual(confidence_from_score(-0.6), 53)
        self.assertEqual(confidence_from_score(20), 99)


class PlayerProps(unittest.TestCase):
    def test_projection_stacks_factors(self):
        pick = score_game(GAME, TeamContext(margin=10.0), TeamContext(), {})   # Boston favored
        teammate_out = [Injury("Derrick White", "Out")]
        prop = project_player(STAR, "Boston Celtics", "Brooklyn Nets", pick, 120.0, teammate_out, PLAYERS)
        self.assertAlmostEqual(prop.proj_pts, round(27 * 120 / 115 * 1.02 * 1.15, 1))
        self.assertEqual(prop.confidence, 99)   # 50+20+10+15+10 = 105, capped
        self.assertEqual(prop.reasons, ["Bad Def", "Usage Bump (1 Out)"])

    def test_projected_loss_and_filter(self):
        pick = score_game(GAME, TeamContext(), TeamContext(margin=10.0), {})   # Brooklyn favored
        prop = project_player(KEY, "Boston Celtics", "Brooklyn Nets", pick, 112.0, [], PLAYERS)
        self.assertEqual(prop.confidence, 50)
        self.assertIn("Proj Loss", prop.reasons)

    def test_out_player_gets_no_prop(self):
        pick = score_game(GAME, TeamContext(margin=10.0), TeamContext(), {})
        ctx = {"Boston Celtics": TeamContext(), "Brooklyn Nets": TeamContext(opp_pts_allowed=121.0)}
        injuries = {"Boston Celtics": [Injury("Jaylen Brown", "Out")]}
        props = player_props([pick], ctx, [STAR, KEY], injuries)
        self.assertEqual([p.player for p in props], ["Derrick White"])


class Tiers(unittest.TestCase):
    def test_grade_pick(self):
        self.assertEqual([grade_pick("Toss Up", "A"), grade_pick("A", "A"), grade_pick("B", "A")],
                         ["Skipped", "Win", "Loss"])

    def test_tier_boundaries(self):
        rows = [{"Confidence": c, "Result": r} for c, r in
                [("76", "Win"), ("75", "Loss"), ("50", "Win"), ("49", "Win"), ("90", "Pending")]]
        s = tier_stats(rows)
        self.assertEqual(s["High (76-100%)"], {"wins": 1, "total": 1})
        self.assertEqual(s["Med (50-75%)"], {"wins": 1, "total": 2})
        self.assertEqual(s["Low (0-49%)"], {"wins": 1, "total": 1})


if __name__ == "__main__":
    unittest.main()
