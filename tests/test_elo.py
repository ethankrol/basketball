import unittest

from etl.elo import ELO_VERSION, season_initial_ratings
from etl.features import compute_features


class EloCarryoverTests(unittest.TestCase):
    def state(self):
        return {"season": 2024, "elo_version": ELO_VERSION, "through_date_exclusive": "2024-07-01",
                "ratings": {"1": 1700, "2": 1300}}

    def test_regression_and_new_team(self):
        self.assertEqual(season_initial_ratings({1, 2, 3}, 2025, self.state()),
                         {1: 1650, 2: 1350, 3: 1500})
        self.assertEqual(season_initial_ratings({1}, 2025, self.state(), 0), {1: 1500})
        self.assertEqual(season_initial_ratings({1}, 2025, self.state(), 1), {1: 1700})

    def test_requires_explicit_bootstrap_or_previous_season(self):
        with self.assertRaises(ValueError):
            season_initial_ratings({1}, 2025)
        self.assertEqual(season_initial_ratings({1}, 2025, bootstrap=True), {1: 1500})
        for state in [{**self.state(), "season": 2025}, {**self.state(), "elo_version": "other"},
                      {**self.state(), "through_date_exclusive": "2024-01-01"},
                      {**self.state(), "ratings": {"1": float("nan")}}]:
            with self.assertRaises(ValueError):
                season_initial_ratings({1}, 2025, state)
        for fraction in [-.1, 1.1, float("nan")]:
            with self.assertRaises(ValueError):
                season_initial_ratings({1}, 2025, self.state(), fraction)

    def test_features_and_sos_use_seeded_ratings(self):
        seed = season_initial_ratings({1, 2}, 2025, self.state())
        games = [{"date": "2024-11-01", "a": 1, "b": 2, "score_a": 80, "score_b": 70, "venue_a": "N"}]
        rows = compute_features(games, {1: "A", 2: "B"}, "2024-11-02", {}, "2024-10-14", initial_ratings=seed)
        self.assertEqual(rows[0]["d1_mean_opponent_pregame_elo"], 1350)
        self.assertGreater(rows[0]["d1_elo"], 1650)
        self.assertLess(rows[0]["d1_elo"], 1660)
        self.assertAlmostEqual(sum(r["d1_elo"] for r in rows), 3000)
