import copy
import unittest

from etl.features import alias_map, canonical_games, compute_features, validate_poll
from etl.export_season import fetch_table


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.teams = {1: "Alpha", 2: "Beta", 3: "Gamma"}
        self.games = [
            {"date": "2024-11-01", "a": 1, "b": 2, "score_a": 80, "score_b": 70, "venue_a": "N"},
            {"date": "2024-11-03", "a": 2, "b": 3, "score_a": 60, "score_b": 75, "venue_a": "H"},
        ]

    def features(self, games=None):
        return compute_features(self.games if games is None else games, self.teams,
                                "2024-11-04", {1: {"score": .5, "rank": 20}}, "2024-10-14")

    def test_future_games_and_cutoff_day_do_not_leak(self):
        added = [{"date": d, "a": 1, "b": 3, "score_a": 110, "score_b": 50, "venue_a": "A"}
                 for d in ["2024-11-04", "2024-11-05"]]
        self.assertEqual(self.features(), self.features(self.games + added))

    def test_elo_updated_once_and_conserved(self):
        rows = self.features(self.games[:1])
        self.assertEqual(rows[0]["d1_elo"], 1510)
        self.assertEqual(rows[1]["d1_elo"], 1490)
        self.assertAlmostEqual(sum(r["d1_elo"] for r in self.features()), 4500)

    def test_permutation_and_same_day_order_are_irrelevant(self):
        games = copy.deepcopy(self.games)
        games[1]["date"] = games[0]["date"]
        self.assertEqual(self.features(games), self.features(list(reversed(games))))

    def test_recent_window_includes_prior_release_day(self):
        rows = compute_features(self.games, self.teams, "2024-11-04", {}, "2024-11-03")
        self.assertEqual(rows[0]["since_poll_d1_games"], 0)
        self.assertEqual(rows[1]["since_poll_d1_games"], 1)

    def test_empty_history_and_neutral_venue(self):
        rows = self.features(self.games[:1])
        self.assertIsNone(rows[2]["d1_win_pct"])
        self.assertIsNone(rows[2]["rest_days"])
        self.assertEqual(rows[0]["d1_neutral_wins"], 1)
        self.assertEqual(rows[0]["d1_home_games"], 0)

    def test_alias_collision_rejected(self):
        teams = [{"team_id": 1, "team_name": "A"}, {"team_id": 2, "team_name": "B"}]
        with self.assertRaisesRegex(ValueError, "Conflicting alias"):
            alias_map(teams, [{"team_spelling": " a ", "team_id": 2}])

    def test_margins_are_capped_per_game_and_symmetrically(self):
        games = [
            {"date": "2024-11-01", "a": 1, "b": 2, "score_a": 100, "score_b": 50, "venue_a": "N"},
            {"date": "2024-11-03", "a": 1, "b": 3, "score_a": 60, "score_b": 70, "venue_a": "N"},
        ]
        rows = self.features(games)
        self.assertEqual(rows[0]["d1_mean_margin"], 20)
        self.assertEqual(rows[0]["d1_mean_capped_margin"], 5)
        self.assertEqual(rows[1]["d1_mean_capped_margin"], -20)
        recent = compute_features(games, self.teams, "2024-11-04", {}, "2024-11-03")
        self.assertEqual(recent[0]["since_poll_d1_mean_capped_margin"], -10)
        self.assertIsNone(recent[1]["since_poll_d1_mean_capped_margin"])

    def test_sos_uses_opponent_rating_before_game(self):
        rows = self.features()
        # Beta lost its first game: Gamma faces a 1490 opponent, not its
        # lower rating after the Gamma game. Alpha faced Beta at 1500.
        self.assertEqual(rows[0]["d1_mean_opponent_pregame_elo"], 1500)
        self.assertEqual(rows[2]["d1_mean_opponent_pregame_elo"], 1490)
        self.assertEqual(rows[2]["since_poll_d1_mean_opponent_pregame_elo"], 1490)

    def test_previous_point_gaps_and_nearby_team_results(self):
        previous = {1: {"score": .50, "rank": 10},
                    2: {"score": .49, "rank": 11},
                    3: {"score": .10, "rank": 24}}
        alpha = compute_features(self.games, self.teams, "2024-11-04", previous, "2024-10-14")[0]
        self.assertAlmostEqual(alpha["previous_points_gap_to_next_lower"], .01)
        self.assertIsNone(alpha["previous_points_gap_to_next_higher"])
        self.assertEqual(alpha["previous_nearby_team_count"], 1)
        self.assertEqual(alpha["nearby_teams_since_poll_wins"], 0)
        self.assertEqual(alpha["nearby_teams_since_poll_losses"], 2)
        self.assertAlmostEqual(alpha["since_poll_best_win_opponent_previous_normal_points"], .49)
        self.assertEqual(alpha["since_poll_losses_to_higher_point_teams"], 0)

    def test_preseason_anchor_and_missingness(self):
        preseason = {1: {"rank": 1, "score": .98}, 2: {"rank": 30, "score": .02}}
        rows = compute_features(self.games, self.teams, "2024-11-04", {}, "2024-10-28",
                                preseason_poll=preseason, preseason_date="2024-10-14")
        self.assertEqual(rows[0]["preseason_rank"], 1)
        self.assertEqual(rows[0]["preseason_normal_points"], .98)
        self.assertIsNone(rows[0]["previous_rank"])
        self.assertEqual(rows[0]["days_since_preseason_poll"], 21)
        self.assertIsNone(rows[1]["preseason_rank"])
        self.assertFalse(rows[1]["preseason_ranked"])
        self.assertEqual(rows[1]["preseason_normal_points"], .02)
        self.assertEqual(rows[2]["preseason_normal_points"], 0)
        missing = self.features()[0]
        self.assertFalse(missing["preseason_poll_available"])
        self.assertIsNone(missing["preseason_normal_points"])
        self.assertIsNone(missing["preseason_ranked"])

    def test_preseason_requires_known_past_release(self):
        for day in [None, "2024-11-05"]:
            with self.assertRaises(ValueError):
                compute_features(self.games, self.teams, "2024-11-04", {}, "2024-10-14",
                                 preseason_poll={1: {"rank": 1, "score": 1}}, preseason_date=day)

    def test_mirrored_game_validation(self):
        a = {"season": 25, "date": "2024-11-01", "team": "A", "opponent": "B",
             "team_score": 80, "opponent_score": 70, "home": True, "neutral": False, "overtime": False}
        b = {**a, "team": "B", "opponent": "A", "team_score": 70, "opponent_score": 80, "home": False}
        games, audit = canonical_games([a, b], {"a": 1, "b": 2}, {1: "A", 2: "B"}, 2025)
        self.assertEqual(len(games), 1)
        for invalid in [[a], [a, b, a], [a, {**b, "team_score": 71}]]:
            with self.assertRaises(ValueError):
                canonical_games(invalid, {"a": 1, "b": 2}, {1: "A", 2: "B"}, 2025)

    def test_poll_validation_and_normalization(self):
        rows = [{"season": 25, "week": 2, "date": "2024-11-11", "team": str(i),
                 "votes": 26 - i, "first_votes": int(i == 1)} for i in range(1, 26)]
        aliases = {str(i): i for i in range(1, 26)}
        result = validate_poll(rows, aliases, aliases.values())
        self.assertEqual(result[1]["score"], 1)
        self.assertEqual(result[25]["score"], .04)
        with self.assertRaisesRegex(ValueError, "Incomplete poll"):
            validate_poll(rows[:-1], aliases, aliases.values())
        with self.assertRaisesRegex(ValueError, "Duplicate poll team"):
            validate_poll(rows + [rows[0]], aliases, aliases.values())


class PaginationTests(unittest.TestCase):
    class Response:
        ok = True
        def __init__(self, rows, total):
            self.rows = rows
            self.headers = {"Content-Range": f"0-1/{total}"}
        def json(self):
            return self.rows

    def test_server_page_cap_smaller_than_requested(self):
        outer = self
        class Session:
            def get(self, url, params, timeout):
                offset = params["offset"]
                return outer.Response(list(range(5))[offset:offset + 2], 5)
        self.assertEqual(fetch_table(Session(), "https://example.invalid", "teams", "team_id"), list(range(5)))

    def test_count_changes_fail_export(self):
        outer = self
        class Session:
            def get(self, url, params, timeout):
                return outer.Response([1], 3 if params["offset"] else 2)
        with self.assertRaisesRegex(ValueError, "changed during export"):
            fetch_table(Session(), "https://example.invalid", "teams", "team_id")


if __name__ == "__main__":
    unittest.main()
