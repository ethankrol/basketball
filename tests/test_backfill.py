import copy
import unittest

from etl.backfill_season import build


def fixture():
    teams = [{"team_id": i, "team_name": str(i), "first_d1_season": 2000, "last_d1_season": 2025}
             for i in range(1, 26)]
    games = []
    for i in range(1, 25):
        row = {"season": 25, "date": "2024-11-05", "team": str(i), "opponent": str(i + 1),
               "team_score": 80, "opponent_score": 70, "home": True, "neutral": False, "overtime": False}
        games.extend([row, {**row, "team": str(i + 1), "opponent": str(i), "team_score": 70,
                            "opponent_score": 80, "home": False}])
    polls = [{"season": 25, "week": week, "date": day, "team": str(i), "votes": 26 - i,
              "first_votes": int(i == 1)} for week, day in [(1, "2024-10-14"), (2, "2024-11-11"), (3, "2024-11-18")]
             for i in range(1, 26)]
    return {"season": 2025, "tables": {"teams": teams, "team_spellings": [], "team_memberships": [{"team_id": t["team_id"], "first_season": 2000, "last_season": None} for t in teams], "games": games, "polls": polls}}


class BackfillTests(unittest.TestCase):
    def test_reproducible_complete_season(self):
        payload = fixture()
        original = copy.deepcopy(payload)
        first = build(payload, {"season": 2025}, bootstrap=True)
        self.assertEqual(first, build(payload, {"season": 2025}, bootstrap=True))
        self.assertEqual(payload, original)
        self.assertEqual(first[2]["feature_rows"], 50)
        self.assertEqual(first[2]["mean_top25_overlap"], 25)
        self.assertEqual({r["preseason_rank"] for r in first[0] if r["team_id"] == 1}, {1})
        self.assertEqual({r["feature_version"] for r in first[0]}, {"d1-v4"})

    def test_invalid_target_is_not_zero_filled(self):
        payload = fixture()
        payload["tables"]["polls"][-1]["votes"] = 0
        rows, predictions, report = build(payload, {"season": 2025}, bootstrap=True)
        self.assertEqual({r["target_week"] for r in rows}, {2})
        self.assertIn(3, report["invalid_polls"])

    def test_target_labels_do_not_change_features(self):
        payload = fixture()
        original_rows = build(payload, {"season": 2025}, bootstrap=True)[0]
        last = [r for r in payload["tables"]["polls"] if r["week"] == 3]
        last[0]["votes"], last[1]["votes"] = last[1]["votes"], last[0]["votes"]
        last[0]["first_votes"], last[1]["first_votes"] = 0, 1
        new_rows = build(payload, {"season": 2025}, bootstrap=True)[0]
        # Source revision changes run IDs, but feature values must not change.
        strip_id = lambda rows: [{k: v for k, v in r.items() if k != "run_id"} for r in rows]
        self.assertEqual(strip_id(original_rows), strip_id(new_rows))


if __name__ == "__main__":
    unittest.main()
