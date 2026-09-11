import unittest

import numpy as np
import pandas as pd

from ml.train_model import metrics, rank_miss_analysis, summarize_rolling_validation
from ml.rank_position_diagnostics import aggregate, prediction_records
from ml.walk_forward_experiments import sample_weights


class RankingMetricTests(unittest.TestCase):
    def frame(self):
        rows = [{"season": 2025, "target_week": 2, "team_id": f"team-{rank}",
                 "actual_score": 1 - rank / 100, "actual_rank": rank,
                 "previous_rank": rank, "since_poll_d1_wins": 1,
                 "since_poll_d1_losses": 0}
                for rank in range(1, 26)]
        rows.append({"season": 2025, "target_week": 2, "team_id": "votes-only",
                     "actual_score": .01, "actual_rank": np.nan, "previous_rank": np.nan,
                     "since_poll_d1_wins": 0, "since_poll_d1_losses": 1})
        return pd.DataFrame(rows)

    def test_within_one_is_top25_only(self):
        frame = self.frame()
        prediction = np.array([.98, .99] + [1 - rank / 100 for rank in range(3, 26)] + [.001])
        result = metrics(frame, prediction)
        self.assertEqual(result["top25_exact_rank_accuracy"], 23 / 25)
        self.assertEqual(result["top25_weekly_exact_rank_accuracy"], 23 / 25)
        self.assertEqual(result["top25_within_one_rank_accuracy"], 1.0)

    def test_miss_analysis_reports_adjacent_swaps(self):
        frame = self.frame()
        prediction = np.array([.98, .99] + [1 - rank / 100 for rank in range(3, 26)] + [.001])
        result = rank_miss_analysis(frame, prediction)
        self.assertEqual(result["adjacent_rank_swaps"], 2)
        self.assertEqual(result["larger_rank_errors"], 0)
        self.assertEqual(result["top25_entries"], 0)
        self.assertEqual(result["top25_exits"], 0)

    def test_rolling_summary_weights_seasons_by_poll_count(self):
        rows = [{"models": {"model": {"polls": 1, "top25_weekly_exact_rank_accuracy": 1.0}}},
                {"models": {"model": {"polls": 3, "top25_weekly_exact_rank_accuracy": 0.0}}}]
        result = summarize_rolling_validation(rows)["model"]
        self.assertEqual(result["polls"], 4)
        self.assertEqual(result["top25_weekly_exact_rank_accuracy"], .25)

    def test_rank_focused_sample_weights(self):
        frame = self.frame()
        weights = sample_weights(frame)
        self.assertEqual(weights[0], 6)
        self.assertEqual(weights[-1], 2)

    def test_position_diagnostics_include_unranked_bucket(self):
        frame = self.frame()
        # A receiving-votes team displaces actual #25 from the predicted poll.
        prediction = np.array([1 - rank / 100 for rank in range(1, 26)] + [.99])
        records = prediction_records(frame, prediction)
        rank_25 = next(row for row in records if row["actual_rank"] == 25)
        self.assertEqual(rank_25["predicted_rank"], 26)
        self.assertFalse(rank_25["retained"])
        report = aggregate(records, bootstrap_samples=100)
        self.assertEqual(report["confusion_matrix"]["counts"][24][25], 1)
        self.assertEqual(report["per_position"][24]["top25_retention"], 0.0)
        self.assertEqual(report["per_position"][24]["mean_signed_error"], 1.0)


if __name__ == "__main__":
    unittest.main()
