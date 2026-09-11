"""Nested poll-level walk-forward model selection without touching holdout seasons."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.experiment_rank_accuracy import pooled_metrics
from ml.train_model import (EXCLUDE, POINT_CHANGE_BLEND_WEIGHT, RICH_GAME_CONTEXT_FEATURES,
                            TARGET, TRAJECTORY_FEATURES, load_rows, metrics)


WHOLE_POLL_CONTEXT = {"previous_ranked_teams_with_losses", "previous_ranked_teams_with_wins",
                      "previous_top10_teams_with_losses", "higher_ranked_teams_with_losses"}
POINT_GAPS = {"previous_points_gap_to_next_higher", "previous_points_gap_to_next_lower"}
V8_NEW_FEATURES = {
    "previous_nearby_team_count", "nearby_teams_since_poll_wins",
    "nearby_teams_since_poll_losses", "nearby_teams_since_poll_elo_change",
    "nearby_teams_since_poll_mean_opponent_elo", "since_poll_losses_to_higher_point_teams",
    "since_poll_mean_opponent_previous_normal_points",
    "since_poll_best_win_opponent_previous_normal_points",
    "since_poll_worst_loss_opponent_previous_normal_points",
}

CONFIGS = {
    "original_v7_uniform": {"max_iter": 200, "max_leaf_nodes": 15, "min_samples_leaf": 20,
                            "l2_regularization": 5, "weighted": False, "features": "original_v7"},
    "current_uniform": {"max_iter": 200, "max_leaf_nodes": 15, "min_samples_leaf": 20,
                        "l2_regularization": 5, "weighted": False, "features": "whole_poll"},
    "regularized_uniform": {"max_iter": 150, "max_leaf_nodes": 8, "min_samples_leaf": 75,
                            "l2_regularization": 20, "weighted": False, "features": "whole_poll"},
    "regularized_weighted": {"max_iter": 150, "max_leaf_nodes": 8, "min_samples_leaf": 75,
                             "l2_regularization": 20, "weighted": True, "features": "whole_poll"},
    "strong_regularized_weighted": {"max_iter": 120, "max_leaf_nodes": 5, "min_samples_leaf": 150,
                                    "l2_regularization": 50, "weighted": True, "features": "whole_poll"},
    "weighted_without_whole_poll": {"max_iter": 150, "max_leaf_nodes": 8, "min_samples_leaf": 75,
                                    "l2_regularization": 20, "weighted": True, "features": "base"},
    "weighted_plus_trajectory": {"max_iter": 150, "max_leaf_nodes": 8, "min_samples_leaf": 75,
                                 "l2_regularization": 20, "weighted": True,
                                 "features": "whole_poll_trajectory"},
    "weighted_plus_rich_game": {"max_iter": 150, "max_leaf_nodes": 8, "min_samples_leaf": 75,
                                "l2_regularization": 20, "weighted": True,
                                "features": "whole_poll_rich_game"},
    "weighted_all_context": {"max_iter": 150, "max_leaf_nodes": 8, "min_samples_leaf": 75,
                             "l2_regularization": 20, "weighted": True, "features": "all_context"},
}


def feature_sets(data):
    available = set(data.columns) - EXCLUDE - POINT_GAPS - {TARGET, "actual_rank"}
    base = available - TRAJECTORY_FEATURES - RICH_GAME_CONTEXT_FEATURES - WHOLE_POLL_CONTEXT
    return {"original_v7": sorted(base - V8_NEW_FEATURES),
            "base": sorted(base), "whole_poll": sorted(base | WHOLE_POLL_CONTEXT),
            "whole_poll_trajectory": sorted(base | WHOLE_POLL_CONTEXT | TRAJECTORY_FEATURES),
            "whole_poll_rich_game": sorted(base | WHOLE_POLL_CONTEXT | RICH_GAME_CONTEXT_FEATURES),
            "all_context": sorted(available)}


def sample_weights(frame):
    weights = np.ones(len(frame), dtype=float)
    weights[frame[TARGET].to_numpy() > 0] = 2.0
    weights[frame.actual_rank.between(1, 25).to_numpy()] = 6.0
    return weights


def fit_predict(data, features, config, train_mask, validation_mask, include_training=False, seed=42):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer

    X = data[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    train_X = imputer.fit_transform(X.loc[train_mask])
    validation_X = imputer.transform(X.loc[validation_mask])
    model_args = {key: config[key] for key in
                  ("max_iter", "max_leaf_nodes", "min_samples_leaf", "l2_regularization")}
    model_args.update({"learning_rate": .04, "random_state": seed, "early_stopping": False})
    score_model = HistGradientBoostingRegressor(**model_args)
    delta_model = HistGradientBoostingRegressor(**model_args)
    target = data[TARGET].to_numpy()
    prior = data.previous_normal_points.to_numpy()
    weight = sample_weights(data.loc[train_mask]) if config["weighted"] else None
    score_model.fit(train_X, target[train_mask], sample_weight=weight)
    delta_model.fit(train_X, target[train_mask] - prior[train_mask], sample_weight=weight)

    def predict(matrix, mask):
        return (POINT_CHANGE_BLEND_WEIGHT * (delta_model.predict(matrix) + prior[mask]) +
                (1 - POINT_CHANGE_BLEND_WEIGHT) * score_model.predict(matrix))

    validation_prediction = predict(validation_X, validation_mask)
    training_prediction = predict(train_X, train_mask) if include_training else None
    return validation_prediction, training_prediction


def annual_candidate_results(data, sets):
    seasons = sorted(data.season.unique())
    results = {name: {} for name in CONFIGS}
    for validation_season in [season for season in seasons if 2011 <= season <= 2023]:
        train_mask = data.season.isin(range(validation_season - 4, validation_season))
        validation_mask = data.season.eq(validation_season)
        for name, config in CONFIGS.items():
            prediction, _ = fit_predict(data, sets[config["features"]], config,
                                        train_mask, validation_mask)
            results[name][validation_season] = {
                "season": int(validation_season), **metrics(data.loc[validation_mask], prediction)}
    return results


def select_candidate(candidate_results, outer_season):
    inner_seasons = list(range(outer_season - 4, outer_season))
    summaries = {name: pooled_metrics([rows[season] for season in inner_seasons])
                 for name, rows in candidate_results.items()}
    selected = min(summaries,
                   key=lambda name: (-summaries[name]["top25_weekly_exact_rank_accuracy"],
                                     summaries[name]["top25_mean_absolute_rank_error"],
                                     -summaries[name]["mean_top25_overlap"]))
    return selected, summaries


def missingness_audit(data, features):
    result = []
    for season, frame in data.groupby("season", sort=True):
        availability = frame[features].notna().mean()
        result.append({"season": int(season), "rows": len(frame),
                       "mean_feature_availability": float(availability.mean()),
                       "features_below_50pct_availability": int((availability < .5).sum()),
                       "availability": {name: float(value) for name, value in availability.items()}})
    return result


def run(data):
    # All tuning and evaluation ends in 2024. The final two seasons are never selected below.
    experiment = data[data.season <= 2024].copy()
    sets = feature_sets(experiment)
    candidates = annual_candidate_results(experiment, sets)
    outer, poll_rows, persistence_rows = [], [], []
    for outer_season in range(2015, 2025):
        selected, inner_summaries = select_candidate(candidates, outer_season)
        config = CONFIGS[selected]
        validation_predictions, validation_frames, training_accuracies = [], [], []
        weeks = sorted(experiment.loc[experiment.season.eq(outer_season), "target_week"].unique())
        for week in weeks:
            train_mask = (experiment.season.isin(range(outer_season - 4, outer_season)) |
                          (experiment.season.eq(outer_season) & experiment.target_week.lt(week)))
            validation_mask = experiment.season.eq(outer_season) & experiment.target_week.eq(week)
            prediction, training_prediction = fit_predict(
                experiment, sets[config["features"]], config, train_mask, validation_mask,
                include_training=True)
            validation_frame = experiment.loc[validation_mask]
            poll_metric = metrics(validation_frame, prediction)
            persistence_metric = metrics(validation_frame,
                                         validation_frame.previous_normal_points.to_numpy())
            training_metric = metrics(experiment.loc[train_mask], training_prediction)
            poll_rows.append({"season": outer_season, "week": int(week), "candidate": selected,
                              "training_exact_rank_accuracy": training_metric["top25_exact_rank_accuracy"],
                              **poll_metric})
            persistence_rows.append({"season": outer_season, "week": int(week), **persistence_metric})
            training_accuracies.append(training_metric["top25_exact_rank_accuracy"])
            validation_predictions.extend(prediction)
            validation_frames.append(validation_frame)
        season_frame = pd.concat(validation_frames)
        season_metric = metrics(season_frame, np.asarray(validation_predictions))
        outer.append({"season": outer_season, "polls": season_metric["polls"],
                      "selected_candidate": selected,
                      "mean_training_exact_rank_accuracy": float(np.mean(training_accuracies)),
                      **{key: value for key, value in season_metric.items() if key != "polls"},
                      "inner_selection_summary": inner_summaries})
    pooled = pooled_metrics([{key: value for key, value in row.items()
                              if key not in {"selected_candidate", "mean_training_exact_rank_accuracy",
                                             "inner_selection_summary"}} for row in outer])
    training_exact = float(np.average([row["mean_training_exact_rank_accuracy"] for row in outer],
                                      weights=[row["polls"] for row in outer]))
    by_week = {}
    for bucket, predicate in {
        "early_weeks_2_5": lambda row: row["week"] <= 5,
        "middle_weeks_6_12": lambda row: 6 <= row["week"] <= 12,
        "late_weeks_13_plus": lambda row: row["week"] >= 13,
    }.items():
        selected_rows = [row for row in poll_rows if predicate(row)]
        by_week[bucket] = {"polls": len(selected_rows),
                           "exact_rank_accuracy": float(np.mean(
                               [row["top25_exact_rank_accuracy"] for row in selected_rows]))}
    feature_ablation = {}
    for name in ("original_v7_uniform", "current_uniform", "regularized_uniform",
                 "regularized_weighted", "strong_regularized_weighted", "weighted_without_whole_poll",
                 "weighted_plus_trajectory", "weighted_plus_rich_game", "weighted_all_context"):
        rows = [candidates[name][season] for season in sorted(candidates[name])]
        feature_ablation[name] = {"summary": pooled_metrics(rows), "per_season": rows}
    return {"protocol": {"outer_seasons": [2015, 2024], "inner_selection_seasons": 4,
                          "training_window_seasons": 4, "retrain_each_poll": True,
                          "current_season_prior_polls_in_training": True,
                          "excluded_holdout_seasons": [2025, 2026]},
            "configs": CONFIGS, "outer_results": outer, "poll_results": poll_rows,
            "pooled_walk_forward": {**pooled, "training_exact_rank_accuracy": training_exact,
                                    "generalization_gap": training_exact - pooled["top25_exact_rank_accuracy"]},
            "persistence_baseline": pooled_metrics(
                [{key: value for key, value in row.items() if key != "week"}
                 for row in persistence_rows]),
            "accuracy_by_season_stage": by_week, "feature_ablation": feature_ablation,
            "feature_availability_by_season": missingness_audit(experiment, sorted(set().union(*sets.values())))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/history"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/model_experiments/walk_forward_nested.json"))
    args = parser.parse_args()
    report = run(load_rows(args.root, 2003, 2026))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"protocol": report["protocol"], "pooled_walk_forward": report["pooled_walk_forward"],
                      "persistence_baseline": report["persistence_baseline"],
                      "accuracy_by_season_stage": report["accuracy_by_season_stage"],
                      "outer_results": [{key: row[key] for key in
                                         ("season", "polls", "selected_candidate",
                                          "mean_training_exact_rank_accuracy",
                                          "top25_exact_rank_accuracy")} for row in report["outer_results"]],
                      "feature_ablation": {name: value["summary"] for name, value in
                                           report["feature_ablation"].items()}}, indent=2))


if __name__ == "__main__":
    main()
