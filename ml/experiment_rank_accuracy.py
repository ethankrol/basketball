"""Validation-only feature ablations, prediction blends, and adjacent-pair analysis."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.train_model import (EXCLUDE, POINT_CHANGE_BLEND_WEIGHT, RICH_GAME_CONTEXT_FEATURES,
                            TARGET, TRAJECTORY_FEATURES, load_rows, metrics)


NEW_FEATURE_GROUPS = {
    "previous_point_spacing": {
        "previous_points_gap_to_next_higher", "previous_points_gap_to_next_lower",
    },
    "nearby_team_results": {
        "previous_nearby_team_count", "nearby_teams_since_poll_wins",
        "nearby_teams_since_poll_losses", "nearby_teams_since_poll_elo_change",
        "nearby_teams_since_poll_mean_opponent_elo",
    },
    "opponent_poll_strength": {
        "since_poll_losses_to_higher_point_teams",
        "since_poll_mean_opponent_previous_normal_points",
        "since_poll_best_win_opponent_previous_normal_points",
        "since_poll_worst_loss_opponent_previous_normal_points",
    },
}
V9_FEATURES = (TRAJECTORY_FEATURES | RICH_GAME_CONTEXT_FEATURES |
               {"previous_ranked_teams_with_losses", "previous_ranked_teams_with_wins",
                "previous_top10_teams_with_losses", "higher_ranked_teams_with_losses"})


def model_prediction(data, features, train_mask, validation_mask, target_kind, seed):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer

    X = data[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    train = imputer.fit_transform(X.loc[train_mask])
    validation = imputer.transform(X.loc[validation_mask])
    prior = data["previous_normal_points"].to_numpy()
    target = data[TARGET].to_numpy()
    if target_kind == "delta":
        target = target - prior
    model = HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=15,
                                           learning_rate=.04, l2_regularization=5,
                                           random_state=seed)
    model.fit(train, target[train_mask])
    prediction = model.predict(validation)
    if target_kind == "delta":
        prediction += prior[validation_mask]
    return prediction


def pooled_metrics(per_season):
    polls = np.asarray([row["polls"] for row in per_season])
    result = {"polls": int(polls.sum())}
    for key in per_season[0]:
        if key not in {"season", "polls"}:
            result[key] = float(np.average([row[key] for row in per_season], weights=polls))
    return result


def adjacent_pair_analysis(frames, predictions):
    groups = {name: defaultdict(lambda: [0, 0]) for name in
              ("predicted_point_gap", "previous_order", "weekly_results", "poll_position")}
    for frame, prediction in zip(frames, predictions):
        values = frame[["season", "target_week", "team_id", "actual_rank", "previous_rank",
                        "since_poll_d1_wins", "since_poll_d1_losses"]].copy()
        values["prediction"] = np.clip(prediction, 0, 1)
        for _, poll in values.groupby(["season", "target_week"], sort=True):
            ordered = poll.sort_values(["prediction", "team_id"], ascending=[False, True]).head(25)
            records = list(ordered.itertuples(index=False))
            for position, (upper, lower) in enumerate(zip(records, records[1:]), 1):
                if pd.isna(upper.actual_rank) or pd.isna(lower.actual_rank):
                    continue
                reversed_order = upper.actual_rank > lower.actual_rank
                gap = upper.prediction - lower.prediction
                gap_name = "<=0.005" if gap <= .005 else "<=0.01" if gap <= .01 else "<=0.025" if gap <= .025 else ">0.025"
                previous = ("agrees_with_previous" if pd.notna(upper.previous_rank) and
                            pd.notna(lower.previous_rank) and upper.previous_rank < lower.previous_rank
                            else "changes_previous_or_unranked")
                if upper.since_poll_d1_losses and lower.since_poll_d1_wins:
                    results = "upper_lost_lower_won"
                elif upper.since_poll_d1_wins and lower.since_poll_d1_losses:
                    results = "upper_won_lower_lost"
                else:
                    results = "other_results"
                poll_position = "top_10" if position <= 10 else "ranks_11_20" if position <= 20 else "cutoff_21_25"
                for group, label in [("predicted_point_gap", gap_name), ("previous_order", previous),
                                     ("weekly_results", results), ("poll_position", poll_position)]:
                    groups[group][label][0] += 1
                    groups[group][label][1] += int(reversed_order)
    return {group: {label: {"pairs": counts[0], "reversals": counts[1],
                            "reversal_rate": counts[1] / counts[0]}
                    for label, counts in sorted(rows.items())}
            for group, rows in groups.items()}


def run(data, seed=42, minimum_training_seasons=8):
    # The final two seasons are excluded before any experiment or blend choice.
    experiment_data = data[data.season <= sorted(data.season.unique())[-3]].copy()
    seasons = sorted(experiment_data.season.unique())
    all_features = sorted(set(experiment_data.columns) - EXCLUDE - V9_FEATURES - {TARGET, "actual_rank"})
    new_features = set().union(*NEW_FEATURE_GROUPS.values())
    feature_sets = {"original_v7_features": sorted(set(all_features) - new_features),
                    "all_v8_features": all_features}
    for name, group in NEW_FEATURE_GROUPS.items():
        feature_sets[f"v8_without_{name}"] = sorted(set(all_features) - group)

    per_season = {name: [] for name in feature_sets}
    per_season.update({"direct_points": [], "persistence": []})
    blend_per_season = {weight: [] for weight in (0.0, .25, .5, .75, 1.0)}
    adjacent_frames, adjacent_predictions = [], []
    for index in range(minimum_training_seasons, len(seasons)):
        validation_season = seasons[index]
        train_mask = experiment_data.season.isin(seasons[:index])
        validation_mask = experiment_data.season.eq(validation_season)
        frame = experiment_data.loc[validation_mask]
        predictions = {}
        for name, features in feature_sets.items():
            predictions[name] = model_prediction(experiment_data, features, train_mask,
                                                 validation_mask, "delta", seed)
            per_season[name].append({"season": int(validation_season),
                                     **metrics(frame, predictions[name])})
        recommended_features = feature_sets["v8_without_previous_point_spacing"]
        predictions["direct_points"] = model_prediction(experiment_data, recommended_features, train_mask,
                                                         validation_mask, "score", seed)
        per_season["direct_points"].append({"season": int(validation_season),
                                             **metrics(frame, predictions["direct_points"])})
        per_season["persistence"].append({"season": int(validation_season),
                                          **metrics(frame, frame.previous_normal_points.to_numpy())})
        for weight in blend_per_season:
            blended = (weight * predictions["v8_without_previous_point_spacing"] +
                       (1 - weight) * predictions["direct_points"])
            blend_per_season[weight].append({"season": int(validation_season), **metrics(frame, blended)})
        adjacent_frames.append(frame)
        adjacent_predictions.append(
            POINT_CHANGE_BLEND_WEIGHT * predictions["v8_without_previous_point_spacing"] +
            (1 - POINT_CHANGE_BLEND_WEIGHT) * predictions["direct_points"])

    summary = {name: pooled_metrics(rows) for name, rows in per_season.items()}
    blends = {str(weight): pooled_metrics(rows) for weight, rows in blend_per_season.items()}
    best_blend = min(blends, key=lambda weight: (-blends[weight]["top25_weekly_exact_rank_accuracy"],
                                                 blends[weight]["top25_mean_absolute_rank_error"]))
    return {"selection_data": {"first_validation_season": int(seasons[minimum_training_seasons]),
                                "last_validation_season": int(seasons[-1]),
                                "excluded_holdout_seasons": [int(season) for season in
                                                             sorted(data.season.unique())[-2:]]},
            "feature_groups": {name: sorted(values) for name, values in NEW_FEATURE_GROUPS.items()},
            "summary": summary, "per_season": per_season,
            "blends": blends,
            "blend_per_season": {str(weight): rows for weight, rows in blend_per_season.items()},
            "best_blend_weight_on_point_change": float(best_blend),
            "adjacent_pair_analysis": adjacent_pair_analysis(adjacent_frames, adjacent_predictions)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/history"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/model_experiments/rank_accuracy.json"))
    parser.add_argument("--start", type=int, default=2003)
    parser.add_argument("--end", type=int, default=2026)
    args = parser.parse_args()
    report = run(load_rows(args.root, args.start, args.end))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selection_data": report["selection_data"], "summary": report["summary"],
                      "blends": report["blends"], "best_blend_weight_on_point_change":
                      report["best_blend_weight_on_point_change"],
                      "adjacent_pair_analysis": report["adjacent_pair_analysis"]}, indent=2))


if __name__ == "__main__":
    main()
