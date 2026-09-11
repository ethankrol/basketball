"""Run prioritized, pre-holdout experiments for AP exact-rank accuracy."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.experiment_rank_accuracy import model_prediction, pooled_metrics
from ml.train_model import (EXCLUDE, POINT_CHANGE_BLEND_WEIGHT, RICH_GAME_CONTEXT_FEATURES,
                            TARGET, TRAJECTORY_FEATURES, load_rows, metrics, rank_metrics)


TRAJECTORY = TRAJECTORY_FEATURES
RICH_GAME_CONTEXT = RICH_GAME_CONTEXT_FEATURES
WHOLE_POLL_CONTEXT = {"previous_ranked_teams_with_losses", "previous_ranked_teams_with_wins",
                      "previous_top10_teams_with_losses", "higher_ranked_teams_with_losses"}
V9_CANDIDATES = TRAJECTORY | RICH_GAME_CONTEXT | WHOLE_POLL_CONTEXT


def evaluate_blend(data, features, training_window, seed=42):
    seasons = sorted(data.season.unique())
    rows, predictions = [], {}
    for index in range(8, len(seasons)):
        season = seasons[index]
        start = 0 if training_window is None else max(0, index - training_window)
        train_mask = data.season.isin(seasons[start:index])
        validation_mask = data.season.eq(season)
        delta = model_prediction(data, features, train_mask, validation_mask, "delta", seed)
        points = model_prediction(data, features, train_mask, validation_mask, "score", seed)
        prediction = POINT_CHANGE_BLEND_WEIGHT * delta + (1 - POINT_CHANGE_BLEND_WEIGHT) * points
        rows.append({"season": int(season), **metrics(data.loc[validation_mask], prediction)})
        predictions[int(season)] = prediction
    return rows, predictions


def pairwise_prediction(data, features, train_mask, validation_mask, baseline_prediction, seed=42):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer

    X = data[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    train_X = imputer.fit_transform(X.loc[train_mask])
    validation_X = imputer.transform(X.loc[validation_mask])
    train_frame, validation_frame = data.loc[train_mask], data.loc[validation_mask]
    train_positions = {index: position for position, index in enumerate(train_frame.index)}
    pairs, labels = [], []
    for _, poll in train_frame[train_frame.actual_rank.between(1, 25)].groupby(
            ["season", "target_week"], sort=False):
        ordered = poll.sort_values("actual_rank")
        indices = list(ordered.index)
        for left in range(len(indices)):
            for right in range(left + 1, min(left + 6, len(indices))):
                difference = train_X[train_positions[indices[left]]] - train_X[train_positions[indices[right]]]
                pairs.extend((difference, -difference))
                labels.extend((1, 0))
    classifier = HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=15,
                                                 learning_rate=.04, l2_regularization=5,
                                                 random_state=seed)
    classifier.fit(np.asarray(pairs), np.asarray(labels))
    predicted_ranks = np.empty(len(validation_frame), dtype=float)
    validation_positions = {index: position for position, index in enumerate(validation_frame.index)}
    offset = 0
    for _, poll in validation_frame.groupby(["season", "target_week"], sort=True):
        count = len(poll)
        base_order = np.argsort(-baseline_prediction[offset:offset + count], kind="stable")
        ranks = np.empty(count, dtype=float)
        ranks[base_order] = np.arange(1, count + 1)
        candidate_local = base_order[:25]
        scores = np.zeros(len(candidate_local))
        for left in range(len(candidate_local)):
            for right in range(left + 1, len(candidate_local)):
                left_global = validation_positions[poll.index[candidate_local[left]]]
                right_global = validation_positions[poll.index[candidate_local[right]]]
                probability = classifier.predict_proba(
                    [(validation_X[left_global] - validation_X[right_global])])[0, 1]
                scores[left] += probability
                scores[right] += 1 - probability
        pair_order = np.argsort(-scores, kind="stable")
        ranks[candidate_local[pair_order]] = np.arange(1, 26)
        predicted_ranks[offset:offset + count] = ranks
        offset += count
    return predicted_ranks


def run(data):
    # Remove both held-out seasons before constructing any training mask.
    experiment = data[data.season <= sorted(data.season.unique())[-3]].copy()
    all_features = set(experiment.columns) - EXCLUDE - {TARGET, "actual_rank",
                                                        "previous_points_gap_to_next_higher",
                                                        "previous_points_gap_to_next_lower"}
    base_features = sorted(all_features - V9_CANDIDATES)
    history = {}
    cached = {}
    for window in (4, 8, 12, None):
        rows, predictions = evaluate_blend(experiment, base_features, window)
        name = "all_available" if window is None else f"last_{window}_seasons"
        history[name] = {"summary": pooled_metrics(rows), "per_season": rows}
        cached[name] = predictions
    best_history = min(history, key=lambda name: (-history[name]["summary"]["top25_weekly_exact_rank_accuracy"],
                                                  history[name]["summary"]["top25_mean_absolute_rank_error"]))
    window = None if best_history == "all_available" else int(best_history.split("_")[1])
    feature_sets = {
        "base": base_features,
        "base_plus_trajectory": sorted(set(base_features) | TRAJECTORY),
        "base_plus_rich_game_context": sorted(set(base_features) | RICH_GAME_CONTEXT),
        "base_plus_whole_poll_context": sorted(set(base_features) | WHOLE_POLL_CONTEXT),
        "base_plus_all_v9_context": sorted(set(base_features) | V9_CANDIDATES),
    }
    context, context_predictions = {}, {}
    for name, features in feature_sets.items():
        if name == "base":
            rows, predictions = history[best_history]["per_season"], cached[best_history]
        else:
            rows, predictions = evaluate_blend(experiment, features, window)
        context[name] = {"summary": pooled_metrics(rows), "per_season": rows}
        context_predictions[name] = predictions
    best_context = min(context, key=lambda name: (-context[name]["summary"]["top25_weekly_exact_rank_accuracy"],
                                                  context[name]["summary"]["top25_mean_absolute_rank_error"]))

    pairwise_rows = []
    seasons = sorted(experiment.season.unique())
    for index in range(8, len(seasons)):
        season = seasons[index]
        start = 0 if window is None else max(0, index - window)
        train_mask = experiment.season.isin(seasons[start:index])
        validation_mask = experiment.season.eq(season)
        predicted_rank = pairwise_prediction(experiment, feature_sets[best_context], train_mask,
                                             validation_mask, context_predictions[best_context][int(season)])
        pairwise_rows.append({"season": int(season),
                              **rank_metrics(experiment.loc[validation_mask], predicted_rank)})
    return {"selection_seasons": [int(seasons[8]), int(seasons[-1])],
            "excluded_holdout_seasons": [int(season) for season in sorted(data.season.unique())[-2:]],
            "history_window_experiment": history, "selected_history_window": best_history,
            "context_experiment": context, "selected_context": best_context,
            "pairwise_ranking": {"summary": pooled_metrics(pairwise_rows), "per_season": pairwise_rows},
            "point_change_blend_comparison": context[best_context]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/history"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/model_experiments/prioritized.json"))
    args = parser.parse_args()
    report = run(load_rows(args.root, 2003, 2026))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    compact = {"selection_seasons": report["selection_seasons"],
               "excluded_holdout_seasons": report["excluded_holdout_seasons"],
               "history": {name: value["summary"] for name, value in report["history_window_experiment"].items()},
               "selected_history_window": report["selected_history_window"],
               "context": {name: value["summary"] for name, value in report["context_experiment"].items()},
               "selected_context": report["selected_context"],
               "pairwise_ranking": report["pairwise_ranking"]["summary"]}
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
