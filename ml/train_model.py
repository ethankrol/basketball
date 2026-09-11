"""Train chronological AP-poll models from historical backfill artifacts.

The script intentionally keeps the last two seasons out of model selection and
reports them as an untouched final test. It trains numeric regressors only; team
names/IDs are excluded to avoid memorizing historical poll reputation.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from etl.features import FEATURE_VERSION, content_hash


EXCLUDE = {"run_id", "season", "target_week", "cutoff_date_exclusive", "previous_poll_date",
           "feature_version", "team_id", "team_name"}
TRAJECTORY_FEATURES = {"previous_normal_points_lag_2", "previous_normal_points_lag_3",
                       "previous_normal_points_lag_4", "previous_points_trend_3_poll",
                       "previous_points_trend_4_poll", "previous_points_change_volatility_4_poll"}
RICH_GAME_CONTEXT_FEATURES = {
    "since_poll_overtime_games", "since_poll_overtime_wins", "since_poll_overtime_losses",
    "since_poll_stronger_opponent_games", "since_poll_stronger_opponent_wins",
    "since_poll_stronger_opponent_mean_capped_margin", "since_poll_weaker_opponent_games",
    "since_poll_weaker_opponent_losses", "since_poll_weaker_opponent_mean_capped_margin",
    "days_since_latest_win", "days_since_latest_loss",
    *{f"since_poll_{venue}_{field}" for venue in ("home", "away", "neutral")
      for field in ("games", "wins", "losses", "mean_capped_margin")},
}
MODEL_EXCLUDE = ({"previous_points_gap_to_next_higher", "previous_points_gap_to_next_lower"} |
                 TRAJECTORY_FEATURES | RICH_GAME_CONTEXT_FEATURES)
TARGET = "actual_score"
POINT_CHANGE_BLEND_WEIGHT = .75
TRAINING_WINDOW_SEASONS = 4


def load_rows(root, start=2003, end=2026, feature_version=FEATURE_VERSION):
    rows = []
    for path in sorted(root.glob("*/runs/*/*/features.json")):
        try:
            season = int(path.parts[-5])
        except (ValueError, IndexError):
            continue
        if not start <= season <= end:
            continue
        directory = path.parent
        report = json.loads((directory / "report.json").read_text())
        # A feature-version bump leaves prior immutable artifacts on disk. They
        # remain useful for audit, but must not enter a training run mixed with
        # the current feature definition.
        if report.get("feature_version") != feature_version:
            continue
        predictions = { (p["run_id"], p["team_id"]): p
                        for p in json.loads((directory / "predictions.json").read_text()) }
        for row in json.loads(path.read_text()):
            p = predictions.get((row["run_id"], row["team_id"]))
            if p is None:
                raise ValueError(f"Missing/mismatched prediction for {path}")
            rows.append({**row, TARGET: p[TARGET], "actual_rank": p["actual_rank"]})
    if not rows:
        raise ValueError("No feature artifacts found")
    data = pd.DataFrame(rows)
    data = data.drop_duplicates(["season", "target_week", "team_id"], keep="last")
    return data.sort_values(["season", "target_week", "team_id"]).reset_index(drop=True)


def metrics(frame, prediction):
    values = frame[["season", "target_week", "team_id", TARGET, "actual_rank"]].copy()
    values["prediction"] = np.clip(prediction, 0, 1)
    scores = []
    errors = []
    rank_errors = []
    exact_ranks = []
    within_one = []
    within_two = []
    weekly_exact = []
    for _, poll in values.groupby(["season", "target_week"], sort=True):
        # Only official Top-25 teams count toward rank-accuracy metrics.
        # The feature rows also include teams receiving votes, whose
        # ``actual_rank`` is null and must not enter this denominator.
        actual = set(poll.loc[poll.actual_rank.between(1, 25), "team_id"])
        predicted = set(poll.nlargest(25, "prediction").team_id)
        scores.append(len(actual & predicted))
        errors.extend((poll.prediction - poll[TARGET]).to_numpy())
        predicted_order = poll.sort_values(["prediction", "team_id"], ascending=[False, True])
        predicted_ranks = {team: rank for rank, team in enumerate(predicted_order.team_id, 1)}
        poll_exact = []
        for _, row in poll[poll.team_id.isin(actual)].iterrows():
            error = abs(predicted_ranks[row.team_id] - int(row.actual_rank))
            rank_errors.append(error)
            exact_ranks.append(error == 0)
            within_one.append(error <= 1)
            within_two.append(error <= 2)
            poll_exact.append(error == 0)
        weekly_exact.append(float(np.mean(poll_exact)))
    err = np.asarray(errors)
    return {"polls": len(scores), "mean_top25_overlap": float(np.mean(scores)),
            "mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "top25_mean_absolute_rank_error": float(np.mean(rank_errors)),
            "top25_exact_rank_accuracy": float(np.mean(exact_ranks)),
            "top25_weekly_exact_rank_accuracy": float(np.mean(weekly_exact)),
            "top25_within_one_rank_accuracy": float(np.mean(within_one)),
            "top25_within_two_rank_accuracy": float(np.mean(within_two))}


def inclusion_metrics(frame, probability):
    """Metrics for the separate probability-of-appearing-in-the-poll model."""
    from sklearn.metrics import brier_score_loss, roc_auc_score
    values = frame[["season", "target_week", "team_id", TARGET]].copy()
    values["probability"] = np.clip(probability, 0, 1)
    overlaps = []
    for _, poll in values.groupby(["season", "target_week"], sort=True):
        actual = set(poll.nlargest(25, TARGET).team_id)
        predicted = set(poll.nlargest(25, "probability").team_id)
        overlaps.append(len(actual & predicted))
    labels = (values[TARGET].to_numpy() > 0).astype(int)
    probabilities = values["probability"].to_numpy()
    return {"polls": len(overlaps), "mean_top25_overlap": float(np.mean(overlaps)),
            "brier": float(brier_score_loss(labels, probabilities)),
            "roc_auc": float(roc_auc_score(labels, probabilities))}


def rank_metrics(frame, predicted_rank):
    """Evaluate an ordinal rank model without treating ranks as vote scores."""
    values = frame[["season", "target_week", "team_id", TARGET, "actual_rank"]].copy()
    values["predicted_rank"] = predicted_rank
    overlaps, errors, exact, within_one, within_two, weekly_exact = [], [], [], [], [], []
    for _, poll in values.groupby(["season", "target_week"], sort=True):
        # Exclude teams that received votes but were not officially ranked.
        actual = set(poll.loc[poll.actual_rank.between(1, 25), "team_id"])
        predicted = set(poll.nsmallest(25, "predicted_rank").team_id)
        overlaps.append(len(actual & predicted))
        order = poll.sort_values(["predicted_rank", "team_id"])
        ranks = {team: rank for rank, team in enumerate(order.team_id, 1)}
        poll_exact = []
        for _, row in poll[poll.team_id.isin(actual)].iterrows():
            error = abs(ranks[row.team_id] - int(row.actual_rank))
            errors.append(error)
            exact.append(error == 0)
            within_one.append(error <= 1)
            within_two.append(error <= 2)
            poll_exact.append(error == 0)
        weekly_exact.append(float(np.mean(poll_exact)))
    return {"polls": len(overlaps), "mean_top25_overlap": float(np.mean(overlaps)),
            "top25_mean_absolute_rank_error": float(np.mean(errors)),
            "top25_exact_rank_accuracy": float(np.mean(exact)),
            "top25_weekly_exact_rank_accuracy": float(np.mean(weekly_exact)),
            "top25_within_one_rank_accuracy": float(np.mean(within_one)),
            "top25_within_two_rank_accuracy": float(np.mean(within_two))}


def rank_miss_analysis(frame, prediction, higher_is_better=True):
    """Classify Top-25 ordering misses and movement bias by weekly results."""
    columns = ["season", "target_week", "team_id", "actual_rank", "previous_rank",
               "since_poll_d1_wins", "since_poll_d1_losses"]
    values = frame[columns].copy()
    values["prediction"] = prediction
    totals = {"polls": 0, "top25_entries": 0, "top25_exits": 0,
              "adjacent_rank_swaps": 0, "larger_rank_errors": 0}
    movement_bias = defaultdict(list)
    for _, poll in values.groupby(["season", "target_week"], sort=True):
        ordered = poll.sort_values(["prediction", "team_id"],
                                   ascending=[not higher_is_better, True])
        predicted_ranks = {team: rank for rank, team in enumerate(ordered.team_id, 1)}
        actual = set(poll.loc[poll.actual_rank.between(1, 25), "team_id"])
        predicted = set(ordered.head(25).team_id)
        totals["polls"] += 1
        totals["top25_entries"] += len(predicted - actual)
        totals["top25_exits"] += len(actual - predicted)
        for _, row in poll[poll.team_id.isin(actual)].iterrows():
            predicted_rank = predicted_ranks[row.team_id]
            error = abs(predicted_rank - int(row.actual_rank))
            if error == 1:
                totals["adjacent_rank_swaps"] += 1
            elif error > 1:
                totals["larger_rank_errors"] += 1
            if pd.notna(row.previous_rank) and 1 <= row.previous_rank <= 25:
                actual_change = row.previous_rank - row.actual_rank
                predicted_change = row.previous_rank - predicted_rank
                if row.since_poll_d1_losses:
                    movement_bias["after_loss"].append(predicted_change - actual_change)
                if row.since_poll_d1_wins:
                    movement_bias["after_win"].append(predicted_change - actual_change)
    return {**totals,
            "mean_rank_movement_bias": {name: float(np.mean(values))
                                        for name, values in movement_bias.items()},
            "movement_bias_interpretation": "Positive means the prediction moved teams upward more than the poll did."}


def poll_diagnostics(frame, prediction):
    """Return poll-level accuracy and per-position records for a prediction set."""
    values = frame[["season", "target_week", "team_id", "actual_rank"]].copy()
    values["prediction"] = np.clip(np.asarray(prediction), 0, 1)
    poll_results, position_records = [], []
    for (season, week), poll in values.groupby(["season", "target_week"], sort=True):
        metric = metrics(poll, poll.prediction.to_numpy())
        poll_results.append({"season": int(season), "week": int(week), **metric})
        ordered = poll.sort_values(["prediction", "team_id"], ascending=[False, True])
        ranks = {team_id: rank for rank, team_id in enumerate(ordered.team_id, 1)}
        for row in poll[poll.actual_rank.between(1, 25)].itertuples(index=False):
            raw_rank = ranks[row.team_id]
            position_records.append({"season": int(season), "week": int(week),
                                     "poll_id": f"{int(season)}-{int(week)}",
                                     "team_id": row.team_id,
                                     "actual_rank": int(row.actual_rank),
                                     "predicted_rank": int(raw_rank if raw_rank <= 25 else 26),
                                     "raw_predicted_rank": int(raw_rank),
                                     "retained": bool(raw_rank <= 25),
                                     "rank_error": int(raw_rank - row.actual_rank)})
    return poll_results, position_records


def ordinal_targets(data):
    """Map every team in each poll to its full order; 26 means unranked."""
    target = np.zeros(len(data), dtype=float)
    for _, indices in data.groupby(["season", "target_week"], sort=False).groups.items():
        poll = data.loc[indices].sort_values([TARGET, "team_id"], ascending=[False, True])
        target[poll.index.to_numpy()] = np.minimum(np.arange(1, len(poll) + 1), 26)
    return target


def rolling_validation(data, seed=42, minimum_training_seasons=8, holdout_seasons=2):
    """Compare models on future polls using a strict poll-by-poll time split.

    Each target poll is evaluated after fitting on the previous four seasons
    plus earlier polls from the target season. The final holdout seasons never
    enter this selection routine.
    """
    seasons = sorted(data.season.unique())[:-holdout_seasons]
    ordinal_target = ordinal_targets(data)
    results = []
    for index in range(minimum_training_seasons, len(seasons)):
        validation_season = seasons[index]
        train_seasons = seasons[max(0, index - TRAINING_WINDOW_SEASONS):index]
        features = sorted(set(data.columns) - EXCLUDE - MODEL_EXCLUDE - {TARGET, "actual_rank"})
        X = data[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.impute import SimpleImputer
        prior = data["previous_normal_points"].to_numpy()
        weeks = sorted(data.loc[data.season.eq(validation_season), "target_week"].unique())
        for week in weeks:
            train_mask = (data.season.isin(train_seasons) |
                          (data.season.eq(validation_season) & data.target_week.lt(week)))
            val_mask = data.season.eq(validation_season) & data.target_week.eq(week)
            imputer = SimpleImputer(strategy="median", add_indicator=True)
            transformed_train = imputer.fit_transform(X.loc[train_mask])
            transformed_validation = imputer.transform(X.loc[val_mask])

            def model():
                return HistGradientBoostingRegressor(
                    max_iter=200, max_leaf_nodes=15, learning_rate=.04,
                    l2_regularization=5, min_samples_leaf=20,
                    early_stopping=False, random_state=seed)

            point_model, delta_model, ordinal_model = model(), model(), model()
            point_model.fit(transformed_train, data.loc[train_mask, TARGET])
            delta_model.fit(transformed_train, data.loc[train_mask, TARGET] - prior[train_mask])
            ordinal_model.fit(transformed_train, ordinal_target[train_mask])
            validation_frame = data.loc[val_mask]
            point_prediction = point_model.predict(transformed_validation)
            delta_prediction = delta_model.predict(transformed_validation) + prior[val_mask]
            ordinal_prediction = ordinal_model.predict(transformed_validation)
            results.append({"season": int(validation_season), "week": int(week), "models": {
                "hgb_score": metrics(validation_frame, point_prediction),
                "hgb_delta": metrics(validation_frame, delta_prediction),
                "hgb_score_delta_blend": metrics(
                    validation_frame, POINT_CHANGE_BLEND_WEIGHT * delta_prediction +
                    (1 - POINT_CHANGE_BLEND_WEIGHT) * point_prediction),
                "hgb_rank": rank_metrics(validation_frame, ordinal_prediction),
            }})
    return results


def summarize_rolling_validation(results):
    """Pool weekly metrics while retaining the number of evaluated polls."""
    names = sorted({name for result in results for name in result["models"]})
    summary = {}
    for name in names:
        polls = np.asarray([result["models"][name]["polls"] for result in results])
        summary[name] = {"polls": int(polls.sum())}
        for metric in results[0]["models"][name]:
            if metric != "polls":
                values = [result["models"][name][metric] for result in results]
                summary[name][metric] = float(np.average(values, weights=polls))
    return summary


def train(data, test_seasons=2, seed=42):
    seasons = sorted(data.season.unique())
    if len(seasons) <= test_seasons + 1:
        raise ValueError("Need more seasons for chronological train/test")
    test = seasons[-test_seasons:]
    validation = seasons[-test_seasons-1:-test_seasons]
    available_train = [s for s in seasons if s not in set(test + validation)]
    train = available_train[-TRAINING_WINDOW_SEASONS:]
    features = sorted(set(data.columns) - EXCLUDE - MODEL_EXCLUDE - {TARGET, "actual_rank"})
    X = data[features].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    # Fit imputation on training rows only. Explicit missing indicators retain
    # the difference between no games and a missing source statistic.
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    train_mask = data.season.isin(train)
    X_train = imputer.fit_transform(X.loc[train_mask])
    X_val = imputer.transform(X.loc[data.season.isin(validation)])
    X_test = imputer.transform(X.loc[data.season.isin(test)])
    y = data[TARGET].to_numpy()
    prior = data["previous_normal_points"].to_numpy()
    y_delta = y - prior
    y_rank = ordinal_targets(data)

    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.ensemble import HistGradientBoostingClassifier
    models = {
        "ridge_score": (Ridge(alpha=10.0), y[train_mask]),
        "hgb_score": (HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=15,
                                    learning_rate=.04, l2_regularization=5,
                                    min_samples_leaf=20, early_stopping=False,
                                    random_state=seed), y[train_mask]),
        "hgb_delta": (HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=15,
                                    learning_rate=.04, l2_regularization=5,
                                    min_samples_leaf=20, early_stopping=False,
                                    random_state=seed), y_delta[train_mask]),
    }
    report = {"feature_version": FEATURE_VERSION, "train_seasons": train,
              "validation_seasons": validation, "test_seasons": test, "features": features,
              "models": {}, "target": TARGET, "seed": seed}
    fitted = {}
    validation_predictions, test_predictions = {}, {}
    for name, (model, target) in models.items():
        model.fit(X_train, target)
        val_pred = model.predict(X_val)
        if name == "hgb_delta": val_pred += prior[data.season.isin(validation)]
        val_frame = data[data.season.isin(validation)]
        test_pred = model.predict(X_test)
        if name == "hgb_delta": test_pred += prior[data.season.isin(test)]
        report["models"][name] = {
            "validation": metrics(val_frame, val_pred),
            "validation_miss_analysis": rank_miss_analysis(val_frame, val_pred),
            "test": metrics(data[data.season.isin(test)], test_pred),
            "test_miss_analysis": rank_miss_analysis(data[data.season.isin(test)], test_pred),
        }
        fitted[name] = model
        validation_predictions[name] = val_pred
        test_predictions[name] = test_pred
    validation_blend = (POINT_CHANGE_BLEND_WEIGHT * validation_predictions["hgb_delta"] +
                        (1 - POINT_CHANGE_BLEND_WEIGHT) * validation_predictions["hgb_score"])
    test_blend = (POINT_CHANGE_BLEND_WEIGHT * test_predictions["hgb_delta"] +
                  (1 - POINT_CHANGE_BLEND_WEIGHT) * test_predictions["hgb_score"])
    report["models"]["hgb_score_delta_blend"] = {
        "point_change_weight": POINT_CHANGE_BLEND_WEIGHT,
        "validation": metrics(data[data.season.isin(validation)], validation_blend),
        "validation_miss_analysis": rank_miss_analysis(data[data.season.isin(validation)], validation_blend),
        "test": metrics(data[data.season.isin(test)], test_blend),
        "test_miss_analysis": rank_miss_analysis(data[data.season.isin(test)], test_blend),
    }
    # A separate inclusion model learns the first AP decision: whether a team
    # receives any votes. Its probability is a useful ranking signal near the
    # Top-25 cutoff, while the regressors retain calibrated vote-share output.
    inclusion = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=15,
                                                learning_rate=.04, l2_regularization=5,
                                                random_state=seed)
    inclusion.fit(X_train, (y[train_mask] > 0).astype(int))
    inclusion_pred = inclusion.predict_proba(X_val)[:, 1]
    report["models"]["hgb_top25_inclusion"] = {
        "validation": inclusion_metrics(data[data.season.isin(validation)], inclusion_pred),
        "test": inclusion_metrics(data[data.season.isin(test)], inclusion.predict_proba(X_test)[:, 1]),
    }
    rank_model = HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=15,
                                                learning_rate=.04, l2_regularization=5,
                                                min_samples_leaf=20, early_stopping=False,
                                                random_state=seed)
    rank_model.fit(X_train, y_rank[train_mask])
    report["models"]["hgb_rank"] = {
        "validation": rank_metrics(data[data.season.isin(validation)], rank_model.predict(X_val)),
        "test": rank_metrics(data[data.season.isin(test)], rank_model.predict(X_test)),
    }
    rank_validation_prediction = rank_model.predict(X_val)
    rank_test_prediction = rank_model.predict(X_test)
    report["models"]["hgb_rank"].update({
        "validation_miss_analysis": rank_miss_analysis(data[data.season.isin(validation)], rank_validation_prediction,
                                                         higher_is_better=False),
        "test_miss_analysis": rank_miss_analysis(data[data.season.isin(test)], rank_test_prediction,
                                                   higher_is_better=False),
    })
    # Select across all earlier rolling seasons. The final two seasons never
    # participate in model or blend selection.
    report["rolling_validation"] = rolling_validation(data, seed=seed, holdout_seasons=test_seasons)
    report["rolling_validation_summary"] = summarize_rolling_validation(report["rolling_validation"])
    candidates = {"hgb_score", "hgb_delta", "hgb_rank", "hgb_score_delta_blend"}
    selected = min(candidates,
                   key=lambda n: (-report["rolling_validation_summary"][n]["top25_weekly_exact_rank_accuracy"],
                                  report["rolling_validation_summary"][n]["top25_mean_absolute_rank_error"],
                                  -report["rolling_validation_summary"][n]["mean_top25_overlap"]))
    refit_seasons = (train + validation)[-TRAINING_WINDOW_SEASONS:]
    refit_mask = data.season.isin(refit_seasons)
    X_refit = imputer.fit_transform(X.loc[refit_mask])
    X_test = imputer.transform(X.loc[data.season.isin(test)])
    if selected == "hgb_score_delta_blend":
        score_model = models["hgb_score"][0].__class__(**models["hgb_score"][0].get_params())
        delta_model = models["hgb_delta"][0].__class__(**models["hgb_delta"][0].get_params())
        score_model.fit(X_refit, y[refit_mask])
        delta_model.fit(X_refit, y[refit_mask] - prior[refit_mask])
        final_model = {"score_model": score_model, "delta_model": delta_model,
                       "point_change_weight": POINT_CHANGE_BLEND_WEIGHT}
        final_prediction = (POINT_CHANGE_BLEND_WEIGHT *
                            (delta_model.predict(X_test) + prior[data.season.isin(test)]) +
                            (1 - POINT_CHANGE_BLEND_WEIGHT) * score_model.predict(X_test))
    else:
        target = (y[refit_mask] - prior[refit_mask] if selected == "hgb_delta" else
                  y_rank[refit_mask] if selected == "hgb_rank" else y[refit_mask])
        selected_estimator = rank_model if selected == "hgb_rank" else models[selected][0]
        final_model = selected_estimator.__class__(**selected_estimator.get_params())
        final_model.fit(X_refit, target)
        final_prediction = final_model.predict(X_test)
        if selected == "hgb_delta":
            final_prediction += prior[data.season.isin(test)]
    report["selected_model"] = selected
    report["refit_seasons"] = refit_seasons
    selected_metrics = (rank_metrics(data[data.season.isin(test)], final_prediction) if selected == "hgb_rank"
                        else metrics(data[data.season.isin(test)], final_prediction))
    report["test_selected_after_refit"] = selected_metrics
    report["test_selected_miss_analysis"] = rank_miss_analysis(
        data[data.season.isin(test)], final_prediction, higher_is_better=selected != "hgb_rank")
    test_poll_results, test_position_records = poll_diagnostics(
        data[data.season.isin(test)], final_prediction)
    report["test_poll_results"] = test_poll_results
    report["test_position_records"] = test_position_records
    report["baseline_persistence_test"] = metrics(data[data.season.isin(test)], prior[data.season.isin(test)])
    return final_model, inclusion, rank_model, imputer, features, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/history"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/model"))
    parser.add_argument("--start", type=int, default=2003)
    parser.add_argument("--end", type=int, default=2026)
    args = parser.parse_args()
    data = load_rows(args.root, args.start, args.end)
    model, inclusion, rank_model, imputer, features, report = train(data)
    args.output.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump({"model": model, "inclusion_model": inclusion, "rank_model": rank_model,
                 "imputer": imputer, "features": features,
                 "target": "ordinal_rank" if report["selected_model"] == "hgb_rank" else report["target"],
                 "model_name": report["selected_model"]},
                args.output / "model.joblib")
    (args.output / "features.json").write_text(json.dumps(features, indent=2) + "\n")
    # Pandas/numpy scalars can appear in metric values; normalize them before
    # hashing and writing the report so the artifact is portable across Python
    # environments.
    report = json.loads(json.dumps(report, default=lambda value: value.item()
                                   if hasattr(value, "item") else str(value)))
    report["artifact_hash"] = content_hash(report)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
