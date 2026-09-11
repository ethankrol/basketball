"""Inspect adjacent ordering errors, fit gap, and feature coverage without using holdout seasons."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from etl.features import alias_map, eligible_teams, normalize_name
from ml.experiment_rank_accuracy import model_prediction
from ml.train_model import EXCLUDE, MODEL_EXCLUDE, POINT_CHANGE_BLEND_WEIGHT, TARGET, load_rows, metrics


def game_details(source, frame, team_id, start, end, aliases):
    lookup = frame.set_index("team_id")
    details = []
    for game in source["tables"]["games"]:
        if not start <= game["date"] < end or aliases.get(normalize_name(game["team"])) != team_id:
            continue
        opponent_id = aliases.get(normalize_name(game["opponent"]))
        opponent = lookup.loc[opponent_id] if opponent_id in lookup.index else None
        details.append({"date": game["date"], "result": "W" if game["team_score"] > game["opponent_score"] else "L",
                        "score": f"{game['team_score']}-{game['opponent_score']}",
                        "margin": game["team_score"] - game["opponent_score"],
                        "location": "N" if game["neutral"] else "H" if game["home"] else "A",
                        "overtime": game["overtime"], "opponent": game["opponent"],
                        "opponent_previous_rank": (None if opponent is None or pd.isna(opponent.previous_rank)
                                                   else int(opponent.previous_rank)),
                        "opponent_previous_points": None if opponent is None else float(opponent.previous_normal_points),
                        "opponent_elo_at_cutoff": None if opponent is None else float(opponent.d1_elo)})
    return details


def adjacent_cases(data, prediction, source, sample_size=40):
    frame = data.copy()
    frame["prediction"] = np.clip(prediction, 0, 1)
    aliases = alias_map(source["tables"]["teams"], source["tables"]["team_spellings"])
    candidates = []
    for (_, week), poll in frame.groupby(["season", "target_week"], sort=True):
        ordered = poll.sort_values(["prediction", "team_id"], ascending=[False, True]).head(25)
        records = list(ordered.itertuples(index=False))
        for predicted_rank, (upper, lower) in enumerate(zip(records, records[1:]), 1):
            if pd.isna(upper.actual_rank) or pd.isna(lower.actual_rank) or upper.actual_rank < lower.actual_rank:
                continue
            candidates.append((week, upper.prediction - lower.prediction, predicted_rank, upper, lower, poll))
    # Inspect each poll before taking additional smallest-gap cases.
    chosen, seen = [], set()
    by_week = defaultdict(list)
    for case in sorted(candidates, key=lambda item: item[1]):
        by_week[case[0]].append(case)
    for offset in range(3):
        for week in sorted(by_week):
            if offset < len(by_week[week]) and len(chosen) < sample_size:
                case = by_week[week][offset]
                chosen.append(case)
                seen.add((case[0], case[2]))
    for case in sorted(candidates, key=lambda item: item[1]):
        if len(chosen) >= sample_size:
            break
        if (case[0], case[2]) not in seen:
            chosen.append(case)
    result = []
    for week, gap, predicted_rank, upper, lower, poll in chosen:
        teams = []
        for row in (upper, lower):
            teams.append({"team": row.team_name, "team_id": int(row.team_id),
                          "previous_rank": None if pd.isna(row.previous_rank) else int(row.previous_rank),
                          "previous_points": float(row.previous_normal_points),
                          "predicted_rank": predicted_rank if row.team_id == upper.team_id else predicted_rank + 1,
                          "predicted_points": float(row.prediction),
                          "actual_rank": int(row.actual_rank), "actual_points": float(row.actual_score),
                          "games": game_details(source, poll, row.team_id, row.previous_poll_date,
                                                row.cutoff_date_exclusive, aliases)})
        result.append({"week": int(week), "previous_point_gap": abs(teams[0]["previous_points"] - teams[1]["previous_points"]),
                       "predicted_point_gap": float(gap), "predicted_order_was_reversed": True,
                       "teams": teams})
    return result


def pattern_summary(cases):
    counts = Counter()
    for case in cases:
        upper, lower = case["teams"]
        upper_games, lower_games = upper["games"], lower["games"]
        if not any(g["result"] == "L" for g in upper_games + lower_games):
            counts["both_undefeated_since_previous_poll"] += 1
        if upper["previous_rank"] and lower["previous_rank"] and upper["previous_rank"] < lower["previous_rank"]:
            counts["model_retained_previous_order"] += 1
        if any(g["result"] == "L" for g in upper_games):
            counts["predicted_upper_team_had_loss"] += 1
        if any(g["result"] == "W" and ((g["opponent_previous_rank"] or 99) <= 25 or
                                       (g["opponent_elo_at_cutoff"] or 0) >= 1550) for g in lower_games):
            counts["actual_upper_team_had_strong_win"] += 1
        if any(g["location"] in {"A", "N"} for g in lower_games):
            counts["actual_upper_team_played_away_or_neutral"] += 1
        if any(g["overtime"] for g in upper_games + lower_games):
            counts["pair_included_overtime"] += 1
    return {"cases": len(cases), **counts}


def missingness(data, features):
    rows = []
    for season, frame in data.groupby("season", sort=True):
        missing = frame[features].isna().mean()
        rows.append({"season": int(season), "rows": len(frame),
                     "features_over_50pct_missing": int((missing > .5).sum()),
                     "mean_missing_fraction": float(missing.mean()),
                     "worst_features": {name: float(value) for name, value in
                                        missing.sort_values(ascending=False).head(10).items()}})
    return rows


def run(root, season=2024, sample_size=40):
    data = load_rows(root, 2003, 2026)
    if season in sorted(data.season.unique())[-2:]:
        raise ValueError("Audit season must not be one of the two reserved holdout seasons")
    features = sorted(set(data.columns) - EXCLUDE - MODEL_EXCLUDE - {TARGET, "actual_rank"})
    prior_seasons = sorted(value for value in data.season.unique() if value < season)[-4:]
    train_mask, validation_mask = data.season.isin(prior_seasons), data.season.eq(season)
    validation = data.loc[validation_mask]
    delta = model_prediction(data, features, train_mask, validation_mask, "delta", 42)
    points = model_prediction(data, features, train_mask, validation_mask, "score", 42)
    prediction = POINT_CHANGE_BLEND_WEIGHT * delta + (1 - POINT_CHANGE_BLEND_WEIGHT) * points
    # In-sample accuracy is diagnostic only and never used for selection.
    train_delta = model_prediction(data, features, train_mask, train_mask, "delta", 42)
    train_points = model_prediction(data, features, train_mask, train_mask, "score", 42)
    train_prediction = POINT_CHANGE_BLEND_WEIGHT * train_delta + (1 - POINT_CHANGE_BLEND_WEIGHT) * train_points
    source = json.loads((root / str(season) / "source.json").read_text())
    cases = adjacent_cases(validation, prediction, source, sample_size)
    weekly = []
    for week, indices in validation.groupby("target_week", sort=True).groups.items():
        positions = validation.index.get_indexer(indices)
        weekly.append({"week": int(week), **metrics(validation.loc[indices], prediction[positions])})
    all_numeric_features = sorted(set(data.columns) - EXCLUDE - {TARGET, "actual_rank"})
    return {"audit_season": season, "holdout_used": False,
            "training_seasons": [int(value) for value in prior_seasons], "features": features,
            "fit": {"training": metrics(data.loc[train_mask], train_prediction),
                    "chronological_validation": metrics(validation, prediction)},
            "validation_by_week": weekly,
            "model_feature_missingness_by_season": missingness(data[data.season <= season], features),
            "all_candidate_feature_missingness_by_season": missingness(
                data[data.season <= season], all_numeric_features),
            "adjacent_swap_patterns": pattern_summary(cases), "adjacent_swap_cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/history"))
    parser.add_argument("--season", type=int, default=2024)
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--output", type=Path, default=Path("artifacts/model_experiments/rank_error_audit.json"))
    args = parser.parse_args()
    report = run(args.root, args.season, args.sample_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("audit_season", "holdout_used", "fit",
                                                    "adjacent_swap_patterns")}, indent=2))


if __name__ == "__main__":
    main()
