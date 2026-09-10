"""Build local research snapshots and persistence predictions from an export."""

import argparse
from collections import defaultdict
from datetime import date
import json
from pathlib import Path

from .features import (FEATURE_VERSION, alias_map, canonical_games, compute_features,
                       content_hash, eligible_teams, validate_poll)
from .elo import ELO_VERSION, season_initial_ratings


def build(payload, overrides, prior_state=None, carryover=0.75, bootstrap=False):
    season, tables = payload["season"], payload["tables"]
    if overrides.get("season") != season:
        raise ValueError("Overrides must explicitly match the exported season")
    aliases = alias_map(tables["teams"], tables["team_spellings"], overrides.get("aliases", []))
    eligible = eligible_teams(tables["teams"], season)
    initial_ratings = season_initial_ratings(eligible, season, prior_state, carryover, bootstrap)
    games, game_audit = canonical_games(tables["games"], aliases, eligible, season)
    groups = defaultdict(list)
    for row in tables["polls"]:
        if row["season"] not in {season, season % 100}:
            raise ValueError("Mixed seasons in poll input")
        groups[row["week"]].append(row)
    if not groups or 1 not in groups:
        raise ValueError("Missing preseason poll")
    polls, issues, dates = {}, {}, {}
    for week, rows in sorted(groups.items()):
        try:
            polls[week] = validate_poll(rows, aliases, eligible)
            dates[week] = overrides.get("release_dates", {}).get(str(week), rows[0]["date"])
            day = date.fromisoformat(dates[week])
            if not date(season - 1, 7, 1) <= day < date(season, 7, 1):
                raise ValueError("Poll date outside season")
        except ValueError as exc:
            polls.pop(week, None)
            issues[week] = str(exc)
    identity = {"season": season, "tables_hash": content_hash(tables),
                "overrides": overrides, "feature_version": FEATURE_VERSION,
                "prior_elo_state": prior_state, "elo_carryover": carryover, "elo_bootstrap": bootstrap}
    source_hash = content_hash(identity)
    snapshots, predictions, metrics, skipped = [], [], [], []
    for week in range(2, max(groups) + 1):
        if week not in polls or week - 1 not in polls:
            skipped.append({"week": week, "reason": "Target or immediately preceding poll missing/invalid"})
            continue
        cutoff = dates[week]
        prior_date = dates[week - 1]
        rows = compute_features(games, eligible, cutoff, polls[week - 1], prior_date,
                                preseason_poll=polls.get(1), preseason_date=dates.get(1), initial_ratings=initial_ratings)
        run_id = content_hash({"source": source_hash, "week": week, "cutoff": cutoff})
        # Tie-break by stable ID for reproducible display; scores retain ties.
        ordered = sorted(rows, key=lambda r: (-r["previous_normal_points"], r["team_id"]))
        ranks = {r["team_id"]: rank for rank, r in enumerate(ordered, 1)}
        actual = polls[week]
        actual_top = {t for t, r in actual.items() if r["rank"] <= 25}
        predicted_top = {r["team_id"] for r in ordered[:25]}
        errors, recipient_errors = [], []
        for r in rows:
            t = r["team_id"]
            snapshots.append({"run_id": run_id, "season": season, "target_week": week,
                              "cutoff_date_exclusive": cutoff, "previous_poll_date": prior_date,
                              "feature_version": FEATURE_VERSION, **r})
            target = actual.get(t, {"score": 0.0, "rank": None})
            error = abs(r["previous_normal_points"] - target["score"])
            errors.append(error)
            if target["score"] > 0:
                recipient_errors.append(error)
            predictions.append({"run_id": run_id, "season": season, "target_week": week,
                                "team_id": t, "model_version": "persistence-v1",
                                "predicted_score": r["previous_normal_points"], "predicted_rank": ranks[t],
                                "actual_score": target["score"],
                                "actual_rank": target["rank"] if t in actual_top else None})
        metrics.append({"week": week, "date": cutoff,
                        "top25_overlap": len(actual_top & predicted_top),
                        "actual_top25_size": len(actual_top),
                        "mae_all_teams": sum(errors) / len(errors),
                        "rmse_all_teams": (sum(e * e for e in errors) / len(errors)) ** 0.5,
                        "mae_vote_recipients": sum(recipient_errors) / len(recipient_errors)})
    if not metrics:
        raise ValueError(f"No valid adjacent polls to evaluate: {issues}")
    report = {"season": season, "source_hash": source_hash, "feature_version": FEATURE_VERSION,
              "mode": "research_only", "eligible_teams": len(eligible), **game_audit,
              "polls_found": len(groups), "invalid_polls": issues, "skipped_predictions": skipped,
              "evaluated_polls": len(metrics), "feature_rows": len(snapshots),
              "mean_top25_overlap": sum(m["top25_overlap"] for m in metrics) / len(metrics),
              "mean_mae_vote_recipients": sum(m["mae_vote_recipients"] for m in metrics) / len(metrics),
              "elo_carryover": carryover, "elo_bootstrap": bootstrap,
              "prior_elo_state_hash": content_hash(prior_state) if prior_state is not None else None,
              "metrics": metrics,
              "limitations": ["Legacy weekly dates have not all been verified against source releases.",
                              "Unresolved/noneligible game opponents are excluded; all result features are D1-only.",
                              "Point-total checks are necessary but do not prove source completeness.",
                              "Historical reconstruction, not an archived live forecast or held-out trained model.",
                              "No final postseason poll is synthesized when absent from the source."]}
    return snapshots, predictions, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--overrides", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    seeds = parser.add_mutually_exclusive_group(required=True)
    seeds.add_argument("--prior-elo", type=Path)
    seeds.add_argument("--bootstrap-elo", action="store_true")
    parser.add_argument("--elo-carryover", type=float, default=0.75)
    args = parser.parse_args()
    rows, predictions, report = build(json.loads(args.input.read_text()), json.loads(args.overrides.read_text()),
                                      json.loads(args.prior_elo.read_text()) if args.prior_elo else None,
                                      args.elo_carryover, args.bootstrap_elo)
    # Content-addressed directories prevent corrections from overwriting an earlier run.
    destination = args.output / report["source_hash"]
    destination.mkdir(parents=True, exist_ok=True)
    for name, value in [("features.json", rows), ("predictions.json", predictions), ("report.json", report)]:
        target = destination / name
        temp = target.with_suffix(".tmp")
        temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temp.replace(target)
    print(json.dumps({k: report[k] for k in ["season", "eligible_teams", "physical_d1_games", "evaluated_polls",
                                            "feature_rows", "invalid_polls", "mean_top25_overlap"]}, indent=2))
    print(f"Artifacts: {destination}")


if __name__ == "__main__":
    main()
