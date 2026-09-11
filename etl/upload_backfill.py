"""Validate and upload a complete research backfill through one atomic Supabase RPC."""

import argparse
from collections import defaultdict
from datetime import date
import json
import math
import os
from pathlib import Path

from dotenv import load_dotenv
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .features import FEATURE_VERSION, content_hash

FEATURE_COLUMNS = set("""
run_id season target_week cutoff_date_exclusive previous_poll_date feature_version
team_id team_name d1_games d1_wins d1_losses d1_win_pct d1_mean_margin d1_mean_capped_margin
d1_elo d1_mean_opponent_pregame_elo since_poll_d1_games since_poll_d1_wins
since_poll_d1_losses since_poll_d1_margin since_poll_d1_mean_capped_margin
since_poll_d1_mean_opponent_pregame_elo since_poll_ranked_wins last_five_d1_wins
since_poll_ranked_losses since_poll_unranked_losses since_poll_elo_change d1_elo_change
since_poll_losses_to_higher_point_teams
since_poll_overtime_games since_poll_overtime_wins since_poll_overtime_losses
last_three_d1_wins last_three_d1_win_pct last_five_d1_win_pct
last_five_d1_mean_capped_margin last_five_d1_mean_opponent_pregame_elo
last_ten_d1_wins last_ten_d1_win_pct last_ten_d1_mean_capped_margin
since_poll_ranked_win_pct since_poll_ranked_loss_pct recent_win_pct_change
recent_margin_change recent_sos_change previous_score_change previous_rank_change
since_poll_best_win_opponent_rank since_poll_worst_loss_opponent_rank
last_five_best_win_opponent_rank last_five_worst_loss_opponent_rank
since_poll_best_win_opponent_elo since_poll_worst_loss_opponent_elo
since_poll_mean_opponent_previous_normal_points since_poll_best_win_opponent_previous_normal_points
since_poll_worst_loss_opponent_previous_normal_points
since_poll_stronger_opponent_games since_poll_stronger_opponent_wins
since_poll_stronger_opponent_mean_capped_margin since_poll_weaker_opponent_games
since_poll_weaker_opponent_losses since_poll_weaker_opponent_mean_capped_margin
days_since_latest_win days_since_latest_loss
since_poll_quality_weighted_wins since_poll_quality_weighted_losses
rest_days previous_normal_points previous_rank previous_ranked preseason_poll_available
previous_points_gap_to_next_higher previous_points_gap_to_next_lower previous_nearby_team_count
nearby_teams_since_poll_wins nearby_teams_since_poll_losses nearby_teams_since_poll_elo_change
nearby_teams_since_poll_mean_opponent_elo
previous_ranked_teams_with_losses previous_ranked_teams_with_wins
previous_top10_teams_with_losses higher_ranked_teams_with_losses
previous_normal_points_lag_2 previous_normal_points_lag_3 previous_normal_points_lag_4
previous_points_trend_3_poll previous_points_trend_4_poll previous_points_change_volatility_4_poll
preseason_normal_points preseason_rank preseason_ranked days_since_preseason_poll
d1_home_games d1_home_wins d1_home_win_pct d1_away_games d1_away_wins d1_away_win_pct
d1_neutral_games d1_neutral_wins d1_neutral_win_pct
since_poll_home_games since_poll_home_wins since_poll_home_losses since_poll_home_mean_capped_margin
since_poll_away_games since_poll_away_wins since_poll_away_losses since_poll_away_mean_capped_margin
since_poll_neutral_games since_poll_neutral_wins since_poll_neutral_losses since_poll_neutral_mean_capped_margin
""".split())
PREDICTION_COLUMNS = set("""
run_id season target_week team_id model_version predicted_score predicted_rank actual_score actual_rank
""".split())


def prepare_upload(report, features, predictions):
    """Fail offline before writing anything if the artifact is incomplete/mixed."""
    if report.get("mode") != "research_only" or report.get("feature_version") != FEATURE_VERSION:
        raise ValueError(f"Expected research-only {FEATURE_VERSION} artifacts; rebuild with the current pipeline")
    expected = report["feature_rows"]
    if not features or len(features) != expected or len(predictions) != expected:
        raise ValueError("Feature/prediction row counts do not match the report")
    groups, lookup = defaultdict(list), {}
    for row in features:
        if set(row) != FEATURE_COLUMNS:
            raise ValueError("Feature columns do not match the migration")
        if row["season"] != report["season"] or row["feature_version"] != report["feature_version"]:
            raise ValueError("Mixed feature metadata")
        if date.fromisoformat(row["previous_poll_date"]) >= date.fromisoformat(row["cutoff_date_exclusive"]):
            raise ValueError("Invalid forecast cutoff")
        expected_id = content_hash({"source": report["source_hash"], "week": row["target_week"],
                                    "cutoff": row["cutoff_date_exclusive"]})
        if row["run_id"] != expected_id:
            raise ValueError("Run ID does not match the report and cutoff")
        key = (row["run_id"], row["team_id"])
        if key in lookup:
            raise ValueError("Duplicate feature row")
        lookup[key] = row
        groups[row["target_week"]].append(row)
    if len(groups) != report["evaluated_polls"]:
        raise ValueError("Poll count does not match the report")
    universe = None
    for rows in groups.values():
        teams = {r["team_id"] for r in rows}
        if len(rows) != report["eligible_teams"] or len(teams) != len(rows):
            raise ValueError("Incomplete team coverage")
        if universe is not None and teams != universe:
            raise ValueError("Inconsistent team universe")
        universe = teams
        if len({r["run_id"] for r in rows}) != 1 or len({r["previous_poll_date"] for r in rows}) != 1:
            raise ValueError("Mixed poll snapshot metadata")
    seen = set()
    ranks = defaultdict(set)
    for row in predictions:
        if set(row) != PREDICTION_COLUMNS:
            raise ValueError("Prediction columns do not match the migration")
        key = (row["run_id"], row["team_id"])
        if key in seen or key not in lookup:
            raise ValueError("Duplicate or unmatched prediction")
        seen.add(key)
        feature = lookup[key]
        if row["season"] != feature["season"] or row["target_week"] != feature["target_week"]:
            raise ValueError("Prediction metadata does not match its features")
        if row["model_version"] != "persistence-v1" or row["predicted_score"] != feature["previous_normal_points"]:
            raise ValueError("Unexpected baseline model or prediction")
        if not all(math.isfinite(row[k]) and 0 <= row[k] <= 1 for k in ("predicted_score", "actual_score")):
            raise ValueError("Invalid normalized score")
        ranks[row["target_week"]].add(row["predicted_rank"])
    if any(values != set(range(1, report["eligible_teams"] + 1)) for values in ranks.values()):
        raise ValueError("Prediction ranks are not a complete ordering")
    # Also rejects NaN/Infinity anywhere in the report or features.
    checksum = content_hash({"report": report, "features": features, "predictions": predictions})
    return {"p_report": report, "p_features": features, "p_predictions": predictions,
            "p_content_hash": checksum}


def upload(session, base, payload):
    try:
        response = session.post(f"{base}/rest/v1/rpc/upload_feature_backfill", json=payload, timeout=60)
    except requests.RequestException:
        raise RuntimeError("Supabase connection failed; rerunning the identical upload is safe") from None
    if response.status_code == 404:
        raise RuntimeError("Upload RPC missing. Run supabase/migrations/202609100001_feature_backfills.sql in the SQL Editor first")
    if not response.ok:
        raise RuntimeError(f"Supabase rejected the upload (HTTP {response.status_code}); inspect database logs")
    result = response.json()
    if (result.get("status") not in {"uploaded", "already_uploaded"}
            or result.get("backfill_id") != payload["p_report"]["source_hash"]
            or result.get("feature_rows") != payload["p_report"]["feature_rows"]):
        raise RuntimeError("Unexpected upload acknowledgment")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True, help="Exact directory containing report/features/predictions.json")
    parser.add_argument("--apply", action="store_true", help="Write to Supabase; otherwise only validate locally")
    parser.add_argument("--direct-db", action="store_true", help="Use SUPABASE_DB_URL instead of the REST API")
    args = parser.parse_args()
    report, features, predictions = [json.loads((args.artifact_dir / name).read_text())
                                     for name in ("report.json", "features.json", "predictions.json")]
    payload = prepare_upload(report, features, predictions)
    print(f"Validated season {report['season']}: {len(features)} feature rows, {len(predictions)} predictions, {report['evaluated_polls']} polls")
    print(f"Feature version: {report['feature_version']}; mode: {report['mode']}")
    print(f"Excluded invalid polls: {report['invalid_polls']}")
    print(f"Request size: {len(json.dumps(payload).encode()) / 1024 / 1024:.2f} MiB")
    if not args.apply:
        print("Preview only. Add --apply after installing the migration to upload.")
        return
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    if args.direct_db:
        import psycopg
        from psycopg.types.json import Jsonb
        try:
            with psycopg.connect(os.environ["SUPABASE_DB_URL"], connect_timeout=15, sslmode="require") as conn:
                result = conn.execute(
                    "select public.upload_feature_backfill(%s, %s, %s, %s)",
                    (Jsonb(payload["p_report"]), Jsonb(payload["p_features"]),
                     Jsonb(payload["p_predictions"]), payload["p_content_hash"]),
                ).fetchone()[0]
        except psycopg.Error as exc:
            raise RuntimeError(f"Database upload failed ({type(exc).__name__}, SQLSTATE {exc.sqlstate}); inspect database logs") from None
        print(json.dumps(result, indent=2))
        return
    key = os.environ["SUPABASE_SERVICE_KEY"]
    base = os.environ["SUPABASE_URL"].rstrip("/")
    session = requests.Session()
    session.headers.update({"apikey": key, "Authorization": f"Bearer {key}"})
    # The RPC is idempotent; retrying a lost acknowledgment cannot duplicate rows.
    session.mount("https://", HTTPAdapter(max_retries=Retry(
        total=2, backoff_factor=1, status_forcelist=[429, 502, 503, 504], allowed_methods=["POST"]
    )))
    print(json.dumps(upload(session, base, payload), indent=2))


if __name__ == "__main__":
    main()
