"""Replay a completed season and export ratings for next season's regression."""

import argparse
from datetime import date
import json
from pathlib import Path

from .elo import ELO_VERSION, season_initial_ratings
from .features import alias_map, canonical_games, compute_features, content_hash, eligible_teams
from .membership import season_nonparticipants


def build_state(payload, prior_state=None, carryover=0.75, bootstrap=False, overrides=None):
    season, tables = payload["season"], payload["tables"]
    if date.today() < date(season, 7, 1):
        raise ValueError("Cannot finalize Elo for an unfinished season")
    eligible = eligible_teams(tables["teams"], season, tables.get('team_memberships'))
    overrides = overrides or {"season": season}
    if overrides["season"] != season:
        raise ValueError("Overrides must match the Elo season")
    aliases = alias_map(tables["teams"], tables["team_spellings"], overrides.get("aliases", []))
    games, audit = canonical_games(tables["games"], aliases, eligible, season, season_nonparticipants(tables, season))
    # Catch obviously partial exports; full completeness still requires source audit.
    completion_floor = overrides.get('completion_floor', f'{season}-04-01')
    if max(g["date"] for g in games) < completion_floor:
        raise ValueError("Prior-season games do not extend into April; export appears incomplete")
    initial = season_initial_ratings(eligible, season, prior_state, carryover, bootstrap)
    rows = compute_features(games, eligible, f"{season}-07-01", {}, f"{season - 1}-07-01",
                            initial_ratings=initial)
    return {"season": season, "elo_version": ELO_VERSION, "through_date_exclusive": f"{season}-07-01",
            "source_hash": content_hash({"tables": tables, "overrides": overrides}), "prior_state_hash": content_hash(prior_state) if prior_state else None,
            "carryover": carryover, "bootstrap": bootstrap, "audit": audit,
            "initial_season": prior_state.get('initial_season', prior_state['season']) if prior_state else season,
            "ratings": {str(r["team_id"]): r["d1_elo"] for r in rows}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overrides", type=Path)
    seeds = parser.add_mutually_exclusive_group(required=True)
    seeds.add_argument("--prior-elo", type=Path)
    seeds.add_argument("--bootstrap-elo", action="store_true")
    parser.add_argument("--elo-carryover", type=float, default=0.75)
    args = parser.parse_args()
    state = build_state(json.loads(args.input.read_text()),
                        json.loads(args.prior_elo.read_text()) if args.prior_elo else None,
                        args.elo_carryover, args.bootstrap_elo,
                        json.loads(args.overrides.read_text()) if args.overrides else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(state['ratings'])} season-ending ratings to {args.output}")


if __name__ == "__main__":
    main()
