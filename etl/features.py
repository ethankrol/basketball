"""Pure, date-cutoff feature computation. No network or database side effects."""

from collections import defaultdict
from datetime import date
import hashlib
import json

FEATURE_VERSION = "d1-v4"
MARGIN_CAP = 20


def normalize_name(value):
    return " ".join(value.casefold().split())


def alias_map(teams, spellings, overrides=()):
    result = {}
    valid_ids = {t["team_id"] for t in teams}
    pairs = [(t["team_name"], t["team_id"]) for t in teams]
    pairs += [(r["team_spelling"], r["team_id"]) for r in spellings]
    pairs += [(r["alias"], r["team_id"]) for r in overrides]
    for name, team_id in pairs:
        name = normalize_name(name)
        if team_id not in valid_ids:
            raise ValueError(f"Alias references unknown ID: {team_id}")
        if name in result and result[name] != team_id:
            raise ValueError(f"Conflicting alias: {name}")
        result[name] = team_id
    return result


def eligible_teams(teams, season, memberships=None):
    from .membership import active_teams
    return active_teams(teams, memberships, season)


def canonical_games(rows, aliases, eligible, season, nonparticipants=()):
    """Validate mirrored rows; retain one physical D1-vs-D1 game.

    Unresolved/noneligible opponents are explicitly excluded and reported. Features
    are D1-only, not official overall records. No fuzzy identity matching.
    """
    groups = defaultdict(list)
    unresolved = set()
    excluded_rows = 0
    for r in rows:
        if r["season"] not in {season, season % 100}:
            raise ValueError("Mixed seasons in game input")
        a, b = aliases.get(normalize_name(r["team"])), aliases.get(normalize_name(r["opponent"]))
        for name, team_id in [(r["team"], a), (r["opponent"], b)]:
            if team_id is None:
                unresolved.add(normalize_name(name))
        if a not in eligible or b not in eligible:
            excluded_rows += 1
            continue
        day = date.fromisoformat(r["date"])
        if not date(season - 1, 7, 1) <= day < date(season, 7, 1):
            raise ValueError("Game date outside season")
        if a == b:
            raise ValueError("A team cannot play itself")
        for field in ["team_score", "opponent_score"]:
            if type(r[field]) is not int or r[field] < 0:
                raise ValueError("Invalid final score")
        if r["team_score"] == r["opponent_score"]:
            raise ValueError("Tied final score")
        for field in ["home", "neutral", "overtime"]:
            if type(r[field]) is not bool:
                raise ValueError(f"Invalid {field} flag")
        low, high = sorted((a, b))
        groups[(r["date"], low, high)].append((a, r))
    games = []
    for (day, low, high), pair in sorted(groups.items()):
        if len(pair) != 2 or {a for a, _ in pair} != {low, high}:
            raise ValueError(f"Missing/duplicate mirrored game: {day}, {low}, {high}")
        pair.sort(key=lambda v: v[0])
        left, right = pair[0][1], pair[1][1]
        if (left["team_score"] != right["opponent_score"] or
            left["opponent_score"] != right["team_score"] or
            left["neutral"] != right["neutral"] or
            left["overtime"] != right["overtime"] or
            (not left["neutral"] and left["home"] == right["home"])):
            raise ValueError(f"Inconsistent mirrored game: {day}, {low}, {high}")
        games.append({"date": day, "a": low, "b": high,
                      "score_a": left["team_score"], "score_b": right["team_score"],
                      "venue_a": "N" if left["neutral"] else ("H" if left["home"] else "A")})
    present = {g[k] for g in games for k in ("a", "b")}
    if present & set(nonparticipants):
        raise ValueError("Recorded nonparticipant has D1 games")
    missing = set(eligible) - present - set(nonparticipants)
    if missing:
        raise ValueError(f"Eligible teams without mapped D1 games: {sorted(missing)}")
    return games, {"physical_d1_games": len(games), "excluded_team_game_rows": excluded_rows,
                   "unresolved_game_names": sorted(unresolved)}


def validate_poll(rows, aliases, eligible):
    """Reject incomplete ballots before filling absent teams with zero points."""
    if not rows or len({r["date"] for r in rows}) != 1:
        raise ValueError("Missing poll or mixed release dates")
    if len({(r["season"], r["week"]) for r in rows}) != 1:
        raise ValueError("Mixed poll identifiers")
    mapped = {}
    for r in rows:
        team_id = aliases.get(normalize_name(r["team"]))
        if team_id not in eligible:
            raise ValueError(f"Unresolved/ineligible poll team: {r['team']}")
        if team_id in mapped:
            raise ValueError(f"Duplicate poll team: {r['team']}")
        if any(type(r[k]) is not int or r[k] < 0 for k in ("votes", "first_votes")):
            raise ValueError("Invalid poll points")
        mapped[team_id] = r
    voters = sum(r["first_votes"] for r in rows)
    points = sum(r["votes"] for r in rows)
    if voters <= 0 or points != 325 * voters:
        raise ValueError(f"Incomplete poll: {points} points; expected {325 * voters} for {voters} voters")
    if len(rows) < 25 or any(r["votes"] > 25 * voters or r["votes"] < 25 * r["first_votes"] for r in rows):
        raise ValueError("Invalid ballot totals")
    # Recompute competition ranks: legacy ranks below 25 can be shifted by missing rows.
    values = [r["votes"] for r in rows]
    return {t: {"score": r["votes"] / (25 * voters),
                "rank": 1 + sum(v > r["votes"] for v in values)} for t, r in mapped.items()}


def compute_features(games, eligible, cutoff, previous_poll, previous_date, k=20.0, home_advantage=65.0,
                     *, preseason_poll=None, preseason_date=None, initial_ratings=None):
    """Use game dates strictly before cutoff; replay both sides once per game.

    Same-day ratings use a common start-of-day state. Callers supply regressed
    preseason ratings; omitted ratings bootstrap at 1500. Since-poll window includes the previous release's game date,
    because that day's games were excluded from its morning forecast.
    Capped margins clip each signed game margin to [-20, 20] before averaging.
    SOS is game-weighted opponent pregame Elo. Preseason inputs must come from
    a validated, already-released preseason poll; None means unavailable.
    """
    cutoff = date.fromisoformat(cutoff)
    previous_date = date.fromisoformat(previous_date)
    if previous_date >= cutoff:
        raise ValueError("Previous poll must precede the forecast cutoff")
    if preseason_poll is not None:
        if preseason_date is None:
            raise ValueError("Preseason poll requires its release date")
        preseason_date = date.fromisoformat(preseason_date)
        if preseason_date > previous_date:
            raise ValueError("Preseason poll must be released by the previous poll date")
    ratings = {t: float(initial_ratings[t]) if initial_ratings is not None else 1500.0 for t in eligible}
    history = defaultdict(list)
    days = defaultdict(list)
    for game in games:
        if date.fromisoformat(game["date"]) < cutoff:
            days[game["date"]].append(game)
    for day, daily in sorted(days.items()):
        changes = defaultdict(float)
        for g in sorted(daily, key=lambda g: (g["a"], g["b"])):
            a, b = g["a"], g["b"]
            advantage = {"H": home_advantage, "A": -home_advantage, "N": 0}[g["venue_a"]]
            expected = 1 / (1 + 10 ** ((ratings[b] - ratings[a] - advantage) / 400))
            won = int(g["score_a"] > g["score_b"])
            delta = k * (won - expected)
            changes[a] += delta
            changes[b] -= delta
            for t, opp, win, margin, venue in [
                (a, b, won, g["score_a"] - g["score_b"], g["venue_a"]),
                (b, a, 1 - won, g["score_b"] - g["score_a"], {"H": "A", "A": "H", "N": "N"}[g["venue_a"]])
            ]:
                history[t].append({"day": date.fromisoformat(day), "win": win, "margin": margin,
                                   "venue": venue, "opponent_elo": ratings[opp], "opponent": opp})
        for t, delta in changes.items():
            ratings[t] += delta
    result = []
    for t in sorted(eligible):
        played = history[t]
        recent = [g for g in played if g["day"] >= previous_date]
        old = previous_poll.get(t, {"score": 0.0, "rank": 0})
        preseason = preseason_poll.get(t, {"score": 0.0, "rank": 0}) if preseason_poll is not None else None
        capped = lambda game: max(-MARGIN_CAP, min(MARGIN_CAP, game["margin"]))
        row = {"team_id": t, "team_name": eligible[t], "d1_games": len(played),
               "d1_wins": sum(g["win"] for g in played),
               "d1_losses": sum(1 - g["win"] for g in played),
               "d1_win_pct": sum(g["win"] for g in played) / len(played) if played else None,
               "d1_mean_margin": sum(g["margin"] for g in played) / len(played) if played else None,
               "d1_mean_capped_margin": sum(capped(g) for g in played) / len(played) if played else None,
               "d1_elo": ratings[t],
               "d1_mean_opponent_pregame_elo": sum(g["opponent_elo"] for g in played) / len(played) if played else None,
               "since_poll_d1_games": len(recent), "since_poll_d1_wins": sum(g["win"] for g in recent),
               "since_poll_d1_losses": sum(1 - g["win"] for g in recent),
               "since_poll_d1_margin": sum(g["margin"] for g in recent),
               "since_poll_d1_mean_capped_margin": sum(capped(g) for g in recent) / len(recent) if recent else None,
               "since_poll_d1_mean_opponent_pregame_elo": sum(g["opponent_elo"] for g in recent) / len(recent) if recent else None,
               "since_poll_ranked_wins": sum(g["win"] and 0 < previous_poll.get(g["opponent"], {}).get("rank", 0) <= 25 for g in recent),
               "last_five_d1_wins": sum(g["win"] for g in played[-5:]),
               "rest_days": (cutoff - played[-1]["day"]).days if played else None,
               "previous_normal_points": old["score"],
               "previous_rank": old["rank"] if 0 < old["rank"] <= 25 else None,
               "previous_ranked": 0 < old["rank"] <= 25}
        row.update({
            "preseason_poll_available": preseason is not None,
            "preseason_normal_points": preseason["score"] if preseason is not None else None,
            "preseason_rank": preseason["rank"] if preseason is not None and 0 < preseason["rank"] <= 25 else None,
            "preseason_ranked": 0 < preseason["rank"] <= 25 if preseason is not None else None,
            "days_since_preseason_poll": (cutoff - preseason_date).days if preseason is not None else None,
        })
        for venue, label in [("H", "home"), ("A", "away"), ("N", "neutral")]:
            subset = [g for g in played if g["venue"] == venue]
            row[f"d1_{label}_games"] = len(subset)
            row[f"d1_{label}_wins"] = sum(g["win"] for g in subset)
        result.append(row)
    return result


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()
