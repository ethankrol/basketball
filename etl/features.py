"""Pure, date-cutoff feature computation. No network or database side effects."""

from collections import defaultdict
from datetime import date
import hashlib
import json
import math

FEATURE_VERSION = "d1-v9"
MARGIN_CAP = 20
NEARBY_POINT_WINDOW = .03


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
                      "venue_a": "N" if left["neutral"] else ("H" if left["home"] else "A"),
                      "overtime": left["overtime"]})
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
                     *, preseason_poll=None, preseason_date=None, initial_ratings=None,
                     previous_previous_poll=None, previous_poll_history=None,
                     margin_of_victory=False):
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
            multiplier = 1.0
            if margin_of_victory:
                capped_margin = min(abs(g["score_a"] - g["score_b"]), MARGIN_CAP)
                multiplier = (math.log1p(capped_margin) * 2.2 /
                              (abs(ratings[a] - ratings[b]) * 0.001 + 2.2))
            delta = k * multiplier * (won - expected)
            changes[a] += delta
            changes[b] -= delta
            for t, opp, win, margin, venue, elo_delta in [
                (a, b, won, g["score_a"] - g["score_b"], g["venue_a"], delta),
                (b, a, 1 - won, g["score_b"] - g["score_a"], {"H": "A", "A": "H", "N": "N"}[g["venue_a"]], -delta)
            ]:
                history[t].append({"day": date.fromisoformat(day), "win": win, "margin": margin,
                                   "venue": venue, "opponent_elo": ratings[opp], "opponent": opp,
                                   "team_elo": ratings[t], "elo_delta": elo_delta,
                                   "overtime": g.get("overtime", False)})
        for t, delta in changes.items():
            ratings[t] += delta
    result = []
    previous_scores = {team_id: row["score"] for team_id, row in previous_poll.items()}
    if previous_poll_history is None:
        previous_poll_history = [previous_poll, previous_previous_poll]
    previous_poll_history = list(previous_poll_history[:4])
    recent_by_team = {team: [game for game in history[team] if game["day"] >= previous_date]
                      for team in eligible}
    previous_ranked = {team for team, row in previous_poll.items() if 0 < row["rank"] <= 25}
    previous_top10 = {team for team, row in previous_poll.items() if 0 < row["rank"] <= 10}
    ranked_teams_with_losses = {team for team in previous_ranked
                                if any(not game["win"] for game in recent_by_team[team])}
    ranked_teams_with_wins = {team for team in previous_ranked
                              if any(game["win"] for game in recent_by_team[team])}
    for t in sorted(eligible):
        played = history[t]
        recent = recent_by_team[t]
        old = previous_poll.get(t, {"score": 0.0, "rank": 0})
        prior_old = ((previous_previous_poll or {}).get(t, {"score": 0.0, "rank": 0}))
        preseason = preseason_poll.get(t, {"score": 0.0, "rank": 0}) if preseason_poll is not None else None
        capped = lambda game: max(-MARGIN_CAP, min(MARGIN_CAP, game["margin"]))
        def window(n): return played[-n:]
        def win_pct(items): return sum(g["win"] for g in items) / len(items) if items else None
        def mean(items, key): return sum(key(g) for g in items) / len(items) if items else None
        def venue_pct(label): return win_pct([g for g in played if g["venue"] == label])
        ranked_recent = [g for g in recent if 0 < previous_poll.get(g["opponent"], {}).get("rank", 0) <= 25]
        last3, last5, last10 = window(3), window(5), window(10)
        def poll_rank(game):
            rank = previous_poll.get(game["opponent"], {}).get("rank", 0)
            return rank if 0 < rank <= 25 else 26
        recent_wins = [g for g in recent if g["win"]]
        recent_losses = [g for g in recent if not g["win"]]
        recent_ranked_wins = [g for g in recent_wins if 0 < previous_poll.get(g["opponent"], {}).get("rank", 0) <= 25]
        recent_ranked_losses = [g for g in recent_losses if 0 < previous_poll.get(g["opponent"], {}).get("rank", 0) <= 25]
        recent_unranked_losses = [g for g in recent_losses if g not in recent_ranked_losses]
        old_score = old["score"]
        higher_scores = [score for team, score in previous_scores.items() if team != t and score > old_score]
        lower_scores = [score for team, score in previous_scores.items() if team != t and score < old_score]
        nearby_teams = [team for team, score in previous_scores.items() if team != t and
                        abs(score - old_score) <= NEARBY_POINT_WINDOW]
        nearby_recent = [game for team in nearby_teams
                         for game in recent_by_team[team]]
        losses_to_higher_point_teams = [game for game in recent_losses
                                        if previous_scores.get(game["opponent"], 0.0) > old_score]
        recent_stronger = [game for game in recent if game["opponent_elo"] > game["team_elo"]]
        recent_weaker = [game for game in recent if game["opponent_elo"] <= game["team_elo"]]
        poll_points = [(poll or {}).get(t, {"score": 0.0})["score"]
                       if poll is not None else None for poll in previous_poll_history]
        point_changes = [poll_points[i] - poll_points[i + 1]
                         for i in range(len(poll_points) - 1)
                         if poll_points[i] is not None and poll_points[i + 1] is not None]
        row = {"team_id": t, "team_name": eligible[t], "d1_games": len(played),
               "d1_wins": sum(g["win"] for g in played),
               "d1_losses": sum(1 - g["win"] for g in played),
               "d1_win_pct": win_pct(played),
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
               "since_poll_ranked_losses": sum(not g["win"] for g in ranked_recent),
               "since_poll_unranked_losses": sum(not g["win"] for g in recent if g not in ranked_recent),
               "since_poll_overtime_games": sum(g["overtime"] for g in recent),
               "since_poll_overtime_wins": sum(g["overtime"] and g["win"] for g in recent),
               "since_poll_overtime_losses": sum(g["overtime"] and not g["win"] for g in recent),
               "since_poll_losses_to_higher_point_teams": len(losses_to_higher_point_teams),
               "since_poll_elo_change": sum(g["elo_delta"] for g in recent),
               "d1_elo_change": sum(g["elo_delta"] for g in played),
               "last_three_d1_wins": sum(g["win"] for g in window(3)),
               "last_three_d1_win_pct": win_pct(window(3)),
               "last_five_d1_wins": sum(g["win"] for g in window(5)),
               "last_five_d1_win_pct": win_pct(window(5)),
               "last_five_d1_mean_capped_margin": mean(window(5), capped),
               "last_five_d1_mean_opponent_pregame_elo": mean(window(5), lambda g: g["opponent_elo"]),
               "last_ten_d1_wins": sum(g["win"] for g in window(10)),
               "last_ten_d1_win_pct": win_pct(window(10)),
               "last_ten_d1_mean_capped_margin": mean(window(10), capped),
               "since_poll_ranked_win_pct": (sum(g["win"] for g in ranked_recent) / len(ranked_recent)
                                             if ranked_recent else None),
               "since_poll_ranked_loss_pct": (sum(not g["win"] for g in ranked_recent) / len(ranked_recent)
                                              if ranked_recent else None),
               "recent_win_pct_change": (win_pct(last3) - win_pct(last10)
                                          if last3 and last10 else None),
               "recent_margin_change": (mean(last5, capped) - mean(played, capped)
                                        if last5 and played else None),
               "recent_sos_change": (mean(last5, lambda g: g["opponent_elo"]) -
                                     mean(played, lambda g: g["opponent_elo"])
                                     if last5 and played else None),
               "since_poll_best_win_opponent_rank": min((poll_rank(g) for g in recent_wins), default=None),
               "since_poll_worst_loss_opponent_rank": max((poll_rank(g) for g in recent_losses), default=None),
               "last_five_best_win_opponent_rank": min((poll_rank(g) for g in window(5) if g["win"]), default=None),
               "last_five_worst_loss_opponent_rank": max((poll_rank(g) for g in window(5) if not g["win"]), default=None),
               "since_poll_best_win_opponent_elo": max((g["opponent_elo"] for g in recent_wins), default=None),
               "since_poll_worst_loss_opponent_elo": min((g["opponent_elo"] for g in recent_losses), default=None),
               "since_poll_mean_opponent_previous_normal_points": mean(
                   recent, lambda g: previous_scores.get(g["opponent"], 0.0)),
               "since_poll_best_win_opponent_previous_normal_points": max(
                   (previous_scores.get(g["opponent"], 0.0) for g in recent_wins), default=None),
               "since_poll_worst_loss_opponent_previous_normal_points": max(
                   (previous_scores.get(g["opponent"], 0.0) for g in recent_losses), default=None),
               "since_poll_stronger_opponent_games": len(recent_stronger),
               "since_poll_stronger_opponent_wins": sum(g["win"] for g in recent_stronger),
               "since_poll_stronger_opponent_mean_capped_margin": mean(recent_stronger, capped),
               "since_poll_weaker_opponent_games": len(recent_weaker),
               "since_poll_weaker_opponent_losses": sum(not g["win"] for g in recent_weaker),
               "since_poll_weaker_opponent_mean_capped_margin": mean(recent_weaker, capped),
               "days_since_latest_win": (cutoff - max((g["day"] for g in recent_wins), default=cutoff)).days
                                        if recent_wins else None,
               "days_since_latest_loss": (cutoff - max((g["day"] for g in recent_losses), default=cutoff)).days
                                         if recent_losses else None,
               "since_poll_quality_weighted_wins": sum(27 - poll_rank(g) for g in recent_ranked_wins),
               "since_poll_quality_weighted_losses": (sum(27 - poll_rank(g) for g in recent_ranked_losses)
                                                        + len(recent_unranked_losses)),
               "rest_days": (cutoff - played[-1]["day"]).days if played else None,
               "previous_normal_points": old["score"],
               "previous_points_gap_to_next_higher": min(higher_scores) - old_score if higher_scores else None,
               "previous_points_gap_to_next_lower": old_score - max(lower_scores) if lower_scores else None,
               "previous_nearby_team_count": len(nearby_teams),
               "nearby_teams_since_poll_wins": sum(game["win"] for game in nearby_recent),
               "nearby_teams_since_poll_losses": sum(not game["win"] for game in nearby_recent),
               "nearby_teams_since_poll_elo_change": sum(game["elo_delta"] for game in nearby_recent),
               "nearby_teams_since_poll_mean_opponent_elo": mean(
                   nearby_recent, lambda game: game["opponent_elo"]),
               "previous_ranked_teams_with_losses": len(ranked_teams_with_losses - {t}),
               "previous_ranked_teams_with_wins": len(ranked_teams_with_wins - {t}),
               "previous_top10_teams_with_losses": len(previous_top10 & ranked_teams_with_losses - {t}),
               "higher_ranked_teams_with_losses": sum(
                   team in ranked_teams_with_losses and previous_poll[team]["rank"] < old["rank"]
                   for team in previous_ranked) if old["rank"] else 0,
               "previous_rank": old["rank"] if 0 < old["rank"] <= 25 else None,
               "previous_ranked": 0 < old["rank"] <= 25}
        row["previous_score_change"] = old["score"] - prior_old["score"] if previous_previous_poll is not None else None
        for lag in range(2, 5):
            row[f"previous_normal_points_lag_{lag}"] = (poll_points[lag - 1]
                                                         if len(poll_points) >= lag else None)
        row["previous_points_trend_3_poll"] = ((poll_points[0] - poll_points[2]) / 2
                                                if len(poll_points) >= 3 and poll_points[2] is not None else None)
        row["previous_points_trend_4_poll"] = ((poll_points[0] - poll_points[3]) / 3
                                                if len(poll_points) >= 4 and poll_points[3] is not None else None)
        row["previous_points_change_volatility_4_poll"] = (
            (sum((value - sum(point_changes) / len(point_changes)) ** 2 for value in point_changes) /
             len(point_changes)) ** .5 if len(point_changes) >= 2 else None)
        row["previous_rank_change"] = (prior_old["rank"] - old["rank"]
                                        if previous_previous_poll is not None and old["rank"] and prior_old["rank"] else None)
        row.update({
            "preseason_poll_available": preseason is not None,
            "preseason_normal_points": preseason["score"] if preseason is not None else None,
            "preseason_rank": preseason["rank"] if preseason is not None and 0 < preseason["rank"] <= 25 else None,
            "preseason_ranked": 0 < preseason["rank"] <= 25 if preseason is not None else None,
            "days_since_preseason_poll": (cutoff - preseason_date).days if preseason is not None else None,
        })
        for venue, label in [("H", "home"), ("A", "away"), ("N", "neutral")]:
            subset = [g for g in played if g["venue"] == venue]
            recent_subset = [g for g in recent if g["venue"] == venue]
            row[f"d1_{label}_games"] = len(subset)
            row[f"d1_{label}_wins"] = sum(g["win"] for g in subset)
            row[f"d1_{label}_win_pct"] = win_pct(subset)
            row[f"since_poll_{label}_games"] = len(recent_subset)
            row[f"since_poll_{label}_wins"] = sum(g["win"] for g in recent_subset)
            row[f"since_poll_{label}_losses"] = sum(not g["win"] for g in recent_subset)
            row[f"since_poll_{label}_mean_capped_margin"] = mean(recent_subset, capped)
        result.append(row)
    return result


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()
