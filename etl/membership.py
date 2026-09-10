"""Men's basketball D1 membership periods, independent of data coverage years."""


def active_teams(teams, periods, season):
    if periods is None:
        raise ValueError("Missing team_memberships; re-export after the membership migration")
    names = {t['team_id']: t['team_name'] for t in teams}
    result = {}
    by_team = {}
    for period in periods:
        t, first, last = period['team_id'], period['first_season'], period['last_season']
        if t not in names or (last is not None and last < first):
            raise ValueError("Invalid membership period")
        by_team.setdefault(t, []).append((first, last))
        if first <= season and (last is None or season <= last):
            if t in result:
                raise ValueError(f"Overlapping membership periods for {t}")
            result[t] = names[t]
    for t, intervals in by_team.items():
        intervals.sort()
        for previous, current in zip(intervals, intervals[1:]):
            if previous[1] is None or previous[1] >= current[0]:
                raise ValueError(f"Overlapping membership periods for {t}")
    if not result:
        raise ValueError(f"No D1 members recorded for {season}")
    return result


def season_nonparticipants(tables, season):
    return {r['team_id'] for r in tables.get('team_season_status', [])
            if r['season'] == season and not r['competed']}
