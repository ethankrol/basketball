"""Review a season's entrants/departures and compare against an external D1 roster."""

import argparse
from collections import Counter
import json
from pathlib import Path

from .features import alias_map, normalize_name
from .membership import active_teams


def audit(payload, expected_ids=None, aliases=()):
    season, tables = payload['season'],payload['tables']
    current=active_teams(tables['teams'],tables['team_memberships'],season)
    previous=active_teams(tables['teams'],tables['team_memberships'],season-1)
    names={t['team_id']:t['team_name'] for t in tables['teams']}
    mapping=alias_map(tables['teams'],tables['team_spellings'],aliases)
    unmatched=Counter(normalize_name(r['team']) for r in tables.get('games',[])
                      if normalize_name(r['team']) not in mapping)
    result={'season':season,'active_count':len(current),
            'entrants':[{'team_id':t,'name':names[t]} for t in sorted(set(current)-set(previous))],
            'departures':[{'team_id':t,'name':names[t]} for t in sorted(set(previous)-set(current))],
            'season_status':[r for r in tables.get('team_season_status',[]) if r['season']==season],
            'unmapped_game_names':dict(unmatched.most_common()),
            'external_roster_checked':expected_ids is not None}
    if expected_ids is not None:
        expected=set(expected_ids)
        result['missing_from_memberships']=sorted(expected-set(current))
        result['unexpected_members']=sorted(set(current)-expected)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--expected-ids',type=Path,help='JSON array of independently verified season D1 team IDs')
    parser.add_argument('--config',type=Path,default=Path('etl/configs/history.json'))
    args=parser.parse_args()
    result=audit(json.loads(args.input.read_text()),
                 json.loads(args.expected_ids.read_text()) if args.expected_ids else None,
                 json.loads(args.config.read_text()).get('aliases',[]))
    print(json.dumps(result,indent=2))
    if result.get('missing_from_memberships') or result.get('unexpected_members'):
        raise SystemExit(1)


if __name__=='__main__':
    main()
