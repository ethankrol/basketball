"""Replay a continuous season range; optionally export and publish through Postgres."""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from .backfill_season import build
from .build_elo_state import build_state
from .features import content_hash, eligible_teams
from .upload_backfill import prepare_upload


def export_history(connection, root, start, end):
    from psycopg.rows import dict_row
    from psycopg import sql
    # All seasons and membership data come from one consistent snapshot.
    with connection.transaction():
        connection.execute('set transaction isolation level repeatable read read only')
        with connection.cursor(row_factory=dict_row) as cursor:
            metadata = {}
            for table, order in [('teams','team_id'), ('team_spellings','team_spelling'),
                                 ('team_memberships','team_id,first_season'), ('team_season_status','team_id,season')]:
                cursor.execute(sql.SQL('select * from public.{} order by ').format(sql.Identifier(table)) + sql.SQL(order))
                metadata[table] = cursor.fetchall()
            for season in range(start, end + 1):
                tables = dict(metadata)
                for table, order in [('games','date,team,opponent'), ('polls','week,team')]:
                    cursor.execute(sql.SQL('select * from public.{} where season=%s order by ').format(sql.Identifier(table)) + sql.SQL(order), (season % 100,))
                    tables[table] = cursor.fetchall()
                destination = root / str(season) / 'source.json'
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps({'season':season,'tables':tables},default=str))
                print(f"Exported {season}: {len(tables['games'])} game rows, {len(tables['polls'])} poll rows",flush=True)


def run_history(root, start, end, config, carryover=.75):
    prior = None
    previous_members = None
    manifest = {'initial_season':start,'last_season':end,'carryover':carryover,'seasons':[]}
    for season in range(start, end + 1):
        payload = json.loads((root / str(season) / 'source.json').read_text())
        if payload['season'] != season:
            raise ValueError('Source season does not match the chronological sequence')
        overrides = {'season':season, 'aliases':config.get('aliases',[]), **config.get('seasons',{}).get(str(season),{})}
        state = build_state(payload, prior, carryover, bootstrap=prior is None, overrides=overrides)
        state_id = content_hash(state)
        output = root / str(season) / 'runs' / state_id
        output.mkdir(parents=True, exist_ok=True)
        (output / 'elo_state.json').write_text(json.dumps(state,indent=2)+'\n')
        members = eligible_teams(payload['tables']['teams'],season,payload['tables']['team_memberships'])
        entry = {'season':season,'elo_state':str(output / 'elo_state.json'), 'elo_state_hash':state_id,
                 'teams':len(members),'games':state['audit']['physical_d1_games'],
                 'entrants':sorted(set(members)-previous_members) if previous_members is not None else [],
                 'departures':sorted(previous_members-set(members)) if previous_members is not None else [],
                 'backfill_dir':None}
        if payload['tables']['polls']:
            # Data errors propagate, except the explicitly expected absence of
            # usable adjacent polls. Never reset Elo to get past an error.
            try:
                rows,predictions,report = build(payload,overrides,prior,carryover,bootstrap=prior is None)
            except ValueError as exc:
                if not str(exc).startswith('No valid adjacent polls to evaluate:'):
                    raise
                entry['poll_status'] = str(exc)
            else:
                destination = output / report['source_hash']
                destination.mkdir(exist_ok=True)
                for filename,value in [('features.json',rows),('predictions.json',predictions),('report.json',report)]:
                    (destination / filename).write_text(json.dumps(value,allow_nan=False)+'\n')
                entry.update(backfill_dir=str(destination),feature_rows=len(rows),
                             invalid_polls=report['invalid_polls'],evaluated_polls=report['evaluated_polls'])
        else:
            entry['poll_status']='No source polls; Elo warmup only'
        manifest['seasons'].append(entry)
        prior = state
        previous_members = set(members)
        print(f"Built {season}: {len(members)} teams, {entry['games']} D1 games, {entry.get('feature_rows',0)} feature rows",flush=True)
    manifest_path = root / f'manifest_{start}_{end}.json'
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def publish_history(connection, manifest):
    from psycopg.types.json import Jsonb
    for entry in manifest['seasons']:
        state = json.loads(Path(entry['elo_state']).read_text())
        if content_hash(state) != entry['elo_state_hash']:
            raise ValueError('Elo state changed after the history manifest was built')
        with connection.transaction():
            if entry['backfill_dir']:
                directory = Path(entry['backfill_dir'])
                p = prepare_upload(*[json.loads((directory/f).read_text()) for f in ['report.json','features.json','predictions.json']])
                connection.execute('select public.upload_feature_backfill(%s,%s,%s,%s)',
                                   (Jsonb(p['p_report']),Jsonb(p['p_features']),Jsonb(p['p_predictions']),p['p_content_hash']))
            with connection.cursor() as cursor:
                cursor.executemany('''insert into public.team_elo_seasons
                    (state_hash,season,team_id,ending_elo,initial_season,prior_state_hash,elo_version,carryover,source_hash)
                    values (%s,%s,%s,%s,%s,%s,%s,%s,%s) on conflict (state_hash,team_id) do nothing''',
                    [(entry['elo_state_hash'],state['season'],int(t),rating,state['initial_season'],state['prior_state_hash'],
                      state['elo_version'],state['carryover'],state['source_hash']) for t,rating in state['ratings'].items()])
        print(f"Published {entry['season']}",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start',type=int,default=2001)
    parser.add_argument('--end',type=int,default=2026)
    parser.add_argument('--root',type=Path,default=Path('artifacts/history'))
    parser.add_argument('--config',type=Path,default=Path('etl/configs/history.json'))
    parser.add_argument('--carryover',type=float,default=.75)
    parser.add_argument('--export',action='store_true')
    parser.add_argument('--apply',action='store_true')
    args = parser.parse_args()
    if not 2001 <= args.start <= args.end <= 2099:
        parser.error('Expected a continuous season range starting no earlier than 2001')
    load_dotenv(Path(__file__).resolve().parents[1] / '.env')
    connection = None
    try:
        if args.export or args.apply:
            import psycopg
            connection=psycopg.connect(os.environ['SUPABASE_DB_URL'],sslmode='require',connect_timeout=15,autocommit=True)
        if args.export:
            export_history(connection,args.root,args.start,args.end)
        manifest=run_history(args.root,args.start,args.end,json.loads(args.config.read_text()),args.carryover)
        if args.apply:
            publish_history(connection,manifest)
    finally:
        if connection:
            connection.close()


if __name__ == '__main__':
    main()
