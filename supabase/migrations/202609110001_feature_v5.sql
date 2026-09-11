-- Add rolling, opponent-quality, venue, and Elo-change features.
begin;

alter table public.team_feature_snapshots
    add column since_poll_ranked_losses integer not null default 0,
    add column since_poll_unranked_losses integer not null default 0,
    add column since_poll_elo_change double precision not null default 0,
    add column d1_elo_change double precision not null default 0,
    add column last_three_d1_wins integer not null default 0,
    add column last_three_d1_win_pct double precision,
    add column last_five_d1_win_pct double precision,
    add column last_five_d1_mean_capped_margin double precision,
    add column last_five_d1_mean_opponent_pregame_elo double precision,
    add column last_ten_d1_wins integer not null default 0,
    add column last_ten_d1_win_pct double precision,
    add column last_ten_d1_mean_capped_margin double precision,
    add column d1_home_win_pct double precision,
    add column d1_away_win_pct double precision,
    add column d1_neutral_win_pct double precision;

do $$ begin
 execute replace(pg_get_functiondef('public.upload_feature_backfill(jsonb,jsonb,jsonb,text)'::regprocedure),
                 'is distinct from ''d1-v4''', 'is distinct from ''d1-v5''');
end $$;

notify pgrst, 'reload schema';
commit;
