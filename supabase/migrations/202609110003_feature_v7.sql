-- Add opponent-quality extrema and quality-weighted recent results.
begin;

alter table public.team_feature_snapshots
    add column since_poll_best_win_opponent_rank integer,
    add column since_poll_worst_loss_opponent_rank integer,
    add column last_five_best_win_opponent_rank integer,
    add column last_five_worst_loss_opponent_rank integer,
    add column since_poll_best_win_opponent_elo double precision,
    add column since_poll_worst_loss_opponent_elo double precision,
    add column since_poll_quality_weighted_wins integer not null default 0,
    add column since_poll_quality_weighted_losses integer not null default 0;

do $$ begin
 execute replace(pg_get_functiondef('public.upload_feature_backfill(jsonb,jsonb,jsonb,text)'::regprocedure),
                 'is distinct from ''d1-v6''', 'is distinct from ''d1-v7''');
end $$;

notify pgrst, 'reload schema';
commit;
