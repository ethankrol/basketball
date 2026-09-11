-- Add poll-to-poll movement and recent-trend features.
begin;

alter table public.team_feature_snapshots
    add column since_poll_ranked_win_pct double precision,
    add column since_poll_ranked_loss_pct double precision,
    add column recent_win_pct_change double precision,
    add column recent_margin_change double precision,
    add column recent_sos_change double precision,
    add column previous_score_change double precision,
    add column previous_rank_change integer;

do $$ begin
 execute replace(pg_get_functiondef('public.upload_feature_backfill(jsonb,jsonb,jsonb,text)'::regprocedure),
                 'is distinct from ''d1-v5''', 'is distinct from ''d1-v6''');
end $$;

notify pgrst, 'reload schema';
commit;
