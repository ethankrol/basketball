-- Add prior-point spacing, nearby-team results, and poll-strength features.
begin;

alter table public.team_feature_snapshots
    add column since_poll_losses_to_higher_point_teams integer not null default 0,
    add column since_poll_mean_opponent_previous_normal_points double precision,
    add column since_poll_best_win_opponent_previous_normal_points double precision,
    add column since_poll_worst_loss_opponent_previous_normal_points double precision,
    add column previous_points_gap_to_next_higher double precision,
    add column previous_points_gap_to_next_lower double precision,
    add column previous_nearby_team_count integer not null default 0,
    add column nearby_teams_since_poll_wins integer not null default 0,
    add column nearby_teams_since_poll_losses integer not null default 0,
    add column nearby_teams_since_poll_elo_change double precision not null default 0,
    add column nearby_teams_since_poll_mean_opponent_elo double precision;

do $$ begin
 execute replace(pg_get_functiondef('public.upload_feature_backfill(jsonb,jsonb,jsonb,text)'::regprocedure),
                 'is distinct from ''d1-v7''', 'is distinct from ''d1-v8''');
end $$;

notify pgrst, 'reload schema';
commit;
