-- Add multi-poll trajectories, richer game context, and whole-poll context.
begin;

alter table public.team_feature_snapshots
    add column since_poll_overtime_games integer not null default 0,
    add column since_poll_overtime_wins integer not null default 0,
    add column since_poll_overtime_losses integer not null default 0,
    add column since_poll_stronger_opponent_games integer not null default 0,
    add column since_poll_stronger_opponent_wins integer not null default 0,
    add column since_poll_stronger_opponent_mean_capped_margin double precision,
    add column since_poll_weaker_opponent_games integer not null default 0,
    add column since_poll_weaker_opponent_losses integer not null default 0,
    add column since_poll_weaker_opponent_mean_capped_margin double precision,
    add column days_since_latest_win integer,
    add column days_since_latest_loss integer,
    add column previous_ranked_teams_with_losses integer not null default 0,
    add column previous_ranked_teams_with_wins integer not null default 0,
    add column previous_top10_teams_with_losses integer not null default 0,
    add column higher_ranked_teams_with_losses integer not null default 0,
    add column previous_normal_points_lag_2 double precision,
    add column previous_normal_points_lag_3 double precision,
    add column previous_normal_points_lag_4 double precision,
    add column previous_points_trend_3_poll double precision,
    add column previous_points_trend_4_poll double precision,
    add column previous_points_change_volatility_4_poll double precision,
    add column since_poll_home_games integer not null default 0,
    add column since_poll_home_wins integer not null default 0,
    add column since_poll_home_losses integer not null default 0,
    add column since_poll_home_mean_capped_margin double precision,
    add column since_poll_away_games integer not null default 0,
    add column since_poll_away_wins integer not null default 0,
    add column since_poll_away_losses integer not null default 0,
    add column since_poll_away_mean_capped_margin double precision,
    add column since_poll_neutral_games integer not null default 0,
    add column since_poll_neutral_wins integer not null default 0,
    add column since_poll_neutral_losses integer not null default 0,
    add column since_poll_neutral_mean_capped_margin double precision;

do $$ begin
 execute replace(pg_get_functiondef('public.upload_feature_backfill(jsonb,jsonb,jsonb,text)'::regprocedure),
                 'is distinct from ''d1-v8''', 'is distinct from ''d1-v9''');
end $$;

notify pgrst, 'reload schema';
commit;
