-- Run once in the Supabase SQL Editor as the database owner.
-- Additive: existing games, polls, teams and team_spellings are untouched.
begin;

create table public.feature_backfills (
    backfill_id text primary key check (backfill_id ~ '^[0-9a-f]{64}$'),
    content_hash text not null check (content_hash ~ '^[0-9a-f]{64}$'),
    season integer not null check (season between 2001 and 2099),
    feature_version text not null,
    mode text not null check (mode = 'research_only'),
    expected_rows integer not null check (expected_rows > 0),
    report jsonb not null,
    uploaded_at timestamptz not null default now()
);

create table public.team_feature_snapshots (
    backfill_id text not null references public.feature_backfills(backfill_id),
    run_id text not null,
    season integer not null,
    target_week integer not null check (target_week >= 2),
    cutoff_date_exclusive date not null,
    previous_poll_date date not null,
    feature_version text not null,
    team_id bigint not null,
    team_name text not null,
    d1_games integer not null check (d1_games >= 0),
    d1_wins integer not null check (d1_wins >= 0),
    d1_losses integer not null check (d1_losses >= 0),
    d1_win_pct double precision,
    d1_mean_margin double precision,
    d1_mean_capped_margin double precision check (d1_mean_capped_margin between -20 and 20),
    d1_elo double precision not null,
    d1_mean_opponent_pregame_elo double precision,
    since_poll_d1_games integer not null,
    since_poll_d1_wins integer not null,
    since_poll_d1_losses integer not null,
    since_poll_d1_margin integer not null,
    since_poll_d1_mean_capped_margin double precision check (since_poll_d1_mean_capped_margin between -20 and 20),
    since_poll_d1_mean_opponent_pregame_elo double precision,
    since_poll_ranked_wins integer not null,
    last_five_d1_wins integer not null,
    rest_days integer,
    previous_normal_points double precision not null check (previous_normal_points between 0 and 1),
    previous_rank integer check (previous_rank between 1 and 25),
    previous_ranked boolean not null,
    preseason_poll_available boolean not null,
    preseason_normal_points double precision check (preseason_normal_points between 0 and 1),
    preseason_rank integer check (preseason_rank between 1 and 25),
    preseason_ranked boolean,
    days_since_preseason_poll integer,
    d1_home_games integer not null,
    d1_home_wins integer not null,
    d1_away_games integer not null,
    d1_away_wins integer not null,
    d1_neutral_games integer not null,
    d1_neutral_wins integer not null,
    primary key (backfill_id, run_id, team_id),
    unique (backfill_id, target_week, team_id),
    check (previous_poll_date < cutoff_date_exclusive),
    check (d1_wins + d1_losses = d1_games)
);

create table public.baseline_predictions (
    backfill_id text not null,
    run_id text not null,
    season integer not null,
    target_week integer not null,
    team_id bigint not null,
    model_version text not null check (model_version = 'persistence-v1'),
    predicted_score double precision not null check (predicted_score between 0 and 1),
    predicted_rank integer not null check (predicted_rank > 0),
    actual_score double precision not null check (actual_score between 0 and 1),
    actual_rank integer check (actual_rank between 1 and 25),
    primary key (backfill_id, run_id, team_id),
    foreign key (backfill_id, run_id, team_id)
        references public.team_feature_snapshots(backfill_id, run_id, team_id)
);

create index team_feature_snapshots_team_season on public.team_feature_snapshots(team_id, season, cutoff_date_exclusive);

-- Newest uploaded research revision for each season; then its latest available
-- poll snapshot. This is NOT necessarily today's or the final season record.
create view public.team_season_features_current with (security_invoker = true) as
with latest as (
    select distinct on (season) backfill_id, season
    from public.feature_backfills
    order by season, uploaded_at desc, backfill_id
)
select f.* from public.team_feature_snapshots f
join latest using (backfill_id, season)
where f.cutoff_date_exclusive = (
    select max(s.cutoff_date_exclusive) from public.team_feature_snapshots s
    where s.backfill_id = f.backfill_id
);

-- A single RPC transaction inserts the entire season or rolls back. Repeating
-- an identical backfill is a no-op; changed content under the same ID fails.
create function public.upload_feature_backfill(
    p_report jsonb, p_features jsonb, p_predictions jsonb, p_content_hash text
) returns jsonb language plpgsql security invoker set search_path = '' as $$
declare
    v_id text := p_report->>'source_hash';
    v_season integer := (p_report->>'season')::integer;
    v_expected integer := (p_report->>'feature_rows')::integer;
    v_existing text;
    v_rows integer;
begin
    if p_report->>'mode' is distinct from 'research_only'
       or p_report->>'feature_version' is distinct from 'd1-v3'
       or jsonb_typeof(p_features) is distinct from 'array'
       or jsonb_typeof(p_predictions) is distinct from 'array'
       or v_expected is null or v_expected <= 0 then
        raise exception 'Unsupported or invalid research backfill';
    end if;
    if jsonb_array_length(p_features) <> v_expected
       or jsonb_array_length(p_predictions) <> v_expected then
        raise exception 'Incomplete backfill row counts';
    end if;
    -- Serialize retries of the same immutable dataset.
    perform pg_advisory_xact_lock(hashtextextended(v_id, 0));
    select content_hash into v_existing from public.feature_backfills where backfill_id = v_id;
    if found then
        if v_existing is distinct from p_content_hash then
            raise exception 'Backfill ID already exists with different content';
        end if;
        return jsonb_build_object('status', 'already_uploaded', 'backfill_id', v_id, 'feature_rows', v_expected);
    end if;
    insert into public.feature_backfills(backfill_id, content_hash, season, feature_version, mode, expected_rows, report)
    values (v_id, p_content_hash, v_season, p_report->>'feature_version', 'research_only', v_expected, p_report);

    insert into public.team_feature_snapshots
    select r.* from jsonb_populate_recordset(null::public.team_feature_snapshots,
        (select jsonb_agg(value || jsonb_build_object('backfill_id', v_id)) from jsonb_array_elements(p_features))) r;
    get diagnostics v_rows = row_count;
    if v_rows <> v_expected or exists (
        select 1 from public.team_feature_snapshots
        where backfill_id = v_id and (season <> v_season or feature_version <> p_report->>'feature_version')
    ) then
        raise exception 'Feature metadata mismatch';
    end if;
    if (select count(distinct target_week) from public.team_feature_snapshots where backfill_id = v_id)
         is distinct from (p_report->>'evaluated_polls')::bigint
       or exists (
         select 1 from public.team_feature_snapshots where backfill_id = v_id
         group by target_week
         having count(*) is distinct from (p_report->>'eligible_teams')::bigint
             or count(distinct run_id) <> 1 or count(distinct cutoff_date_exclusive) <> 1
       ) then
        raise exception 'Incomplete poll/team coverage';
    end if;

    insert into public.baseline_predictions
    select r.* from jsonb_populate_recordset(null::public.baseline_predictions,
        (select jsonb_agg(value || jsonb_build_object('backfill_id', v_id)) from jsonb_array_elements(p_predictions))) r;
    if exists (
        select 1 from public.baseline_predictions p
        join public.team_feature_snapshots f using (backfill_id, run_id, team_id)
        where p.backfill_id = v_id and (p.season <> f.season or p.target_week <> f.target_week)
    ) then
        raise exception 'Prediction metadata mismatch';
    end if;
    return jsonb_build_object('status', 'uploaded', 'backfill_id', v_id, 'feature_rows', v_rows);
end;
$$;

alter table public.feature_backfills enable row level security;
alter table public.team_feature_snapshots enable row level security;
alter table public.baseline_predictions enable row level security;
revoke all on public.feature_backfills, public.team_feature_snapshots, public.baseline_predictions,
    public.team_season_features_current from anon, authenticated;
grant select, insert on public.feature_backfills, public.team_feature_snapshots, public.baseline_predictions to service_role;
grant select on public.team_season_features_current to service_role;
revoke all on function public.upload_feature_backfill(jsonb, jsonb, jsonb, text) from public, anon, authenticated;
grant execute on function public.upload_feature_backfill(jsonb, jsonb, jsonb, text) to service_role;

notify pgrst, 'reload schema';
commit;
