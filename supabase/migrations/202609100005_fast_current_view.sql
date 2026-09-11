begin;

-- The old view used a correlated MAX() for every snapshot row. This index and
-- grouped cutoff CTE let Postgres find one latest cutoff per uploaded revision.
create index if not exists team_feature_snapshots_backfill_cutoff_idx
    on public.team_feature_snapshots (backfill_id, cutoff_date_exclusive desc);

drop view if exists public.team_season_features_current;
create view public.team_season_features_current with (security_invoker = true) as
with latest_backfills as (
    select distinct on (season) backfill_id, season
    from public.feature_backfills
    order by season, uploaded_at desc, backfill_id
), latest_cutoffs as (
    select backfill_id, max(cutoff_date_exclusive) as cutoff_date_exclusive
    from public.team_feature_snapshots
    group by backfill_id
)
select f.*
from public.team_feature_snapshots f
join latest_backfills b on b.backfill_id = f.backfill_id and b.season = f.season
join latest_cutoffs c on c.backfill_id = f.backfill_id
                         and c.cutoff_date_exclusive = f.cutoff_date_exclusive;

notify pgrst, 'reload schema';
commit;
