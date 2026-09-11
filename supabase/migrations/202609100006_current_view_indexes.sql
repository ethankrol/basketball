begin;
create index if not exists feature_backfills_season_uploaded_idx
    on public.feature_backfills (season, uploaded_at desc, backfill_id);
create index if not exists team_feature_snapshots_backfill_cutoff_team_idx
    on public.team_feature_snapshots (backfill_id, cutoff_date_exclusive, team_id);
notify pgrst, 'reload schema';
commit;
