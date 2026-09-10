# AP basketball poll predictions

Python ETL and Supabase data for a men's college basketball AP poll website.
See [the completion plan](PROJECT_PLAN.md) for the broader roadmap.

## First working slice: historical features and baseline

The new pipeline exports one season from the existing Supabase tables, validates
canonical D1 games and poll totals, builds dated feature snapshots, and evaluates
a persistence forecast (repeat the previous poll's normalized points).

It is **research-only**: backfilling does not write to Supabase or publish predictions.
The separate upload command can store these artifacts in private research tables.
The existing daily ingestion scripts are separate and have not been replaced.

```sh
python -m venv .venv
.venv/bin/pip install -r requirements-etl.txt
```

Set `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` in your environment or root `.env`
file. These are only required for the export; backfills and tests run offline.

```sh
.venv/bin/python -m etl.export_season --season 2025 --output artifacts/2025/source.json
.venv/bin/python -m etl.export_season --season 2024 --output artifacts/2024/source.json
.venv/bin/python -m etl.build_elo_state --input artifacts/2024/source.json --overrides etl/configs/research_2024.json --bootstrap-elo --output artifacts/2024/elo_state.json
.venv/bin/python -m etl.backfill_season --input artifacts/2025/source.json --overrides etl/configs/research_2025.json --prior-elo artifacts/2024/elo_state.json --output artifacts/2025/backfill
.venv/bin/python -m unittest discover -s tests -v
```

`--season` uses the four-digit ending year. The export adapter translates it to
the legacy database's two-digit season. Exports page through all rows with stable
ordering and count checks. Run exports when ingestion is idle: REST pagination
does not provide a transactionally consistent snapshot across tables or detect
every in-place source edit.

Generated files live under ignored `artifacts/`. Each backfill uses a directory
derived from the input content, configuration and feature version:

- `features.json`: one team row per target poll, strictly excluding game dates on
  or after that poll's cutoff date. Contains prior poll inputs but no target labels.
- `predictions.json`: persistence predictions plus actual labels for evaluation.
- `report.json`: coverage, unresolved names, invalid/skipped polls and metrics.

Repeated identical inputs produce identical outputs. A changed source/configuration
creates a separate directory. Bump `FEATURE_VERSION` when feature semantics change.
These are reconstructed historical runs, not evidence that a forecast was issued
before its target poll.

## Upload the computed data to Supabase

1. Open your Supabase project -> **SQL Editor** -> **New query**. Paste and run
   [the feature backfill migration](supabase/migrations/202609100001_feature_backfills.sql)
   once. It creates new tables, a view and an upload function in one transaction.
   The API service key cannot install arbitrary SQL by itself. This migration
   was applied to the configured project on September 10, 2026. Do not rerun
   this initial migration there; it is needed only when setting up another database.
2. Preview an exact backfill directory (the final directory printed by the
   backfill command), then add `--apply` to upload:

```sh
.venv/bin/python -m etl.upload_backfill --artifact-dir artifacts/2025/backfill/8fefd77820d0b6e78ccf1b50c3a397007cb64b35ce90b66047992dc2dbec095e
.venv/bin/python -m etl.upload_backfill --artifact-dir artifacts/2025/backfill/8fefd77820d0b6e78ccf1b50c3a397007cb64b35ce90b66047992dc2dbec095e --apply
```

That directory contains the initial 2025 `d1-v3` build with 2024 Elo carryover.
Use your newly printed directory instead if rebuilding changes the source hash.
This upload contains **6,188 feature rows and 6,188 persistence prediction rows**.
No re-export or new scrape is needed to upload an existing artifact.

For a PostgreSQL connection (including the session pooler), set `SUPABASE_DB_URL`
in `.env`, install `requirements-db.txt`, and add `--direct-db --apply` to the
upload command. This calls the same atomic function without the REST API key.
The initial 2025 upload used this route successfully on September 10, 2026;
the REST route returned HTTP 401 and its API credentials still need checking
before relying on it for automated uploads.

New relations:

| Relation | Contents |
|---|---|
| `feature_backfills` | One immutable dataset revision with report, Elo provenance and research status |
| `team_feature_snapshots` | Typed feature columns for every team and usable poll cutoff |
| `baseline_predictions` | Reconstructed persistence forecasts and evaluation labels |
| `team_season_features_current` | Latest available cutoff in the most recently uploaded revision of each season |

The current view will contain 364 rows for this 2025 backfill and uses March 17,
2025 snapshots. It is **not** an end-of-season or live September 2026 summary.
Uploading an older research revision later makes that revision current; upload
the intended revision last. Re-uploading an identical existing revision is a
no-op and does not change its timestamp.

The uploader checks columns, IDs, team coverage and paired predictions before
making a request. The database function then inserts the entire dataset in one
transaction. Duplicate retries are harmless; the same dataset ID with different
content is rejected. A failed transaction rolls back. If a network timeout loses
the acknowledgment, rerun the same command. Files must be from one complete build.

All new tables remain private to server/service-role access with row-level
security enabled; the browser gets no new read or write permissions. Existing
`games`, `polls`, `teams` and `team_spellings` are untouched. Skipped polls stay
skipped, and the raw December 16 poll still needs source repair. This command
does not upload the separate season-end Elo state file; that file remains an
input to future backfills, whose provenance is recorded in the report.

After upload, verify in the SQL Editor:

```sql
select season, feature_version, expected_rows, mode, uploaded_at
from public.feature_backfills order by uploaded_at desc;

select season, target_week, count(*)
from public.team_season_features_current
group by season, target_week;

select team_name, d1_elo, d1_mean_capped_margin,
       d1_mean_opponent_pregame_elo, preseason_rank
from public.team_season_features_current
where season = 2025 order by d1_elo desc limit 25;
```

This is a manual research upload. Daily Actions still run the original ingestion
jobs; scheduled feature publication is not wired in yet. Python tests cover the
upload contract and error handling. The migration and PostgreSQL upload were
verified on September 10, 2026: 6,188 feature rows, 6,188 prediction rows and
364 current-view rows. A repeated upload returned `already_uploaded`. RLS was
enabled on all three tables and anon/authenticated roles could not execute the RPC.

## Feature definitions

All game features are **D1 vs D1 only**, based on season eligibility metadata.
Unmapped/noneligible opponents are excluded and listed in the audit, not assumed
to be non-D1. This means the displayed feature records are not official overall
team records. Every eligible team must have at least one mapped game in the export.

Feature version `d1-v2` adds per-game signed margins capped at **-20 to +20**,
averaged over the season and since the previous poll. Capping happens before
averaging, so a 50-point win and a 10-point loss average +5, not +20. Raw margins
remain available for model comparisons; the cap does not change Elo updates.

Strength of schedule is `d1_mean_opponent_pregame_elo`: the game-weighted average
of opponents' ratings before each matchup. Repeat opponents count once per game;
higher means a tougher schedule. The same measure is now computed for games since
the previous poll. These are Elo-based SOS proxies, not an official SOS rating.
With equal 1500 season starts, early-season SOS has limited information.

Preseason rank, ranked status and normalized points persist from the validated
preseason poll throughout the season. `days_since_preseason_poll` lets a future
model learn how that anchor changes with time rather than imposing a fixed decay.
Unranked teams have null rank but retain any preseason points. Teams absent from
a valid poll have zero points; an unavailable/invalid preseason poll produces null
anchor fields and `preseason_poll_available=false`. Future polls are rejected.
The persistence baseline still uses only the previous poll; a trained model and
chronological validation are needed to establish the value of these extra features.

Feature version `d1-v3` starts returning teams at
`1500 + carryover * (previous_season_ending_elo - 1500)`, with a configurable
`--elo-carryover` default of 0.75. Teams absent from prior state start at 1500.
The backfill requires either `--prior-elo` or explicit `--bootstrap-elo` for the
first historical season. The example bootstraps 2024, then carries into 2025;
for a longer rating history, build season states chronologically, passing each
state into the following year's `build_elo_state` command. Prior-state hashes and
carryover settings are included in run identity. A prior state must match the
immediately preceding season and Elo algorithm version. April coverage is checked,
but this is not proof of full source completeness. Bootstrap initialization can
still affect subsequent seasons; a longer warmup is preferable for model training.

Elo uses K=20 and a 65-point home adjustment, and
updates once per physical game. Both teams use pregame ratings. Games on the same
date use start-of-day ratings because the source has no reliable tipoff timestamps.
These parameters are initial defaults, not tuned estimates.

Features include D1 games/wins/losses, win percentage, mean margin, Elo, mean
opponent pregame Elo, venue splits, recent five-game wins, rest days, results since
the prior poll, wins over teams ranked in that prior poll, and prior normalized
points/rank. Empty history produces null rates and rest days, not fabricated zeros.

Polls must have unique mapped eligible teams, consistent identifiers/dates, valid
points, and total points equal to 325 times the first-place-vote sum. A failed poll
is excluded both as a target and as the previous-poll input to the following week.
Absent teams receive zero labels only after these checks pass. Checks do not
replace comparison with the original source. Do not silently repair a failed total.

Normalized points are recomputed from raw points and validated voter counts. They
are not ranking probabilities. Tied prediction scores use team ID for deterministic
display order. Actual ranks use competition ranking; rank >25 is not an official
AP rank. Actual ties at the boundary can make the actual top-25 set larger than 25;
the report includes its size.

## Initial 2025 audit (September 9, 2026)

- 380 team records and 1,193 aliases exported; 364 season-eligible teams.
- 12,584 team-game rows reduced to 5,764 validated physical D1 games.
- 876 poll rows across 20 releases; the final postseason release is absent.
- December 16, 2024 (week 7) contains 20,132 points versus 20,150 expected.
  Weeks 7 and 8 are excluded from prediction evaluation.
- A season-scoped `Miami` -> Miami FL alias and sourced October 14 preseason date
  are kept in the research configuration. Other legacy weekly dates still need
  source verification.
- 17 evaluated polls, 6,188 feature rows. Persistence averages **22.47 teams in
  common with the actual top 25**. This is a pipeline baseline on provisionally
  dated historical data, not held-out model performance.

## Next implementation steps

1. Verify the release calendar and repair the missing vote recipient from a cited
   source; audit excluded game names and historical coverage.
2. Apply the feature-backfill migration above and verify a real upload. Audit the
   existing source-table constraints before changing the ingestion jobs.
3. Backfill additional verified seasons and add chronological model evaluation.
4. Refresh team-season eligibility for 2026/2027 before running those seasons.
5. Wire validated feature/inference runs into scheduled jobs and build the UI.

Your earlier `etl/weekly_stats_agg.py` experiment is preserved. The working,
side-effect-free feature implementation is now `etl/features.py`.
