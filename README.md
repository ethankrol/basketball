# AP basketball poll predictions

Python ETL and Supabase data for a men's college basketball AP poll website.
See [the completion plan](PROJECT_PLAN.md) for the broader roadmap.

## First working slice: historical features and baseline

The new pipeline exports one season from the existing Supabase tables, validates
canonical D1 games and poll totals, builds dated feature snapshots, and evaluates
a persistence forecast (repeat the previous poll's normalized points).

It is **research-only**: it does not write to Supabase or publish predictions.
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
2. Export the deployed schema/constraints and add versioned feature-run tables,
   typed snapshot columns, prediction tables and atomic publication. The current
   repository contains no applied migration for these tables.
3. Backfill additional verified seasons and add chronological model evaluation.
4. Refresh team-season eligibility for 2026/2027 before running those seasons.
5. Wire validated feature/inference runs into scheduled jobs and build the UI.

Your earlier `etl/weekly_stats_agg.py` experiment is preserved. The working,
side-effect-free feature implementation is now `etl/features.py`.
