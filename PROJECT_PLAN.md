# AP basketball website completion plan

Reviewed September 9, 2026. Scope: men's Division I basketball, matching the existing ESPN scraper. This is a source-code and local-data review, not an audit of the deployed Supabase database or GitHub run history. Existing uncommitted work was preserved.

## Recommended product contract

Predict the next AP poll's normalized points and sort teams into a projected ranking. Launch with a forecast frozen on Monday morning after Sunday's results have been reconciled, before the actual poll becomes available. Show official results separately and score the frozen prediction when they arrive.

A midweek estimate using completed games is a different prediction horizon. Initially label it “projection based on completed games.” Forecasting the full next poll midweek requires remaining schedules and either horizon-specific training or game simulations. Do not evaluate those forecasts as though they had Sunday's complete information.

Treat preseason prediction as a separate later project: roster turnover, transfers, recruiting and expectations require additional data. The official preseason poll can be an input to subsequent regular-season predictions once published. Handle postseason polls explicitly in the calendar rather than assuming every week is identical.

## Repository assessment

Useful foundation:

- Python/pandas ingestion and Supabase storage fit this workload.
- Local game files cover seasons ending 2001–2026; the 2026 local copy ends November 21, 2025 and needs refreshing. File presence does not establish historical completeness.
- Game parsing produces both team perspectives, which helps aggregations.
- Poll ingestion includes others receiving votes and normalized points, valuable training labels.
- Team alias and historical conference work has started.
- GitHub Actions already provides scheduled execution with credentials in secrets.

Priority gaps and findings:

| Area | Evidence | Action |
|---|---|---|
| Feature computation | `etl/weekly_stats_agg.py` is a stub; `get_data('team_id, conference_2001')` omits its required table argument | Build pure feature functions, with explicit database reads at the job boundary |
| Team identity | Game/poll ingestion writes names; neither applies the alias map; others receiving votes are not lowercased consistently | Resolve source aliases to stable IDs before writing facts; quarantine unknown names |
| Poll dates | Preseason uses October 1; other dates are calculated; skipped weeks compress dates and IDs | Store actual release dates and stable source identifiers in a poll calendar |
| Upcoming season | Calendar ends at 2026; poll workflow hardcodes 2026 and excludes October | Add explicit season configuration and preseason/postseason releases; use four-digit ending years throughout |
| Scheduling | Games run once daily; polls run once Monday at 19:00 UTC | Add reconciliation before forecast freeze and repeated bounded poll checks |
| Parse reliability | First HTML table/paragraph assumptions, no poll HTTP status check, no request timeouts | Match AP section explicitly; validate page identity and poll date; archive representative fixtures |
| Normalization | Voter count inferred from all parsed points and result clipped | Validate complete poll and denominator first; do not hide parse errors by clipping |
| Venue | Neutral flag examines only first token; home is true on one side even for neutral games | Parse full marker field, test overtime plus neutral examples, expose H/A/N venue |
| DB reads | `get_data` and alias audit lack pagination | Paginate with stable ordering or perform server-side queries |
| DB writes | Upsert conflict targets and schema migrations absent from repo | Inspect deployed constraints, export migrations, specify natural-key conflicts and batch writes |
| Failure status | Poll-not-found can print failure while job exits successfully | Separate expected not-yet-released from overdue/malformed data; fail and alert appropriately |
| Application | `sttest.py` is a generic demo; no product routes, model training or model artifacts found | Build the website and reproducible modeling layer after trustworthy data |

All ETL Python files compiled in a read-only check; regex escape warnings were observed. Sample fixed-width parsing matched the first local game rows. No ETL jobs were executed against production, and live constraints, policies, table completeness and workflow health remain unverified.

## Database and feature architecture

One mutable row per team-season is useful for a current summary, but cannot serve as training history. Use historical snapshots plus a current-summary view.

| Relation | Grain / key | Purpose |
|---|---|---|
| `teams` | team ID | Stable identity |
| `team_aliases` | source + alias, with validity if needed | Explicit ESPN/KenPom name mapping |
| `team_seasons` | team ID + season | Division eligibility, season conference, season metadata |
| `games` | stable canonical game ID | One physical game, scores, participants, date/time, venue, status, source |
| `team_games` view | game ID + team ID | Two perspectives for aggregation |
| `poll_releases` | poll ID | Season, source week, release timestamp/date, forecast cutoff, status, voters |
| `poll_results` | poll ID + team ID | Official rank, points, first-place votes, normalized points |
| `feature_runs` | run ID | Cutoff, target poll, feature version, source revision/hash, status |
| `team_feature_snapshots` | run ID + team ID | Typed numeric features for each eligible team |
| `model_versions` | model version | Artifact location, training boundary, feature version, metrics, code revision |
| `prediction_runs` / `predictions` | run ID / run ID + team ID | Target poll, issue time, model and feature run, predicted score/rank, frozen status |
| `etl_runs` | run ID | Inputs, counts, duration, failures and freshness |

Expose `team_season_features_current` as a view over the latest successful feature run for each season. For reproducibility, identify an equivalent feature run by season, cutoff, target poll, feature version and source revision. Retrying it should not duplicate data. Corrections create a new revision and never silently rewrite the inputs of an already-published forecast.

Keep features in Python/pandas initially; use Postgres for constraints, storage, joins and serving. Elo requires sequential state, so one Python feature implementation shared by training/backfills and production reduces inconsistency. Avoid a separate feature-store service at this scale.

Population procedure:

1. Fetch and archive source responses with fetch time and content hash. Validate schemas, dates and counts before modifying published data.
2. Canonicalize teams, normalize types, resolve game identity, and upsert facts using verified unique keys. A source without game IDs needs a documented deterministic key and correction/reconciliation policy.
3. Read the active season completely, plus prior-season rating state and previously released polls. Do not assume one Supabase select returns the entire table.
4. Sort physical games chronologically. Compute both teams' pregame ratings, then update both once per physical game. Never update Elo twice from mirrored rows. With date-only sources, use start-of-day states for same-day games to avoid arbitrary ordering effects.
5. Recompute current-season features from the beginning initially. This is easier to audit and handles score corrections; benchmark before implementing incremental state. If a prior-season correction changes carried ratings, rebuild downstream seasons for the corrected research revision.
6. Write a complete run in staging, validate coverage and invariants, then atomically mark it published through a database transaction/function. The website reads only successful runs.
7. For a historical backfill, replay games once chronologically and emit snapshots at each poll cutoff. Use the same functions as live inference. Keep reconstructed historical data distinct from truly archived as-of source data: historical corrections may not have been known then.

## Initial feature set

Start with roughly 20–35 explainable features, all computed strictly at the forecast cutoff:

- Poll history: previous normalized points, previous official rank plus unranked indicator, previous point change, consecutive ranked weeks, preseason points once known.
- Season results: games, wins, losses, win percentage, average and capped scoring margin, home/away/neutral splits.
- Since last poll: games, wins/losses, margins, road wins, strongest opponent beaten, weakest opponent lost to, wins against previously ranked teams.
- Strength: current Elo, change since last poll, mean opponent pregame Elo, quality wins and bad losses using fixed or training-tuned thresholds.
- Context: week of season, rest days, recent five-game record, prior-poll point gaps to nearby teams including the top-25 boundary.

Define windows and missing values explicitly: no previous poll is different from zero previous votes. Opponent strength must be measured as of the relevant historical time, never from final-season ratings. Do not put the target poll's rank, votes or first-place votes in features. Define Division I vs non-Division I result handling once and apply consistently.

Start Elo with a common rating and season-to-season regression toward the mean. Compare the existing conference-based initialization as a historical-validation experiment; the 2001 conference file is not a current conference table. Tune home advantage, K and offseason carryover using training/validation years only. Possession-based efficiency cannot be recovered reliably from final scores alone and requires more data.

## Target, models and evaluation

Target for team i in poll p: `points_i / (25 * number_of_voters_p)`. This represents fraction of maximum possible points, not probability of being ranked. Rank predictions within each target poll.

The existing denominator estimate `sum(points) / 325` assumes a complete set of all vote recipients and complete 25-team ballots. Prefer a reported voter count; otherwise validate inferred count against first-place totals, integer tolerance and source completeness. Preserve official ranks/ties; do not present computed positions 26+ as official AP ranks.

Build one training row for every eligible team at every valid target release. Teams absent from a verified complete poll get zero points; missing or failed polls must never become all-zero labels. Many zero labels make overall RMSE alone misleading.

Model sequence:

1. Persistence baseline: next points equal previous points; retain every eligible team in the candidate universe.
2. Regularized linear baseline using prior points and recent results.
3. First nonlinear candidate: CatBoost regression with RMSE, comparing direct score prediction with prediction of the change from prior score. Clip model outputs to [0,1] for serving and evaluate that exact transformation.
4. If needed, compare a two-stage model, P(any points) times expected normalized points given positive points. Calibrate and evaluate out of time.
5. Compare a ranking objective grouped by poll if ordering remains weak. Ranking scores do not automatically provide meaningful normalized points, so preserve a regression head/model if points are displayed.

CatBoost is a proposed candidate, not a demonstrated winner. No need for a neural network before these experiments. Avoid raw team-name/ID features initially; test identity-related features separately for memorization and era drift. Compare recent training windows against all available history.

Use expanding chronological validation by entire season, with the last two complete, validated seasons held untouched for final evaluation. Keep every team in a poll in the same split. Model training and hyperparameter selection must precede the evaluation period; prior official polls within that period are legitimate inference inputs once released. Report preseason/early season, normal weeks, postseason and unusual seasons separately.

Report per-poll top-25 overlap (out of 25), NDCG@25, rank error over the union of actual/predicted top 25 with an explicit unranked convention, normalized-point MAE/RMSE across all teams and across vote recipients, and top-1 accuracy. Compare paired poll errors to persistence. Do not promise a numerical accuracy level until backtests exist. Define a launch gate of repeatable held-out improvement in top-25 overlap/order without material score degradation. Use season or block-based uncertainty estimates rather than treating team rows as independent.

## Release operations

Keep Actions for the MVP's batch jobs, with manual dispatch, concurrency control, timeouts, bounded HTTP retries, structured counts and alerts. Separate ingestion, feature/inference, actual-poll ingestion and evaluation stages.

Normal daily run: reconcile games -> validate -> build features -> predict -> publish. Rebuild predictions daily; retraining can be offseason or a deliberate versioned cadence, not every ETL run.

Release day: reconcile Sunday games and freeze the official forecast at a documented Monday-morning Eastern cutoff. A game-date-only dataset should explicitly use games through Sunday; do not invent completion timestamps. If late Sunday results are still missing, preserve the last valid output with an incomplete/stale status rather than calling the forecast final. Check expected game coverage using a schedule feed or an independent completeness signal; a successful fetch does not prove completeness.

Fetch the target poll repeatedly in a bounded release window and later retry window, validate its identity/date/completeness, insert actuals, and evaluate the frozen prediction. Do not replace the frozen forecast with predictions that used the newly published target. Configure actual seasonal release dates; account for October preseason, holidays, gaps and postseason. Store timestamps in UTC and express the product policy in America/New_York.

GitHub documents that scheduled jobs can be delayed or dropped. Schedule away from the hour boundary and monitor data age; use a managed scheduler/worker if precise publication latency becomes a product requirement. A once-weekly scrape cannot guarantee prompt updates.

## Website MVP

Use Supabase as the storage/read backend and serve precomputed outputs; no separate Python web server is required just to read rankings. Choose the UI framework when implementation starts; a TypeScript/React frontend is a reasonable product direction, while Streamlit can serve as an internal exploration tool.

Pages: latest official poll and next projection; team season history with games and point/rank trends; season/week archive; published model accuracy and methodology. Show projected vs official labels, forecast issue time, games-through cutoff, target release, previous-rank movement, others receiving votes and freshness status. Keep probability claims out until a probability model has been evaluated.

Use a narrow read API or reviewed read-only Supabase policies. Service credentials stay in ETL/server secrets. Add query indexes, caching, accessible mobile tables, empty/offseason/error states, and a stable published-run pointer. Validate source reuse/attribution and branding permissions before public launch; this review did not establish them.

## Delivery order and completion gates

| Milestone | Work | Completion gate |
|---|---|---|
| 1: data contract | Audit live tables/runs, migrations, canonical teams, real poll calendar, source validation | One representative season reconciles; unmatched identities and missing releases are accounted for |
| 2: feature pipeline | Pure computations, Elo replay, historical snapshots, current view, run metadata | Repeated build identical; later games cannot change an earlier cutoff; full team coverage |
| 3: modeling | Zero-vote labels, baselines, CatBoost experiments, chronological evaluation, artifact versioning | Reproducible held-out report and selected serving model |
| 4: automation | Daily pipeline, release reconciliation/freeze, poll retries, evaluation, freshness monitoring | Reruns do not duplicate facts; failed/partial runs do not publish; no target leakage |
| 5: public website | Rankings, team pages, archives, methodology and accuracy, deployment | Mobile UI and deployed reads agree with a selected frozen run; actual poll update is rehearsed |

Planning estimate for one developer: roughly 5–8 focused weeks, depending heavily on historical poll cleanup and available time. Build a thin website against a small validated dataset alongside modeling once the schema is stable; avoid polishing a full UI before data contracts settle.

First concrete implementation task: export/audit the deployed schema, then make one recent complete season flow from canonical facts through dated feature snapshots to a persistence forecast and its actual poll comparison. That establishes the complete path before scaling to every season or optimizing a model.

## Sources checked

- [Supabase Python select: default 1,000-row response limit](https://supabase.com/docs/reference/python/select)
- [Supabase upsert: conflict columns and unique constraints](https://supabase.com/docs/reference/python/upsert)
- [GitHub scheduled workflow limitations](https://docs.github.com/en/actions/how-tos/troubleshoot-workflows)
- [CatBoost regression objectives](https://catboost.ai/docs/en/concepts/loss-functions-regression)
- [Current KenPom 2026 game text source](https://kenpom.com/cbbga26.txt)
- [AP's October 13, 2025 men's preseason release](https://apnews.com/article/d0ecf6a1b386ac7868c20fe6b09f78ea), demonstrating why the October 1 placeholder and November-only schedule are insufficient.

The ESPN historical page could not be opened through the research browser; current HTML compatibility remains unverified.
