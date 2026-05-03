# TODO

Running list of follow-ups for the F1 Markov model project.

## Open

### Dashboard: actuals vs. prediction comparison

For races that have happened, show the actual finishing order alongside each model's prediction (and the agent forecast, if present). Want a per-race "How did we do?" view: green check / red X for each podium slot, top-3 hit count, top-10 overlap, maybe a delta column for predicted vs. actual position.

Implementation notes:
- The dashboard already has a `calibration` block driven from `build_calibration` (prior-year same round). Reuse that pattern, but condition on this season's actuals from `data/results.csv` joined to `data/races.csv` (when a result exists for the same `(season, round)`).
- A race is "scored" once `results.csv` has a row for `raceId = RACE_ID_START + round - 1`. If absent, dashboard hides the section (forward-looking races stay forward-looking).
- Score *every* model in `data["models"]` plus `agent_prediction.podium` if present. Common metrics: top-3 hits, top-10 overlap, mean abs position error, Spearman ρ.

### Season summary tab

A second top-level tab (or a sidebar selector that switches view) showing season-wide model performance and the upcoming race at a glance. Two-pane layout:
- **Left**: rolling per-race scorecard for each model (top-3 hits, top-10 overlap, calibration metric like LL or Brier) — line chart over rounds + small table.
- **Right**: next race preview card — pulled from the most recent forward-looking sim_*.json (the one whose date is `>= today`). Show podium pick, agent forecast if present, key narrative.

Pre-reqs: actuals-vs-prediction scoring (above) needs to land first since the season-summary scorecard reuses the same metrics.

### Backfill earlier-race predictions without Stage 8

The first three sim files (`sim_2026_australia.json`, `sim_2026_china.json`, `sim_2026_japan.json`) were generated when Stage 8 (Time-Varying Plackett-Luce) was still in the prediction pipeline. Stage 8 was consistently degenerate and excluded from the composite, but its presence still clutters the dashboard's model selector and comparison charts for those races.

Re-run each prior race using the current Stage 8-less pipeline, with the same conditioning the original prediction had access to (training data 2015-2025, plus only the in-season races completed *before* the target race). No agent prediction layer — raw script output only.

| Round | Target | Conditioning |
|---|---|---|
| 1 | Australia | Training 2015-2025 only; no in-season results |
| 2 | China | Training + R1 Australia results |
| 3 | Japan | Training + R1 Australia + R2 China results |
| 4 | Miami | Training + R1 + R2 + R3 Japan results |

Steps for each:
1. Temporarily remove the later races from `data/{races,results,qualifying}.csv` (or run from a checkout at the appropriate commit)
2. `uv run python simulate_race.py --season 2026 --round N --race-name "..." --circuit "..." --date ...`
3. Restore the data files
4. The output `data/sim_2026_<slug>.json` overwrites the original

Cleanest implementation: a wrapper script that masks out post-target rows in-memory before training (or filters by `--train-end-race`), so we don't have to mutate the CSVs. Worth adding only if we'll do this more than once.

## Done

- [x] Remove Stage 8 from active prediction pipeline (kept as Stage 9 dep)
- [x] Add agent forecast scaffolding (`add_agent_prediction.py` + dashboard rendering)
- [x] Renumber 2026 calendar after Bahrain/Saudi cancellations
