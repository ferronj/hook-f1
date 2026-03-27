---
name: predict-race
description: Predict the next F1 race using Markov models. Use when asked to predict, simulate, or forecast an upcoming F1 race.
---

# F1 Race Prediction Workflow

Run this workflow to generate Monte Carlo predictions for the next upcoming F1 race.

## Step 1: Determine Target Race

Read `config_2026.py` and find the next race from `CALENDAR_2026` whose date >= today.
That round number is the **target**. All rounds before it must have results data.

```python
from config_2026 import CALENDAR_2026
from datetime import date
today = date.today()
for rnd in sorted(CALENDAR_2026):
    race_date = date.fromisoformat(CALENDAR_2026[rnd]["date"])
    if race_date >= today:
        target_round = rnd
        break
```

Print: "Target race: Round {N} — {race_name} ({date})"

## Step 2: Check Prior Race Results

Read `generate_2026_data.py` and check which rounds have entries in both
`QUALIFYING_RESULTS` and `RACE_RESULTS` dicts. **All rounds < target must have data.**

For any missing round:
1. **Web search** for "{year} {race_name} Grand Prix results" and "{year} {race_name} qualifying results"
2. Add `QUALIFYING_RESULTS[round]` — list of `(driver_name, constructor_name, grid_position)` tuples
3. Add `RACE_RESULTS[round]` — list of `(driver_name, constructor_name, position, positionText, laps, time_str, points, statusId)` tuples
   - `statusId`: 1=Finished, 130=DNF, 20=DNS
   - `points`: 25,18,15,12,10,8,6,4,2,1 for P1-P10; 0 otherwise
   - For DNF: use actual laps completed, `"\\N"` for time, `"R"` for positionText
   - For DNS: laps=0, `"\\N"` for time, `"DNS"` for positionText
4. **Show the user** the results you found and ask them to confirm before saving

## Step 3: Check Roster

Verify no mid-season driver changes for the target race:
- Cross-check `DRIVER_MAP` in `generate_2026_data.py` against `DRIVERS_2026` in `config_2026.py`
- Web search for "{year} F1 driver changes" if unsure
- If changes needed, update both files and the roster in the race results

## Step 4: Populate CSVs

```bash
uv run python generate_2026_data.py
```

This appends new rounds to `data/races.csv`, `data/results.csv`, `data/qualifying.csv`.
Safe to run multiple times (skips existing rounds).

## Step 5: Run Simulation

```bash
uv run python simulate_race.py --season {year} --round {target_round}
```

If the target race is not yet in `races.csv` (future race), add explicit metadata:

```bash
uv run python simulate_race.py --season {year} --round {target_round} \
    --race-name "{race_name}" \
    --circuit "{circuit}" \
    --date {date}
```

Get race_name, circuit, and date from `config_2026.py` `CALENDAR_2026[target_round]`.

This outputs `data/sim_{year}_{slug}.json` for the dashboard.

## Step 6: Current Form Analysis

Web search for recent F1 news to provide context:
- "{year} F1 standings after round {target_round - 1}"
- "{year} {race_name} Grand Prix preview"
- Team performance trends, reliability issues, weather forecast

Summarize key narratives for the user alongside the model predictions.

## Step 7: Verify Dashboard

```bash
uv run streamlit run dashboard.py
```

Check that:
- The new sim file appears in the race selector
- All models load (Stage 6, Stage 9, Composite)
- Podium predictions, heatmap, and constructor analysis render correctly

If running locally, use the Claude Preview MCP tool to verify the dashboard visually.

## Step 8: Commit and Push

Stage the changed files and push to main:

```bash
git add generate_2026_data.py data/sim_*.json
git commit -m "Add R{prior_rounds} results and R{target_round} predictions"
git push origin main
```

This deploys the updated dashboard to Streamlit Cloud.
