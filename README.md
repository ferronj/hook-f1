# F1 Markov Model Project

Dirichlet-Multinomial Markov and Bayesian state-space models for predicting F1 finishing positions, trained on historical race results (2015-2025).

## Environment Setup

```bash
# Install dependencies with uv
uv sync

# Run any script
uv run python <script>

# Launch the dashboard
uv run streamlit run dashboard.py
```

## Model Stages

Three production models are trained on 2015-2025 data and ensembled:

| Stage | Description | Strength |
|-------|------------|----------|
| 6 | Year-weighted constructor (Dirichlet-Multinomial) | Best calibration (LL/race = -59.0) |
| 8 | Time-varying Plackett-Luce | Best ranking (Spearman rho = 0.800) |
| 9 | Bayesian state-space (MAP) | Best top-3 accuracy (1.59/3) |

A **composite** model blends Stage 6 + Stage 9 probabilities (Stage 8 excluded when degenerate).

Archived models (stages 1-5, 7) are preserved in `archive/` for reference.

## Race Prediction Workflow

### Quick start: predict the next race

```bash
# 1. Add prior race results to generate_2026_data.py, then populate CSVs:
uv run python generate_2026_data.py

# 2. Run simulation for the target round:
uv run python simulate_race.py --season 2026 --round 3

# 3. Launch dashboard to view predictions:
uv run streamlit run dashboard.py
```

### Adding race results

Edit `generate_2026_data.py` and add entries to `QUALIFYING_RESULTS` and `RACE_RESULTS` dicts:

```python
QUALIFYING_RESULTS[3] = [
    ("Driver Name", "Constructor Name", grid_position),
    ...
]
RACE_RESULTS[3] = [
    ("Driver Name", "Constructor Name", position, "posText", laps, "time", points, statusId),
    ...
]
# statusId: 1=Finished, 130=DNF, 20=DNS
```

Then run `uv run python generate_2026_data.py` to append to CSVs.

### Simulation CLI

```bash
# Auto-detect everything from CSVs:
uv run python simulate_race.py --season 2026 --round 3

# Explicit metadata for races not yet in CSVs:
uv run python simulate_race.py --season 2026 --round 3 \
    --race-name "Japanese Grand Prix" --circuit "Suzuka Circuit" --date 2026-03-29
```

Results are saved to `data/sim_*.json` and auto-discovered by the dashboard.

## Key Files

| File | Description |
|------|-------------|
| `simulate_race.py` | General-purpose Monte Carlo race simulation CLI |
| `config_2026.py` | 2026 season constants (drivers, constructors, calendar) |
| `generate_2026_data.py` | Append 2026 race results to Ergast CSVs |
| `dashboard.py` | Streamlit multi-race dashboard |
| `models/stage{3,6,8,9}_*.py` | Production model implementations |
| `data/sim_*.json` | Simulation results consumed by dashboard |
| `pyproject.toml` | Python dependencies (managed by uv) |
