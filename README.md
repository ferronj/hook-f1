# F1 Markov Model Project

Dirichlet-Multinomial Markov and Bayesian state-space models for predicting F1 finishing positions, trained on historical race results (2010-2025).

## Environment Setup

```bash
# Install dependencies with uv
uv sync

# Run any script
uv run python <script>

# Launch the dashboard
uv run streamlit run dashboard.py
```

For full Bayesian NUTS sampling (optional, requires PyMC):
```bash
uv sync --extra nuts
```

## Model Stages

| Stage | Description | Key Idea |
|-------|------------|----------|
| 1 | Global transition matrix | Baseline, can't differentiate drivers |
| 2 | Driver partial pooling | Empirical Bayes with shared kappa |
| 3 | Constructor + driver | Additive constructor pseudo-count priors |
| 4 | Recency + grid | Adds recency weighting + grid position (collapsed to Stage 3) |
| 5 | Circuit-specific | Circuit-specific transition matrices |
| 6 | Year-weighted constructor | Geometric decay on constructor priors, CV for w (**best calibration**) |
| 7 | Constructor-tier HMM | 4 hidden competitiveness tiers, Baum-Welch EM |
| 8 | Time-varying Plackett-Luce | Exponential smoothing on log-strengths (**best ranking**) |
| 9 | Bayesian state-space | Random walk + PL, MAP or NUTS inference (**best top-3 accuracy**) |

## Race Simulations

Each simulation script trains Stage 6, Stage 8, and Stage 9 models on 2015-2025 data, runs 10,000 Monte Carlo races per model, and outputs a JSON file consumed by the dashboard.

```bash
# 2026 Australian GP
uv run python simulate_2026_australia.py

# 2026 Japanese GP
uv run python simulate_2026_japan.py

# Sequential race-by-race simulation with in-season updates
uv run python simulate_2026_race.py
```

Results are saved to `data/sim_*.json` and auto-discovered by the dashboard.

## Stage 9 NUTS Training (Full Bayesian)

Stage 9 with MAP point estimates overconcentrates probability (e.g., McLaren 100% WCC for 2026). Full NUTS sampling produces posterior distributions over ~3000 parameters, naturally widening prediction intervals and improving calibration.

### Step 1: Train the NUTS model

```bash
uv run python run_stage9_nuts.py \
  --min-year 2020 --max-year 2024 \
  --draws 1000 --tune 2000 --chains 4 --cores 4 \
  --target-accept 0.9 --max-treedepth 12 \
  --output stage9_nuts_2025.nc
```

**Estimated runtime: 2-4 hours** on Apple Silicon (4 chains x 3000 draws, ~1400 parameters).

This will:
1. Run MAP cross-validation (16 fits) to select sigma_d, sigma_c (~2 min)
2. MAP fit on full data for NUTS warm start (~10 sec)
3. Build vectorized PyMC model (<1 sec)
4. NUTS sampling (~2-4 hours)
5. Save trace to `.nc` + metadata to `.pkl`
6. Print convergence diagnostics (R-hat, ESS, divergences)

#### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--min-year` | 2020 | First training year |
| `--max-year` | 2024 | Last training year |
| `--draws` | 1000 | NUTS draws per chain |
| `--tune` | 2000 | Tuning steps per chain |
| `--chains` | 4 | Number of chains |
| `--cores` | 4 | CPU cores |
| `--target-accept` | 0.9 | Target acceptance rate |
| `--max-treedepth` | 12 | Max NUTS tree depth |
| `--output` | `stage9_nuts_trace.nc` | Output trace path |
| `--checkpoint-every` | 200 | Progress report interval |
| `--n-mc-samples` | 3000 | MC samples for position marginals |

#### Quick test run (~1 min)

```bash
uv run python run_stage9_nuts.py \
  --min-year 2023 --max-year 2024 \
  --draws 50 --tune 100 --chains 1 --cores 1 \
  --output test_trace.nc
```

### Step 2: Run season simulations (MAP vs NUTS comparison)

```bash
uv run python simulate_seasons_nuts.py \
  --trace stage9_nuts_2025.nc \
  --n-sims 10000 --n-posterior-draws 200
```

Runs 10,000 Monte Carlo seasons for both 2025 (retrospective, compared to actual results) and 2026 (predictive), comparing Stage 9 MAP vs NUTS predictions side-by-side.

### Running Long Jobs on macOS

The NUTS training takes 2-4 hours. Here are options to ensure it completes without interruption:

#### Option 1: `caffeinate` (simplest)

Prevents your Mac from sleeping while the command runs. The process dies if you close the terminal.

```bash
caffeinate -i uv run python run_stage9_nuts.py \
  --min-year 2020 --max-year 2024 \
  --draws 1000 --tune 2000 --chains 4 --cores 4 \
  --output stage9_nuts_2025.nc
```

#### Option 2: `nohup` + `caffeinate` (survives terminal close)

Runs in the background, survives closing the terminal, prevents sleep. Output goes to `nuts_training.log`.

```bash
nohup caffeinate -i uv run python run_stage9_nuts.py \
  --min-year 2020 --max-year 2024 \
  --draws 1000 --tune 2000 --chains 4 --cores 4 \
  --output stage9_nuts_2025.nc \
  > nuts_training.log 2>&1 &

# Monitor progress:
tail -f nuts_training.log
```

#### Option 3: `tmux` or `screen` (recommended for interactive monitoring)

Persistent terminal session that survives disconnects. You can detach/reattach at will.

```bash
# Start a tmux session
tmux new -s nuts

# Run the training inside tmux
caffeinate -i uv run python run_stage9_nuts.py \
  --min-year 2020 --max-year 2024 \
  --draws 1000 --tune 2000 --chains 4 --cores 4 \
  --output stage9_nuts_2025.nc

# Detach: Ctrl+B then D
# Reattach later: tmux attach -t nuts
```

#### macOS sleep prevention notes

- `caffeinate -i` prevents idle sleep (process keeps running even if you walk away)
- `caffeinate -s` also prevents system sleep when on AC power
- Energy Saver settings: System Preferences > Battery > Options > "Prevent automatic sleeping on power adapter" can also help
- Closing the laptop lid will still sleep the Mac unless you use `caffeinate -s` on AC power or an external display

## Other Scripts

| Script | Description |
|--------|-------------|
| `simulate_seasons.py` | Monte Carlo simulation comparing Stage 6 vs Stage 9 MAP |
| `evaluate_multi_era.py` | Multi-era evaluation across 4 eras x all stages |
| `evaluate_2025_r1.py` | 2025 Australian GP prediction comparison |
| `generate_2025_data.py` | Scrapes and appends 2025 season data |
| `predict_2026_shanghai.py` | 2026 Shanghai GP predictions (Stage 6 + Stage 9) |

## Key Files

- `models/stage{1-9}_*.py` — Model implementations
- `data/` — Ergast-style relational CSVs
- `data/sim_*.json` — Simulation results consumed by dashboard
- `pyproject.toml` — Python dependencies (managed by uv)
- `SIMULATION_RESULTS.md` — Stage 6 vs Stage 9 MAP simulation analysis
- `PREDICTIONS_2026.md` — 2026 season predictions and model analysis
