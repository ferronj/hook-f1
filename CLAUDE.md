# F1 Markov Model Project

## Environment
- Python environment: `uv run python <script>` (or `micromamba run -n f1-markov python3 <script>`)
- Project uses `uv` with `pyproject.toml` for dependency management

## Project Structure

### Production Pipeline
- `simulate_race.py` — General-purpose Monte Carlo race simulation with CLI args (replaces hardcoded per-race scripts)
- `dashboard.py` — Multi-race Streamlit dashboard, auto-discovers `data/sim_*.json` files
- `generate_2025_data.py` — Scrapes 2025 F1 season data and appends to Ergast CSVs

### Models (`models/`)
Only the production models are kept here. Stage 3 serves as the shared utility hub.
- `stage3_constructor.py` — Shared base: F1DataLoader, prepare_transitions, build_count_matrices, constants (DNF/START/N_OUTCOMES)
- `stage6_recency_constructor.py` — Year-weighted constructor priors, best calibration (LL/race = -59.0)
- `stage8_plackett_luce.py` — Time-varying Plackett-Luce, best ranking (Spearman rho = 0.800), also provides PL utilities for Stage 9
- `stage9_bayesian_ss.py` — Bayesian state-space with MAP inference, best top-3 accuracy (1.59/3)

### Data (`data/`)
- Ergast-style relational CSVs (circuits, constructors, drivers, races, results, qualifying, etc.)
- `sim_*.json` — Simulation output files consumed by the dashboard

### Archive (`archive/`)
Deprecated models (stages 1, 2, 4, 5, 7) and one-off evaluation/prediction scripts preserved for reference.
- `archive/models/` — stage1 (baseline), stage2 (driver pooling), stage4 (recency grid), stage5 (circuit), stage7 (HMM)
- `archive/` — evaluation scripts, predict scripts, retrospective simulations, notebook, old configs

## Model Architecture
- **State space**: 0=DNF, 1-20=positions, 21=START (first race of season)
- **Transition matrix shape**: (22 prev states) x (21 outcomes) — prev includes START, outcomes are DNF + P1-P20
- Dirichlet-Multinomial conjugate updates (Stages 3/6), Plackett-Luce (Stages 8/9)
- Training data: 2015-2025 (11 years) for 2026 predictions
- Hyperparameters optimized via leave-last-year-out CV

## Import Dependency Chain
```
stage3_constructor.py  (shared: F1DataLoader, constants, prepare_transitions, count matrices)
  ├── stage6_recency_constructor.py  (adds year-weighted constructor priors)
  ├── stage8_plackett_luce.py        (adds PL ranking model, DNF model)
  │     └── stage9_bayesian_ss.py    (adds Bayesian state-space, imports PL utilities from stage8)
  └── stage9_bayesian_ss.py          (also imports directly from stage3)
```
`simulate_race.py` adds `models/` to sys.path and imports from stages 6, 8, 9.
`dashboard.py` has NO model imports — reads only JSON.

## Key Learnings

### Driver IDs
- Max Verstappen = **830** (NOT 50, which is his father Jos Verstappen)
- New 2025 rookies: Antonelli=863, Hadjar=864, Bortoleto=865

### Constructor IDs (2026 season)
McLaren=1, Williams=3, Ferrari=6, Red Bull=9, Audi=15 (was Sauber), Aston Martin=117, Mercedes=131, Haas=210, Alpine=214, Racing Bulls=215, Cadillac=216 (new)

### Recency Weighting
- **Key insight**: Recency weighting helps out-of-sample prediction, not in-sample fit. You cannot learn temporal decay from training data alone — you need held-out future data (leave-last-year-out CV).
- Stage 6 uses geometric decay `w^(ref_year - y)` on constructor prior only; w=0.7 consistently selected.
- Stage 6 needs ≥10 years of training data for stable kappa_c estimation.

### Model Comparison Summary
| Model | Best At | LL/race | Avg T3 | Weakness |
|-------|---------|---------|--------|----------|
| Stage 6 | Calibration | -59.0 | 1.32 | Lower ranking accuracy |
| Stage 8 | Ranking | -94.7 | 1.36 | Overconcentrates probability, poor calibration |
| Stage 9 | Top-3 prediction | -60.0 | 1.59 | Slower to train |
| Ensemble | Balance | — | — | Blends strengths of all three |

### Why Archived Models Were Dropped
- **Stage 1**: Baseline only, can't differentiate drivers
- **Stage 2**: Superseded by Stage 6 (no constructor effect)
- **Stage 4**: Parameter collapse (lambda→0, kappa_grid→0), degenerates to Stage 3
- **Stage 5**: Circuit prior interacts destructively with constructor recency weighting
- **Stage 7 (HMM)**: Too coarse — 4 tiers can't distinguish within-constructor driver variation

## Data Notes
- Original Ergast data covers 1950-2024
- 2025 data generated via `generate_2025_data.py` (scrapes Wikipedia)
- The 2025 data is appended to the original CSVs — to regenerate, remove 2025 entries first
- Simulation results stored as JSON in `data/sim_*.json` (consumed by dashboard)

## Simulation CLI
```bash
# Round 1 (auto-detects race metadata and roster from CSVs):
uv run python simulate_race.py --season 2026 --round 1

# Round 2 (uses R1 results as starting state):
uv run python simulate_race.py --season 2026 --round 2

# With roster override for new/changed drivers:
uv run python simulate_race.py --season 2026 --round 1 --roster data/roster_2026.json

# Explicit metadata for future races not yet in CSVs:
uv run python simulate_race.py --season 2026 --round 1 \
    --race-name "Australian Grand Prix" --circuit "Albert Park, Melbourne" --date 2026-03-08
```
- Auto-detects race name, circuit, date from `races.csv` + `circuits.csv`
- Auto-detects driver roster from most recent race results
- For round > 1, uses previous round finishing positions as starting state
- Outputs `data/sim_{season}_{race_slug}.json` for the dashboard
- `--roster` accepts a JSON override: `{"driver_id": {"constructor_id": N, "name": "...", "abbreviation": "..."}}`

## Dashboard
- **Run**: `uv run streamlit run dashboard.py`
- **Dependencies**: streamlit, plotly (in pyproject.toml)
- **Features**: Model selector (Stage 6/Ensemble/Stage 9), podium prediction, full grid table, position distributions, model comparison, constructor analysis, position heatmap
