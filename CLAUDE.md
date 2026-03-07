# F1 Markov Model Project

## Environment
- Python environment: `micromamba run -n f1-markov python3 <script>`
- Do NOT use `python` (not found); always use `python3` via micromamba

## Project Structure
- `data/` — Ergast-style relational CSVs (circuits, constructors, drivers, races, results, qualifying, etc.)
- `models/` — F1 finishing position models of increasing complexity:
  - `stage1_global_transition.py` — Global transition matrix (baseline, can't differentiate drivers)
  - `stage2_driver_pooling.py` — Driver-specific partial pooling via Empirical Bayes (shared kappa)
  - `stage3_constructor.py` — Adds constructor effect via additive pseudo-count priors (kappa_g, kappa_c)
  - `stage4_recency_grid.py` — Adds recency weighting on priors + grid position prior (kappa_g, kappa_c, kappa_grid, lambda)
  - `stage5_circuit.py` — Adds circuit-specific transition matrices (kappa_g, kappa_c, kappa_circuit)
  - `stage6_recency_constructor.py` — Year-weighted constructor priors with leave-last-year-out CV for decay selection (kappa_g, kappa_c, w); optional circuit prior mode (kappa_k) when meta_df has 'circuit' column
  - `stage7_hmm.py` — Constructor-tier HMM with 4 hidden states (dominant/frontrunner/midfield/backmarker), Baum-Welch EM, driver offsets in log-odds space
  - `stage8_plackett_luce.py` — Time-varying Plackett-Luce with exponential smoothing on log-strengths, MC position marginals, Markov correction
  - `stage9_bayesian_ss.py` — Bayesian state-space with MAP inference: random walk on driver/constructor log-strengths, PL observations, L-BFGS-B optimization (sigma_d, sigma_c)
- `generate_2025_data.py` — Scrapes 2025 F1 season data and appends to Ergast CSVs
- `evaluate_2025_r1.py` — Compares all 4 stages on 2025 Australian GP top-3 prediction
- `evaluate_multi_era.py` — Multi-era evaluation: tests all 10 stages across 4 eras (2000, 2010, 2020, 2025)
- `simulate_2025_season.py` — Retrospective Monte Carlo simulation of 2025 season (Stages 2, 3, 6, 7, 8)
- `simulate_2026_australia.py` — Monte Carlo simulation of 2026 Australian GP (Stage 6 + Stage 9 + Ensemble), outputs JSON
- `dashboard_2026_australia.py` — Streamlit interactive dashboard for 2026 Australian GP results (`streamlit run dashboard_2026_australia.py`)
- `predict_2026_r1.py` — 2026 Australian GP predictions (Stage 2 + Stage 3)
- `predict_2026_season.py` — Full 2026 season simulation (Stage 3)
- `f1.ipynb` — Original exploration notebook

## Model Architecture
- **State space**: 0=DNF, 1-20=positions, 21=START (first race of season)
- **Transition matrix shape**: (22 prev states) x (21 outcomes) — prev includes START, outcomes are DNF + P1-P20
- All models use Dirichlet-Multinomial conjugate updates
- Training data: 2010-2024 seasons by default (configurable)
- Hyperparameters optimized via marginal log-likelihood

## Key Learnings

### Driver IDs
- Max Verstappen = **830** (NOT 50, which is his father Jos Verstappen)
- All other 2025 driver IDs verified correct against drivers.csv
- New 2025 rookies: Antonelli=863, Hadjar=864, Bortoleto=865

### Recency Weighting Pitfalls and Solutions
- **Stage 4 failure (exponential decay on priors)**: Applying `exp(-lambda * age)` to prior-level matrices. Optimizer found lambda≈0 (no decay) because in-sample marginal likelihood doesn't benefit from recency. Also, applying decay to driver-level counts causes degeneracy (lambda→∞ zeroes out counts).
- **Stage 6 w collapse (joint optimization)**: Optimizing `w` jointly with kappas via L-BFGS-B pushed w→1.0 (no decay). Same root cause: DM marginal likelihood on training data doesn't incentivize recency because all training observations are equally valid within the training set.
- **Stage 6 solution (leave-last-year-out CV)**: Split training data into inner-train (years 1..N-1) and validation (year N). For each w candidate, optimize kappas on inner-train, evaluate DM log-likelihood on validation. This correctly selects w=0.7 because the validation year is out-of-sample.
- **Key insight**: Recency weighting helps out-of-sample prediction, not in-sample fit. You cannot learn temporal decay from training data alone — you need held-out future data.

### Stage 6: Year-Weighted Constructor Priors (Best Model)
- Architecture: `alpha = kappa_g * pi^{global} + kappa_c * pi^{(C,w)} + n_{i,C,s}`
- Only the constructor prior gets year-weighted; global prior stays unweighted (stable baseline); driver counts stay unweighted
- Geometric decay: each observation in season y gets weight `w^(ref_year - y)` where `ref_year = max(training_years) + 1`
- w selected via leave-last-year-out CV from candidates (0.3, 0.5, 0.7, 0.85, 1.0); w=0.7 consistently selected
- Fitted on 2020-2024: kappa_g=18.93, kappa_c=15.13, w=0.7 (vs Stage 3's kappa_g=0.10, kappa_c=391.50)
- The 26x reduction in kappa_c breaks constructor prior dominance, allowing driver data to contribute meaningfully

### Stage 7: Constructor-Tier HMM
- 4 hidden competitiveness tiers per constructor: dominant / frontrunner / midfield / backmarker
- Baum-Welch EM with K-means initialization + 5 random restarts
- Shared tier emissions shifted by location parameter; driver offsets in log-odds space with shrinkage weight = n/(n+20)
- Sensible tier assignments (2020-2024): Ferrari=dominant, Red Bull=dominant, McLaren=frontrunner, Mercedes=frontrunner, Williams=backmarker
- **Weakness**: Hamilton (driver 1) massively overrated (sim rank 1.1, actual 7) because Ferrari=dominant tier; HMM can't distinguish within-constructor driver variation well
- **Weakness**: Tsunoda overrated (sim rank 2.5, actual 17) because Red Bull=dominant tier

### Stage 8: Time-Varying Plackett-Luce
- Models full race ranking jointly (positions are mutually exclusive) — fundamentally different from DM-Markov stages
- Driver strengths: latent log_λ_i(t) with exponential smoothing; α=0.99 selected via leave-last-year-out CV
- Constructor effect: additive on log-strength; two-stage DNF model (empirical P(DNF) + PL for finisher ranking)
- MC sampling (3000 rankings) for position marginals; Markov correction via global transition matrix ratio
- **Critical fix**: Must work in log space with normalization to prevent strength divergence (multiplicative decomposition + exponential smoothing causes unbounded growth)
- **Best Spearman rho**: 0.800 (vs Stage 6's 0.577) — best at ranking order among all models
- **Worst calibration**: Avg calibration error 0.5886, avg LL/race -97.2 — overconcentrates probability mass on too few outcomes
- Verstappen massively overrated: P(WDC) = 92.0% because Verstappen's 2020-2024 strength dominates PL; model can't anticipate team changes

### Stage 9: Bayesian State-Space (Best Top-3 Accuracy)
- Architecture: Random walk on driver log-strengths `mu_i(t) ~ N(mu_i(t-1), sigma_d^2)` and constructor effects `beta_C(t) ~ N(beta_C(t-1), sigma_c^2)`, observed via Plackett-Luce rankings
- MAP inference via L-BFGS-B with analytic gradients over full trajectory (~3000 params for 5 years)
- Gap-aware random walk: innovation variance scales with number of races between appearances
- Hyperparameters (sigma_d, sigma_c) selected via leave-last-year-out CV; consistently selects sigma_d=0.02, sigma_c=0.05 (slow evolution)
- Soft centering penalty (0.01) for identifiability of log-strengths
- DNF model: same empirical shrinkage as Stage 8; MC sampling (3000) for position marginals; Markov correction
- **Best Avg T3**: 1.59/3 (best top-3 prediction accuracy across all models, beating Stage 8's 1.36)
- **Good calibration**: LL/race = -60.0 (close to Stage 6's -59.0, vastly better than Stage 8's -94.8)
- **Era 2000 standout**: 2.00/3 avg T3 with 5 perfect races — joint optimization captures late-90s dynamics well
- **Key advantage over Stage 8**: Joint optimization over full trajectory prevents the greedy overconcentration that plagues Stage 8
- **Trade-off**: Slower to train (16 MAP fits for CV grid), but produces both good ranking AND good calibration

### Stage 6c: Circuit-Aware Year-Weighted Constructor (Negative Result)
- Architecture: `alpha = kappa_g * pi^{global} + kappa_c * pi^{(C,w)} + kappa_k * pi^{(K)} + n_{i,C,s}`
- Circuit prior is static (unweighted) — tracks don't change physically across years
- Uses Stage 5's data loading (meta_df with `circuit` column); backward compatible (no circuit column → identical to Stage 6)
- **Result: Stage 6c does NOT improve over Stage 6** — avg LL/race -59.4 vs Stage 6's -59.0, avg T3 1.28 vs 1.32
- In 2 of 4 eras (2020, 2025), Stage 6c collapsed to Stage 5 behavior (w=1.0, kappa_c dominant, recency weighting disabled)
- Era 2000: kappa_k=128.67 absorbed signal from kappa_c (pushed to 0.01), but overall LL improved slightly (-869.5 vs -875.6)
- Era 2010: kappa_k=74.81 with w=0.85, but LL worse (-1295.2 vs -1284.2)
- **Key insight**: Circuit effects interact with constructor recency weighting in destructive ways. When both are present, the optimizer struggles to allocate prior weight correctly. The leave-last-year-out CV can't jointly optimize w + 3 kappas reliably.
- Stage 6 (without circuit) remains the best overall model

### Multi-Era Evaluation Results (9 Stages × 4 Eras)
- Tested all 9 stages across 4 eras: Train 1995-1999→Eval 2000, Train 2005-2009→Eval 2010, Train 2015-2019→Eval 2020, Train 2020-2024→Eval 2025
- **Stage 6 best LL/race**: -59.0 (best calibrated probabilities across eras)
- **Stage 8 best Avg T3**: 1.36/3 correct in top 3 (best ranking accuracy) but worst LL/race (-94.7)
- Stage 6c underperforms Stage 6: avg T3 = 1.28, LL/race = -59.4 — circuit prior disrupts constructor recency balance
- Stage 7 (HMM) disappoints: Avg T3 = 1.00, LL/race = -63.1 — too coarse (4 tiers insufficient)
- Stages 4 and 5 consistently underperform due to parameter collapse / overfitting

### 2025 Retrospective Simulation (10K Monte Carlo, 5 Models)
- **Spearman rho**: Stage 8 (0.800) > Stage 2 (0.710) > Stage 6 (0.577) > Stage 7 (0.562) > Stage 3 (0.475)
- **Rank MAE**: Stage 8 (3.01) > Stage 7 (3.76) > Stage 3 (3.82) > Stage 6 (4.05) > Stage 2 (4.44)
- **P(Norris WDC)**: Stage 6 (5.9%) > Stage 2 (5.1%) > Stage 3 (2.7%) > Stage 8 (0.1%) > Stage 7 (0.0%)
- Stage 8 has best ranking but worst championship probability calibration — overconcentrates on Verstappen (92.0% WDC)
- Stage 7 badly overrates Ferrari (P(WCC)=71.7%) and Hamilton (sim rank 1.1 vs actual 7) due to coarse tier modeling
- **Stage 6 remains best overall**: balanced calibration + reasonable ranking + best championship probabilities

### Model Evaluation (2025 Australian GP)
- Stage 2 (driver pooling) performed best: 1/3 exact match, 1/3 in top 3, best calibration
- Stage 3 and Stage 4 both predicted Mercedes drivers too high (Antonelli P1, Russell P2) due to strong constructor prior from 2010-2024 Mercedes dominance
- Stage 4 collapsed to Stage 3 behavior (lambda→0, kappa_grid→0)
- Actual top 3: Norris (McLaren), Verstappen (Red Bull), Russell (Mercedes)

### Stage 6 Training Window Sensitivity
- **5-year window (2021-2025)**: kappa_c collapses to 0.01 (constructor prior zeroed out). Too few years for leave-last-year-out CV to separate kappa_c from kappa_g.
- **11-year window (2015-2025)**: kappa_g=29.85, kappa_c=4.17, w=0.70 — healthy constructor prior.
- **16-year window (2010-2025)**: kappa_g=32.17, kappa_c=2.01, w=0.70 — similar but slightly less constructor weight.
- **Key insight**: Stage 6 needs ≥10 years of training data for stable kappa_c estimation. Use 2015-2025 for 2026 predictions.

### 2026 Australian GP Simulation (Stage 6 + Stage 9 + Ensemble)
- Training: 2015-2025 (11 years)
- Stage 6 (kappa_g=29.85, kappa_c=4.17, w=0.70): Flat distribution, VER 8.3% win, NOR 6.5%, LEC 6.7%
- Stage 9 (sigma_d=0.100, sigma_c=0.100): Concentrated on McLaren/VER — NOR 27.7%, VER 25.1%, PIA 22.1%
- Ensemble (50/50 blend): VER 16.8%, NOR 16.9%, PIA 12.9% — balanced view
- Cadillac (new team, ID=216): No constructor history, driver-prior only (BOT/PER)
- New rookies: Lindblad (866, Racing Bulls), Colapinto (861, Alpine)
- Dashboard: `micromamba run -n f1-markov streamlit run dashboard_2026_australia.py`

### Constructor IDs (2026 season)
McLaren=1, Williams=3, Ferrari=6, Red Bull=9, Audi=15 (was Sauber), Aston Martin=117, Mercedes=131, Haas=210, Alpine=214, Racing Bulls=215, Cadillac=216 (new)

## Data Notes
- Original Ergast data covers 1950-2024
- 2025 data generated via `generate_2025_data.py` (scrapes Wikipedia)
- The 2025 data is appended to the original CSVs — to regenerate, remove 2025 entries first
- Grid positions for qualifying come from `qualifying.csv` (position column)
- Simulation results stored as JSON in `data/sim_2026_australia.json` (consumed by dashboard)

## Dashboard
- **Streamlit app**: `dashboard_2026_australia.py`
- **Run**: `micromamba run -n f1-markov streamlit run dashboard_2026_australia.py`
- **Dependencies**: streamlit, plotly (installed in f1-markov env)
- **Data source**: `data/sim_2026_australia.json` (generated by `simulate_2026_australia.py`)
- **Features**: Model selector (Stage 6/Ensemble/Stage 9), podium prediction, full grid table, position distributions, model comparison, 2025 calibration check, constructor analysis, position heatmap
