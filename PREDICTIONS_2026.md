# 2026 F1 Season Predictions — Model Analysis

## Method

Dirichlet-Multinomial Markov models trained on 2010–2025 F1 results (6,915 race transitions). We evaluated four model stages on the 2025 Australian GP and then used the two most informative stages to predict 2026.

**Stage 2** (driver partial pooling) won the 2025 validation test — it correctly placed Verstappen at P2 (1/3 exact position match). **Stage 3** (constructor + driver) provides the most structurally interesting predictions by incorporating team strength, so we use it as the primary model for 2026. Both are reported for comparison.

Season predictions use 10,000 Monte Carlo simulated seasons via Stage 3, sampling race-by-race from predicted Dirichlet-Multinomial distributions and accumulating championship points.

## 2026 Australian GP Predictions

| Pos | Driver | Team | P(top3) |
|-----|--------|------|---------|
| P1 | Kimi Antonelli | Mercedes | 33.9% |
| P2 | George Russell | Mercedes | 33.9% |
| P3 | Charles Leclerc | Ferrari | 27.8% |

Pole: Antonelli or Russell (~13% each).

## 2026 Season Championship Predictions

### Drivers' Championship

| Rank | Driver | Team | P(WDC) | P(Top 3) | Avg Pts | Avg Wins |
|------|--------|------|--------|----------|---------|----------|
| 1 | Max Verstappen | Red Bull | 23.6% | 55.8% | 242.5 | 3.4 |
| 2 | George Russell | Mercedes | 21.8% | 53.4% | 238.1 | 3.1 |
| 3 | Kimi Antonelli | Mercedes | 20.6% | 52.4% | 236.8 | 3.1 |
| 4 | Isack Hadjar | Red Bull | 18.9% | 51.8% | 235.7 | 3.2 |
| 5 | Charles Leclerc | Ferrari | 5.1% | 29.1% | 204.3 | 1.3 |
| 6 | Lewis Hamilton | Ferrari | 4.8% | 27.9% | 202.0 | 1.3 |

### Constructors' Championship

| Rank | Team | P(WCC) | Avg Pts |
|------|------|--------|---------|
| 1 | Red Bull | 44.0% | 478.2 |
| 2 | Mercedes | 42.7% | 474.9 |
| 3 | Ferrari | 11.1% | 406.3 |
| 4 | Cadillac | 1.1% | 250.4 |
| 5 | McLaren | 1.0% | 304.1 |

### Narrative Predictions

- **Good surprise**: Charles Leclerc — 5.1% WDC but 29.1% top-3 championship finish
- **Flop**: George Russell — highest variance among contenders (CV=0.28)
- **Crazy**: Arvid Lindblad wins a Grand Prix (62% P(≥1 win) across the season)

## What the Model Gets Right

**Constructor hierarchy is real.** Stage 3's optimized kappa_c (420.45) dwarfs kappa_g (0.10), meaning the model assigns ~100% of the prior weight to the constructor component. This matches F1 reality: the car matters far more than the driver in determining finishing positions. Over 2010–2025, Mercedes, Red Bull, and Ferrari have dominated, and the model reflects that.

**Teammate parity.** The model predicts extremely tight intra-team battles (Russell 238 vs Antonelli 237 pts, Verstappen 242 vs Hadjar 236, Leclerc 204 vs Hamilton 202). This is a direct consequence of the constructor prior dominating — teammates share the same constructor, so their predictions converge toward the team's historical performance.

**The Markov property carries information.** Drivers who finished well in the previous race tend to finish well in the next one. This isn't just momentum — it captures that cars and drivers tend to be consistently fast or slow within a season. The transition matrix from START (first race) is the weakest predictive state, which is why first-race predictions are hardest.

## What the Model Gets Wrong

**The constructor prior has too much inertia (fixed in Stage 6).** Mercedes dominated 2014–2021 but was merely competitive (not dominant) in 2022–2025. Stage 3 weighs all training seasons equally, inflating Mercedes. Stage 6 addresses this with year-weighted constructor priors (geometric decay w=0.7), reducing kappa_c from 391.5 to 15.1, which breaks constructor dominance and allows driver data to contribute. Stage 6 is now the best model across all 4 evaluation eras (2000, 2010, 2020, 2025). The 2026 predictions above still use Stage 3 and should be regenerated with Stage 6.

**McLaren was drastically underrated (partially fixed in Stage 6).** Norris won the 2025 WDC and McLaren won the 2024 WCC, yet Stage 3 ranks them 7th–8th individually and 5th in WCC predictions. Stage 6's year-weighting improved McLaren's 2025 Monte Carlo WCC probability from 3.6% to 11.5% and Norris WDC from 2.1% to 5.9%, but the model still underrates their trajectory.

**Rookies and team-changers are poorly modeled.** Hadjar has only 24 career observations yet is predicted to be championship-competitive (18.9% WDC) purely because of Red Bull's constructor prior. Similarly, Antonelli (24 observations) is ranked 3rd for WDC. The model is essentially saying "whoever drives the Red Bull/Mercedes will be strong" rather than evaluating the driver. This is arguably correct in F1, but it means the model can't distinguish a generational talent from a pay driver at the same team.

**Cadillac is unmodeled.** As a brand-new constructor, Cadillac has no historical data, so Bottas and Perez fall back to the global prior. The model can't account for GM's investment level, technical partnerships, or infrastructure — it just treats them as average. This is a fundamental limitation of any model trained purely on historical results.

**No regulation change modeling.** 2026 introduces major new technical regulations (new aero rules, active aero, new power units). Historical constructor strength is less predictive across regulation changes. The model has no concept of this discontinuity.

## Stage 4 Post-Mortem

Stage 4 added recency weighting and grid position to Stage 3. Both features collapsed to zero during optimization:

- **lambda → 0** (no recency decay): All historical seasons weighted equally
- **kappa_grid → 0** (no grid effect): Grid position prior contributed nothing

This happened because the marginal log-likelihood found no improvement from these additions — the constructor prior already explains most of the variance. The grid position effect was likely absorbed by the constructor prior (top teams qualify at the front and finish at the front).

A key implementation lesson: applying recency decay to the driver-level counts used in the marginal likelihood creates a degenerate optimum where lambda → ∞ and all counts shrink to zero, trivially maximizing the likelihood by making the model predict only the prior. The fix is to apply decay only to the prior-level matrices (global, constructor, grid).

## Stage 6: Year-Weighted Constructor Priors

Stage 6 solved Stage 4's recency collapse by: (1) weighting only the constructor counts (not global prior or driver counts), and (2) selecting the decay parameter w via leave-last-year-out cross-validation instead of in-sample optimization.

- **Architecture**: `alpha = kappa_g * pi^{global} + kappa_c * pi^{(C,w)} + n_{i,C,s}` where pi^{(C,w)} uses year-weighted constructor counts with geometric decay `w^(ref_year - y)`
- **w=0.7** consistently selected across all 4 evaluation eras via CV
- **Fitted params**: kappa_g=18.93, kappa_c=15.13 (vs Stage 3's kappa_g=0.10, kappa_c=391.50)
- **Best model overall**: wins both avg correct-in-top-3 (1.32/3) and avg LL/race (-59.0) across 4 eras
- **2025 Monte Carlo**: P(Norris WDC) improved from 2.1% to 5.9%, P(McLaren WCC) from 3.6% to 11.5%
- **Key insight**: In-sample marginal likelihood cannot learn temporal decay because all training observations are equally valid within the training set. Only out-of-sample evaluation (via CV) can select meaningful recency parameters.

## Files

- `predict_2026_r1.py` — Australian GP race predictions (Stage 2 + Stage 3)
- `predict_2026_season.py` — Full season Monte Carlo simulation (10K seasons, Stage 3)
- `evaluate_2025_r1.py` — Retrospective validation on 2025 Australian GP
- `evaluate_multi_era.py` — Multi-era evaluation across 4 eras (2000, 2010, 2020, 2025) × 6 stages
- `simulate_2025_season.py` — Retrospective 2025 Monte Carlo simulation (Stages 2, 3, 6)
- `models/stage{1,2,3,4,5,6}_*.py` — Model implementations
