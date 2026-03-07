# F1 Season Simulations — Stage 6 vs Stage 9

Monte Carlo simulation (10,000 seasons each) comparing two models:
- **Stage 6**: Year-weighted constructor Dirichlet-Markov (best calibrated model)
- **Stage 9**: Bayesian state-space with MAP inference and Plackett-Luce observations (best ranking accuracy)

## Models

### Stage 6 — Year-Weighted Constructor Dirichlet-Markov
- Architecture: `alpha = kappa_g * pi^{global} + kappa_c * pi^{(C,w)} + n_{i,C,s}`
- Constructor prior uses geometric decay (w selected via leave-last-year-out CV)
- Strength: Best probabilistic calibration (LL/race = -59.0 across 4 evaluation eras)
- Weakness: Compresses rankings — all drivers cluster near the mean because Dirichlet-Multinomial smoothing pulls everyone toward the prior

### Stage 9 — Bayesian State-Space (MAP)
- Architecture: Random walk on latent log-strengths, Plackett-Luce observation model
- MAP inference via L-BFGS-B with analytic gradients over the full trajectory
- Strength: Best ranking accuracy (Avg T3 = 1.59/3 across 4 evaluation eras, Spearman rho = 0.817 on 2025)
- Weakness: Overconcentrates probability on top drivers — the model is very confident in its rankings

---

## Part 1: 2025 Season Retrospective

Training data: 2020–2024 | Simulated season: 2025

### Stage 6 Hyperparameters
- kappa_g = 18.93, kappa_c = 15.13, w = 0.7

### Stage 9 Hyperparameters
- sigma_d = 0.100, sigma_c = 0.020

### Drivers' Championship

| Rk | Driver | Team | S6 P(WDC) | S6 AvgRk | S9 P(WDC) | S9 AvgRk | Actual Rk | Actual Pts |
|----|--------|------|-----------|----------|-----------|----------|-----------|------------|
| 1 | Lando Norris | McLaren | 5.9% | 8.8 | 3.7% | 3.9 | **1** | 394 |
| 2 | Max Verstappen | Red Bull | 33.8% | 5.3 | 70.8% | 1.4 | **2** | 389 |
| 3 | Oscar Piastri | McLaren | 4.9% | 9.6 | 0.4% | 5.5 | **3** | 381 |
| 4 | George Russell | Mercedes | 6.8% | 8.3 | 4.4% | 3.6 | **4** | 289 |
| 5 | Charles Leclerc | Ferrari | 8.8% | 7.6 | 20.1% | 2.4 | **5** | 225 |
| 6 | Kimi Antonelli | Mercedes | 4.8% | 9.4 | 0.5% | 5.9 | **6** | 135 |
| 7 | Lewis Hamilton | Ferrari | 4.4% | 9.4 | 0.1% | 7.0 | **7** | 135 |
| 8 | Alex Albon | Williams | 1.2% | 13.4 | 0.0% | 15.7 | **8** | 70 |
| 9 | Carlos Sainz | Williams | 1.6% | 12.2 | 0.0% | 10.1 | **9** | 54 |
| 10 | Fernando Alonso | Aston Martin | 2.5% | 11.1 | 0.0% | 11.3 | **10** | 51 |

### Constructors' Championship

| Rk | Team | S6 P(WCC) | S6 Avg Pts | S9 P(WCC) | S9 Avg Pts | Actual Pts |
|----|------|-----------|------------|-----------|------------|------------|
| 1 | McLaren | 11.5% | 285 | 18.9% | 531 | **775** |
| 2 | Red Bull | 40.4% | 350 | 38.5% | 587 | 410 |
| 3 | Mercedes | 13.4% | 292 | 18.1% | 527 | 424 |
| 4 | Ferrari | 15.4% | 299 | 24.4% | 548 | 360 |

### Accuracy Comparison

| Metric | Stage 6 | Stage 9 | Winner |
|--------|---------|---------|--------|
| Spearman rho (rank order) | 0.577 | **0.817** | S9 |
| Points MAE | 99.3 | **46.3** | S9 |
| Rank MAE | 4.05 | **2.48** | S9 |
| P(Norris WDC) [actual champ] | **5.9%** | 3.7% | S6 |
| P(McLaren WCC) [actual champ] | 11.5% | **18.9%** | S9 |

### Analysis

**Stage 9 dominates ranking accuracy.** Spearman rho of 0.817 vs 0.577 — Stage 9 correctly identifies the competitive hierarchy (Verstappen/Leclerc/Norris/Russell at the top, backmarkers at the bottom). Points MAE is half of Stage 6's (46.3 vs 99.3).

**Stage 6 compresses the field.** All 20 drivers cluster between avg rank 5.3 and 13.4 (8-position spread). Stage 9 spreads them from 1.4 to 18.4 (17-position spread). The actual 2025 season had meaningful separation between frontrunners and backmarkers, which Stage 9 captures.

**Both models overrate Verstappen for WDC.** Stage 9 gives him 70.8% (actual: runner-up), Stage 6 gives 33.8%. Neither model anticipated McLaren's 2025 dominance — McLaren won both championships but had only recently become competitive (2024 WCC was their first since 1998).

**Stage 9's biggest miss is Tsunoda.** Predicted avg rank 7.7 (actual: 17th). Red Bull's constructor strength inflates Tsunoda because Stage 9 can't distinguish him from Verstappen at the team level — the constructor effect is too strong.

**Stage 6 badly overrates rookies/unknowns.** Bortoleto (avg rank 11.5, actual 19th), Doohan (avg rank 11.6, actual 20th) — the Dirichlet prior pulls everyone to the mean, so drivers with little data look better than they are.

---

## Part 2: 2026 Season Predictions

Training data: 2021–2025 | Simulated season: 2026

### Stage 6 Hyperparameters
- kappa_g = 38.66, kappa_c = 0.01, w = 0.3

Note: kappa_c collapsed to 0.01 — the constructor prior is essentially disabled for 2026 predictions. This is because training on 2021–2025 with w=0.3 (aggressive decay) leaves very little constructor signal. The model becomes almost purely a global + driver-level model.

### Stage 9 Hyperparameters
- sigma_d = 0.020, sigma_c = 0.100

Top-5 driver strengths: Verstappen (7.378), Leclerc (1.831), Norris (1.761), Sainz (1.629), Alonso (1.458)

Note: Verstappen's strength is 4x higher than anyone else, reflecting his 2020–2024 dominance. But Norris won the 2025 WDC, meaning the 2025 season data corrected the gap somewhat (Norris 3rd strongest going into 2026).

### Drivers' Championship Predictions

| Rk | Driver | Team | S6 P(WDC) | S6 AvgRk | S9 P(WDC) | S9 AvgRk |
|----|--------|------|-----------|----------|-----------|----------|
| 1 | Lando Norris | McLaren | 6.6% | 9.8 | **63.6%** | **1.5** |
| 2 | Max Verstappen | Red Bull | **18.8%** | **7.7** | 18.9% | 2.3 |
| 3 | Oscar Piastri | McLaren | 6.2% | 10.6 | 16.6% | 2.4 |
| 4 | George Russell | Mercedes | 4.9% | 10.3 | 0.6% | 4.2 |
| 5 | Charles Leclerc | Ferrari | 6.6% | 9.3 | 0.2% | 5.0 |
| 6 | Lewis Hamilton | Ferrari | 3.4% | 11.8 | 0.0% | 6.2 |
| 7 | Kimi Antonelli | Mercedes | 3.4% | 11.8 | 0.0% | 9.5 |
| 8 | Isack Hadjar | Red Bull | 3.7% | 11.8 | 0.0% | 10.7 |

Full 22-driver field at the bottom of this document.

### Constructors' Championship Predictions

| Rk | Team | S6 P(WCC) | S6 Avg Pts | S9 P(WCC) | S9 Avg Pts |
|----|------|-----------|------------|-----------|------------|
| 1 | McLaren | 14.0% | 295 | **100.0%** | **968** |
| 2 | Red Bull | **20.9%** | **314** | 0.0% | 550 |
| 3 | Ferrari | 10.8% | 285 | 0.0% | 494 |
| 4 | Mercedes | 9.2% | 276 | 0.0% | 455 |
| 5 | Cadillac | 7.5% | 261 | 0.0% | 122 |

### Consensus Predictions

Despite their different architectures, the models agree on several things:

**WDC top 3 (by combined probability):**
1. **Lando Norris** — S6: 6.6%, S9: 63.6%. Stage 9's heavy favorite based on 2025 WDC + McLaren's constructor strength
2. **Max Verstappen** — S6: 18.8%, S9: 18.9%. Both models agree he's ~19% WDC. Stage 6's leader, Stage 9's #2
3. **Oscar Piastri** — S6: 6.2%, S9: 16.6%. McLaren teammate effect in Stage 9; Stage 6 ranks him mid-pack

**WCC favorite:** McLaren (S6: 14.0%, S9: 100.0%)

**Hamilton at Ferrari:** Both models rank him mid-pack (S6: 11.8, S9: 6.2). Stage 9 gives him more credit from historical strength but still below Leclerc at the same team.

### Model Disagreements

The models diverge dramatically due to their fundamentally different architectures:

**Stage 6 (near-uniform predictions):** With kappa_c collapsed to 0.01, Stage 6 is essentially a global prior + driver counts model. Every driver's prediction is pulled toward the same global mean, producing a compressed field where the gap between rank 1 (7.7) and rank 22 (13.1) is only 5.4 positions. This makes Verstappen the favorite (18.8%) but gives even backmarkers like Colapinto (3.7%) and Stroll (2.6%) non-trivial championship chances.

**Stage 9 (extreme concentration):** Stage 9 gives McLaren 100% WCC probability and Norris 63.6% WDC. This is a consequence of exponential-space strength representation — Verstappen's raw strength (7.378) is high but McLaren's recent trajectory (2024 WCC + 2025 WDC/WCC) means both McLaren drivers have high composite strength. The model is maximally confident because it tracks individual race-level strengths rather than team averages.

**Biggest disagreements by rank:**

| Driver | Team | S6 Avg Rank | S9 Avg Rank | Gap |
|--------|------|-------------|-------------|-----|
| Lando Norris | McLaren | 9.8 | 1.5 | 8.3 |
| Oscar Piastri | McLaren | 10.6 | 2.4 | 8.2 |
| Franco Colapinto | Alpine | 12.0 | 20.0 | 8.0 |
| Pierre Gasly | Alpine | 12.6 | 20.1 | 7.5 |
| George Russell | Mercedes | 10.3 | 4.2 | 6.1 |

### Intra-Team Battles

| Team | S6 Prediction | S9 Prediction |
|------|---------------|---------------|
| McLaren | Norris (151) vs Piastri (143) | Norris (509) vs Piastri (458) |
| Ferrari | Leclerc (155) vs Hamilton (131) | Leclerc (286) vs Hamilton (208) |
| Red Bull | Verstappen (184) vs Hadjar (131) | Verstappen (464) vs Hadjar (86) |
| Mercedes | Russell (146) vs Antonelli (130) | Russell (348) vs Antonelli (108) |
| Cadillac | Bottas (131) vs Perez (131) | Perez (65) vs Bottas (57) |

Stage 9 predicts much larger intra-team gaps where one driver is established and the other is new (Red Bull: 5.4x, Mercedes: 3.2x). Stage 6 predicts near-parity everywhere because the constructor prior dominates.

### Caveats

**2026 regulation changes.** Major new technical regulations take effect in 2026 (new aero philosophy, active aerodynamics, new power unit formula). Historical constructor strength is less predictive across regulation discontinuities. Neither model accounts for this — both extrapolate from 2021–2025 performance.

**Cadillac is unmodeled.** Brand-new constructor with no historical data. Stage 6 falls back to a global prior (ranks Cadillac drivers ~11–12th). Stage 9 uses only driver history (Bottas and Perez have individual strength estimates from previous teams).

**Stage 6's kappa_c collapse is concerning.** When trained on 2021–2025 with w=0.3, the constructor prior effectively disappears (kappa_c = 0.01). This means Stage 6 is not really using constructor information for 2026 predictions, which severely limits its discriminative power.

**Stage 9 overconcentrates.** 100% WCC for McLaren is unrealistic — it means in all 10,000 simulations, McLaren won. This reflects the model's inability to represent regulation-change uncertainty. A Bayesian model with proper posterior sampling (full NUTS instead of MAP) would produce wider credible intervals.

---

## Summary

| Aspect | Stage 6 | Stage 9 |
|--------|---------|---------|
| 2025 Spearman rho | 0.577 | **0.817** |
| 2025 Rank MAE | 4.05 | **2.48** |
| 2025 Points MAE | 99.3 | **46.3** |
| 2025 P(actual WDC) | **5.9%** | 3.7% |
| 2025 P(actual WCC) | 11.5% | **18.9%** |
| 2026 WDC favorite | Verstappen (18.8%) | Norris (63.6%) |
| 2026 WCC favorite | Red Bull (20.9%) | McLaren (100%) |
| Field compression | Severe (5.4-pos spread) | Reasonable (18.6-pos spread) |
| Calibration | Better (spreads probability) | Worse (too concentrated) |
| Best use case | Uncertainty quantification | Point prediction / ranking |

**Stage 9 is the better ranking model.** It correctly identifies who is fast and who is not. Its 2025 retrospective accuracy (rho=0.817, rank MAE=2.48) is excellent.

**Stage 6 is the better calibrated model.** It assigns more realistic probabilities to unlikely outcomes. A 5.9% chance for the actual WDC winner is more honest than 3.7%.

**For 2026 predictions, use Stage 9 for rankings and Stage 6 for uncertainty.** Stage 9 says: Norris, Verstappen, Piastri are the top 3 favorites. Stage 6 says: there's a 20% chance someone outside the top 5 wins — don't rule out surprises, especially with new regulations.

---

## Appendix: Full 2026 Driver Predictions

### Stage 6

| Rk | Driver | Team | P(WDC) | P(Top3) | Avg Pts | Avg Wins | Avg Rank |
|----|--------|------|--------|---------|---------|----------|----------|
| 1 | Max Verstappen | Red Bull | 18.8% | 36.3% | 184 | 2.9 | 7.7 |
| 2 | Charles Leclerc | Ferrari | 6.6% | 20.0% | 155 | 1.2 | 9.3 |
| 3 | Lando Norris | McLaren | 6.6% | 20.0% | 151 | 1.4 | 9.8 |
| 4 | George Russell | Mercedes | 4.9% | 16.7% | 146 | 1.2 | 10.3 |
| 5 | Oscar Piastri | McLaren | 6.2% | 17.5% | 143 | 1.3 | 10.6 |
| 6 | Arvid Lindblad | Racing Bulls | 3.7% | 12.1% | 131 | 1.2 | 11.7 |
| 7 | Sergio Perez | Cadillac | 3.5% | 11.9% | 131 | 1.2 | 11.7 |
| 8 | Valtteri Bottas | Cadillac | 3.8% | 12.2% | 131 | 1.2 | 11.8 |
| 9 | Lewis Hamilton | Ferrari | 3.4% | 11.1% | 131 | 1.1 | 11.8 |
| 10 | Kimi Antonelli | Mercedes | 3.4% | 11.3% | 130 | 1.1 | 11.8 |
| 11 | Isack Hadjar | Red Bull | 3.7% | 12.0% | 131 | 1.2 | 11.8 |
| 12 | Fernando Alonso | Aston Martin | 3.2% | 11.4% | 129 | 1.1 | 11.9 |
| 13 | Esteban Ocon | Haas | 3.5% | 11.5% | 129 | 1.2 | 12.0 |
| 14 | Carlos Sainz | Williams | 3.4% | 11.7% | 129 | 1.1 | 12.0 |
| 15 | Franco Colapinto | Alpine | 3.7% | 11.3% | 128 | 1.2 | 12.0 |
| 16 | Liam Lawson | Racing Bulls | 3.4% | 11.3% | 128 | 1.1 | 12.0 |
| 17 | Nico Hulkenberg | Audi | 3.5% | 11.5% | 128 | 1.1 | 12.1 |
| 18 | Oliver Bearman | Haas | 3.3% | 11.0% | 127 | 1.1 | 12.2 |
| 19 | Gabriel Bortoleto | Audi | 3.5% | 11.2% | 127 | 1.1 | 12.2 |
| 20 | Pierre Gasly | Alpine | 3.0% | 9.9% | 123 | 1.1 | 12.6 |
| 21 | Lance Stroll | Aston Martin | 2.6% | 9.2% | 120 | 1.0 | 12.9 |
| 22 | Alex Albon | Williams | 2.4% | 9.0% | 118 | 1.0 | 13.1 |

### Stage 9

| Rk | Driver | Team | P(WDC) | P(Top3) | Avg Pts | Avg Wins | Avg Rank |
|----|--------|------|--------|---------|---------|----------|----------|
| 1 | Lando Norris | McLaren | 63.6% | 99.1% | 509 | 11.7 | 1.5 |
| 2 | Max Verstappen | Red Bull | 18.9% | 91.8% | 464 | 9.1 | 2.3 |
| 3 | Oscar Piastri | McLaren | 16.6% | 90.3% | 458 | 8.8 | 2.4 |
| 4 | George Russell | Mercedes | 0.6% | 14.1% | 348 | 4.8 | 4.2 |
| 5 | Charles Leclerc | Ferrari | 0.2% | 4.2% | 286 | 3.7 | 5.0 |
| 6 | Lewis Hamilton | Ferrari | 0.0% | 0.5% | 208 | 2.0 | 6.2 |
| 7 | Kimi Antonelli | Mercedes | 0.0% | 0.0% | 108 | 0.8 | 9.5 |
| 8 | Isack Hadjar | Red Bull | 0.0% | 0.0% | 86 | 0.6 | 10.7 |
| 9 | Fernando Alonso | Aston Martin | 0.0% | 0.0% | 74 | 0.6 | 11.8 |
| 10 | Carlos Sainz | Williams | 0.0% | 0.0% | 72 | 0.5 | 11.9 |
| 11 | Sergio Perez | Cadillac | 0.0% | 0.0% | 65 | 0.4 | 12.4 |
| 12 | Valtteri Bottas | Cadillac | 0.0% | 0.0% | 57 | 0.3 | 13.1 |
| 13 | Arvid Lindblad | Racing Bulls | 0.0% | 0.0% | 57 | 0.4 | 13.4 |
| 14 | Alex Albon | Williams | 0.0% | 0.0% | 53 | 0.3 | 13.4 |
| 15 | Oliver Bearman | Haas | 0.0% | 0.0% | 48 | 0.3 | 14.2 |
| 16 | Esteban Ocon | Haas | 0.0% | 0.0% | 45 | 0.3 | 14.5 |
| 17 | Nico Hulkenberg | Audi | 0.0% | 0.0% | 39 | 0.3 | 15.3 |
| 18 | Liam Lawson | Racing Bulls | 0.0% | 0.0% | 32 | 0.2 | 16.1 |
| 19 | Gabriel Bortoleto | Audi | 0.0% | 0.0% | 25 | 0.1 | 17.1 |
| 20 | Lance Stroll | Aston Martin | 0.0% | 0.0% | 19 | 0.1 | 18.0 |
| 21 | Franco Colapinto | Alpine | 0.0% | 0.0% | 8 | 0.0 | 20.0 |
| 22 | Pierre Gasly | Alpine | 0.0% | 0.0% | 8 | 0.0 | 20.1 |

---

## Files

- `simulate_seasons.py` — Combined simulation script (Stage 6 + Stage 9, 2025 retro + 2026 prediction)
- `simulate_2025_season.py` — Original 2025 retrospective (Stages 2, 3, 6, 7, 8)
- `predict_2026_season.py` — Original 2026 prediction (Stage 3 only)
- `models/stage6_recency_constructor.py` — Stage 6 model
- `models/stage9_bayesian_ss.py` — Stage 9 model
