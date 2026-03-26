"""
2026 Chinese Grand Prix (Shanghai) Predictions
===============================================
Uses Stage 6 (Year-Weighted Constructor) and Stage 9 (Bayesian State-Space MAP)
with observed 2026 Australian GP results incorporated.

Training: 2015-2025 (11 years)
Observed: Round 1 (Australia) — Russell P1, Antonelli P2, Piastri P3
Monte Carlo: 10,000 simulations per model

Run: python3 predict_2026_shanghai.py
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def main():
    sim_path = DATA_DIR / "sim_2026_r02_china.json"
    with open(sim_path) as f:
        data = json.load(f)

    s6 = {d["abbreviation"]: d for d in data["models"]["stage6"]["drivers"]}
    s9 = {d["abbreviation"]: d for d in data["models"]["stage9"]["drivers"]}
    comp = {d["abbreviation"]: d for d in data["models"]["composite"]["drivers"]}

    # Sorted by composite podium probability
    comp_sorted = sorted(comp.values(), key=lambda x: -x["p_podium"])
    s6_sorted = sorted(s6.values(), key=lambda x: -x["p_podium"])
    s9_sorted = sorted(s9.values(), key=lambda x: -x["p_podium"])

    print("=" * 72)
    print("  2026 CHINESE GRAND PRIX — SHANGHAI PREDICTIONS")
    print("  Race 2 of 24 | Shanghai International Circuit | 2026-03-15")
    print("=" * 72)
    print()
    print("Models: Stage 6 (Dirichlet-Multinomial) + Stage 9 (Bayesian SS MAP)")
    print("Training: 2015-2025 | Observed: R1 Australia results incorporated")
    print(f"Simulations: {data['n_sims']:,} per model")
    print()

    # --- POLE POSITION ---
    print("-" * 72)
    print("  POLE POSITION PREDICTION")
    print("-" * 72)
    # Use composite P(win) as proxy for qualifying pace
    top_quali = comp_sorted[:5]
    print(f"  {'Driver':<20} {'Team':<15} {'S6 P(Win)':>10} {'S9 P(Win)':>10} {'Blend':>10}")
    for d in top_quali:
        abbr = d["abbreviation"]
        print(f"  {d['name']:<20} {d['team']:<15} "
              f"{s6[abbr]['p_win']:>9.1%} {s9[abbr]['p_win']:>9.1%} {d['p_win']:>9.1%}")
    print()
    print(f"  >>> POLE: George Russell (Mercedes) — 38.9% P(win)")
    print(f"      Russell qualified P1 at Australia; Mercedes clearly the")
    print(f"      fastest in qualifying trim. Stage 9 gives him 49.6% win prob.")
    print()

    # --- RACE WINNER (P1) ---
    print("-" * 72)
    print("  P1 — RACE WINNER")
    print("-" * 72)
    print(f"  >>> George Russell (Mercedes)")
    print(f"      Stage 6: 27.6% | Stage 9: 49.6% | Composite: 38.9%")
    print(f"      Won Australia from pole by 3.5s. Mercedes dominant on race pace.")
    print(f"      Shanghai's mix of high-speed straights and technical sector 2")
    print(f"      suits the W17's aero efficiency.")
    print()

    # --- PODIUM (P2, P3) ---
    print("-" * 72)
    print("  P2 & P3 — PODIUM PREDICTION")
    print("-" * 72)
    print(f"  {'Driver':<20} {'Team':<15} {'P(Podium)':>10} {'E[Pos]':>8}")
    for d in comp_sorted[:6]:
        print(f"  {d['name']:<20} {d['team']:<15} {d['p_podium']:>9.1%} {d['e_pos']:>7.1f}")
    print()
    print(f"  >>> P2: Oscar Piastri (McLaren)")
    print(f"      Stage 9 loves Piastri (70.5% podium) after his strong P3 in Aus.")
    print(f"      Won Shanghai in 2025. Track specialist.")
    print()
    print(f"  >>> P3: Lando Norris (McLaren)")
    print(f"      Stage 9: 73.2% podium, E[Pos]=2.6. McLaren 1-2 very possible.")
    print(f"      NOR finished P2 at Shanghai 2025. Consistent frontrunner.")
    print()

    # --- SPRINT WINNER ---
    print("-" * 72)
    print("  SPRINT RACE WINNER")
    print("-" * 72)
    print(f"  >>> Lando Norris (McLaren)")
    print(f"      Sprint races reward pure pace without strategic variance.")
    print(f"      Norris has 16.2% composite win probability and McLaren's")
    print(f"      short-stint pace is lethal. Piastri won the 2025 Shanghai")
    print(f"      sprint; expect McLaren to be the sprint threat.")
    print(f"      Alternative: Russell if he nails sprint quali.")
    print()

    # --- SURPRISE ---
    print("-" * 72)
    print("  THE SURPRISE")
    print("-" * 72)
    max_var = sorted(comp.values(),
                     key=lambda x: x["percentiles"]["p95"] - x["percentiles"]["p5"],
                     reverse=True)
    # Find midfielders with upside
    print(f"  >>> Liam Lawson (Racing Bulls) — Points finish or better")
    print(f"      E[Pos]=9.4, but P(points)=46.1% and p5=3 (!)")
    print(f"      Lawson finished P8 in Australia. Racing Bulls had pace.")
    print(f"      Shanghai's long back straight could help the VCARB's")
    print(f"      top-speed advantage. A P6-P7 finish would be a statement.")
    print()

    # --- FLOP ---
    print("-" * 72)
    print("  THE FLOP")
    print("-" * 72)
    print(f"  >>> Max Verstappen (Red Bull) — Expected to struggle again")
    print(f"      Stage 6: E[Pos]=9.3 | Stage 9: E[Pos]=4.3 | Composite: 6.9")
    print(f"      Qualified P20 (!!) and finished P9 in Australia.")
    print(f"      Red Bull's RB22 is not the dominant car anymore.")
    print(f"      P(DNF)=18.0% — reliability also a concern.")
    print(f"      The 4x WDC could easily finish outside the top 5 again.")
    print()

    # --- SOMETHING CRAZY ---
    print("-" * 72)
    print("  SOMETHING CRAZY")
    print("-" * 72)
    print(f"  >>> Cadillac double-points finish")
    print(f"      Bottas P(points)=26.9%, Perez P(points)=15.8%")
    print(f"      A brand-new team scoring points in only their second race")
    print(f"      would be wild. Bottas has massive variance (p5=3, p95=20).")
    print(f"      If Shanghai gets a safety car, Bottas knows how to capitalize.")
    print(f"      The Finnish veteran could channel his 10 wins' experience")
    print(f"      to drag the Cadillac into the points. Perez finished P19 in")
    print(f"      Australia but the car has development potential.")
    print()
    print(f"      Alternatively: Verstappen DNF + Antonelli wins = new era")
    print(f"      confirmed. VER P(DNF)=18%, ANT P(win)=20.3%.")
    print()

    # --- FULL GRID ---
    print("=" * 72)
    print("  FULL GRID PREDICTION (Composite: Stage 6 + Stage 9)")
    print("=" * 72)
    print(f"  {'Pos':>4} {'Driver':<20} {'Team':<15} {'P(Win)':>8} {'P(Pod)':>8} {'P(Pts)':>8} {'E[Pos]':>8}")
    print(f"  {'-'*68}")
    for j, d in enumerate(comp_sorted):
        print(f"  {j+1:>4} {d['name']:<20} {d['team']:<15} "
              f"{d['p_win']:>7.1%} {d['p_podium']:>7.1%} {d['p_points']:>7.1%} {d['e_pos']:>7.1f}")

    print()
    print("=" * 72)
    print("  MODEL PARAMETERS")
    print("=" * 72)
    s6p = data["models"]["stage6"]["params"]
    s9p = data["models"]["stage9"]["params"]
    print(f"  Stage 6: kappa_g={s6p['kappa_g']:.2f}, kappa_c={s6p['kappa_c']:.2f}, w={s6p['w']:.2f}")
    print(f"  Stage 9: sigma_d={s9p['sigma_d']:.3f}, sigma_c={s9p['sigma_c']:.3f}")
    print(f"  Composite: 50/50 blend (Stage 8 excluded — degenerate)")
    print()


if __name__ == "__main__":
    main()
