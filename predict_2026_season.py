"""
Monte Carlo simulation of the 2026 F1 season using Stage 3 Markov model.

Simulates 10,000 seasons to predict:
  - Drivers' Championship probabilities
  - Constructors' Championship probabilities
  - Positive surprise / flop / crazy prediction
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent / "models"))

from stage3_constructor import (
    F1DataLoader as S3Loader,
    ConstructorPooledDirichletF1,
    prepare_transitions as s3_prepare,
    START, N_OUTCOMES,
)


DATA_DIR = Path(__file__).parent / "data"
N_SIMS = 10000
N_RACES = 24
N_SPRINTS = 6

# Points: P1=25, P2=18, ..., P10=1, P11+=0, DNF=0
RACE_POINTS = np.zeros(N_OUTCOMES, dtype=int)
for pos, pts in {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}.items():
    RACE_POINTS[pos] = pts

SPRINT_POINTS = np.zeros(N_OUTCOMES, dtype=int)
for pos, pts in {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}.items():
    SPRINT_POINTS[pos] = pts

# 2026 driver lineup
DRIVERS_2026 = {
    846: (1, "Lando Norris"),
    857: (1, "Oscar Piastri"),
    1:   (6, "Lewis Hamilton"),
    844: (6, "Charles Leclerc"),
    847: (131, "George Russell"),
    863: (131, "Kimi Antonelli"),
    830: (9, "Max Verstappen"),
    864: (9, "Isack Hadjar"),
    4:   (117, "Fernando Alonso"),
    840: (117, "Lance Stroll"),
    832: (3, "Carlos Sainz"),
    848: (3, "Alex Albon"),
    861: (214, "Franco Colapinto"),
    842: (214, "Pierre Gasly"),
    839: (210, "Esteban Ocon"),
    860: (210, "Oliver Bearman"),
    807: (15, "Nico Hulkenberg"),
    865: (15, "Gabriel Bortoleto"),
    859: (215, "Liam Lawson"),
    866: (215, "Arvid Lindblad"),
    822: (216, "Valtteri Bottas"),
    815: (216, "Sergio Perez"),
}

CONSTRUCTOR_NAMES = {
    1: "McLaren", 6: "Ferrari", 131: "Mercedes", 9: "Red Bull",
    117: "Aston Martin", 3: "Williams", 214: "Alpine",
    210: "Haas", 15: "Audi", 215: "Racing Bulls", 216: "Cadillac",
}


def precompute_probs(model):
    """Precompute P(next | prev) for all drivers and all prev states."""
    driver_ids = list(DRIVERS_2026.keys())
    # prob_cache[driver_idx][prev_state] = probability vector of length N_OUTCOMES
    prob_cache = {}
    for i, did in enumerate(driver_ids):
        cid = DRIVERS_2026[did][0]
        cache = np.zeros((START + 1, N_OUTCOMES))  # prev can be 0..START
        for prev in range(START + 1):
            if did in model.driver_constructor_counts_:
                cache[prev] = model.predict_proba(did, prev, constructor_id=cid)
            else:
                cache[prev] = model.predict_proba_new_driver(prev, constructor_id=cid)
        prob_cache[did] = cache
    return prob_cache


def main():
    print("=" * 70)
    print("2026 F1 SEASON SIMULATION")
    print(f"Monte Carlo: {N_SIMS:,} simulated seasons")
    print(f"Model: Stage 3 (Constructor + Driver) trained on 2021-2025")
    print(f"Races: {N_RACES} GPs + {N_SPRINTS} sprints")
    print("=" * 70)

    # Train model
    print("\nTraining model...")
    loader = S3Loader(DATA_DIR)
    df = loader.load_merged(min_year=2021, max_year=2025)
    prev, next_, meta = s3_prepare(df)

    model = ConstructorPooledDirichletF1(
        prior_alpha_global=1.0,
        prior_alpha_constructor=1.0,
        kappa_init=(10.0, 10.0),
        kappa_bounds=((0.1, 500.0), (0.01, 500.0)),
    )
    model.fit(prev, next_, meta)
    print(f"Kappa_g: {model.kappa_g_:.2f}, Kappa_c: {model.kappa_c_:.2f}")

    # Precompute all probability vectors
    print("Precomputing probability distributions...")
    prob_cache = precompute_probs(model)

    driver_ids = list(DRIVERS_2026.keys())
    n_drivers = len(driver_ids)
    did_to_idx = {did: i for i, did in enumerate(driver_ids)}
    cid_list = [DRIVERS_2026[did][0] for did in driver_ids]

    # Run simulations
    print(f"Simulating {N_SIMS:,} seasons...")
    rng = np.random.default_rng(42)

    # Accumulators
    driver_total_pts = np.zeros((N_SIMS, n_drivers))
    driver_total_wins = np.zeros((N_SIMS, n_drivers), dtype=int)
    constructor_total_pts = np.zeros((N_SIMS, len(CONSTRUCTOR_NAMES)))
    cid_list_unique = sorted(CONSTRUCTOR_NAMES.keys())
    cid_to_cidx = {cid: i for i, cid in enumerate(cid_list_unique)}

    for sim in range(N_SIMS):
        if (sim + 1) % 2000 == 0:
            print(f"  ... {sim + 1:,}/{N_SIMS:,}")

        prev_pos = np.full(n_drivers, START, dtype=int)

        for race_num in range(N_RACES):
            # Sample results for all drivers
            results = np.zeros(n_drivers, dtype=int)
            for i, did in enumerate(driver_ids):
                probs = prob_cache[did][prev_pos[i]]
                results[i] = rng.choice(N_OUTCOMES, p=probs)

            # Award points (no collision resolution — independent samples)
            for i in range(n_drivers):
                pos = results[i]
                pts = RACE_POINTS[pos]
                driver_total_pts[sim, i] += pts
                cidx = cid_to_cidx[cid_list[i]]
                constructor_total_pts[sim, cidx] += pts
                if pos == 1:
                    driver_total_wins[sim, i] += 1

            # Update prev_pos
            prev_pos = results

            # Sprint (first N_SPRINTS races)
            if race_num < N_SPRINTS:
                for i, did in enumerate(driver_ids):
                    probs = prob_cache[did][prev_pos[i]]
                    sprint_pos = rng.choice(N_OUTCOMES, p=probs)
                    pts = SPRINT_POINTS[sprint_pos]
                    driver_total_pts[sim, i] += pts
                    cidx = cid_to_cidx[cid_list[i]]
                    constructor_total_pts[sim, cidx] += pts

    # =====================================================================
    # RESULTS
    # =====================================================================

    # Drivers' championship
    driver_champ_idx = np.argmax(driver_total_pts, axis=1)
    driver_champ_count = np.bincount(driver_champ_idx, minlength=n_drivers)

    # Championship top 3
    driver_top3_count = np.zeros(n_drivers, dtype=int)
    for sim in range(N_SIMS):
        top3_idx = np.argsort(-driver_total_pts[sim])[:3]
        for idx in top3_idx:
            driver_top3_count[idx] += 1

    # Constructors' championship
    constr_champ_idx = np.argmax(constructor_total_pts, axis=1)
    constr_champ_count = np.bincount(constr_champ_idx,
                                      minlength=len(cid_list_unique))

    print("\n" + "=" * 70)
    print("DRIVERS' CHAMPIONSHIP PREDICTIONS")
    print("=" * 70)

    driver_stats = []
    for i, did in enumerate(driver_ids):
        cid, name = DRIVERS_2026[did]
        driver_stats.append({
            "driver_id": did,
            "name": name,
            "team": CONSTRUCTOR_NAMES[cid],
            "P(WDC)": driver_champ_count[i] / N_SIMS,
            "P(top3)": driver_top3_count[i] / N_SIMS,
            "avg_pts": driver_total_pts[:, i].mean(),
            "std_pts": driver_total_pts[:, i].std(),
            "avg_wins": driver_total_wins[:, i].mean(),
        })

    ds = pd.DataFrame(driver_stats).sort_values("P(WDC)", ascending=False)

    print(f"\n{'Rank':<5} {'Driver':<22} {'Team':<15} {'P(WDC)':>8} "
          f"{'P(top3)':>8} {'Avg Pts':>8} {'Avg Wins':>9}")
    print("-" * 82)
    for rank, (_, row) in enumerate(ds.iterrows(), 1):
        print(f"{rank:<5} {row['name']:<22} {row['team']:<15} "
              f"{row['P(WDC)']:>8.1%} {row['P(top3)']:>8.1%} "
              f"{row['avg_pts']:>8.1f} {row['avg_wins']:>9.1f}")

    print("\n" + "=" * 70)
    print("CONSTRUCTORS' CHAMPIONSHIP PREDICTIONS")
    print("=" * 70)

    constr_stats = []
    for j, cid in enumerate(cid_list_unique):
        constr_stats.append({
            "team": CONSTRUCTOR_NAMES[cid],
            "P(WCC)": constr_champ_count[j] / N_SIMS,
            "avg_pts": constructor_total_pts[:, j].mean(),
        })

    cs = pd.DataFrame(constr_stats).sort_values("P(WCC)", ascending=False)

    print(f"\n{'Rank':<5} {'Team':<20} {'P(WCC)':>8} {'Avg Pts':>10}")
    print("-" * 45)
    for rank, (_, row) in enumerate(cs.iterrows(), 1):
        print(f"{rank:<5} {row['team']:<20} {row['P(WCC)']:>8.1%} "
              f"{row['avg_pts']:>10.1f}")

    # =====================================================================
    # NARRATIVE PREDICTIONS
    # =====================================================================
    print("\n" + "=" * 70)
    print("2026 SEASON PREDICTIONS")
    print("=" * 70)

    print("\n--- DRIVERS' CHAMPION ---")
    top = ds.iloc[0]
    print(f"  {top['name']} ({top['team']}) — {top['P(WDC)']:.1%}")
    if len(ds) > 1:
        r = ds.iloc[1]
        print(f"  Runner-up: {r['name']} ({r['team']}) — {r['P(WDC)']:.1%}")

    print("\n--- CONSTRUCTORS' CHAMPION ---")
    top_c = cs.iloc[0]
    print(f"  {top_c['team']} — {top_c['P(WCC)']:.1%}")

    # Good surprise: mid-tier driver with best championship top-3 rate
    print("\n--- GOOD SURPRISE ---")
    mid = ds[(ds["P(WDC)"] > 0.001) & (ds["P(WDC)"] < ds.iloc[3]["P(WDC)"])]
    if len(mid) > 0:
        surp = mid.sort_values("P(top3)", ascending=False).iloc[0]
        print(f"  {surp['name']} ({surp['team']})")
        print(f"  P(WDC)={surp['P(WDC)']:.1%}, P(champ top 3)={surp['P(top3)']:.1%}, "
              f"avg wins={surp['avg_wins']:.1f}")

    # Flop: top contender with highest relative variance
    print("\n--- FLOP ---")
    top6 = ds.head(6).copy()
    top6["cv"] = top6["std_pts"] / top6["avg_pts"]
    flop = top6.sort_values("cv", ascending=False).iloc[0]
    print(f"  {flop['name']} ({flop['team']})")
    print(f"  P(WDC)={flop['P(WDC)']:.1%}, but std={flop['std_pts']:.0f} pts "
          f"(CV={flop['cv']:.2f})")

    # Crazy: backmarker with any race win
    print("\n--- CRAZY PREDICTION ---")
    bottom = ds.tail(10).sort_values("avg_wins", ascending=False)
    if len(bottom) > 0:
        crazy = bottom.iloc[0]
        pct_win = (driver_total_wins[:, did_to_idx[crazy["driver_id"]]] > 0).mean()
        print(f"  {crazy['name']} ({crazy['team']}) wins a Grand Prix!")
        print(f"  P(at least 1 win in season) = {pct_win:.1%}")

    # Intra-team battles
    print("\n--- INTRA-TEAM BATTLES (avg season pts) ---")
    teams = defaultdict(list)
    for _, row in ds.iterrows():
        teams[row["team"]].append(row)
    for team in ["Mercedes", "Ferrari", "Red Bull", "McLaren",
                  "Aston Martin", "Cadillac"]:
        if team in teams and len(teams[team]) == 2:
            d1, d2 = teams[team]
            print(f"  {team}: {d1['name']} ({d1['avg_pts']:.0f}) vs "
                  f"{d2['name']} ({d2['avg_pts']:.0f})")


if __name__ == "__main__":
    main()
