"""
Retrospective Monte Carlo simulation of the 2025 F1 season.

Trains Stages 2, 3, 6, 7 (HMM), and 8 (Plackett-Luce) on 2020-2024 data,
simulates 10,000 seasons each, and compares simulated championship outcomes
against actual 2025 results.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent / "models"))

from stage2_driver_pooling import (
    F1DataLoader as S2Loader,
    PartialPooledDirichletF1,
    prepare_transitions as s2_prepare,
    START, N_OUTCOMES,
)
from stage3_constructor import (
    F1DataLoader as S3Loader,
    ConstructorPooledDirichletF1,
    prepare_transitions as s3_prepare,
)
from stage6_recency_constructor import (
    F1DataLoader as S6Loader,
    RecencyConstructorDirichletF1,
    prepare_transitions as s6_prepare,
)
from stage7_hmm import HiddenMarkovF1
from stage8_plackett_luce import TimeVaryingPlackettLuceF1


DATA_DIR = Path(__file__).parent / "data"
N_SIMS = 10000
N_RACES = 24
N_SPRINTS = 6

# Points: P1=25, P2=18, ..., P10=1, P11+=0, DNF=0
RACE_POINTS = np.zeros(N_OUTCOMES, dtype=int)
for pos, pts in {1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
                 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}.items():
    RACE_POINTS[pos] = pts

SPRINT_POINTS = np.zeros(N_OUTCOMES, dtype=int)
for pos, pts in {1: 8, 2: 7, 3: 6, 4: 5, 5: 4,
                 6: 3, 7: 2, 8: 1}.items():
    SPRINT_POINTS[pos] = pts

# 2025 driver lineup (primary constructor for the season)
# Based on actual 2025 race entries
DRIVERS_2025 = {
    846: (1,   "Lando Norris"),         # McLaren
    857: (1,   "Oscar Piastri"),        # McLaren
    830: (9,   "Max Verstappen"),       # Red Bull
    852: (9,   "Yuki Tsunoda"),         # Red Bull (replaced by Lawson mid-season for RB)
    847: (131, "George Russell"),       # Mercedes
    863: (131, "Kimi Antonelli"),       # Mercedes
    844: (6,   "Charles Leclerc"),      # Ferrari
    1:   (6,   "Lewis Hamilton"),       # Ferrari
    4:   (117, "Fernando Alonso"),      # Aston Martin
    840: (117, "Lance Stroll"),         # Aston Martin
    832: (3,   "Carlos Sainz"),         # Williams
    848: (3,   "Alex Albon"),           # Williams
    842: (214, "Pierre Gasly"),         # Alpine
    862: (214, "Jack Doohan"),          # Alpine
    839: (210, "Esteban Ocon"),         # Haas
    860: (210, "Oliver Bearman"),       # Haas
    807: (15,  "Nico Hulkenberg"),      # Kick Sauber
    865: (15,  "Gabriel Bortoleto"),    # Kick Sauber
    859: (215, "Liam Lawson"),          # Racing Bulls
    864: (215, "Isack Hadjar"),         # Racing Bulls
}

CONSTRUCTOR_NAMES = {
    1: "McLaren", 6: "Ferrari", 131: "Mercedes", 9: "Red Bull",
    117: "Aston Martin", 3: "Williams", 214: "Alpine",
    210: "Haas", 15: "Kick Sauber", 215: "Racing Bulls",
}

# Actual 2025 season results
ACTUAL_2025 = {
    846: {"name": "Lando Norris",      "team": "McLaren",       "pts": 394, "wins": 7, "rank": 1},
    830: {"name": "Max Verstappen",    "team": "Red Bull",      "pts": 389, "wins": 8, "rank": 2},
    857: {"name": "Oscar Piastri",     "team": "McLaren",       "pts": 381, "wins": 7, "rank": 3},
    847: {"name": "George Russell",    "team": "Mercedes",      "pts": 289, "wins": 2, "rank": 4},
    844: {"name": "Charles Leclerc",   "team": "Ferrari",       "pts": 225, "wins": 0, "rank": 5},
    863: {"name": "Kimi Antonelli",    "team": "Mercedes",      "pts": 135, "wins": 0, "rank": 6},
    1:   {"name": "Lewis Hamilton",    "team": "Ferrari",       "pts": 135, "wins": 0, "rank": 7},
    848: {"name": "Alex Albon",        "team": "Williams",      "pts": 70,  "wins": 0, "rank": 8},
    832: {"name": "Carlos Sainz",      "team": "Williams",      "pts": 54,  "wins": 0, "rank": 9},
    4:   {"name": "Fernando Alonso",   "team": "Aston Martin",  "pts": 51,  "wins": 0, "rank": 10},
    807: {"name": "Nico Hulkenberg",   "team": "Kick Sauber",   "pts": 51,  "wins": 0, "rank": 11},
    864: {"name": "Isack Hadjar",      "team": "Racing Bulls",  "pts": 50,  "wins": 0, "rank": 12},
    860: {"name": "Oliver Bearman",    "team": "Haas",          "pts": 39,  "wins": 0, "rank": 13},
    859: {"name": "Liam Lawson",       "team": "Racing Bulls",  "pts": 38,  "wins": 0, "rank": 14},
    839: {"name": "Esteban Ocon",      "team": "Haas",          "pts": 34,  "wins": 0, "rank": 15},
    840: {"name": "Lance Stroll",      "team": "Aston Martin",  "pts": 29,  "wins": 0, "rank": 16},
    852: {"name": "Yuki Tsunoda",      "team": "Red Bull",      "pts": 21,  "wins": 0, "rank": 17},
    842: {"name": "Pierre Gasly",      "team": "Alpine",        "pts": 20,  "wins": 0, "rank": 18},
    865: {"name": "Gabriel Bortoleto", "team": "Kick Sauber",   "pts": 19,  "wins": 0, "rank": 19},
    862: {"name": "Jack Doohan",       "team": "Alpine",        "pts": 0,   "wins": 0, "rank": 20},
}

ACTUAL_WCC = {
    "McLaren": 775, "Mercedes": 424, "Red Bull": 410, "Ferrari": 360,
    "Williams": 124, "Racing Bulls": 88, "Aston Martin": 80,
    "Haas": 73, "Kick Sauber": 70, "Alpine": 20,
}


def precompute_probs_s2(model, drivers):
    """Precompute probs for Stage 2 (no constructor)."""
    prob_cache = {}
    for did in drivers:
        cache = np.zeros((START + 1, N_OUTCOMES))
        for prev in range(START + 1):
            if did in model.driver_counts_:
                cache[prev] = model.predict_proba(did, prev)
            else:
                cache[prev] = model.predict_proba_new_driver(prev)
        prob_cache[did] = cache
    return prob_cache


def precompute_probs_s3(model, drivers):
    """Precompute probs for Stage 3 (with constructor)."""
    prob_cache = {}
    for did in drivers:
        cid = drivers[did][0]
        cache = np.zeros((START + 1, N_OUTCOMES))
        for prev in range(START + 1):
            if did in model.driver_constructor_counts_:
                cache[prev] = model.predict_proba(did, prev, constructor_id=cid)
            else:
                cache[prev] = model.predict_proba_new_driver(prev, constructor_id=cid)
        prob_cache[did] = cache
    return prob_cache


def precompute_probs_s7(model, drivers):
    """Precompute probs for Stage 7 (HMM, with constructor)."""
    prob_cache = {}
    for did in drivers:
        cid = drivers[did][0]
        cache = np.zeros((START + 1, N_OUTCOMES))
        for prev in range(START + 1):
            if model.driver_offsets_ is not None and did in model.driver_offsets_:
                cache[prev] = model.predict_proba(did, prev, constructor_id=cid)
            else:
                cache[prev] = model.predict_proba_new_driver(prev, constructor_id=cid)
        prob_cache[did] = cache
    return prob_cache


def precompute_probs_s8(model, drivers):
    """Precompute probs for Stage 8 (Plackett-Luce, with constructor)."""
    prob_cache = {}
    for did in drivers:
        cid = drivers[did][0]
        cache = np.zeros((START + 1, N_OUTCOMES))
        for prev in range(START + 1):
            if did in model.driver_strengths_:
                cache[prev] = model.predict_proba(did, prev, constructor_id=cid)
            else:
                cache[prev] = model.predict_proba_new_driver(prev, constructor_id=cid)
        prob_cache[did] = cache
    return prob_cache


def run_simulation(prob_cache, driver_ids, cid_list, cid_list_unique, rng):
    """Run N_SIMS Monte Carlo season simulations."""
    n_drivers = len(driver_ids)
    cid_to_cidx = {cid: i for i, cid in enumerate(cid_list_unique)}

    driver_total_pts = np.zeros((N_SIMS, n_drivers))
    driver_total_wins = np.zeros((N_SIMS, n_drivers), dtype=int)
    constructor_total_pts = np.zeros((N_SIMS, len(cid_list_unique)))

    for sim in range(N_SIMS):
        prev_pos = np.full(n_drivers, START, dtype=int)

        for race_num in range(N_RACES):
            results = np.zeros(n_drivers, dtype=int)
            for i, did in enumerate(driver_ids):
                probs = prob_cache[did][prev_pos[i]]
                results[i] = rng.choice(N_OUTCOMES, p=probs)

            for i in range(n_drivers):
                pos = results[i]
                pts = RACE_POINTS[pos]
                driver_total_pts[sim, i] += pts
                cidx = cid_to_cidx[cid_list[i]]
                constructor_total_pts[sim, cidx] += pts
                if pos == 1:
                    driver_total_wins[sim, i] += 1

            prev_pos = results

            if race_num < N_SPRINTS:
                for i, did in enumerate(driver_ids):
                    probs = prob_cache[did][prev_pos[i]]
                    sprint_pos = rng.choice(N_OUTCOMES, p=probs)
                    pts = SPRINT_POINTS[sprint_pos]
                    driver_total_pts[sim, i] += pts
                    cidx = cid_to_cidx[cid_list[i]]
                    constructor_total_pts[sim, cidx] += pts

    return driver_total_pts, driver_total_wins, constructor_total_pts


def analyze_results(driver_total_pts, driver_total_wins, constructor_total_pts,
                    driver_ids, cid_list_unique):
    """Compute championship probabilities and stats."""
    n_drivers = len(driver_ids)

    # Driver championship
    driver_champ_idx = np.argmax(driver_total_pts, axis=1)
    driver_champ_count = np.bincount(driver_champ_idx, minlength=n_drivers)

    # Avg championship position per driver
    driver_avg_rank = np.zeros(n_drivers)
    for sim in range(N_SIMS):
        ranks = np.argsort(-driver_total_pts[sim]).argsort() + 1
        driver_avg_rank += ranks
    driver_avg_rank /= N_SIMS

    # Constructor championship
    constr_champ_idx = np.argmax(constructor_total_pts, axis=1)
    constr_champ_count = np.bincount(constr_champ_idx,
                                     minlength=len(cid_list_unique))

    return {
        "driver_champ_count": driver_champ_count,
        "driver_avg_rank": driver_avg_rank,
        "constr_champ_count": constr_champ_count,
    }


def main():
    print("=" * 80)
    print("2025 SEASON RETROSPECTIVE SIMULATION")
    print(f"Monte Carlo: {N_SIMS:,} simulated seasons per model")
    print(f"Training: 2020-2024 | Simulating: 2025 (24 races + 6 sprints)")
    print("=" * 80)

    # =====================================================================
    # Train models
    # =====================================================================
    print("\nTraining models on 2020-2024 data...")

    loader2 = S2Loader(DATA_DIR)
    df2 = loader2.load_merged(min_year=2020, max_year=2024)
    prev2, next2, meta2 = s2_prepare(df2)
    model2 = PartialPooledDirichletF1(
        prior_alpha_global=1.0, kappa_init=10.0, kappa_bounds=(0.1, 500.0),
    )
    model2.fit(prev2, next2, meta2)
    print(f"  Stage 2: kappa={model2.kappa_:.2f}")

    loader3 = S3Loader(DATA_DIR)
    df3 = loader3.load_merged(min_year=2020, max_year=2024)
    prev3, next3, meta3 = s3_prepare(df3)
    model3 = ConstructorPooledDirichletF1(
        prior_alpha_global=1.0, prior_alpha_constructor=1.0,
        kappa_init=(10.0, 10.0), kappa_bounds=((0.1, 500.0), (0.01, 500.0)),
    )
    model3.fit(prev3, next3, meta3)
    print(f"  Stage 3: kappa_g={model3.kappa_g_:.2f}, kappa_c={model3.kappa_c_:.2f}")

    loader6 = S6Loader(DATA_DIR)
    df6 = loader6.load_merged(min_year=2020, max_year=2024)
    prev6, next6, meta6 = s6_prepare(df6)
    model6 = RecencyConstructorDirichletF1(
        prior_alpha_global=1.0, prior_alpha_constructor=1.0,
        kappa_init=(10.0, 10.0), kappa_bounds=((0.1, 500.0), (0.01, 500.0)),
    )
    model6.fit(prev6, next6, meta6)
    print(f"  Stage 6: kappa_g={model6.kappa_g_:.2f}, kappa_c={model6.kappa_c_:.2f}, "
          f"w={model6.w_:.4f}")

    # Stage 7 (HMM) — reuses Stage 3 data
    model7 = HiddenMarkovF1(n_tiers=4, em_iters=50, n_restarts=5)
    model7.fit(prev3, next3, meta3)

    # Stage 8 (PL) — reuses Stage 3 data
    model8 = TimeVaryingPlackettLuceF1(
        alpha_candidates=(0.85, 0.9, 0.95, 0.99),
        n_mc_samples=3000,
    )
    model8.fit(prev3, next3, meta3)

    # =====================================================================
    # Setup
    # =====================================================================
    driver_ids = list(DRIVERS_2025.keys())
    n_drivers = len(driver_ids)
    cid_list = [DRIVERS_2025[did][0] for did in driver_ids]
    cid_list_unique = sorted(CONSTRUCTOR_NAMES.keys())
    did_to_idx = {did: i for i, did in enumerate(driver_ids)}

    # =====================================================================
    # Simulate Stage 2
    # =====================================================================
    print("\nSimulating Stage 2...")
    prob_cache_s2 = precompute_probs_s2(model2, DRIVERS_2025)
    rng2 = np.random.default_rng(42)
    pts2, wins2, cpts2 = run_simulation(
        prob_cache_s2, driver_ids, cid_list, cid_list_unique, rng2
    )
    res2 = analyze_results(pts2, wins2, cpts2, driver_ids, cid_list_unique)
    print("  Done.")

    # =====================================================================
    # Simulate Stage 3
    # =====================================================================
    print("Simulating Stage 3...")
    prob_cache_s3 = precompute_probs_s3(model3, DRIVERS_2025)
    rng3 = np.random.default_rng(42)
    pts3, wins3, cpts3 = run_simulation(
        prob_cache_s3, driver_ids, cid_list, cid_list_unique, rng3
    )
    res3 = analyze_results(pts3, wins3, cpts3, driver_ids, cid_list_unique)
    print("  Done.")

    # =====================================================================
    # Simulate Stage 6
    # =====================================================================
    print("Simulating Stage 6...")
    prob_cache_s6 = precompute_probs_s3(model6, DRIVERS_2025)  # same interface as S3
    rng6 = np.random.default_rng(42)
    pts6, wins6, cpts6 = run_simulation(
        prob_cache_s6, driver_ids, cid_list, cid_list_unique, rng6
    )
    res6 = analyze_results(pts6, wins6, cpts6, driver_ids, cid_list_unique)
    print("  Done.")

    # =====================================================================
    # Simulate Stage 7 (HMM)
    # =====================================================================
    print("Simulating Stage 7 (HMM)...")
    prob_cache_s7 = precompute_probs_s7(model7, DRIVERS_2025)
    rng7 = np.random.default_rng(42)
    pts7, wins7, cpts7 = run_simulation(
        prob_cache_s7, driver_ids, cid_list, cid_list_unique, rng7
    )
    res7 = analyze_results(pts7, wins7, cpts7, driver_ids, cid_list_unique)
    print("  Done.")

    # =====================================================================
    # Simulate Stage 8 (Plackett-Luce)
    # =====================================================================
    print("Simulating Stage 8 (PL)...")
    prob_cache_s8 = precompute_probs_s8(model8, DRIVERS_2025)
    rng8 = np.random.default_rng(42)
    pts8, wins8, cpts8 = run_simulation(
        prob_cache_s8, driver_ids, cid_list, cid_list_unique, rng8
    )
    res8 = analyze_results(pts8, wins8, cpts8, driver_ids, cid_list_unique)
    print("  Done.")

    # =====================================================================
    # Comparison Table: Drivers
    # =====================================================================
    print("\n" + "=" * 80)
    print("DRIVERS' CHAMPIONSHIP — SIMULATED vs ACTUAL")
    print("=" * 80)

    # Build comparison rows
    all_models = [
        ("s2", pts2, wins2, res2),
        ("s3", pts3, wins3, res3),
        ("s6", pts6, wins6, res6),
        ("s7", pts7, wins7, res7),
        ("s8", pts8, wins8, res8),
    ]

    rows = []
    for i, did in enumerate(driver_ids):
        name = DRIVERS_2025[did][1]
        actual = ACTUAL_2025.get(did, {})
        r = {
            "driver_id": did,
            "name": name,
            "team": CONSTRUCTOR_NAMES[DRIVERS_2025[did][0]],
            "actual_pts": actual.get("pts", 0),
            "actual_rank": actual.get("rank", 99),
            "actual_wins": actual.get("wins", 0),
        }
        for tag, pts_m, wins_m, res_m in all_models:
            r[f"{tag}_avg_pts"] = pts_m[:, i].mean()
            r[f"{tag}_avg_rank"] = res_m["driver_avg_rank"][i]
            r[f"{tag}_avg_wins"] = wins_m[:, i].mean()
            r[f"{tag}_p_wdc"] = res_m["driver_champ_count"][i] / N_SIMS
        rows.append(r)

    comp_df = pd.DataFrame(rows).sort_values("actual_rank")

    model_labels = ["STAGE 2", "STAGE 3", "STAGE 6", "STAGE 7", "STAGE 8"]
    model_tags = ["s2", "s3", "s6", "s7", "s8"]

    print(f"\n{'Driver':<20} {'Team':<12} "
          f"{'ACTUAL':>11}  ", end="")
    for ml in model_labels:
        print(f"  {ml:>15}", end="")
    print()
    print(f"{'':20} {'':12} "
          f"{'Pts':>5} {'Rk':>3} {'W':>3}  ", end="")
    for _ in model_labels:
        print(f"  {'Pts':>5} {'Rk':>5} {'W':>3}", end="")
    print()
    print("-" * 180)

    for _, row in comp_df.iterrows():
        print(f"{row['name']:<20} {row['team']:<12} "
              f"{row['actual_pts']:>5.0f} {row['actual_rank']:>3.0f} "
              f"{row['actual_wins']:>3.0f}  ", end="")
        for tag in model_tags:
            print(f"  {row[f'{tag}_avg_pts']:>5.0f} "
                  f"{row[f'{tag}_avg_rank']:>5.1f} "
                  f"{row[f'{tag}_avg_wins']:>3.1f}", end="")
        print()

    # =====================================================================
    # Comparison Table: Constructors
    # =====================================================================
    print("\n" + "=" * 80)
    print("CONSTRUCTORS' CHAMPIONSHIP — SIMULATED vs ACTUAL")
    print("=" * 80)

    cid_to_cidx = {cid: i for i, cid in enumerate(cid_list_unique)}
    wcc_sorted = sorted(ACTUAL_WCC.items(), key=lambda x: -x[1])

    print(f"\n{'Team':<18} {'ACTUAL':>8}  ", end="")
    for ml in model_labels:
        print(f"  {ml:>14}", end="")
    print()
    print(f"{'':18} {'Pts':>8}  ", end="")
    for _ in model_labels:
        print(f"  {'AvgPts':>7} {'P(WCC)':>6}", end="")
    print()
    print("-" * 105)

    cpts_all = [cpts2, cpts3, cpts6, cpts7, cpts8]
    res_all = [res2, res3, res6, res7, res8]

    for team, actual_pts in wcc_sorted:
        cid = [k for k, v in CONSTRUCTOR_NAMES.items() if v == team]
        if not cid:
            continue
        cidx = cid_to_cidx[cid[0]]
        print(f"{team:<18} {actual_pts:>8}  ", end="")
        for cpts_m, res_m in zip(cpts_all, res_all):
            avg = cpts_m[:, cidx].mean()
            pwcc = res_m["constr_champ_count"][cidx] / N_SIMS
            print(f"  {avg:>7.1f} {pwcc:>6.1%}", end="")
        print()

    # =====================================================================
    # Model Accuracy Metrics
    # =====================================================================
    print("\n" + "=" * 80)
    print("MODEL ACCURACY COMPARISON")
    print("=" * 80)

    # Prepare aligned arrays (only drivers in both sim and actual)
    common_dids = [did for did in driver_ids if did in ACTUAL_2025]

    actual_pts_arr = np.array([ACTUAL_2025[d]["pts"] for d in common_dids])
    actual_rank_arr = np.array([ACTUAL_2025[d]["rank"] for d in common_dids])
    actual_wins_arr = np.array([ACTUAL_2025[d]["wins"] for d in common_dids])

    sim_data = {}
    for label, pts_m, wins_m, res_m in [
        ("Stage 2", pts2, wins2, res2),
        ("Stage 3", pts3, wins3, res3),
        ("Stage 6", pts6, wins6, res6),
        ("Stage 7", pts7, wins7, res7),
        ("Stage 8", pts8, wins8, res8),
    ]:
        sim_data[label] = {
            "pts_arr": np.array([pts_m[:, did_to_idx[d]].mean() for d in common_dids]),
            "rank_arr": np.array([res_m["driver_avg_rank"][did_to_idx[d]] for d in common_dids]),
            "wins_arr": np.array([wins_m[:, did_to_idx[d]].mean() for d in common_dids]),
            "res": res_m,
            "pts_m": pts_m,
            "wins_m": wins_m,
        }

    norris_idx = did_to_idx[846]
    ver_idx = did_to_idx[830]
    mclaren_cidx = cid_to_cidx[1]

    header = f"{'Metric':<35}"
    for label in sim_data:
        header += f" {label:>12}"
    print(f"\n{header}")
    print("-" * (35 + 13 * len(sim_data)))

    # Spearman rho
    row = f"{'Spearman rho (rank)':<35}"
    for label, sd in sim_data.items():
        rho, _ = spearmanr(sd["rank_arr"], actual_rank_arr)
        row += f" {rho:>12.3f}"
    print(row)

    # Points MAE
    row = f"{'Points MAE':<35}"
    for label, sd in sim_data.items():
        mae = np.mean(np.abs(sd["pts_arr"] - actual_pts_arr))
        row += f" {mae:>12.1f}"
    print(row)

    # Rank MAE
    row = f"{'Rank MAE':<35}"
    for label, sd in sim_data.items():
        rank_mae = np.mean(np.abs(sd["rank_arr"] - actual_rank_arr))
        row += f" {rank_mae:>12.2f}"
    print(row)

    # Wins MAE
    row = f"{'Wins MAE':<35}"
    for label, sd in sim_data.items():
        wins_mae = np.mean(np.abs(sd["wins_arr"] - actual_wins_arr))
        row += f" {wins_mae:>12.2f}"
    print(row)

    # P(Norris WDC)
    row = f"{'P(Norris WDC) [actual champion]':<35}"
    for label, sd in sim_data.items():
        p = sd["res"]["driver_champ_count"][norris_idx] / N_SIMS
        row += f" {p:>12.1%}"
    print(row)

    # P(Verstappen WDC)
    row = f"{'P(Verstappen WDC)':<35}"
    for label, sd in sim_data.items():
        p = sd["res"]["driver_champ_count"][ver_idx] / N_SIMS
        row += f" {p:>12.1%}"
    print(row)

    # P(McLaren WCC)
    row = f"{'P(McLaren WCC) [actual champion]':<35}"
    for label, sd in sim_data.items():
        p = sd["res"]["constr_champ_count"][mclaren_cidx] / N_SIMS
        row += f" {p:>12.1%}"
    print(row)

    # =====================================================================
    # Biggest Misses
    # =====================================================================
    print("\n" + "=" * 80)
    print("BIGGEST PREDICTION MISSES (by rank difference)")
    print("=" * 80)

    for label, sd in sim_data.items():
        print(f"\n  {label}:")
        rank_diffs = sd["rank_arr"] - actual_rank_arr
        sorted_idx = np.argsort(np.abs(rank_diffs))[::-1]
        for i in sorted_idx[:5]:
            did = common_dids[i]
            name = ACTUAL_2025[did]["name"]
            diff = rank_diffs[i]
            direction = "too low" if diff > 0 else "too high"
            print(f"    {name:<22} actual=#{actual_rank_arr[i]:.0f}  "
                  f"sim=#{sd['rank_arr'][i]:.1f}  ({direction} by {abs(diff):.1f})")


if __name__ == "__main__":
    main()
