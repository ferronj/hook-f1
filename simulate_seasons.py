"""
Monte Carlo simulation of 2025 (retrospective) and 2026 (predictive) F1 seasons.

Models: Stage 6 (year-weighted constructor Dirichlet-Markov) and
        Stage 9 (Bayesian state-space MAP with Plackett-Luce).

2025 training: 2020-2024
2026 training: 2021-2025

10,000 simulated seasons per model per year.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent / "models"))

from stage3_constructor import (
    F1DataLoader as S3Loader,
    prepare_transitions as s3_prepare,
    START, N_OUTCOMES,
)
from stage6_recency_constructor import (
    F1DataLoader as S6Loader,
    RecencyConstructorDirichletF1,
    prepare_transitions as s6_prepare,
)
from stage9_bayesian_ss import BayesianStateSpaceF1

DATA_DIR = Path(__file__).parent / "data"
N_SIMS = 10000
N_RACES = 24
N_SPRINTS = 6

RACE_POINTS = np.zeros(N_OUTCOMES, dtype=int)
for pos, pts in {1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
                 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}.items():
    RACE_POINTS[pos] = pts

SPRINT_POINTS = np.zeros(N_OUTCOMES, dtype=int)
for pos, pts in {1: 8, 2: 7, 3: 6, 4: 5, 5: 4,
                 6: 3, 7: 2, 8: 1}.items():
    SPRINT_POINTS[pos] = pts

# =========================================================================
# Driver lineups
# =========================================================================
DRIVERS_2025 = {
    846: (1,   "Lando Norris"),
    857: (1,   "Oscar Piastri"),
    830: (9,   "Max Verstappen"),
    852: (9,   "Yuki Tsunoda"),
    847: (131, "George Russell"),
    863: (131, "Kimi Antonelli"),
    844: (6,   "Charles Leclerc"),
    1:   (6,   "Lewis Hamilton"),
    4:   (117, "Fernando Alonso"),
    840: (117, "Lance Stroll"),
    832: (3,   "Carlos Sainz"),
    848: (3,   "Alex Albon"),
    842: (214, "Pierre Gasly"),
    862: (214, "Jack Doohan"),
    839: (210, "Esteban Ocon"),
    860: (210, "Oliver Bearman"),
    807: (15,  "Nico Hulkenberg"),
    865: (15,  "Gabriel Bortoleto"),
    859: (215, "Liam Lawson"),
    864: (215, "Isack Hadjar"),
}

DRIVERS_2026 = {
    846: (1,   "Lando Norris"),
    857: (1,   "Oscar Piastri"),
    1:   (6,   "Lewis Hamilton"),
    844: (6,   "Charles Leclerc"),
    847: (131, "George Russell"),
    863: (131, "Kimi Antonelli"),
    830: (9,   "Max Verstappen"),
    864: (9,   "Isack Hadjar"),
    4:   (117, "Fernando Alonso"),
    840: (117, "Lance Stroll"),
    832: (3,   "Carlos Sainz"),
    848: (3,   "Alex Albon"),
    861: (214, "Franco Colapinto"),
    842: (214, "Pierre Gasly"),
    839: (210, "Esteban Ocon"),
    860: (210, "Oliver Bearman"),
    807: (15,  "Nico Hulkenberg"),
    865: (15,  "Gabriel Bortoleto"),
    859: (215, "Liam Lawson"),
    866: (215, "Arvid Lindblad"),
    822: (216, "Valtteri Bottas"),
    815: (216, "Sergio Perez"),
}

CONSTRUCTOR_NAMES_2025 = {
    1: "McLaren", 6: "Ferrari", 131: "Mercedes", 9: "Red Bull",
    117: "Aston Martin", 3: "Williams", 214: "Alpine",
    210: "Haas", 15: "Kick Sauber", 215: "Racing Bulls",
}

CONSTRUCTOR_NAMES_2026 = {
    1: "McLaren", 6: "Ferrari", 131: "Mercedes", 9: "Red Bull",
    117: "Aston Martin", 3: "Williams", 214: "Alpine",
    210: "Haas", 15: "Audi", 215: "Racing Bulls", 216: "Cadillac",
}

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

ACTUAL_WCC_2025 = {
    "McLaren": 775, "Mercedes": 424, "Red Bull": 410, "Ferrari": 360,
    "Williams": 124, "Racing Bulls": 88, "Aston Martin": 80,
    "Haas": 73, "Kick Sauber": 70, "Alpine": 20,
}


# =========================================================================
# Probability precomputation
# =========================================================================
def precompute_probs_s6(model, drivers):
    """Precompute probs for Stage 6 (constructor-aware Dirichlet-Markov)."""
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


def precompute_probs_s9(model, drivers):
    """Precompute probs for Stage 9 (Bayesian state-space)."""
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


# =========================================================================
# Simulation engine
# =========================================================================
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

    driver_champ_idx = np.argmax(driver_total_pts, axis=1)
    driver_champ_count = np.bincount(driver_champ_idx, minlength=n_drivers)

    driver_top3_count = np.zeros(n_drivers, dtype=int)
    driver_avg_rank = np.zeros(n_drivers)
    for sim in range(N_SIMS):
        ranks = np.argsort(-driver_total_pts[sim]).argsort() + 1
        driver_avg_rank += ranks
        top3_idx = np.argsort(-driver_total_pts[sim])[:3]
        for idx in top3_idx:
            driver_top3_count[idx] += 1
    driver_avg_rank /= N_SIMS

    constr_champ_idx = np.argmax(constructor_total_pts, axis=1)
    constr_champ_count = np.bincount(constr_champ_idx,
                                     minlength=len(cid_list_unique))

    return {
        "driver_champ_count": driver_champ_count,
        "driver_avg_rank": driver_avg_rank,
        "driver_top3_count": driver_top3_count,
        "constr_champ_count": constr_champ_count,
    }


# =========================================================================
# Print helpers
# =========================================================================
def print_driver_table(label, drivers, driver_ids, pts, wins, res,
                       constructor_names, did_to_idx, actual=None):
    """Print driver championship results for one model."""
    n_drivers = len(driver_ids)
    rows = []
    for i, did in enumerate(driver_ids):
        name = drivers[did][1]
        cid = drivers[did][0]
        r = {
            "did": did,
            "name": name,
            "team": constructor_names[cid],
            "avg_pts": pts[:, i].mean(),
            "avg_rank": res["driver_avg_rank"][i],
            "avg_wins": wins[:, i].mean(),
            "p_wdc": res["driver_champ_count"][i] / N_SIMS,
            "p_top3": res["driver_top3_count"][i] / N_SIMS,
        }
        if actual and did in actual:
            r["actual_rank"] = actual[did]["rank"]
            r["actual_pts"] = actual[did]["pts"]
            r["actual_wins"] = actual[did]["wins"]
        rows.append(r)

    df = pd.DataFrame(rows).sort_values("avg_rank")

    print(f"\n  {label}")
    if actual:
        print(f"  {'Rk':<4} {'Driver':<22} {'Team':<14} "
              f"{'P(WDC)':>7} {'P(T3)':>7} {'AvgPts':>7} {'AvgW':>5} {'AvgRk':>5} "
              f"{'ActRk':>5} {'ActPts':>6}")
        print(f"  {'-'*95}")
    else:
        print(f"  {'Rk':<4} {'Driver':<22} {'Team':<14} "
              f"{'P(WDC)':>7} {'P(T3)':>7} {'AvgPts':>7} {'AvgW':>5} {'AvgRk':>5}")
        print(f"  {'-'*75}")

    for rank, (_, row) in enumerate(df.iterrows(), 1):
        line = (f"  {rank:<4} {row['name']:<22} {row['team']:<14} "
                f"{row['p_wdc']:>7.1%} {row['p_top3']:>7.1%} "
                f"{row['avg_pts']:>7.0f} {row['avg_wins']:>5.1f} "
                f"{row['avg_rank']:>5.1f}")
        if actual and "actual_rank" in row:
            line += f" {row['actual_rank']:>5.0f} {row['actual_pts']:>6.0f}"
        print(line)

    return df


def print_constructor_table(label, cpts, res, cid_list_unique,
                            constructor_names, actual_wcc=None):
    """Print constructor championship results for one model."""
    rows = []
    cid_to_cidx = {cid: i for i, cid in enumerate(cid_list_unique)}
    for cid in cid_list_unique:
        cidx = cid_to_cidx[cid]
        team = constructor_names[cid]
        r = {
            "team": team,
            "avg_pts": cpts[:, cidx].mean(),
            "p_wcc": res["constr_champ_count"][cidx] / N_SIMS,
        }
        if actual_wcc and team in actual_wcc:
            r["actual_pts"] = actual_wcc[team]
        rows.append(r)

    df = pd.DataFrame(rows).sort_values("avg_pts", ascending=False)

    print(f"\n  {label}")
    if actual_wcc:
        print(f"  {'Rk':<4} {'Team':<18} {'P(WCC)':>7} {'AvgPts':>8} {'ActPts':>8}")
        print(f"  {'-'*50}")
    else:
        print(f"  {'Rk':<4} {'Team':<18} {'P(WCC)':>7} {'AvgPts':>8}")
        print(f"  {'-'*40}")

    for rank, (_, row) in enumerate(df.iterrows(), 1):
        line = f"  {rank:<4} {row['team']:<18} {row['p_wcc']:>7.1%} {row['avg_pts']:>8.0f}"
        if actual_wcc and "actual_pts" in row and not np.isnan(row.get("actual_pts", np.nan)):
            line += f" {row['actual_pts']:>8.0f}"
        print(line)

    return df


def compute_accuracy_metrics(driver_ids, drivers, pts, wins, res,
                             did_to_idx, actual):
    """Compute accuracy metrics against actual results."""
    common = [d for d in driver_ids if d in actual]
    actual_rank = np.array([actual[d]["rank"] for d in common])
    actual_pts = np.array([actual[d]["pts"] for d in common])
    sim_rank = np.array([res["driver_avg_rank"][did_to_idx[d]] for d in common])
    sim_pts = np.array([pts[:, did_to_idx[d]].mean() for d in common])

    rho, _ = spearmanr(sim_rank, actual_rank)
    pts_mae = np.mean(np.abs(sim_pts - actual_pts))
    rank_mae = np.mean(np.abs(sim_rank - actual_rank))

    return {"spearman_rho": rho, "pts_mae": pts_mae, "rank_mae": rank_mae}


# =========================================================================
# Main
# =========================================================================
def main():
    print("=" * 80)
    print("F1 SEASON SIMULATIONS — Stage 6 vs Stage 9")
    print(f"Monte Carlo: {N_SIMS:,} simulated seasons per model")
    print("=" * 80)

    # =====================================================================
    # 2025 RETROSPECTIVE (train 2020-2024)
    # =====================================================================
    print("\n" + "=" * 80)
    print("PART 1: 2025 SEASON RETROSPECTIVE")
    print("Training: 2020-2024 | Simulating: 2025")
    print("=" * 80)

    # Train Stage 6
    print("\nTraining Stage 6 on 2020-2024...")
    loader6 = S6Loader(DATA_DIR)
    df6 = loader6.load_merged(min_year=2020, max_year=2024)
    prev6, next6, meta6 = s6_prepare(df6)
    model6_25 = RecencyConstructorDirichletF1(
        prior_alpha_global=1.0, prior_alpha_constructor=1.0,
        kappa_init=(10.0, 10.0), kappa_bounds=((0.1, 500.0), (0.01, 500.0)),
    )
    model6_25.fit(prev6, next6, meta6)
    print(f"  Stage 6: kappa_g={model6_25.kappa_g_:.2f}, "
          f"kappa_c={model6_25.kappa_c_:.2f}, w={model6_25.w_:.4f}")

    # Train Stage 9
    print("\nTraining Stage 9 on 2020-2024...")
    loader3 = S3Loader(DATA_DIR)
    df3 = loader3.load_merged(min_year=2020, max_year=2024)
    prev3, next3, meta3 = s3_prepare(df3)
    model9_25 = BayesianStateSpaceF1(
        sigma_d_candidates=(0.02, 0.05, 0.1, 0.2),
        sigma_c_candidates=(0.02, 0.05, 0.1, 0.2),
        n_mc_samples=3000,
    )
    model9_25.fit(prev3, next3, meta3)

    # Setup 2025 simulation
    driver_ids_25 = list(DRIVERS_2025.keys())
    cid_list_25 = [DRIVERS_2025[d][0] for d in driver_ids_25]
    cid_unique_25 = sorted(CONSTRUCTOR_NAMES_2025.keys())
    did_to_idx_25 = {d: i for i, d in enumerate(driver_ids_25)}

    # Simulate Stage 6 - 2025
    print("\nSimulating 2025 — Stage 6...")
    pc6_25 = precompute_probs_s6(model6_25, DRIVERS_2025)
    pts6_25, wins6_25, cpts6_25 = run_simulation(
        pc6_25, driver_ids_25, cid_list_25, cid_unique_25,
        np.random.default_rng(42)
    )
    res6_25 = analyze_results(pts6_25, wins6_25, cpts6_25, driver_ids_25, cid_unique_25)

    # Simulate Stage 9 - 2025
    print("Simulating 2025 — Stage 9...")
    pc9_25 = precompute_probs_s9(model9_25, DRIVERS_2025)
    pts9_25, wins9_25, cpts9_25 = run_simulation(
        pc9_25, driver_ids_25, cid_list_25, cid_unique_25,
        np.random.default_rng(42)
    )
    res9_25 = analyze_results(pts9_25, wins9_25, cpts9_25, driver_ids_25, cid_unique_25)

    # Print 2025 results
    print("\n" + "-" * 80)
    print("2025 DRIVERS' CHAMPIONSHIP")
    print("-" * 80)

    df6_drv_25 = print_driver_table(
        "Stage 6 (Year-Weighted Constructor Dirichlet-Markov)",
        DRIVERS_2025, driver_ids_25, pts6_25, wins6_25, res6_25,
        CONSTRUCTOR_NAMES_2025, did_to_idx_25, ACTUAL_2025
    )

    df9_drv_25 = print_driver_table(
        "Stage 9 (Bayesian State-Space)",
        DRIVERS_2025, driver_ids_25, pts9_25, wins9_25, res9_25,
        CONSTRUCTOR_NAMES_2025, did_to_idx_25, ACTUAL_2025
    )

    print("\n" + "-" * 80)
    print("2025 CONSTRUCTORS' CHAMPIONSHIP")
    print("-" * 80)

    print_constructor_table(
        "Stage 6", cpts6_25, res6_25, cid_unique_25,
        CONSTRUCTOR_NAMES_2025, ACTUAL_WCC_2025
    )
    print_constructor_table(
        "Stage 9", cpts9_25, res9_25, cid_unique_25,
        CONSTRUCTOR_NAMES_2025, ACTUAL_WCC_2025
    )

    # Accuracy comparison
    print("\n" + "-" * 80)
    print("2025 MODEL ACCURACY COMPARISON")
    print("-" * 80)

    acc6_25 = compute_accuracy_metrics(
        driver_ids_25, DRIVERS_2025, pts6_25, wins6_25, res6_25,
        did_to_idx_25, ACTUAL_2025
    )
    acc9_25 = compute_accuracy_metrics(
        driver_ids_25, DRIVERS_2025, pts9_25, wins9_25, res9_25,
        did_to_idx_25, ACTUAL_2025
    )

    norris_idx = did_to_idx_25[846]
    ver_idx = did_to_idx_25[830]
    mclaren_cidx = {cid: i for i, cid in enumerate(cid_unique_25)}[1]

    print(f"\n  {'Metric':<35} {'Stage 6':>12} {'Stage 9':>12}")
    print(f"  {'-'*60}")
    print(f"  {'Spearman rho (rank order)':<35} {acc6_25['spearman_rho']:>12.3f} {acc9_25['spearman_rho']:>12.3f}")
    print(f"  {'Points MAE':<35} {acc6_25['pts_mae']:>12.1f} {acc9_25['pts_mae']:>12.1f}")
    print(f"  {'Rank MAE':<35} {acc6_25['rank_mae']:>12.2f} {acc9_25['rank_mae']:>12.2f}")

    p_norris_6 = res6_25["driver_champ_count"][norris_idx] / N_SIMS
    p_norris_9 = res9_25["driver_champ_count"][norris_idx] / N_SIMS
    print(f"  {'P(Norris WDC) [actual champ]':<35} {p_norris_6:>12.1%} {p_norris_9:>12.1%}")

    p_ver_6 = res6_25["driver_champ_count"][ver_idx] / N_SIMS
    p_ver_9 = res9_25["driver_champ_count"][ver_idx] / N_SIMS
    print(f"  {'P(Verstappen WDC)':<35} {p_ver_6:>12.1%} {p_ver_9:>12.1%}")

    p_mclaren_6 = res6_25["constr_champ_count"][mclaren_cidx] / N_SIMS
    p_mclaren_9 = res9_25["constr_champ_count"][mclaren_cidx] / N_SIMS
    print(f"  {'P(McLaren WCC) [actual champ]':<35} {p_mclaren_6:>12.1%} {p_mclaren_9:>12.1%}")

    # Biggest misses
    print("\n  Biggest rank misses:")
    for label, res, pts_m in [("Stage 6", res6_25, pts6_25), ("Stage 9", res9_25, pts9_25)]:
        common = [d for d in driver_ids_25 if d in ACTUAL_2025]
        diffs = []
        for d in common:
            sim_r = res["driver_avg_rank"][did_to_idx_25[d]]
            act_r = ACTUAL_2025[d]["rank"]
            diffs.append((d, sim_r - act_r))
        diffs.sort(key=lambda x: abs(x[1]), reverse=True)
        print(f"    {label}:")
        for d, diff in diffs[:3]:
            name = ACTUAL_2025[d]["name"]
            direction = "too low" if diff > 0 else "too high"
            print(f"      {name:<22} ({direction} by {abs(diff):.1f})")

    # =====================================================================
    # 2026 PREDICTION (train 2021-2025)
    # =====================================================================
    print("\n\n" + "=" * 80)
    print("PART 2: 2026 SEASON PREDICTION")
    print("Training: 2021-2025 | Simulating: 2026")
    print("=" * 80)

    # Train Stage 6
    print("\nTraining Stage 6 on 2021-2025...")
    loader6b = S6Loader(DATA_DIR)
    df6b = loader6b.load_merged(min_year=2021, max_year=2025)
    prev6b, next6b, meta6b = s6_prepare(df6b)
    model6_26 = RecencyConstructorDirichletF1(
        prior_alpha_global=1.0, prior_alpha_constructor=1.0,
        kappa_init=(10.0, 10.0), kappa_bounds=((0.1, 500.0), (0.01, 500.0)),
    )
    model6_26.fit(prev6b, next6b, meta6b)
    print(f"  Stage 6: kappa_g={model6_26.kappa_g_:.2f}, "
          f"kappa_c={model6_26.kappa_c_:.2f}, w={model6_26.w_:.4f}")

    # Train Stage 9
    print("\nTraining Stage 9 on 2021-2025...")
    loader3b = S3Loader(DATA_DIR)
    df3b = loader3b.load_merged(min_year=2021, max_year=2025)
    prev3b, next3b, meta3b = s3_prepare(df3b)
    model9_26 = BayesianStateSpaceF1(
        sigma_d_candidates=(0.02, 0.05, 0.1, 0.2),
        sigma_c_candidates=(0.02, 0.05, 0.1, 0.2),
        n_mc_samples=3000,
    )
    model9_26.fit(prev3b, next3b, meta3b)

    # Setup 2026 simulation
    driver_ids_26 = list(DRIVERS_2026.keys())
    cid_list_26 = [DRIVERS_2026[d][0] for d in driver_ids_26]
    cid_unique_26 = sorted(CONSTRUCTOR_NAMES_2026.keys())
    did_to_idx_26 = {d: i for i, d in enumerate(driver_ids_26)}

    # Simulate Stage 6 - 2026
    print("\nSimulating 2026 — Stage 6...")
    pc6_26 = precompute_probs_s6(model6_26, DRIVERS_2026)
    pts6_26, wins6_26, cpts6_26 = run_simulation(
        pc6_26, driver_ids_26, cid_list_26, cid_unique_26,
        np.random.default_rng(42)
    )
    res6_26 = analyze_results(pts6_26, wins6_26, cpts6_26, driver_ids_26, cid_unique_26)

    # Simulate Stage 9 - 2026
    print("Simulating 2026 — Stage 9...")
    pc9_26 = precompute_probs_s9(model9_26, DRIVERS_2026)
    pts9_26, wins9_26, cpts9_26 = run_simulation(
        pc9_26, driver_ids_26, cid_list_26, cid_unique_26,
        np.random.default_rng(42)
    )
    res9_26 = analyze_results(pts9_26, wins9_26, cpts9_26, driver_ids_26, cid_unique_26)

    # Print 2026 results
    print("\n" + "-" * 80)
    print("2026 DRIVERS' CHAMPIONSHIP PREDICTIONS")
    print("-" * 80)

    print_driver_table(
        "Stage 6 (Year-Weighted Constructor Dirichlet-Markov)",
        DRIVERS_2026, driver_ids_26, pts6_26, wins6_26, res6_26,
        CONSTRUCTOR_NAMES_2026, did_to_idx_26
    )

    print_driver_table(
        "Stage 9 (Bayesian State-Space)",
        DRIVERS_2026, driver_ids_26, pts9_26, wins9_26, res9_26,
        CONSTRUCTOR_NAMES_2026, did_to_idx_26
    )

    print("\n" + "-" * 80)
    print("2026 CONSTRUCTORS' CHAMPIONSHIP PREDICTIONS")
    print("-" * 80)

    print_constructor_table(
        "Stage 6", cpts6_26, res6_26, cid_unique_26, CONSTRUCTOR_NAMES_2026
    )
    print_constructor_table(
        "Stage 9", cpts9_26, res9_26, cid_unique_26, CONSTRUCTOR_NAMES_2026
    )

    # Key 2026 storylines
    print("\n" + "-" * 80)
    print("2026 KEY COMPARISONS")
    print("-" * 80)

    # WDC favorites
    print("\n  WDC Favorites:")
    print(f"  {'Driver':<22} {'Team':<14} {'S6 P(WDC)':>10} {'S9 P(WDC)':>10}")
    print(f"  {'-'*58}")
    # Merge and sort by S9 WDC
    wdc_rows = []
    for i, did in enumerate(driver_ids_26):
        p6 = res6_26["driver_champ_count"][i] / N_SIMS
        p9 = res9_26["driver_champ_count"][i] / N_SIMS
        if p6 > 0.01 or p9 > 0.01:
            wdc_rows.append((DRIVERS_2026[did][1], CONSTRUCTOR_NAMES_2026[DRIVERS_2026[did][0]], p6, p9))
    wdc_rows.sort(key=lambda x: -(x[2] + x[3]))
    for name, team, p6, p9 in wdc_rows:
        print(f"  {name:<22} {team:<14} {p6:>10.1%} {p9:>10.1%}")

    # WCC favorites
    print("\n  WCC Favorites:")
    print(f"  {'Team':<18} {'S6 P(WCC)':>10} {'S9 P(WCC)':>10}")
    print(f"  {'-'*40}")
    cid_to_cidx_26 = {cid: i for i, cid in enumerate(cid_unique_26)}
    wcc_rows = []
    for cid in cid_unique_26:
        cidx = cid_to_cidx_26[cid]
        p6 = res6_26["constr_champ_count"][cidx] / N_SIMS
        p9 = res9_26["constr_champ_count"][cidx] / N_SIMS
        if p6 > 0.005 or p9 > 0.005:
            wcc_rows.append((CONSTRUCTOR_NAMES_2026[cid], p6, p9))
    wcc_rows.sort(key=lambda x: -(x[1] + x[2]))
    for team, p6, p9 in wcc_rows:
        print(f"  {team:<18} {p6:>10.1%} {p9:>10.1%}")

    # Model disagreements
    print("\n  Biggest model disagreements (rank difference):")
    for i, did in enumerate(driver_ids_26):
        r6 = res6_26["driver_avg_rank"][i]
        r9 = res9_26["driver_avg_rank"][i]
        diff = abs(r6 - r9)
        if diff > 2.0:
            name = DRIVERS_2026[did][1]
            team = CONSTRUCTOR_NAMES_2026[DRIVERS_2026[did][0]]
            print(f"    {name:<22} ({team}): S6 rank {r6:.1f} vs S9 rank {r9:.1f}")

    # Intra-team battles
    print("\n  Intra-team battles (avg season points):")
    teams_6 = defaultdict(list)
    teams_9 = defaultdict(list)
    for i, did in enumerate(driver_ids_26):
        name = DRIVERS_2026[did][1]
        team = CONSTRUCTOR_NAMES_2026[DRIVERS_2026[did][0]]
        teams_6[team].append((name, pts6_26[:, i].mean()))
        teams_9[team].append((name, pts9_26[:, i].mean()))

    for team in ["McLaren", "Ferrari", "Red Bull", "Mercedes",
                 "Aston Martin", "Williams", "Cadillac"]:
        if team in teams_6 and len(teams_6[team]) == 2:
            d1_6, d2_6 = teams_6[team]
            d1_9, d2_9 = teams_9[team]
            print(f"    {team}:")
            print(f"      S6: {d1_6[0]} ({d1_6[1]:.0f}) vs {d2_6[0]} ({d2_6[1]:.0f})")
            print(f"      S9: {d1_9[0]} ({d1_9[1]:.0f}) vs {d2_9[0]} ({d2_9[1]:.0f})")


if __name__ == "__main__":
    main()
