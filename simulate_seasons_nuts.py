"""
Monte Carlo simulation of 2025 (retrospective) and 2026 (predictive) F1 seasons.

Compares Stage 9 MAP (point estimate) vs Stage 9 NUTS (full Bayesian posterior).

Usage:
  micromamba run -n f1-markov python3 simulate_seasons_nuts.py \
    --trace stage9_nuts_2025.nc

  # Or with custom options:
  micromamba run -n f1-markov python3 simulate_seasons_nuts.py \
    --trace stage9_nuts_2025.nc --n-sims 5000 --n-posterior-draws 200

The NUTS trace must be generated first via run_stage9_nuts.py.
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent / "models"))

from stage3_constructor import (
    F1DataLoader,
    prepare_transitions,
    START, N_OUTCOMES,
)
from stage9_bayesian_ss import BayesianStateSpaceF1

DATA_DIR = Path(__file__).parent / "data"

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
def precompute_probs_map(model, drivers):
    """Precompute MAP-based probs for Stage 9."""
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


def precompute_probs_bayes(model, drivers, n_posterior_draws=200):
    """Precompute Bayesian (posterior-averaged) probs for Stage 9."""
    prob_cache = {}
    for did in drivers:
        cid = drivers[did][0]
        cache = np.zeros((START + 1, N_OUTCOMES))
        for prev in range(START + 1):
            if did in model.driver_strengths_:
                cache[prev] = model.predict_proba_bayesian(
                    did, prev, constructor_id=cid,
                    n_posterior_draws=n_posterior_draws,
                )
            else:
                cache[prev] = model.predict_proba_new_driver(prev, constructor_id=cid)
        prob_cache[did] = cache
    return prob_cache


# =========================================================================
# Simulation engine
# =========================================================================
def run_simulation(prob_cache, driver_ids, cid_list, cid_list_unique, rng,
                   n_sims=10000):
    """Run Monte Carlo season simulations."""
    n_drivers = len(driver_ids)
    cid_to_cidx = {cid: i for i, cid in enumerate(cid_list_unique)}

    driver_total_pts = np.zeros((n_sims, n_drivers))
    driver_total_wins = np.zeros((n_sims, n_drivers), dtype=int)
    constructor_total_pts = np.zeros((n_sims, len(cid_list_unique)))

    for sim in range(n_sims):
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


def analyze_results(driver_total_pts, driver_total_wins,
                    constructor_total_pts, driver_ids, cid_list_unique,
                    n_sims=10000):
    """Compute championship probabilities and stats."""
    n_drivers = len(driver_ids)

    driver_champ_idx = np.argmax(driver_total_pts, axis=1)
    driver_champ_count = np.bincount(driver_champ_idx, minlength=n_drivers)

    driver_top3_count = np.zeros(n_drivers, dtype=int)
    driver_avg_rank = np.zeros(n_drivers)
    for sim in range(n_sims):
        ranks = np.argsort(-driver_total_pts[sim]).argsort() + 1
        driver_avg_rank += ranks
        top3_idx = np.argsort(-driver_total_pts[sim])[:3]
        for idx in top3_idx:
            driver_top3_count[idx] += 1
    driver_avg_rank /= n_sims

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
                       constructor_names, did_to_idx, actual=None,
                       n_sims=10000):
    """Print driver championship results."""
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
            "p_wdc": res["driver_champ_count"][i] / n_sims,
            "p_top3": res["driver_top3_count"][i] / n_sims,
        }
        if actual and did in actual:
            r["actual_rank"] = actual[did]["rank"]
            r["actual_pts"] = actual[did]["pts"]
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
                            constructor_names, actual_wcc=None,
                            n_sims=10000):
    """Print constructor championship results."""
    rows = []
    cid_to_cidx = {cid: i for i, cid in enumerate(cid_list_unique)}
    for cid in cid_list_unique:
        cidx = cid_to_cidx[cid]
        team = constructor_names[cid]
        r = {
            "team": team,
            "avg_pts": cpts[:, cidx].mean(),
            "p_wcc": res["constr_champ_count"][cidx] / n_sims,
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
                             did_to_idx, actual, n_sims=10000):
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
    parser = argparse.ArgumentParser(
        description="Simulate F1 seasons with Stage 9 MAP vs NUTS"
    )
    parser.add_argument("--trace", type=str, required=True,
                        help="Path to NUTS trace file (.nc)")
    parser.add_argument("--n-sims", type=int, default=10000,
                        help="Number of Monte Carlo simulations (default: 10000)")
    parser.add_argument("--n-posterior-draws", type=int, default=200,
                        help="Posterior draws for Bayesian predictions (default: 200)")
    parser.add_argument("--n-mc-samples", type=int, default=3000,
                        help="MC samples for position marginals (default: 3000)")
    parser.add_argument("--year", type=str, default="both",
                        choices=["2025", "2026", "both"],
                        help="Which season to simulate (default: both)")
    args = parser.parse_args()

    n_sims = args.n_sims

    print("=" * 80)
    print("F1 SEASON SIMULATIONS — Stage 9 MAP vs NUTS (Full Bayesian)")
    print(f"Monte Carlo: {n_sims:,} simulated seasons")
    print(f"Posterior draws for Bayesian predictions: {args.n_posterior_draws}")
    print(f"NUTS trace: {args.trace}")
    print("=" * 80)

    # Load NUTS trace
    print("\nLoading NUTS trace...")
    model_nuts = BayesianStateSpaceF1.load_nuts_trace(
        args.trace, n_mc_samples=args.n_mc_samples
    )
    print(f"  NUTS model loaded: {len(model_nuts.driver_strengths_)} drivers, "
          f"{len(model_nuts.constructor_strengths_)} constructors, "
          f"{model_nuts.n_posterior_samples_} posterior samples")

    # Train MAP model with same training data
    # Infer training years from the trace metadata
    import pickle
    meta_path = Path(args.trace).with_suffix('.pkl')
    with open(meta_path, 'rb') as f:
        trace_meta = pickle.load(f)

    # =====================================================================
    # 2025 RETROSPECTIVE
    # =====================================================================
    if args.year in ("2025", "both"):
        print("\n" + "=" * 80)
        print("PART 1: 2025 SEASON RETROSPECTIVE")
        print("Training: 2020-2024 | Simulating: 2025")
        print("=" * 80)

        # Train MAP model
        print("\nTraining Stage 9 MAP on 2020-2024...")
        loader = F1DataLoader(DATA_DIR)
        df = loader.load_merged(min_year=2020, max_year=2024)
        prev, nxt, meta = prepare_transitions(df)
        model_map = BayesianStateSpaceF1(
            sigma_d_candidates=(0.02, 0.05, 0.1, 0.2),
            sigma_c_candidates=(0.02, 0.05, 0.1, 0.2),
            n_mc_samples=args.n_mc_samples,
        )
        model_map.fit(prev, nxt, meta)

        # Setup
        driver_ids = list(DRIVERS_2025.keys())
        cid_list = [DRIVERS_2025[d][0] for d in driver_ids]
        cid_unique = sorted(CONSTRUCTOR_NAMES_2025.keys())
        did_to_idx = {d: i for i, d in enumerate(driver_ids)}

        # MAP simulation
        print("\nPrecomputing MAP probabilities...")
        pc_map = precompute_probs_map(model_map, DRIVERS_2025)
        print("Simulating 2025 — Stage 9 MAP...")
        pts_map, wins_map, cpts_map = run_simulation(
            pc_map, driver_ids, cid_list, cid_unique,
            np.random.default_rng(42), n_sims=n_sims
        )
        res_map = analyze_results(pts_map, wins_map, cpts_map, driver_ids,
                                  cid_unique, n_sims=n_sims)

        # NUTS simulation
        print("Precomputing Bayesian (NUTS) probabilities...")
        pc_nuts = precompute_probs_bayes(model_nuts, DRIVERS_2025,
                                          n_posterior_draws=args.n_posterior_draws)
        print("Simulating 2025 — Stage 9 NUTS...")
        pts_nuts, wins_nuts, cpts_nuts = run_simulation(
            pc_nuts, driver_ids, cid_list, cid_unique,
            np.random.default_rng(42), n_sims=n_sims
        )
        res_nuts = analyze_results(pts_nuts, wins_nuts, cpts_nuts, driver_ids,
                                   cid_unique, n_sims=n_sims)

        # Print
        print("\n" + "-" * 80)
        print("2025 DRIVERS' CHAMPIONSHIP")
        print("-" * 80)

        print_driver_table(
            "Stage 9 MAP", DRIVERS_2025, driver_ids, pts_map, wins_map,
            res_map, CONSTRUCTOR_NAMES_2025, did_to_idx, ACTUAL_2025,
            n_sims=n_sims
        )
        print_driver_table(
            "Stage 9 NUTS (Full Bayesian)", DRIVERS_2025, driver_ids,
            pts_nuts, wins_nuts, res_nuts, CONSTRUCTOR_NAMES_2025,
            did_to_idx, ACTUAL_2025, n_sims=n_sims
        )

        print("\n" + "-" * 80)
        print("2025 CONSTRUCTORS' CHAMPIONSHIP")
        print("-" * 80)

        print_constructor_table("Stage 9 MAP", cpts_map, res_map, cid_unique,
                                CONSTRUCTOR_NAMES_2025, ACTUAL_WCC_2025,
                                n_sims=n_sims)
        print_constructor_table("Stage 9 NUTS", cpts_nuts, res_nuts, cid_unique,
                                CONSTRUCTOR_NAMES_2025, ACTUAL_WCC_2025,
                                n_sims=n_sims)

        # Accuracy comparison
        print("\n" + "-" * 80)
        print("2025 MODEL ACCURACY COMPARISON")
        print("-" * 80)

        acc_map = compute_accuracy_metrics(
            driver_ids, DRIVERS_2025, pts_map, wins_map, res_map,
            did_to_idx, ACTUAL_2025, n_sims=n_sims
        )
        acc_nuts = compute_accuracy_metrics(
            driver_ids, DRIVERS_2025, pts_nuts, wins_nuts, res_nuts,
            did_to_idx, ACTUAL_2025, n_sims=n_sims
        )

        norris_idx = did_to_idx[846]
        ver_idx = did_to_idx[830]
        mclaren_cidx = {cid: i for i, cid in enumerate(cid_unique)}[1]

        print(f"\n  {'Metric':<35} {'MAP':>12} {'NUTS':>12}")
        print(f"  {'-'*60}")
        print(f"  {'Spearman rho (rank order)':<35} {acc_map['spearman_rho']:>12.3f} {acc_nuts['spearman_rho']:>12.3f}")
        print(f"  {'Points MAE':<35} {acc_map['pts_mae']:>12.1f} {acc_nuts['pts_mae']:>12.1f}")
        print(f"  {'Rank MAE':<35} {acc_map['rank_mae']:>12.2f} {acc_nuts['rank_mae']:>12.2f}")

        p_norris_map = res_map["driver_champ_count"][norris_idx] / n_sims
        p_norris_nuts = res_nuts["driver_champ_count"][norris_idx] / n_sims
        print(f"  {'P(Norris WDC) [actual champ]':<35} {p_norris_map:>12.1%} {p_norris_nuts:>12.1%}")

        p_ver_map = res_map["driver_champ_count"][ver_idx] / n_sims
        p_ver_nuts = res_nuts["driver_champ_count"][ver_idx] / n_sims
        print(f"  {'P(Verstappen WDC)':<35} {p_ver_map:>12.1%} {p_ver_nuts:>12.1%}")

        p_mcl_map = res_map["constr_champ_count"][mclaren_cidx] / n_sims
        p_mcl_nuts = res_nuts["constr_champ_count"][mclaren_cidx] / n_sims
        print(f"  {'P(McLaren WCC) [actual champ]':<35} {p_mcl_map:>12.1%} {p_mcl_nuts:>12.1%}")

        # Probability spread analysis
        print("\n  Probability spread (P(WDC) max driver):")
        max_p_map = max(res_map["driver_champ_count"]) / n_sims
        max_p_nuts = max(res_nuts["driver_champ_count"]) / n_sims
        print(f"    MAP:  {max_p_map:.1%}  |  NUTS: {max_p_nuts:.1%}")
        print(f"    {'→ NUTS reduces overconcentration' if max_p_nuts < max_p_map else '→ Unexpected: NUTS more concentrated'}")

    # =====================================================================
    # 2026 PREDICTION
    # =====================================================================
    if args.year in ("2026", "both"):
        print("\n\n" + "=" * 80)
        print("PART 2: 2026 SEASON PREDICTION")
        print("Training: 2021-2025 | Simulating: 2026")
        print("=" * 80)

        # Train MAP model
        print("\nTraining Stage 9 MAP on 2021-2025...")
        loader = F1DataLoader(DATA_DIR)
        df = loader.load_merged(min_year=2021, max_year=2025)
        prev, nxt, meta = prepare_transitions(df)
        model_map_26 = BayesianStateSpaceF1(
            sigma_d_candidates=(0.02, 0.05, 0.1, 0.2),
            sigma_c_candidates=(0.02, 0.05, 0.1, 0.2),
            n_mc_samples=args.n_mc_samples,
        )
        model_map_26.fit(prev, nxt, meta)

        # Setup
        driver_ids = list(DRIVERS_2026.keys())
        cid_list = [DRIVERS_2026[d][0] for d in driver_ids]
        cid_unique = sorted(CONSTRUCTOR_NAMES_2026.keys())
        did_to_idx = {d: i for i, d in enumerate(driver_ids)}

        # MAP simulation
        print("\nPrecomputing MAP probabilities...")
        pc_map = precompute_probs_map(model_map_26, DRIVERS_2026)
        print("Simulating 2026 — Stage 9 MAP...")
        pts_map, wins_map, cpts_map = run_simulation(
            pc_map, driver_ids, cid_list, cid_unique,
            np.random.default_rng(42), n_sims=n_sims
        )
        res_map = analyze_results(pts_map, wins_map, cpts_map, driver_ids,
                                  cid_unique, n_sims=n_sims)

        # NUTS simulation
        print("Precomputing Bayesian (NUTS) probabilities...")
        pc_nuts = precompute_probs_bayes(model_nuts, DRIVERS_2026,
                                          n_posterior_draws=args.n_posterior_draws)
        print("Simulating 2026 — Stage 9 NUTS...")
        pts_nuts, wins_nuts, cpts_nuts = run_simulation(
            pc_nuts, driver_ids, cid_list, cid_unique,
            np.random.default_rng(42), n_sims=n_sims
        )
        res_nuts = analyze_results(pts_nuts, wins_nuts, cpts_nuts, driver_ids,
                                   cid_unique, n_sims=n_sims)

        # Print
        print("\n" + "-" * 80)
        print("2026 DRIVERS' CHAMPIONSHIP PREDICTIONS")
        print("-" * 80)

        print_driver_table(
            "Stage 9 MAP", DRIVERS_2026, driver_ids, pts_map, wins_map,
            res_map, CONSTRUCTOR_NAMES_2026, did_to_idx, n_sims=n_sims
        )
        print_driver_table(
            "Stage 9 NUTS (Full Bayesian)", DRIVERS_2026, driver_ids,
            pts_nuts, wins_nuts, res_nuts, CONSTRUCTOR_NAMES_2026,
            did_to_idx, n_sims=n_sims
        )

        print("\n" + "-" * 80)
        print("2026 CONSTRUCTORS' CHAMPIONSHIP PREDICTIONS")
        print("-" * 80)

        print_constructor_table("Stage 9 MAP", cpts_map, res_map, cid_unique,
                                CONSTRUCTOR_NAMES_2026, n_sims=n_sims)
        print_constructor_table("Stage 9 NUTS", cpts_nuts, res_nuts, cid_unique,
                                CONSTRUCTOR_NAMES_2026, n_sims=n_sims)

        # Concentration analysis
        print("\n" + "-" * 80)
        print("2026 CONCENTRATION ANALYSIS (MAP vs NUTS)")
        print("-" * 80)

        max_p_map = max(res_map["driver_champ_count"]) / n_sims
        max_p_nuts = max(res_nuts["driver_champ_count"]) / n_sims
        max_c_map = max(res_map["constr_champ_count"]) / n_sims
        max_c_nuts = max(res_nuts["constr_champ_count"]) / n_sims

        n_contenders_map = sum(1 for c in res_map["driver_champ_count"] if c > 0.01 * n_sims)
        n_contenders_nuts = sum(1 for c in res_nuts["driver_champ_count"] if c > 0.01 * n_sims)

        print(f"\n  {'Metric':<40} {'MAP':>10} {'NUTS':>10}")
        print(f"  {'-'*62}")
        print(f"  {'Max P(WDC) (any driver)':<40} {max_p_map:>10.1%} {max_p_nuts:>10.1%}")
        print(f"  {'Max P(WCC) (any team)':<40} {max_c_map:>10.1%} {max_c_nuts:>10.1%}")
        print(f"  {'WDC contenders (P > 1%)':<40} {n_contenders_map:>10} {n_contenders_nuts:>10}")
        print(f"\n  {'→ Lower max P and more contenders = less overconcentration = better calibration'}")


if __name__ == "__main__":
    main()
