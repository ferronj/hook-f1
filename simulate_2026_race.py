"""
Sequential race-by-race simulation for the 2026 F1 season.

For Race N, trains models on 2015-2025 historical data (hyperparameters fixed),
then incorporates observed 2026 results from Races 1..N-1 into model state,
and simulates Race N via Monte Carlo.

Usage:
    micromamba run -n f1-markov python3 simulate_2026_race.py --race N [--n-sims 10000]
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "models"))

from stage6_recency_constructor import (
    F1DataLoader as S6Loader,
    RecencyConstructorDirichletF1,
    prepare_transitions as s6_prepare,
    START, N_OUTCOMES, DNF,
)
from stage8_plackett_luce import TimeVaryingPlackettLuceF1
from stage9_bayesian_ss import BayesianStateSpaceF1

from config_2026 import (
    DRIVERS_2026, CONSTRUCTOR_NAMES, TEAM_COLORS,
    CALENDAR_2026, race_slug,
)

DATA_DIR = Path(__file__).parent / "data"
ACTIVE_STAGES = ["stage6", "stage8", "stage9"]


# ---------------------------------------------------------------------------
# Load observed 2026 race results from CSVs
# ---------------------------------------------------------------------------
def load_observed_2026_races(up_to_round):
    """
    Load observed 2026 race results for rounds 1..up_to_round-1 from CSVs.

    Returns list of dicts, each with:
        round: int
        results: list of (driver_id, constructor_id, finish_position, is_dnf)
        grid: dict of driver_id -> grid_position
    """
    races_csv = pd.read_csv(DATA_DIR / "races.csv")
    results_csv = pd.read_csv(DATA_DIR / "results.csv")

    # Find 2026 race IDs for rounds < up_to_round
    races_2026 = races_csv[
        (races_csv["year"] == 2026) & (races_csv["round"] < up_to_round)
    ].sort_values("round")

    observed = []
    for _, race_row in races_2026.iterrows():
        race_id = race_row["raceId"]
        round_num = race_row["round"]

        race_results = results_csv[results_csv["raceId"] == race_id]
        if race_results.empty:
            continue

        entries = []
        for _, r in race_results.iterrows():
            did = int(r["driverId"])
            cid = int(r["constructorId"])
            pos = int(r["positionOrder"])
            # Clamp positions >20 to 20 (model state space is P1-P20)
            pos = min(pos, 20)
            status = int(r["statusId"])
            is_dnf = status != 1  # anything other than "Finished"
            if is_dnf:
                pos = DNF  # 0
            entries.append((did, cid, pos, is_dnf))

        grid = {}
        for _, r in race_results.iterrows():
            did = int(r["driverId"])
            grid[did] = int(r["grid"]) if r["grid"] != "\\N" else 22

        observed.append({
            "round": int(round_num),
            "results": entries,
            "grid": grid,
        })

    return observed


# ---------------------------------------------------------------------------
# Simulation helpers (from simulate_2026_australia.py)
# ---------------------------------------------------------------------------
def simulate_single_race(model, driver_ids, prev_positions, n_sims, rng):
    """Simulate n_sims races and return position distributions."""
    n_drivers = len(driver_ids)
    all_results = np.zeros((n_sims, n_drivers), dtype=int)

    # Precompute probabilities from each driver's prev_position
    probs_list = []
    for i, did in enumerate(driver_ids):
        cid = DRIVERS_2026[did][0]
        prev_pos = prev_positions[i]

        known = False
        if hasattr(model, 'driver_constructor_counts_') and did in model.driver_constructor_counts_:
            known = True
        elif hasattr(model, 'driver_strengths_') and did in model.driver_strengths_:
            known = True

        if known:
            probs = model.predict_proba(did, prev_pos, constructor_id=cid)
        else:
            probs = model.predict_proba_new_driver(prev_pos, constructor_id=cid)
        probs_list.append(probs)

    probs_array = np.array(probs_list)

    for sim in range(n_sims):
        for i in range(n_drivers):
            all_results[sim, i] = rng.choice(N_OUTCOMES, p=probs_array[i])

    return all_results, probs_array


def compute_stats(all_results, probs_array, driver_ids, n_sims):
    """Compute detailed statistics from simulation results."""
    stats = []
    for i, did in enumerate(driver_ids):
        cid, name, abbr = DRIVERS_2026[did]
        team = CONSTRUCTOR_NAMES[cid]
        probs = probs_array[i]

        pos_counts = np.bincount(all_results[:, i], minlength=N_OUTCOMES)
        pos_dist = pos_counts / n_sims

        p_win = pos_dist[1]
        p_podium = pos_dist[1] + pos_dist[2] + pos_dist[3]
        p_points = sum(pos_dist[1:11])
        p_dnf = pos_dist[0]

        finishing = all_results[:, i][all_results[:, i] > 0]
        e_pos = finishing.mean() if len(finishing) > 0 else 20.0

        p5 = np.percentile(finishing, 5) if len(finishing) > 0 else 1
        p25 = np.percentile(finishing, 25) if len(finishing) > 0 else 5
        p50 = np.percentile(finishing, 50) if len(finishing) > 0 else 10
        p75 = np.percentile(finishing, 75) if len(finishing) > 0 else 15
        p95 = np.percentile(finishing, 95) if len(finishing) > 0 else 20

        stats.append({
            "driver_id": int(did),
            "name": name,
            "abbreviation": abbr,
            "team": team,
            "team_color": TEAM_COLORS.get(team, "#888888"),
            "constructor_id": int(cid),
            "p_win": float(p_win),
            "p_podium": float(p_podium),
            "p_points": float(p_points),
            "p_dnf": float(p_dnf),
            "e_pos": float(e_pos),
            "percentiles": {
                "p5": float(p5), "p25": float(p25), "p50": float(p50),
                "p75": float(p75), "p95": float(p95),
            },
            "position_distribution": [float(x) for x in pos_dist],
            "model_probs": [float(x) for x in probs],
        })

    return sorted(stats, key=lambda x: -x["p_podium"])


def build_composite(all_probs, driver_ids, n_sims):
    """Build composite model by equal-weight blending."""
    n_models = len(all_probs)
    composite_probs = sum(all_probs) / n_models
    for i in range(composite_probs.shape[0]):
        s = composite_probs[i].sum()
        if s > 0:
            composite_probs[i] /= s

    rng_comp = np.random.default_rng(42)
    results_comp = np.zeros((n_sims, len(driver_ids)), dtype=int)
    for sim in range(n_sims):
        for i in range(len(driver_ids)):
            results_comp[sim, i] = rng_comp.choice(N_OUTCOMES, p=composite_probs[i])

    return results_comp, composite_probs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Simulate 2026 F1 race N with sequential model updates"
    )
    parser.add_argument("--race", type=int, required=True,
                        help="Race number to predict (1-24)")
    parser.add_argument("--n-sims", type=int, default=10000,
                        help="Number of Monte Carlo simulations (default: 10000)")
    args = parser.parse_args()

    race_num = args.race
    n_sims = args.n_sims

    if race_num not in CALENDAR_2026:
        print(f"Error: Race {race_num} not in 2026 calendar (1-24)")
        sys.exit(1)

    race_info = CALENDAR_2026[race_num]
    race_name = f"2026 {race_info['name']}"
    prediction_type = "pre_season" if race_num == 1 else "in_season"

    print("=" * 70)
    print(f"2026 RACE {race_num}: {race_info['name'].upper()}")
    print(f"Circuit: {race_info['circuit']}")
    print(f"Date: {race_info['date']}")
    print(f"Monte Carlo: {n_sims:,} simulated races per model")
    print(f"Active models: {', '.join(ACTIVE_STAGES)}")
    print("=" * 70)

    rng = np.random.default_rng(42)
    driver_ids = list(DRIVERS_2026.keys())

    # ==================================================================
    # Step 1: Train models on 2015-2025 historical data
    # ==================================================================
    print("\nStep 1: Training models on 2015-2025 data...")

    loader = S6Loader(DATA_DIR)
    df = loader.load_merged(min_year=2015, max_year=2025)
    prev, nxt, meta = s6_prepare(df)

    print("  Training Stage 6 (Year-Weighted Constructor)...")
    model6 = RecencyConstructorDirichletF1(
        prior_alpha_global=1.0,
        prior_alpha_constructor=1.0,
        kappa_init=(10.0, 10.0),
        kappa_bounds=((0.1, 500.0), (0.01, 500.0)),
        w_candidates=(0.3, 0.5, 0.7, 0.85, 1.0),
    )
    model6.fit(prev, nxt, meta)
    print(f"    kappa_g={model6.kappa_g_:.2f}, kappa_c={model6.kappa_c_:.2f}, w={model6.w_:.2f}")

    print("  Training Stage 8 (Time-Varying Plackett-Luce)...")
    model8 = TimeVaryingPlackettLuceF1(
        alpha_candidates=(0.9, 0.95, 0.99),
        n_mc_samples=3000,
    )
    model8.fit(prev, nxt, meta)
    print(f"    alpha={model8.alpha_:.3f}")

    print("  Training Stage 9 (Bayesian State-Space)...")
    model9 = BayesianStateSpaceF1(
        sigma_d_candidates=(0.02, 0.05, 0.1, 0.2),
        sigma_c_candidates=(0.02, 0.05, 0.1, 0.2),
        sigma_0=1.0,
        dnf_shrinkage=20.0,
        n_mc_samples=3000,
        center_penalty=0.01,
        maxiter=500,
    )
    model9.fit(prev, nxt, meta)
    print(f"    sigma_d={model9.sigma_d_:.3f}, sigma_c={model9.sigma_c_:.3f}")

    models = {"stage6": model6, "stage8": model8, "stage9": model9}

    # ==================================================================
    # Step 2: Incorporate observed 2026 races (1..N-1)
    # ==================================================================
    prev_positions = {did: START for did in driver_ids}

    if race_num > 1:
        observed = load_observed_2026_races(race_num)
        n_observed = len(observed)
        print(f"\nStep 2: Incorporating {n_observed} observed 2026 race(s)...")

        for obs_race in observed:
            rnd = obs_race["round"]
            race_name_obs = CALENDAR_2026[rnd]["name"]
            print(f"  Round {rnd} ({race_name_obs}): {len(obs_race['results'])} entries")

            # Stage 6: needs (did, cid, prev_pos, finish_pos)
            s6_results = []
            for did, cid, pos, is_dnf in obs_race["results"]:
                s6_results.append((did, cid, prev_positions.get(did, START), pos))
            model6.incorporate_race(s6_results, 2026)

            # Stage 8 & 9: need (did, cid, finish_pos, is_dnf)
            model8.incorporate_race(obs_race["results"])
            model9.incorporate_race(obs_race["results"])

            # Update prev_positions for next race
            for did, cid, pos, is_dnf in obs_race["results"]:
                prev_positions[did] = pos
    else:
        print("\nStep 2: No observed races (predicting Race 1 from START)")

    # ==================================================================
    # Step 3: Simulate Race N
    # ==================================================================
    print(f"\nStep 3: Simulating Race {race_num} ({race_info['name']})...")

    # Build prev_position array for all drivers
    prev_pos_array = np.array([prev_positions.get(did, START) for did in driver_ids])

    stage_probs = {}
    stage_stats = {}
    stage_params = {}
    stage_meta = {}

    # Stage 6
    print(f"  Simulating {n_sims:,} races with Stage 6...")
    r6, p6 = simulate_single_race(model6, driver_ids, prev_pos_array, n_sims, rng)
    stage_probs["stage6"] = p6
    stage_stats["stage6"] = compute_stats(r6, p6, driver_ids, n_sims)
    stage_params["stage6"] = {
        "kappa_g": float(model6.kappa_g_),
        "kappa_c": float(model6.kappa_c_),
        "w": float(model6.w_),
    }
    stage_meta["stage6"] = {
        "name": "Stage 6: Year-Weighted Constructor",
        "description": "Dirichlet-Multinomial with recency-weighted constructor priors. Trained 2015-2025.",
    }

    # Stage 8
    print(f"  Simulating {n_sims:,} races with Stage 8...")
    r8, p8 = simulate_single_race(model8, driver_ids, prev_pos_array, n_sims, rng)
    stage_probs["stage8"] = p8
    stage_stats["stage8"] = compute_stats(r8, p8, driver_ids, n_sims)
    stage_params["stage8"] = {"alpha": float(model8.alpha_)}
    stage_meta["stage8"] = {
        "name": "Stage 8: Time-Varying Plackett-Luce",
        "description": "Time-varying driver strengths with Plackett-Luce ranking model. Trained 2015-2025.",
    }

    # Stage 9
    print(f"  Simulating {n_sims:,} races with Stage 9...")
    r9, p9 = simulate_single_race(model9, driver_ids, prev_pos_array, n_sims, rng)
    stage_probs["stage9"] = p9
    stage_stats["stage9"] = compute_stats(r9, p9, driver_ids, n_sims)
    stage_params["stage9"] = {
        "sigma_d": float(model9.sigma_d_),
        "sigma_c": float(model9.sigma_c_),
    }
    stage_meta["stage9"] = {
        "name": "Stage 9: Bayesian State-Space",
        "description": "Random walk on driver/constructor log-strengths with Plackett-Luce observations. Trained 2015-2025.",
    }

    # ==================================================================
    # Composite
    # ==================================================================
    COMPOSITE_STAGES = []
    for sk in ACTIVE_STAGES:
        max_pwin = max(d["p_win"] for d in stage_stats[sk])
        if max_pwin > 0.50:
            print(f"  WARNING: {stage_meta[sk]['name']} degenerate "
                  f"(max P(win) = {max_pwin:.1%}). Excluded from composite.")
            stage_meta[sk]["description"] += " Warning: Degenerate — excluded from composite."
        else:
            COMPOSITE_STAGES.append(sk)

    if not COMPOSITE_STAGES:
        COMPOSITE_STAGES = ACTIVE_STAGES

    active_probs = [stage_probs[k] for k in COMPOSITE_STAGES]
    stage_names_for_composite = [
        stage_meta[k]["name"].split(":")[0].strip() for k in COMPOSITE_STAGES
    ]
    blend_desc = " + ".join(stage_names_for_composite)
    n_active = len(active_probs)
    weight = f"1/{n_active}"

    print(f"\n  Computing composite ({weight} each: {blend_desc})...")
    results_comp, probs_comp = build_composite(active_probs, driver_ids, n_sims)
    stats_comp = compute_stats(results_comp, probs_comp, driver_ids, n_sims)

    # ==================================================================
    # Build output JSON
    # ==================================================================
    models_dict = {}
    for sk in ACTIVE_STAGES:
        models_dict[sk] = {
            "name": stage_meta[sk]["name"],
            "description": stage_meta[sk]["description"],
            "params": stage_params[sk],
            "drivers": stage_stats[sk],
        }

    models_dict["composite"] = {
        "name": f"Composite ({n_active}-Model Blend)",
        "description": f"Equal-weight blend of {blend_desc}.",
        "params": {"blend": f"{weight} each of {blend_desc}"},
        "drivers": stats_comp,
    }

    output = {
        "race": race_name,
        "race_number": race_num,
        "circuit": race_info["circuit"],
        "date": race_info["date"],
        "prediction_type": prediction_type,
        "observed_races": race_num - 1,
        "n_sims": n_sims,
        "training_years": "2015-2025",
        "active_stages": ACTIVE_STAGES,
        "models": models_dict,
        "team_colors": TEAM_COLORS,
        "constructor_names": {str(k): v for k, v in CONSTRUCTOR_NAMES.items()},
    }

    slug = race_slug(race_num)
    out_path = DATA_DIR / f"sim_2026_r{race_num:02d}_{slug}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {out_path}")

    # Print summary
    for model_key in ACTIVE_STAGES + ["composite"]:
        drivers = output["models"][model_key]["drivers"]
        model_name_str = output["models"][model_key]["name"]
        print(f"\n{'='*60}")
        print(f"{model_name_str} — Top 10")
        print(f"{'='*60}")
        print(f"{'Pos':>4} {'Driver':<20} {'Team':<15} {'P(Win)':>8} {'P(Pod)':>8} {'E[Pos]':>8}")
        for j, d in enumerate(drivers[:10]):
            print(f"{j+1:>4} {d['name']:<20} {d['team']:<15} "
                  f"{d['p_win']:>7.1%} {d['p_podium']:>7.1%} {d['e_pos']:>7.1f}")


if __name__ == "__main__":
    main()
