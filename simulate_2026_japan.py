"""
Monte Carlo simulation of the 2026 Japanese GP using Stage 6, Stage 8, and Stage 9 models.

Runs 10,000 simulated races per model and saves detailed results as JSON
for the dashboard to consume. Also computes a composite model (equal-weight blend
of all active stage models).
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "models"))

from stage6_recency_constructor import (
    F1DataLoader as S6Loader,
    RecencyConstructorDirichletF1,
    prepare_transitions as s6_prepare,
    START, N_OUTCOMES,
)
from stage8_plackett_luce import TimeVaryingPlackettLuceF1
from stage9_bayesian_ss import BayesianStateSpaceF1

DATA_DIR = Path(__file__).parent / "data"
N_SIMS = 10000

# 2026 Japanese GP drivers: driverId -> (constructorId, name, abbreviation)
DRIVERS_2026 = {
    846: (1, "Lando Norris", "NOR"),
    857: (1, "Oscar Piastri", "PIA"),
    1:   (6, "Lewis Hamilton", "HAM"),
    844: (6, "Charles Leclerc", "LEC"),
    847: (131, "George Russell", "RUS"),
    863: (131, "Kimi Antonelli", "ANT"),
    830: (9, "Max Verstappen", "VER"),
    864: (9, "Isack Hadjar", "HAD"),
    4:   (117, "Fernando Alonso", "ALO"),
    840: (117, "Lance Stroll", "STR"),
    832: (3, "Carlos Sainz", "SAI"),
    848: (3, "Alex Albon", "ALB"),
    861: (214, "Franco Colapinto", "COL"),
    842: (214, "Pierre Gasly", "GAS"),
    839: (210, "Esteban Ocon", "OCO"),
    860: (210, "Oliver Bearman", "BEA"),
    807: (15, "Nico Hulkenberg", "HUL"),
    865: (15, "Gabriel Bortoleto", "BOR"),
    859: (215, "Liam Lawson", "LAW"),
    866: (215, "Arvid Lindblad", "LIN"),
    822: (216, "Valtteri Bottas", "BOT"),
    815: (216, "Sergio Perez", "PER"),
}

CONSTRUCTOR_NAMES = {
    1: "McLaren", 6: "Ferrari", 131: "Mercedes", 9: "Red Bull",
    117: "Aston Martin", 3: "Williams", 214: "Alpine",
    210: "Haas", 15: "Audi", 215: "Racing Bulls", 216: "Cadillac",
}

TEAM_COLORS = {
    "McLaren": "#FF8000", "Ferrari": "#E8002D", "Mercedes": "#27F4D2",
    "Red Bull": "#3671C6", "Aston Martin": "#229971", "Williams": "#64C4FF",
    "Alpine": "#0093CC", "Haas": "#B6BABD", "Audi": "#00594F",
    "Racing Bulls": "#6692FF", "Cadillac": "#1B3D2F",
}

ACTIVE_STAGES = ["stage6", "stage8", "stage9"]


def simulate_single_race(model, model_name, driver_ids, rng):
    """Simulate N_SIMS races and return position distributions."""
    n_drivers = len(driver_ids)
    all_results = np.zeros((N_SIMS, n_drivers), dtype=int)

    probs_list = []
    for did in driver_ids:
        cid = DRIVERS_2026[did][0]
        known = False
        if hasattr(model, 'driver_constructor_counts_') and did in model.driver_constructor_counts_:
            known = True
        elif hasattr(model, 'driver_strengths_') and did in model.driver_strengths_:
            known = True

        if known:
            probs = model.predict_proba(did, START, constructor_id=cid)
        else:
            probs = model.predict_proba_new_driver(START, constructor_id=cid)
        probs_list.append(probs)

    probs_array = np.array(probs_list)

    for sim in range(N_SIMS):
        for i in range(n_drivers):
            all_results[sim, i] = rng.choice(N_OUTCOMES, p=probs_array[i])

    return all_results, probs_array


def compute_stats(all_results, probs_array, driver_ids):
    """Compute detailed statistics from simulation results."""
    stats = []

    for i, did in enumerate(driver_ids):
        cid, name, abbr = DRIVERS_2026[did]
        team = CONSTRUCTOR_NAMES[cid]
        probs = probs_array[i]

        pos_counts = np.bincount(all_results[:, i], minlength=N_OUTCOMES)
        pos_dist = pos_counts / N_SIMS

        p_win = pos_dist[1]
        p_podium = pos_dist[1] + pos_dist[2] + pos_dist[3]
        p_points = sum(pos_dist[1:11])
        p_dnf = pos_dist[0]

        finishing_positions = all_results[:, i][all_results[:, i] > 0]
        e_pos = finishing_positions.mean() if len(finishing_positions) > 0 else 20.0

        p5 = np.percentile(finishing_positions, 5) if len(finishing_positions) > 0 else 1
        p25 = np.percentile(finishing_positions, 25) if len(finishing_positions) > 0 else 5
        p50 = np.percentile(finishing_positions, 50) if len(finishing_positions) > 0 else 10
        p75 = np.percentile(finishing_positions, 75) if len(finishing_positions) > 0 else 15
        p95 = np.percentile(finishing_positions, 95) if len(finishing_positions) > 0 else 20

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


def build_composite(all_probs, driver_ids):
    """Build composite model by equal-weight blending all stage model probabilities."""
    n_models = len(all_probs)
    composite_probs = sum(all_probs) / n_models
    for i in range(composite_probs.shape[0]):
        s = composite_probs[i].sum()
        if s > 0:
            composite_probs[i] /= s

    rng_comp = np.random.default_rng(42)
    results_comp = np.zeros((N_SIMS, len(driver_ids)), dtype=int)
    for sim in range(N_SIMS):
        for i in range(len(driver_ids)):
            results_comp[sim, i] = rng_comp.choice(N_OUTCOMES, p=composite_probs[i])

    return results_comp, composite_probs


def main():
    print("=" * 70)
    print("2026 JAPANESE GP SIMULATION")
    print(f"Monte Carlo: {N_SIMS:,} simulated races per model")
    print(f"Active models: {', '.join(ACTIVE_STAGES)}")
    print("=" * 70)

    rng = np.random.default_rng(42)
    driver_ids = list(DRIVERS_2026.keys())

    stage_probs = {}
    stage_stats = {}
    stage_results = {}
    stage_params = {}
    stage_meta = {}

    # ===================================================================
    # Stage 6: Year-Weighted Constructor
    # ===================================================================
    print("\nTraining Stage 6 (Year-Weighted Constructor, 2015-2025)...")
    loader6 = S6Loader(DATA_DIR)
    df6 = loader6.load_merged(min_year=2015, max_year=2025)
    prev6, next6, meta6 = s6_prepare(df6)

    model6 = RecencyConstructorDirichletF1(
        prior_alpha_global=1.0,
        prior_alpha_constructor=1.0,
        kappa_init=(10.0, 10.0),
        kappa_bounds=((0.1, 500.0), (0.01, 500.0)),
        w_candidates=(0.3, 0.5, 0.7, 0.85, 1.0),
    )
    model6.fit(prev6, next6, meta6)
    print(f"  kappa_g={model6.kappa_g_:.2f}, kappa_c={model6.kappa_c_:.2f}, w={model6.w_:.2f}")

    print(f"Simulating {N_SIMS:,} races with Stage 6...")
    r6, p6 = simulate_single_race(model6, "Stage 6", driver_ids, rng)
    stage_probs["stage6"] = p6
    stage_results["stage6"] = r6
    stage_stats["stage6"] = compute_stats(r6, p6, driver_ids)
    stage_params["stage6"] = {
        "kappa_g": float(model6.kappa_g_),
        "kappa_c": float(model6.kappa_c_),
        "w": float(model6.w_),
    }
    stage_meta["stage6"] = {
        "name": "Stage 6: Year-Weighted Constructor",
        "description": "Dirichlet-Multinomial with recency-weighted constructor priors. Best calibrated model (LL/race = -59.0). Trained 2015-2025.",
    }

    # ===================================================================
    # Stage 8: Time-Varying Plackett-Luce
    # ===================================================================
    print("\nTraining Stage 8 (Time-Varying Plackett-Luce, 2015-2025)...")
    model8 = TimeVaryingPlackettLuceF1(
        alpha_candidates=(0.9, 0.95, 0.99),
        n_mc_samples=3000,
    )
    model8.fit(prev6, next6, meta6)
    print(f"  alpha={model8.alpha_:.3f}")

    print(f"Simulating {N_SIMS:,} races with Stage 8...")
    r8, p8 = simulate_single_race(model8, "Stage 8", driver_ids, rng)
    stage_probs["stage8"] = p8
    stage_results["stage8"] = r8
    stage_stats["stage8"] = compute_stats(r8, p8, driver_ids)
    stage_params["stage8"] = {"alpha": float(model8.alpha_)}
    stage_meta["stage8"] = {
        "name": "Stage 8: Time-Varying Plackett-Luce",
        "description": "Time-varying driver strengths with Plackett-Luce ranking model. Best Spearman rho (0.800) but overconcentrates probability.",
    }

    # ===================================================================
    # Stage 9: Bayesian State-Space
    # ===================================================================
    print("\nTraining Stage 9 (Bayesian State-Space, 2015-2025)...")
    model9 = BayesianStateSpaceF1(
        sigma_d_candidates=(0.02, 0.05, 0.1, 0.2),
        sigma_c_candidates=(0.02, 0.05, 0.1, 0.2),
        sigma_0=1.0,
        dnf_shrinkage=20.0,
        n_mc_samples=3000,
        center_penalty=0.01,
        maxiter=500,
    )
    model9.fit(prev6, next6, meta6)
    print(f"  sigma_d={model9.sigma_d_:.3f}, sigma_c={model9.sigma_c_:.3f}")

    print(f"Simulating {N_SIMS:,} races with Stage 9...")
    r9, p9 = simulate_single_race(model9, "Stage 9", driver_ids, rng)
    stage_probs["stage9"] = p9
    stage_results["stage9"] = r9
    stage_stats["stage9"] = compute_stats(r9, p9, driver_ids)
    stage_params["stage9"] = {
        "sigma_d": float(model9.sigma_d_),
        "sigma_c": float(model9.sigma_c_),
    }
    stage_meta["stage9"] = {
        "name": "Stage 9: Bayesian State-Space",
        "description": "Random walk on driver/constructor log-strengths with Plackett-Luce observations. Best top-3 accuracy (1.59/3).",
    }

    # ===================================================================
    # Sanity check: flag degenerate models
    # ===================================================================
    COMPOSITE_STAGES = []
    for sk in ACTIVE_STAGES:
        max_pwin = max(d["p_win"] for d in stage_stats[sk])
        if max_pwin > 0.50:
            print(f"  WARNING: {stage_meta[sk]['name']} is degenerate "
                  f"(max P(win) = {max_pwin:.1%}). Excluded from composite.")
            stage_meta[sk]["description"] += " \u26a0\ufe0f Degenerate predictions \u2014 excluded from composite."
        else:
            COMPOSITE_STAGES.append(sk)

    if not COMPOSITE_STAGES:
        COMPOSITE_STAGES = ACTIVE_STAGES

    # ===================================================================
    # Composite Model (equal-weight blend of non-degenerate stage models)
    # ===================================================================
    active_probs = [stage_probs[k] for k in COMPOSITE_STAGES]
    stage_names_for_composite = [stage_meta[k]["name"].split(":")[0].strip() for k in COMPOSITE_STAGES]
    blend_desc = " + ".join(stage_names_for_composite)
    n_active = len(active_probs)
    weight = f"1/{n_active}"

    print(f"\nComputing composite ({weight} each: {blend_desc})...")
    results_comp, probs_comp = build_composite(active_probs, driver_ids)
    stats_comp = compute_stats(results_comp, probs_comp, driver_ids)

    # ===================================================================
    # 2025 Japanese GP actual result (calibration reference)
    # ===================================================================
    calibration_2025 = {
        "race": "2025 Japanese GP",
        "top3": ["Max Verstappen", "Lando Norris", "Oscar Piastri"],
        "top10": [
            "Max Verstappen", "Lando Norris", "Oscar Piastri",
            "Charles Leclerc", "George Russell", "Kimi Antonelli",
            "Lewis Hamilton", "Isack Hadjar", "Alex Albon", "Oliver Bearman",
        ],
    }

    # ===================================================================
    # Build output JSON
    # ===================================================================
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
        "description": f"Equal-weight blend of {blend_desc} probabilities, renormalized. Balances calibration, ranking accuracy, and robustness.",
        "params": {"blend": f"{weight} each of {blend_desc}"},
        "drivers": stats_comp,
    }

    output = {
        "race": "2026 Japanese Grand Prix",
        "circuit": "Suzuka Circuit, Suzuka",
        "date": "2026-04-05",
        "n_sims": N_SIMS,
        "training_years": "2015-2025",
        "active_stages": ACTIVE_STAGES,
        "calibration_2025": calibration_2025,
        "models": models_dict,
        "team_colors": TEAM_COLORS,
        "constructor_names": {str(k): v for k, v in CONSTRUCTOR_NAMES.items()},
    }

    out_path = Path(__file__).parent / "data" / "sim_2026_japan.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {out_path}")

    # Print summary for all models + composite
    all_model_keys = ACTIVE_STAGES + ["composite"]
    for model_key in all_model_keys:
        drivers = output["models"][model_key]["drivers"]
        model_name = output["models"][model_key]["name"]
        print(f"\n{'='*60}")
        print(f"{model_name} \u2014 Top 10")
        print(f"{'='*60}")
        print(f"{'Rank':<5} {'Driver':<20} {'Team':<15} {'P(win)':>8} {'P(pod)':>8} {'E[pos]':>7}")
        print("-" * 65)
        for rank, d in enumerate(drivers[:10], 1):
            print(f"{rank:<5} {d['name']:<20} {d['team']:<15} "
                  f"{d['p_win']:>8.1%} {d['p_podium']:>8.1%} {d['e_pos']:>7.1f}")


if __name__ == "__main__":
    main()
