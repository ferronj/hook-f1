"""
General-purpose Monte Carlo race simulation for any F1 race.

Trains Stage 6, 8, and 9 models, runs N simulated races, and saves
results as JSON for the dashboard to consume.

Usage:
    # Round 1 of 2026 (auto-detects race metadata from CSVs):
    uv run python simulate_race.py --season 2026 --round 1

    # Round 2 (uses R1 finishing positions as starting state):
    uv run python simulate_race.py --season 2026 --round 2

    # With roster override for new/changed drivers:
    uv run python simulate_race.py --season 2026 --round 1 --roster data/roster_2026.json

    # Explicit race metadata (for future races not yet in CSVs):
    uv run python simulate_race.py --season 2026 --round 1 \
        --race-name "Australian Grand Prix" \
        --circuit "Albert Park, Melbourne" \
        --date 2026-03-08
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
    START, N_OUTCOMES,
)
from stage8_plackett_luce import TimeVaryingPlackettLuceF1
from stage9_bayesian_ss import BayesianStateSpaceF1

DATA_DIR = Path(__file__).parent / "data"

TEAM_COLORS = {
    "McLaren": "#FF8000", "Ferrari": "#E8002D", "Mercedes": "#27F4D2",
    "Red Bull": "#3671C6", "Aston Martin": "#229971", "Williams": "#64C4FF",
    "Alpine": "#0093CC", "Haas": "#B6BABD", "Sauber": "#00594F",
    "Audi": "#00594F", "Racing Bulls": "#6692FF", "Cadillac": "#1B3D2F",
    "AlphaTauri": "#6692FF", "Alfa Romeo": "#B12845",
    "Racing Point": "#F596C8", "Renault": "#FFF500",
    "Toro Rosso": "#469BFF", "Force India": "#F596C8",
}

CLASSIFIED_STATUS_KEYWORDS = {"finished", "lap", "laps"}

# =====================================================================
# MODEL REGISTRY
# =====================================================================
MODEL_REGISTRY = {
    "stage6": {
        "class": RecencyConstructorDirichletF1,
        "kwargs": {
            "prior_alpha_global": 1.0,
            "prior_alpha_constructor": 1.0,
            "kappa_init": (10.0, 10.0),
            "kappa_bounds": ((0.1, 500.0), (0.01, 500.0)),
            "w_candidates": (0.3, 0.5, 0.7, 0.85, 1.0),
        },
        "params_fn": lambda m: {"kappa_g": m.kappa_g_, "kappa_c": m.kappa_c_, "w": m.w_},
        "meta": {
            "name": "Stage 6: Year-Weighted Constructor",
            "description": "Dirichlet-Multinomial with recency-weighted constructor priors. Best calibrated model (LL/race = -59.0).",
        },
    },
    "stage8": {
        "class": TimeVaryingPlackettLuceF1,
        "kwargs": {
            "alpha_candidates": (0.9, 0.95, 0.99),
            "n_mc_samples": 3000,
        },
        "params_fn": lambda m: {"alpha": m.alpha_},
        "meta": {
            "name": "Stage 8: Time-Varying Plackett-Luce",
            "description": "Time-varying driver strengths with Plackett-Luce ranking model. Best Spearman rho (0.800) but overconcentrates probability.",
        },
    },
    "stage9": {
        "class": BayesianStateSpaceF1,
        "kwargs": {
            "sigma_d_candidates": (0.02, 0.05, 0.1, 0.2),
            "sigma_c_candidates": (0.02, 0.05, 0.1, 0.2),
            "sigma_0": 1.0,
            "dnf_shrinkage": 20.0,
            "n_mc_samples": 3000,
            "center_penalty": 0.01,
            "maxiter": 500,
        },
        "params_fn": lambda m: {"sigma_d": m.sigma_d_, "sigma_c": m.sigma_c_},
        "meta": {
            "name": "Stage 9: Bayesian State-Space",
            "description": "Random walk on driver/constructor log-strengths with Plackett-Luce observations. Best top-3 accuracy (1.59/3).",
        },
    },
}


# =====================================================================
# ROSTER & RACE METADATA AUTO-DETECTION
# =====================================================================

def load_race_metadata(season, round_num):
    """Look up race name, circuit, and date from CSVs."""
    races_df = pd.read_csv(DATA_DIR / "races.csv")
    circuits_df = pd.read_csv(DATA_DIR / "circuits.csv")

    race_row = races_df[
        (races_df["year"] == season) & (races_df["round"] == round_num)
    ]
    if race_row.empty:
        return None, None, None

    race_row = race_row.iloc[0]
    circuit_row = circuits_df[circuits_df["circuitId"] == race_row["circuitId"]]
    if circuit_row.empty:
        circuit_str = "Unknown Circuit"
    else:
        c = circuit_row.iloc[0]
        circuit_str = f"{c['name']}, {c['location']}"

    return race_row["name"], circuit_str, str(race_row["date"])


def build_roster(season, round_num, roster_override_path=None):
    """Auto-detect driver roster from results data.

    Returns:
        drivers_dict: {driver_id: (constructor_id, full_name, abbreviation)}
        constructor_names: {constructor_id: name}
    """
    results_df = pd.read_csv(DATA_DIR / "results.csv")
    races_df = pd.read_csv(DATA_DIR / "races.csv")
    drivers_df = pd.read_csv(DATA_DIR / "drivers.csv")
    constructors_df = pd.read_csv(DATA_DIR / "constructors.csv")

    # Find the most recent race before this one to get the roster
    if round_num > 1:
        # Use previous round in same season
        prev_race = races_df[
            (races_df["year"] == season) & (races_df["round"] == round_num - 1)
        ]
    else:
        prev_race = pd.DataFrame()

    if prev_race.empty:
        # Fall back to the last race of the most recent completed season
        prior_seasons = races_df[races_df["year"] < season]
        if prior_seasons.empty:
            prior_seasons = races_df[races_df["year"] <= season]
        max_year = prior_seasons["year"].max()
        last_round = prior_seasons[prior_seasons["year"] == max_year]["round"].max()
        prev_race = races_df[
            (races_df["year"] == max_year) & (races_df["round"] == last_round)
        ]

    prev_race_id = prev_race.iloc[0]["raceId"]
    roster_results = results_df[results_df["raceId"] == prev_race_id]

    # Build driver dict
    drivers_dict = {}
    for _, row in roster_results.iterrows():
        did = int(row["driverId"])
        cid = int(row["constructorId"])
        drv = drivers_df[drivers_df["driverId"] == did]
        if drv.empty:
            continue
        drv = drv.iloc[0]
        name = f"{drv['forename']} {drv['surname']}"
        abbr = str(drv.get("code", "???"))
        if abbr == "nan" or abbr == "":
            abbr = drv["surname"][:3].upper()
        drivers_dict[did] = (cid, name, abbr)

    # Build constructor names
    constructor_names = {}
    for cid in set(v[0] for v in drivers_dict.values()):
        c_row = constructors_df[constructors_df["constructorId"] == cid]
        if not c_row.empty:
            constructor_names[cid] = c_row.iloc[0]["name"]
        else:
            constructor_names[cid] = f"Constructor {cid}"

    # Apply roster override if provided
    if roster_override_path:
        with open(roster_override_path) as f:
            override = json.load(f)
        for did_str, info in override.items():
            did = int(did_str)
            drivers_dict[did] = (
                info["constructor_id"],
                info["name"],
                info["abbreviation"],
            )
            cid = info["constructor_id"]
            if "constructor_name" in info:
                constructor_names[cid] = info["constructor_name"]

    return drivers_dict, constructor_names


def get_prev_positions(season, round_num, driver_ids):
    """Get each driver's finishing position from the previous round.

    Returns dict of {driver_id: prev_state} where prev_state is:
        START (21) for round 1 or unknown drivers
        0 for DNF
        1-20 for finishing position
    """
    if round_num <= 1:
        return {did: START for did in driver_ids}

    races_df = pd.read_csv(DATA_DIR / "races.csv")
    results_df = pd.read_csv(DATA_DIR / "results.csv")
    status_df = pd.read_csv(DATA_DIR / "status.csv")

    prev_race = races_df[
        (races_df["year"] == season) & (races_df["round"] == round_num - 1)
    ]
    if prev_race.empty:
        return {did: START for did in driver_ids}

    prev_race_id = prev_race.iloc[0]["raceId"]
    prev_results = results_df[results_df["raceId"] == prev_race_id]

    # Merge with status to determine DNF
    prev_results = prev_results.merge(status_df, on="statusId", how="left")

    prev_positions = {}
    for did in driver_ids:
        drv_result = prev_results[prev_results["driverId"] == did]
        if drv_result.empty:
            prev_positions[did] = START
            continue

        row = drv_result.iloc[0]
        status = str(row.get("status", "")).lower()
        is_classified = any(kw in status for kw in CLASSIFIED_STATUS_KEYWORDS)

        if is_classified:
            pos = int(row.get("positionOrder", 20))
            prev_positions[did] = min(pos, 20)  # cap at P20
        else:
            prev_positions[did] = 0  # DNF

    return prev_positions


def build_calibration(season, round_num):
    """Auto-generate calibration data from prior year's same race."""
    races_df = pd.read_csv(DATA_DIR / "races.csv")
    results_df = pd.read_csv(DATA_DIR / "results.csv")
    drivers_df = pd.read_csv(DATA_DIR / "drivers.csv")
    status_df = pd.read_csv(DATA_DIR / "status.csv")

    # Find the same round in the previous season
    prev_year = season - 1
    prev_race = races_df[
        (races_df["year"] == prev_year) & (races_df["round"] == round_num)
    ]
    if prev_race.empty:
        return None

    prev_race_row = prev_race.iloc[0]
    prev_race_id = prev_race_row["raceId"]
    prev_results = results_df[results_df["raceId"] == prev_race_id].copy()
    prev_results = prev_results.merge(status_df, on="statusId", how="left")
    prev_results = prev_results.merge(
        drivers_df[["driverId", "forename", "surname"]], on="driverId", how="left"
    )
    prev_results["full_name"] = prev_results["forename"] + " " + prev_results["surname"]

    # Filter to classified finishers and sort
    prev_results["is_classified"] = prev_results["status"].str.lower().apply(
        lambda s: any(kw in s for kw in CLASSIFIED_STATUS_KEYWORDS)
    )
    classified = prev_results[prev_results["is_classified"]].sort_values("positionOrder")

    if classified.empty:
        return None

    top3 = classified.head(3)["full_name"].tolist()
    top10 = classified.head(10)["full_name"].tolist()

    return {
        "race": f"{prev_year} {prev_race_row['name']}",
        "top3": top3,
        "top10": top10,
    }


# =====================================================================
# SIMULATION LOGIC
# =====================================================================

def simulate_single_race(model, driver_ids, drivers_dict, prev_positions, rng, n_sims):
    """Simulate n_sims races and return position distributions."""
    n_drivers = len(driver_ids)
    all_results = np.zeros((n_sims, n_drivers), dtype=int)

    probs_list = []
    for did in driver_ids:
        cid = drivers_dict[did][0]
        prev_pos = prev_positions[did]

        known = False
        if hasattr(model, "driver_constructor_counts_") and did in model.driver_constructor_counts_:
            known = True
        elif hasattr(model, "driver_strengths_") and did in model.driver_strengths_:
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


def compute_stats(all_results, probs_array, driver_ids, drivers_dict, constructor_names, n_sims):
    """Compute detailed statistics from simulation results."""
    stats = []

    for i, did in enumerate(driver_ids):
        cid, name, abbr = drivers_dict[did]
        team = constructor_names.get(cid, f"Team {cid}")
        probs = probs_array[i]

        pos_counts = np.bincount(all_results[:, i], minlength=N_OUTCOMES)
        pos_dist = pos_counts / n_sims

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


def build_composite(all_probs, driver_ids, n_sims):
    """Build composite model by equal-weight blending all stage model probabilities."""
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


# =====================================================================
# MAIN
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="F1 Monte Carlo race simulation")
    parser.add_argument("--season", type=int, required=True, help="Season year (e.g. 2026)")
    parser.add_argument("--round", type=int, required=True, help="Race round number")
    parser.add_argument("--race-name", type=str, default=None, help="Race name (auto-detected from CSVs)")
    parser.add_argument("--circuit", type=str, default=None, help="Circuit name (auto-detected from CSVs)")
    parser.add_argument("--date", type=str, default=None, help="Race date YYYY-MM-DD (auto-detected)")
    parser.add_argument("--train-start", type=int, default=None, help="Training start year (default: season-11)")
    parser.add_argument("--train-end", type=int, default=None, help="Training end year (default: season-1)")
    parser.add_argument("--n-sims", type=int, default=10000, help="Number of MC simulations (default: 10000)")
    parser.add_argument("--roster", type=str, default=None, help="Path to roster override JSON")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path (auto-generated)")
    parser.add_argument("--stages", type=str, default="stage6,stage8,stage9",
                        help="Comma-separated model stages (default: stage6,stage8,stage9)")
    return parser.parse_args()


def main():
    args = parse_args()
    season = args.season
    round_num = args.round
    n_sims = args.n_sims
    stage_keys = [s.strip() for s in args.stages.split(",")]
    train_start = args.train_start or (season - 11)
    train_end = args.train_end or (season - 1)

    # --- Race metadata ---
    race_name = args.race_name
    circuit = args.circuit
    date = args.date

    if not all([race_name, circuit, date]):
        auto_name, auto_circuit, auto_date = load_race_metadata(season, round_num)
        race_name = race_name or auto_name
        circuit = circuit or auto_circuit
        date = date or auto_date

    if not race_name:
        print(f"ERROR: Cannot find race metadata for {season} round {round_num}.")
        print("Provide --race-name, --circuit, and --date explicitly.")
        sys.exit(1)

    # --- Driver roster ---
    drivers_dict, constructor_names = build_roster(season, round_num, args.roster)
    driver_ids = list(drivers_dict.keys())

    # --- Starting states ---
    prev_positions = get_prev_positions(season, round_num, driver_ids)
    state_desc = "START" if round_num <= 1 else f"Round {round_num - 1} results"

    print("=" * 70)
    print(f"{season} {race_name} — RACE SIMULATION")
    print(f"Circuit: {circuit} · Date: {date}")
    print(f"Monte Carlo: {n_sims:,} simulated races per model")
    print(f"Training: {train_start}-{train_end} · Starting state: {state_desc}")
    print(f"Models: {', '.join(stage_keys)} · Drivers: {len(driver_ids)}")
    print("=" * 70)

    # --- Load training data ---
    loader = S6Loader(DATA_DIR)
    df = loader.load_merged(min_year=train_start, max_year=train_end)
    prev_arr, next_arr, meta_df = s6_prepare(df)

    rng = np.random.default_rng(42)
    stage_probs = {}
    stage_stats = {}
    stage_results = {}
    stage_params = {}
    stage_meta = {}

    # --- Train and simulate each model ---
    for sk in stage_keys:
        if sk not in MODEL_REGISTRY:
            print(f"WARNING: Unknown stage '{sk}', skipping.")
            continue

        cfg = MODEL_REGISTRY[sk]
        print(f"\nTraining {cfg['meta']['name']} ({train_start}-{train_end})...")

        model = cfg["class"](**cfg["kwargs"])
        model.fit(prev_arr, next_arr, meta_df)

        params = cfg["params_fn"](model)
        param_str = ", ".join(f"{k}={v:.3f}" for k, v in params.items())
        print(f"  {param_str}")

        print(f"Simulating {n_sims:,} races...")
        r, p = simulate_single_race(model, driver_ids, drivers_dict, prev_positions, rng, n_sims)

        stage_probs[sk] = p
        stage_results[sk] = r
        stage_stats[sk] = compute_stats(r, p, driver_ids, drivers_dict, constructor_names, n_sims)
        stage_params[sk] = {k: float(v) for k, v in params.items()}

        desc = cfg["meta"]["description"] + f" Trained {train_start}-{train_end}."
        stage_meta[sk] = {"name": cfg["meta"]["name"], "description": desc}

    active_stages = [sk for sk in stage_keys if sk in stage_probs]

    # --- Sanity check: flag degenerate models ---
    composite_stages = []
    for sk in active_stages:
        max_pwin = max(d["p_win"] for d in stage_stats[sk])
        if max_pwin > 0.50:
            print(f"  WARNING: {stage_meta[sk]['name']} is degenerate "
                  f"(max P(win) = {max_pwin:.1%}). Excluded from composite.")
            stage_meta[sk]["description"] += " ⚠️ Degenerate — excluded from composite."
        else:
            composite_stages.append(sk)

    if not composite_stages:
        composite_stages = active_stages

    # --- Composite model ---
    active_probs = [stage_probs[k] for k in composite_stages]
    stage_names_for_composite = [stage_meta[k]["name"].split(":")[0].strip() for k in composite_stages]
    blend_desc = " + ".join(stage_names_for_composite)
    n_active = len(active_probs)
    weight = f"1/{n_active}"

    print(f"\nComputing composite ({weight} each: {blend_desc})...")
    results_comp, probs_comp = build_composite(active_probs, driver_ids, n_sims)
    stats_comp = compute_stats(results_comp, probs_comp, driver_ids, drivers_dict, constructor_names, n_sims)

    # --- Calibration (prior year same round) ---
    calibration = build_calibration(season, round_num)

    # --- Build output JSON ---
    models_dict = {}
    for sk in active_stages:
        models_dict[sk] = {
            "name": stage_meta[sk]["name"],
            "description": stage_meta[sk]["description"],
            "params": stage_params[sk],
            "drivers": stage_stats[sk],
        }

    models_dict["composite"] = {
        "name": f"Composite ({n_active}-Model Blend)",
        "description": f"Equal-weight blend of {blend_desc} probabilities, renormalized.",
        "params": {"blend": f"{weight} each of {blend_desc}"},
        "drivers": stats_comp,
    }

    output = {
        "race": f"{season} {race_name}",
        "circuit": circuit,
        "date": date,
        "season": season,
        "round": round_num,
        "n_sims": n_sims,
        "training_years": f"{train_start}-{train_end}",
        "prev_state": state_desc,
        "active_stages": active_stages,
        "models": models_dict,
        "team_colors": TEAM_COLORS,
        "constructor_names": {str(k): v for k, v in constructor_names.items()},
    }
    if calibration:
        output["calibration"] = calibration

    # --- Write output ---
    if args.output:
        out_path = Path(args.output)
    else:
        slug = race_name.lower().replace(" ", "_").replace("'", "")
        out_path = DATA_DIR / f"sim_{season}_{slug}.json"

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {out_path}")

    # --- Print summary ---
    all_model_keys = active_stages + ["composite"]
    for model_key in all_model_keys:
        drivers = output["models"][model_key]["drivers"]
        model_name = output["models"][model_key]["name"]
        print(f"\n{'='*60}")
        print(f"{model_name} — Top 10")
        print(f"{'='*60}")
        print(f"{'Rank':<5} {'Driver':<20} {'Team':<15} {'P(win)':>8} {'P(pod)':>8} {'E[pos]':>7}")
        print("-" * 65)
        for rank, d in enumerate(drivers[:10], 1):
            print(f"{rank:<5} {d['name']:<20} {d['team']:<15} "
                  f"{d['p_win']:>8.1%} {d['p_podium']:>8.1%} {d['e_pos']:>7.1f}")


if __name__ == "__main__":
    main()
