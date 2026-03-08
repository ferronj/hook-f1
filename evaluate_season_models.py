"""
Focused evaluation of Stage 6, Stage 8, Stage 9 MAP, and Stage 9 NUTS
across three eras (2010, 2020, 2025).

Stage 9 NUTS requires a pre-computed trace; only evaluated on 2025
(trace trained on 2020-2024).

Usage:
  micromamba run -n f1-markov python3 evaluate_season_models.py

  # With NUTS trace:
  micromamba run -n f1-markov python3 evaluate_season_models.py \
      --nuts-trace stage9_nuts_2025.nc
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "models"))

from stage3_constructor import (
    F1DataLoader,
    prepare_transitions,
    START, N_OUTCOMES,
)
from stage6_recency_constructor import (
    F1DataLoader as S6Loader,
    RecencyConstructorDirichletF1,
    prepare_transitions as s6_prepare,
)
from stage8_plackett_luce import (
    TimeVaryingPlackettLuceF1,
)
from stage9_bayesian_ss import (
    BayesianStateSpaceF1,
)

DATA_DIR = Path(__file__).parent / "data"

ERAS = [
    {"name": "2010", "train_min": 2005, "train_max": 2009, "eval_year": 2010},
    {"name": "2020", "train_min": 2015, "train_max": 2019, "eval_year": 2020},
    {"name": "2025", "train_min": 2020, "train_max": 2024, "eval_year": 2025},
]


# =========================================================================
# Helpers
# =========================================================================
def prob_top3(probs):
    """P(finish in P1, P2, or P3)."""
    return probs[1] + probs[2] + probs[3]


def score_top3(predicted_top3_ids, actual_top3_ids):
    """Score predicted vs actual top 3."""
    pred_set = set(predicted_top3_ids)
    actual_set = set(actual_top3_ids)
    exact = sum(1 for i, d in enumerate(predicted_top3_ids)
                if i < len(actual_top3_ids) and d == actual_top3_ids[i])
    in_top3 = len(pred_set & actual_set)
    return exact, in_top3


# =========================================================================
# Prediction functions
# =========================================================================
def predict_stage6(model, drivers):
    results = []
    for did, cid, name in drivers:
        if did in model.driver_constructor_counts_:
            probs = model.predict_proba(did, START, constructor_id=cid)
        else:
            probs = model.predict_proba_new_driver(START, constructor_id=cid)
        results.append((did, name, cid, prob_top3(probs), probs.copy()))
    return results


def predict_stage8(model, drivers):
    results = []
    # Temporarily reduce MC samples for faster evaluation
    orig_mc = model.n_mc_samples
    model.n_mc_samples = min(orig_mc, 1000)
    for did, cid, name in drivers:
        if did in model.driver_strengths_:
            probs = model.predict_proba(did, START, constructor_id=cid)
        else:
            probs = model.predict_proba_new_driver(START, constructor_id=cid)
        results.append((did, name, cid, prob_top3(probs), probs.copy()))
    model.n_mc_samples = orig_mc
    return results


def predict_stage9_map(model, drivers):
    results = []
    orig_mc = model.n_mc_samples
    model.n_mc_samples = min(orig_mc, 1000)
    for did, cid, name in drivers:
        if did in model.driver_strengths_:
            probs = model.predict_proba(did, START, constructor_id=cid)
        else:
            probs = model.predict_proba_new_driver(START, constructor_id=cid)
        results.append((did, name, cid, prob_top3(probs), probs.copy()))
    model.n_mc_samples = orig_mc
    return results


def predict_stage9_nuts(model, drivers, n_posterior_draws=50):
    """Stage 9 NUTS: posterior-averaged predictions.

    Uses reduced MC samples per draw for evaluation speed.
    """
    results = []
    orig_mc = model.n_mc_samples
    model.n_mc_samples = min(orig_mc, 1000)
    for did, cid, name in drivers:
        if did in model.driver_strengths_:
            probs = model.predict_proba_bayesian(
                did, START, constructor_id=cid,
                n_posterior_draws=n_posterior_draws,
            )
        else:
            probs = model.predict_proba_new_driver(START, constructor_id=cid)
        results.append((did, name, cid, prob_top3(probs), probs.copy()))
    model.n_mc_samples = orig_mc
    return results


# =========================================================================
# Evaluation
# =========================================================================
def evaluate_predictions(predictions, actual_results):
    """Evaluate predictions for a single race."""
    ranked = sorted(predictions, key=lambda x: x[3], reverse=True)
    pred_top3_ids = [r[0] for r in ranked[:3]]

    actual_sorted = actual_results.sort_values("positionOrder")
    actual_top3 = actual_sorted.head(3)
    actual_top3_ids = actual_top3["driverId"].tolist()

    exact, in_top3 = score_top3(pred_top3_ids, actual_top3_ids)

    pred_by_driver = {r[0]: r[4] for r in predictions}

    log_lik = 0.0
    for _, row in actual_results.iterrows():
        did = row["driverId"]
        actual_pos = int(row["position_mapped"])
        if did in pred_by_driver and actual_pos < N_OUTCOMES:
            p = pred_by_driver[did][actual_pos]
            log_lik += np.log(max(p, 1e-300))

    # Calibration: sum of predicted probabilities for actual top-3 positions
    calibration = 0.0
    for _, row in actual_top3.iterrows():
        did = row["driverId"]
        actual_pos = int(row["position_mapped"])
        if did in pred_by_driver and actual_pos < N_OUTCOMES:
            calibration += pred_by_driver[did][actual_pos]

    return {
        "pred_top3": pred_top3_ids,
        "actual_top3": actual_top3_ids,
        "exact_matches": exact,
        "correct_in_top3": in_top3,
        "calibration": calibration,
        "log_likelihood": log_lik,
    }


def train_models(train_min, train_max, include_nuts=False, nuts_model=None):
    """Train Stage 6, 8, 9 MAP on the given year range."""
    models = {}

    # Stage 6
    print("  Training Stage 6...")
    loader6 = S6Loader(DATA_DIR)
    df6 = loader6.load_merged(min_year=train_min, max_year=train_max)
    prev6, next6, meta6 = s6_prepare(df6)
    model6 = RecencyConstructorDirichletF1(
        prior_alpha_global=1.0, prior_alpha_constructor=1.0,
        kappa_init=(10.0, 10.0), kappa_bounds=((0.1, 500.0), (0.01, 500.0)),
    )
    model6.fit(prev6, next6, meta6)
    models["Stage 6"] = model6

    # Stage 8 (uses Stage 3's data loader)
    print("  Training Stage 8...")
    loader3 = F1DataLoader(DATA_DIR)
    df3 = loader3.load_merged(min_year=train_min, max_year=train_max)
    prev3, next3, meta3 = prepare_transitions(df3)
    model8 = TimeVaryingPlackettLuceF1(
        alpha_candidates=(0.85, 0.9, 0.95, 0.99),
        n_mc_samples=3000,
    )
    model8.fit(prev3, next3, meta3)
    models["Stage 8 (PL)"] = model8

    # Stage 9 MAP
    print("  Training Stage 9 MAP...")
    model9 = BayesianStateSpaceF1(
        sigma_d_candidates=(0.02, 0.05, 0.1, 0.2),
        sigma_c_candidates=(0.02, 0.05, 0.1, 0.2),
        n_mc_samples=3000,
    )
    model9.fit(prev3, next3, meta3)
    models["Stage 9 MAP"] = model9

    # Stage 9 NUTS (pre-loaded, not trained here)
    if include_nuts and nuts_model is not None:
        models["Stage 9 NUTS"] = nuts_model

    return models


def evaluate_era(era, models, nuts_posterior_draws=200):
    """Evaluate models on a single era's races."""
    eval_year = era["eval_year"]

    loader = F1DataLoader(DATA_DIR)
    results_df = loader.load_merged(min_year=eval_year, max_year=eval_year)
    races = pd.read_csv(DATA_DIR / "races.csv")
    races_eval = races[races["year"] == eval_year].sort_values("round")

    stage_names = list(models.keys())
    all_metrics = {name: [] for name in stage_names}

    race_count = 0
    total_races = len(races_eval)
    for _, race in races_eval.iterrows():
        race_id = race["raceId"]
        race_results = results_df[results_df["raceId"] == race_id]
        if len(race_results) == 0:
            continue
        race_count += 1
        print(f"    Race {race_count}/{total_races}...", end="", flush=True)

        drivers = []
        for _, r in race_results.iterrows():
            did = r["driverId"]
            cid = r["constructorId"]
            name = r.get("driver_name", str(did))
            drivers.append((did, cid, name))

        preds = {}
        if "Stage 6" in models:
            preds["Stage 6"] = predict_stage6(models["Stage 6"], drivers)
        if "Stage 8 (PL)" in models:
            preds["Stage 8 (PL)"] = predict_stage8(models["Stage 8 (PL)"], drivers)
        if "Stage 9 MAP" in models:
            preds["Stage 9 MAP"] = predict_stage9_map(models["Stage 9 MAP"], drivers)
        if "Stage 9 NUTS" in models:
            preds["Stage 9 NUTS"] = predict_stage9_nuts(
                models["Stage 9 NUTS"], drivers,
                n_posterior_draws=nuts_posterior_draws,
            )
        print(" done", flush=True)

        for stage_name in stage_names:
            if stage_name in preds:
                metrics = evaluate_predictions(preds[stage_name], race_results)
                all_metrics[stage_name].append(metrics)

    return all_metrics, races_eval


# =========================================================================
# Race-by-race detail
# =========================================================================
def print_race_detail(era, all_metrics, races_eval, stage_names):
    """Print race-by-race top-3 predictions vs actuals."""
    races_list = races_eval.to_dict("records")

    loader = F1DataLoader(DATA_DIR)
    results_df = loader.load_merged(
        min_year=era["eval_year"], max_year=era["eval_year"]
    )
    driver_names = (
        results_df[["driverId", "driver_name"]]
        .drop_duplicates()
        .set_index("driverId")["driver_name"]
        .to_dict()
    )

    n_races = len(all_metrics[stage_names[0]])
    race_idx = 0
    for _, race in races_eval.iterrows():
        race_id = race["raceId"]
        race_results = results_df[results_df["raceId"] == race_id]
        if len(race_results) == 0:
            continue
        if race_idx >= n_races:
            break

        race_name = race.get("name", f"Round {race['round']}")
        actual = all_metrics[stage_names[0]][race_idx]["actual_top3"]
        actual_names = [driver_names.get(d, str(d))[:15] for d in actual]

        print(f"\n  R{race['round']:>2} {race_name[:25]:<25} "
              f"Actual: {', '.join(actual_names)}")

        for sn in stage_names:
            m = all_metrics[sn][race_idx]
            pred_names = [driver_names.get(d, str(d))[:12] for d in m["pred_top3"]]
            marker = "✓" if m["correct_in_top3"] >= 2 else " "
            print(f"      {sn:<14} [{m['correct_in_top3']}/3] {marker} "
                  f"{', '.join(pred_names)}")

        race_idx += 1


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Stage 6/8/9 MAP/NUTS across eras"
    )
    parser.add_argument("--nuts-trace", type=str, default=None,
                        help="Path to NUTS trace (.nc) for Stage 9 NUTS")
    parser.add_argument("--n-posterior-draws", type=int, default=200,
                        help="Posterior draws for NUTS predictions (default: 200)")
    parser.add_argument("--race-detail", action="store_true",
                        help="Print race-by-race predictions")
    args = parser.parse_args()

    print("=" * 75)
    print("MODEL EVALUATION — Stage 6 / Stage 8 / Stage 9 MAP / Stage 9 NUTS")
    print("Eras: 2010, 2020, 2025")
    print("=" * 75)

    # Load NUTS model if trace provided
    nuts_model = None
    if args.nuts_trace:
        print(f"\nLoading NUTS trace from {args.nuts_trace}...")
        nuts_model = BayesianStateSpaceF1.load_nuts_trace(
            args.nuts_trace, n_mc_samples=3000
        )
        print(f"  {nuts_model.n_posterior_samples_} posterior samples, "
              f"{len(nuts_model.driver_strengths_)} drivers, "
              f"{len(nuts_model.constructor_strengths_)} constructors")
        print("  (NUTS will only be evaluated on 2025 — trace trained on 2020-2024)")

    cross_era = {}  # stage_name -> list of era dicts

    for era in ERAS:
        print(f"\n{'=' * 75}")
        print(f"ERA: {era['name']} — Train {era['train_min']}-{era['train_max']}, "
              f"Eval {era['eval_year']}")
        print("=" * 75)

        # Only include NUTS for 2025 era
        include_nuts = (era["eval_year"] == 2025 and nuts_model is not None)

        print(f"\nTraining models on {era['train_min']}-{era['train_max']}...")
        models = train_models(
            era["train_min"], era["train_max"],
            include_nuts=include_nuts, nuts_model=nuts_model,
        )

        # Print hyperparameters
        m6 = models["Stage 6"]
        m8 = models["Stage 8 (PL)"]
        m9 = models["Stage 9 MAP"]
        print(f"\n  Hyperparameters:")
        print(f"    Stage 6:     kappa_g={m6.kappa_g_:.2f}, "
              f"kappa_c={m6.kappa_c_:.2f}, w={m6.w_:.2f}")
        print(f"    Stage 8:     alpha={m8.alpha_:.2f}")
        print(f"    Stage 9 MAP: sigma_d={m9.sigma_d_:.3f}, "
              f"sigma_c={m9.sigma_c_:.3f}")
        if include_nuts:
            print(f"    Stage 9 NUTS: sigma_d={nuts_model.sigma_d_:.3f}, "
                  f"sigma_c={nuts_model.sigma_c_:.3f} (from trace)")

        # Evaluate
        print(f"\n  Evaluating on {era['eval_year']}...")
        all_metrics, races_eval = evaluate_era(
            era, models, nuts_posterior_draws=args.n_posterior_draws
        )
        stage_names = list(models.keys())
        n_races = len(all_metrics[stage_names[0]])
        print(f"  {n_races} races evaluated")

        # Era summary table
        print(f"\n  {'Model':<16} {'Avg T3':>7} {'Avg Exact':>10} "
              f"{'LL/race':>9} {'Avg Cal':>8} {'Perfect':>8}")
        print(f"  {'-' * 62}")

        for sn in stage_names:
            metrics = all_metrics[sn]
            avg_t3 = np.mean([m["correct_in_top3"] for m in metrics])
            avg_exact = np.mean([m["exact_matches"] for m in metrics])
            total_ll = sum(m["log_likelihood"] for m in metrics)
            ll_per_race = total_ll / len(metrics)
            avg_cal = np.mean([m["calibration"] for m in metrics])
            perfect = sum(1 for m in metrics if m["correct_in_top3"] == 3)

            if sn not in cross_era:
                cross_era[sn] = []
            cross_era[sn].append({
                "era": era["name"],
                "avg_t3": avg_t3,
                "avg_exact": avg_exact,
                "ll_per_race": ll_per_race,
                "total_ll": total_ll,
                "avg_cal": avg_cal,
                "perfect": perfect,
                "n_races": len(metrics),
            })

            print(f"  {sn:<16} {avg_t3:>7.2f} {avg_exact:>10.2f} "
                  f"{ll_per_race:>9.1f} {avg_cal:>8.4f} {perfect:>8}")

        # Race-by-race detail
        if args.race_detail:
            print(f"\n  --- Race-by-Race Detail ---")
            print_race_detail(era, all_metrics, races_eval, stage_names)

    # =====================================================================
    # CROSS-ERA SUMMARY
    # =====================================================================
    print("\n\n" + "=" * 75)
    print("CROSS-ERA SUMMARY")
    print("=" * 75)

    # Determine all stages that appeared
    all_stages = ["Stage 6", "Stage 8 (PL)", "Stage 9 MAP", "Stage 9 NUTS"]
    active_stages = [s for s in all_stages if s in cross_era]

    era_names = [e["name"] for e in ERAS]

    print(f"\n  {'Model':<16}", end="")
    for en in era_names:
        print(f" {'T3-'+en:>8}", end="")
    print(f" {'Avg T3':>8} {'Avg LL/r':>9}")
    print(f"  {'-' * (16 + 8*len(era_names) + 8 + 9 + 2)}")

    for sn in active_stages:
        era_data = cross_era[sn]
        print(f"  {sn:<16}", end="")

        for en in era_names:
            matching = [ed for ed in era_data if ed["era"] == en]
            if matching:
                print(f" {matching[0]['avg_t3']:>8.2f}", end="")
            else:
                print(f" {'—':>8}", end="")

        avg_t3 = np.mean([ed["avg_t3"] for ed in era_data])
        total_ll = sum(ed["total_ll"] for ed in era_data)
        total_races = sum(ed["n_races"] for ed in era_data)
        avg_ll = total_ll / total_races
        print(f" {avg_t3:>8.2f} {avg_ll:>9.1f}")

    print(f"\n  {'Model':<16}", end="")
    for en in era_names:
        print(f" {'LL-'+en:>8}", end="")
    print(f" {'Avg Cal':>8} {'Perfect':>8}")
    print(f"  {'-' * (16 + 8*len(era_names) + 8 + 8 + 2)}")

    for sn in active_stages:
        era_data = cross_era[sn]
        print(f"  {sn:<16}", end="")

        for en in era_names:
            matching = [ed for ed in era_data if ed["era"] == en]
            if matching:
                print(f" {matching[0]['ll_per_race']:>8.1f}", end="")
            else:
                print(f" {'—':>8}", end="")

        avg_cal = np.mean([ed["avg_cal"] for ed in era_data])
        total_perfect = sum(ed["perfect"] for ed in era_data)
        print(f" {avg_cal:>8.4f} {total_perfect:>8}")

    # Best model summary
    print(f"\n  --- Best Models ---")

    # For T3 comparison, only use stages present in all eras
    common_stages = [s for s in active_stages
                     if len(cross_era[s]) == len(ERAS)]
    if common_stages:
        best_t3 = max(common_stages,
                      key=lambda s: np.mean([ed["avg_t3"]
                                             for ed in cross_era[s]]))
        best_t3_val = np.mean([ed["avg_t3"] for ed in cross_era[best_t3]])
        print(f"  Best Avg T3 (all eras): {best_t3} = {best_t3_val:.2f}/3")

        best_ll = max(common_stages,
                      key=lambda s: sum(ed["total_ll"]
                                        for ed in cross_era[s])
                                    / sum(ed["n_races"]
                                          for ed in cross_era[s]))
        best_ll_val = (sum(ed["total_ll"] for ed in cross_era[best_ll])
                       / sum(ed["n_races"] for ed in cross_era[best_ll]))
        print(f"  Best Avg LL/race (all eras): {best_ll} = {best_ll_val:.1f}")

    # NUTS vs MAP direct comparison on 2025
    if "Stage 9 NUTS" in cross_era and "Stage 9 MAP" in cross_era:
        nuts_2025 = [ed for ed in cross_era["Stage 9 NUTS"]
                     if ed["era"] == "2025"]
        map_2025 = [ed for ed in cross_era["Stage 9 MAP"]
                    if ed["era"] == "2025"]
        if nuts_2025 and map_2025:
            print(f"\n  --- Stage 9: MAP vs NUTS (2025) ---")
            print(f"  {'Metric':<25} {'MAP':>10} {'NUTS':>10} {'Δ':>10}")
            print(f"  {'-' * 57}")
            m = map_2025[0]
            n = nuts_2025[0]
            for label, mk, fmt in [
                ("Avg Top-3", "avg_t3", ".2f"),
                ("Avg Exact", "avg_exact", ".2f"),
                ("LL/race", "ll_per_race", ".1f"),
                ("Avg Calibration", "avg_cal", ".4f"),
                ("Perfect races", "perfect", "d"),
            ]:
                mv = m[mk]
                nv = n[mk]
                delta = nv - mv
                print(f"  {label:<25} {mv:>10{fmt}} {nv:>10{fmt}} "
                      f"{'+' if delta > 0 else ''}{delta:>9{fmt}}")


if __name__ == "__main__":
    main()
