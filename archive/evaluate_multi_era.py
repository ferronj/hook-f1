"""
Multi-era blind evaluation of 10 model stages across 4 F1 eras.

Tests model robustness by training on 5-year windows and evaluating on the
following season across different periods of F1 history:

    Era 1: Train 1995-1999, Eval 2000  (pre-Schumacher dominance ending)
    Era 2: Train 2005-2009, Eval 2010  (Brawn → Red Bull era shift)
    Era 3: Train 2015-2019, Eval 2020  (Mercedes → COVID season)
    Era 4: Train 2020-2024, Eval 2025  (McLaren breakthrough)

Each model is trained ONCE per era on the training window, then predicts
top-3 finishers for every race in the evaluation year without retraining.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "models"))

from stage1_global_transition import (
    F1DataLoader as S1Loader,
    DirichletMultinomialF1,
    prepare_transitions as s1_prepare,
    START, N_OUTCOMES,
)
from stage2_driver_pooling import (
    F1DataLoader as S2Loader,
    PartialPooledDirichletF1,
    prepare_transitions as s2_prepare,
)
from stage3_constructor import (
    F1DataLoader as S3Loader,
    ConstructorPooledDirichletF1,
    prepare_transitions as s3_prepare,
)
from stage4_recency_grid import (
    F1DataLoader as S4Loader,
    RecencyGridDirichletF1,
    prepare_transitions as s4_prepare,
)
from stage5_circuit import (
    F1DataLoader as S5Loader,
    CircuitDirichletF1,
    prepare_transitions as s5_prepare,
)
from stage6_recency_constructor import (
    F1DataLoader as S6Loader,
    RecencyConstructorDirichletF1,
    prepare_transitions as s6_prepare,
)
from stage7_hmm import (
    HiddenMarkovF1,
)
from stage8_plackett_luce import (
    TimeVaryingPlackettLuceF1,
)
from stage9_bayesian_ss import (
    BayesianStateSpaceF1,
)


DATA_DIR = Path(__file__).parent / "data"

ERAS = [
    {"name": "2000", "train_min": 1995, "train_max": 1999, "eval_year": 2000},
    {"name": "2010", "train_min": 2005, "train_max": 2009, "eval_year": 2010},
    {"name": "2020", "train_min": 2015, "train_max": 2019, "eval_year": 2020},
    {"name": "2025", "train_min": 2020, "train_max": 2024, "eval_year": 2025},
]


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


# ---------------------------------------------------------------------------
# Prediction functions (one per stage)
# ---------------------------------------------------------------------------
def predict_stage1(model, drivers):
    probs_start = model.predict_proba(START)
    return [(did, name, cid, prob_top3(probs_start), probs_start.copy())
            for did, cid, name in drivers]


def predict_stage2(model, drivers):
    results = []
    for did, cid, name in drivers:
        if did in model.driver_counts_:
            probs = model.predict_proba(did, START)
        else:
            probs = model.predict_proba_new_driver(START)
        results.append((did, name, cid, prob_top3(probs), probs.copy()))
    return results


def predict_stage3(model, drivers):
    results = []
    for did, cid, name in drivers:
        if did in model.driver_constructor_counts_:
            probs = model.predict_proba(did, START, constructor_id=cid)
        else:
            probs = model.predict_proba_new_driver(START, constructor_id=cid)
        results.append((did, name, cid, prob_top3(probs), probs.copy()))
    return results


def predict_stage5(model, drivers, circuit_id):
    results = []
    for did, cid, name in drivers:
        if did in model.driver_constructor_counts_:
            probs = model.predict_proba(
                did, START, constructor_id=cid, circuit_id=circuit_id
            )
        else:
            probs = model.predict_proba_new_driver(
                START, constructor_id=cid, circuit_id=circuit_id
            )
        results.append((did, name, cid, prob_top3(probs), probs.copy()))
    return results


# Stage 4 and Stage 6 use same interface as Stage 3
predict_stage4 = predict_stage3
predict_stage6 = predict_stage3

# Stage 6c uses same interface as Stage 5 (passes circuit_id)
predict_stage6c = predict_stage5


def predict_stage7(model, drivers):
    """Stage 7: HMM. Uses constructor_id, checks driver_offsets_."""
    results = []
    for did, cid, name in drivers:
        if did in model.driver_offsets_:
            probs = model.predict_proba(did, START, constructor_id=cid)
        else:
            probs = model.predict_proba_new_driver(START, constructor_id=cid)
        results.append((did, name, cid, prob_top3(probs), probs.copy()))
    return results


def predict_stage8(model, drivers):
    """Stage 8: Plackett-Luce. Uses constructor_id, checks driver_strengths_."""
    results = []
    for did, cid, name in drivers:
        if did in model.driver_strengths_:
            probs = model.predict_proba(did, START, constructor_id=cid)
        else:
            probs = model.predict_proba_new_driver(START, constructor_id=cid)
        results.append((did, name, cid, prob_top3(probs), probs.copy()))
    return results


# Stage 9 uses same interface as Stage 8 (checks driver_strengths_)
predict_stage9 = predict_stage8


def evaluate_predictions(predictions, actual_results):
    """Evaluate predictions for a single race."""
    ranked = sorted(predictions, key=lambda x: x[3], reverse=True)
    pred_top3_ids = [r[0] for r in ranked[:3]]

    actual_sorted = actual_results.sort_values("positionOrder")
    actual_top3 = actual_sorted.head(3)
    actual_top3_ids = actual_top3["driverId"].tolist()

    exact, in_top3 = score_top3(pred_top3_ids, actual_top3_ids)

    pred_by_driver = {r[0]: r[4] for r in predictions}
    calibration = 0.0
    for _, row in actual_top3.iterrows():
        did = row["driverId"]
        actual_pos = int(row["position_mapped"])
        if did in pred_by_driver and actual_pos < N_OUTCOMES:
            calibration += pred_by_driver[did][actual_pos]

    log_lik = 0.0
    for _, row in actual_results.iterrows():
        did = row["driverId"]
        actual_pos = int(row["position_mapped"])
        if did in pred_by_driver and actual_pos < N_OUTCOMES:
            p = pred_by_driver[did][actual_pos]
            log_lik += np.log(max(p, 1e-300))

    return {
        "pred_top3": pred_top3_ids,
        "actual_top3": actual_top3_ids,
        "exact_matches": exact,
        "correct_in_top3": in_top3,
        "calibration": calibration,
        "log_likelihood": log_lik,
    }


def train_all_models(train_min, train_max):
    """Train all 6 model stages on the given year range."""
    models = {}

    # Stage 1
    loader1 = S1Loader(DATA_DIR)
    df1 = loader1.load_merged(min_year=train_min, max_year=train_max)
    prev1, next1, meta1 = s1_prepare(df1)
    model1 = DirichletMultinomialF1(prior_alpha=1.0)
    model1.fit(prev1, next1)
    models["Stage 1"] = model1

    # Stage 2
    loader2 = S2Loader(DATA_DIR)
    df2 = loader2.load_merged(min_year=train_min, max_year=train_max)
    prev2, next2, meta2 = s2_prepare(df2)
    model2 = PartialPooledDirichletF1(
        prior_alpha_global=1.0, kappa_init=10.0, kappa_bounds=(0.1, 500.0),
    )
    model2.fit(prev2, next2, meta2)
    models["Stage 2"] = model2

    # Stage 3
    loader3 = S3Loader(DATA_DIR)
    df3 = loader3.load_merged(min_year=train_min, max_year=train_max)
    prev3, next3, meta3 = s3_prepare(df3)
    model3 = ConstructorPooledDirichletF1(
        prior_alpha_global=1.0, prior_alpha_constructor=1.0,
        kappa_init=(10.0, 10.0), kappa_bounds=((0.1, 500.0), (0.01, 500.0)),
    )
    model3.fit(prev3, next3, meta3)
    models["Stage 3"] = model3

    # Stage 4
    loader4 = S4Loader(DATA_DIR)
    df4 = loader4.load_merged(min_year=train_min, max_year=train_max)
    prev4, next4, meta4 = s4_prepare(df4)
    model4 = RecencyGridDirichletF1(
        prior_alpha_global=1.0, prior_alpha_constructor=1.0,
        kappa_init=(5.0, 10.0), kappa_bounds=((0.01, 500.0), (0.01, 500.0)),
        lambda_init=0.15, lambda_bounds=(0.01, 0.7),
    )
    model4.fit(prev4, next4, meta4)
    models["Stage 4"] = model4

    # Stage 5
    loader5 = S5Loader(DATA_DIR)
    df5 = loader5.load_merged(min_year=train_min, max_year=train_max)
    prev5, next5, meta5 = s5_prepare(df5)
    model5 = CircuitDirichletF1(
        prior_alpha_global=1.0, prior_alpha_constructor=1.0,
        prior_alpha_circuit=1.0,
        kappa_init=(10.0, 10.0, 1.0),
        kappa_bounds=((0.1, 500.0), (0.01, 500.0), (0.01, 200.0)),
    )
    model5.fit(prev5, next5, meta5)
    models["Stage 5"] = model5

    # Stage 6
    loader6 = S6Loader(DATA_DIR)
    df6 = loader6.load_merged(min_year=train_min, max_year=train_max)
    prev6, next6, meta6 = s6_prepare(df6)
    model6 = RecencyConstructorDirichletF1(
        prior_alpha_global=1.0, prior_alpha_constructor=1.0,
        kappa_init=(10.0, 10.0), kappa_bounds=((0.1, 500.0), (0.01, 500.0)),
    )
    model6.fit(prev6, next6, meta6)
    models["Stage 6"] = model6

    # Stage 6c (circuit-aware Stage 6) — reuses Stage 5 data (has circuit column)
    model6c = RecencyConstructorDirichletF1(
        prior_alpha_global=1.0, prior_alpha_constructor=1.0,
        prior_alpha_circuit=1.0,
        kappa_init=(10.0, 10.0), kappa_bounds=((0.1, 500.0), (0.01, 500.0)),
        kappa_k_init=1.0, kappa_k_bounds=(0.01, 200.0),
    )
    model6c.fit(prev5, next5, meta5)  # meta5 has 'circuit' column
    models["Stage 6c"] = model6c

    # Stage 7 (HMM) — reuses Stage 3 data
    model7 = HiddenMarkovF1(n_tiers=4, em_iters=50, n_restarts=5)
    model7.fit(prev3, next3, meta3)
    models["Stage 7 (HMM)"] = model7

    # Stage 8 (Plackett-Luce) — reuses Stage 3 data
    model8 = TimeVaryingPlackettLuceF1(
        alpha_candidates=(0.85, 0.9, 0.95, 0.99),
        n_mc_samples=3000,
    )
    model8.fit(prev3, next3, meta3)
    models["Stage 8 (PL)"] = model8

    # Stage 9 (Bayesian State-Space) — reuses Stage 3 data
    model9 = BayesianStateSpaceF1(
        sigma_d_candidates=(0.02, 0.05, 0.1, 0.2),
        sigma_c_candidates=(0.02, 0.05, 0.1, 0.2),
        n_mc_samples=3000,
    )
    model9.fit(prev3, next3, meta3)
    models["Stage 9 (BSS)"] = model9

    return models


def evaluate_era(era, models):
    """Evaluate all models on a single era's races."""
    eval_year = era["eval_year"]

    loader = S1Loader(DATA_DIR)
    results_df = loader.load_merged(min_year=eval_year, max_year=eval_year)
    races = pd.read_csv(DATA_DIR / "races.csv")
    races_eval = races[races["year"] == eval_year].sort_values("round")

    driver_names = (
        results_df[["driverId", "driver_name"]]
        .drop_duplicates()
        .set_index("driverId")["driver_name"]
        .to_dict()
    )

    stage_names = [
        "Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5", "Stage 6",
        "Stage 6c", "Stage 7 (HMM)", "Stage 8 (PL)", "Stage 9 (BSS)",
    ]
    all_metrics = {name: [] for name in stage_names}

    for _, race in races_eval.iterrows():
        race_id = race["raceId"]
        circuit_id = int(race["circuitId"])
        race_results = results_df[results_df["raceId"] == race_id]
        if len(race_results) == 0:
            continue

        drivers = []
        for _, r in race_results.iterrows():
            did = r["driverId"]
            cid = r["constructorId"]
            name = r.get("driver_name", str(did))
            drivers.append((did, cid, name))

        preds = {
            "Stage 1": predict_stage1(models["Stage 1"], drivers),
            "Stage 2": predict_stage2(models["Stage 2"], drivers),
            "Stage 3": predict_stage3(models["Stage 3"], drivers),
            "Stage 4": predict_stage4(models["Stage 4"], drivers),
            "Stage 5": predict_stage5(models["Stage 5"], drivers, circuit_id),
            "Stage 6": predict_stage6(models["Stage 6"], drivers),
            "Stage 6c": predict_stage6c(models["Stage 6c"], drivers, circuit_id),
            "Stage 7 (HMM)": predict_stage7(models["Stage 7 (HMM)"], drivers),
            "Stage 8 (PL)": predict_stage8(models["Stage 8 (PL)"], drivers),
            "Stage 9 (BSS)": predict_stage9(models["Stage 9 (BSS)"], drivers),
        }

        for stage_name in stage_names:
            metrics = evaluate_predictions(preds[stage_name], race_results)
            all_metrics[stage_name].append(metrics)

    return all_metrics, driver_names, races_eval


def main():
    print("=" * 70)
    print("MULTI-ERA EVALUATION — 10 Model Stages × 4 Eras")
    print("=" * 70)

    stage_names = [
        "Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5", "Stage 6",
        "Stage 6c", "Stage 7 (HMM)", "Stage 8 (PL)", "Stage 9 (BSS)",
    ]
    cross_era_metrics = {name: [] for name in stage_names}

    for era in ERAS:
        print(f"\n{'=' * 70}")
        print(f"ERA: {era['name']} — Train {era['train_min']}-{era['train_max']}, "
              f"Eval {era['eval_year']}")
        print("=" * 70)

        # Train
        print(f"\nTraining all models on {era['train_min']}-{era['train_max']}...")
        models = train_all_models(era["train_min"], era["train_max"])

        # Print fitted hyperparameters
        m2 = models["Stage 2"]
        m3 = models["Stage 3"]
        m4 = models["Stage 4"]
        m5 = models["Stage 5"]
        m6 = models["Stage 6"]
        m6c = models["Stage 6c"]
        m8 = models["Stage 8 (PL)"]
        print(f"  Stage 2: kappa={m2.kappa_:.2f}")
        print(f"  Stage 3: kappa_g={m3.kappa_g_:.2f}, kappa_c={m3.kappa_c_:.2f}")
        print(f"  Stage 4: kappa_g={m4.kappa_g_:.2f}, kappa_c={m4.kappa_c_:.2f}, "
              f"lambda={m4.lambda_:.4f}")
        print(f"  Stage 5: kappa_g={m5.kappa_g_:.2f}, kappa_c={m5.kappa_c_:.2f}, "
              f"kappa_k={m5.kappa_k_:.2f}")
        print(f"  Stage 6: kappa_g={m6.kappa_g_:.2f}, kappa_c={m6.kappa_c_:.2f}, "
              f"w={m6.w_:.4f}")
        print(f"  Stage 6c: kappa_g={m6c.kappa_g_:.2f}, kappa_c={m6c.kappa_c_:.2f}, "
              f"kappa_k={m6c.kappa_k_:.2f}, w={m6c.w_:.4f}")
        print(f"  Stage 8: alpha={m8.alpha_:.2f}")
        m9 = models["Stage 9 (BSS)"]
        print(f"  Stage 9: sigma_d={m9.sigma_d_:.3f}, sigma_c={m9.sigma_c_:.3f}")

        # Evaluate
        all_metrics, driver_names, races_eval = evaluate_era(era, models)
        n_races = len(all_metrics["Stage 1"])
        print(f"\nEvaluated {n_races} races")

        # Era summary
        print(f"\n{'Model':<16} {'Avg T3':>7} {'Avg Exact':>10} {'Avg Cal':>8} "
              f"{'Total LL':>10} {'Perfect':>8}")
        print("-" * 65)

        for stage_name in stage_names:
            metrics = all_metrics[stage_name]
            avg_in_top3 = np.mean([m["correct_in_top3"] for m in metrics])
            avg_exact = np.mean([m["exact_matches"] for m in metrics])
            avg_cal = np.mean([m["calibration"] for m in metrics])
            total_ll = sum(m["log_likelihood"] for m in metrics)
            perfect = sum(1 for m in metrics if m["correct_in_top3"] == 3)

            cross_era_metrics[stage_name].append({
                "era": era["name"],
                "avg_in_top3": avg_in_top3,
                "avg_exact": avg_exact,
                "avg_cal": avg_cal,
                "total_ll": total_ll,
                "perfect": perfect,
                "n_races": len(metrics),
            })

            print(f"{stage_name:<16} {avg_in_top3:>7.2f} {avg_exact:>10.2f} "
                  f"{avg_cal:>8.4f} {total_ll:>10.1f} {perfect:>8}")

    # =========================================================================
    # CROSS-ERA SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("CROSS-ERA SUMMARY")
    print("=" * 70)

    # Per-era breakdown
    print(f"\n{'Model':<16}", end="")
    for era in ERAS:
        print(f" {'T3-'+era['name']:>8}", end="")
    print(f" {'Avg T3':>8} {'Avg Cal':>8} {'Avg LL/race':>12}")
    print("-" * 88)

    for stage_name in stage_names:
        era_data = cross_era_metrics[stage_name]
        print(f"{stage_name:<16}", end="")
        for ed in era_data:
            print(f" {ed['avg_in_top3']:>8.2f}", end="")
        avg_t3 = np.mean([ed["avg_in_top3"] for ed in era_data])
        avg_cal = np.mean([ed["avg_cal"] for ed in era_data])
        total_races = sum(ed["n_races"] for ed in era_data)
        total_ll = sum(ed["total_ll"] for ed in era_data)
        avg_ll_per_race = total_ll / total_races
        print(f" {avg_t3:>8.2f} {avg_cal:>8.4f} {avg_ll_per_race:>12.1f}")

    # Best model
    print(f"\n--- Best Model (by cross-era avg top-3) ---")
    best_name = max(
        stage_names,
        key=lambda s: np.mean([ed["avg_in_top3"]
                               for ed in cross_era_metrics[s]])
    )
    best_avg = np.mean([ed["avg_in_top3"]
                        for ed in cross_era_metrics[best_name]])
    print(f"  {best_name}: avg {best_avg:.2f}/3 correct in top 3")

    print(f"\n--- Best Model (by cross-era avg LL/race) ---")
    best_ll_name = max(
        stage_names,
        key=lambda s: sum(ed["total_ll"] for ed in cross_era_metrics[s])
                      / sum(ed["n_races"] for ed in cross_era_metrics[s])
    )
    total_races = sum(ed["n_races"] for ed in cross_era_metrics[best_ll_name])
    total_ll = sum(ed["total_ll"] for ed in cross_era_metrics[best_ll_name])
    print(f"  {best_ll_name}: avg {total_ll/total_races:.1f} LL/race")


if __name__ == "__main__":
    main()
