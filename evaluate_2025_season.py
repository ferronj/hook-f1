"""
Full-season blind evaluation of 5 model stages on the 2025 F1 season.

Trains each model ONCE on 2020-2024 data, then predicts the top 3 finishers
for each of 24 2025 races without retraining. Aggregates metrics across the
season and picks the best model.
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


DATA_DIR = Path(__file__).parent / "data"


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


def load_2025_races():
    """Load 2025 race info and results."""
    races = pd.read_csv(DATA_DIR / "races.csv")
    races_2025 = races[races["year"] == 2025].sort_values("round")

    loader = S1Loader(DATA_DIR)
    results_df = loader.load_merged(min_year=2025, max_year=2025)

    return races_2025, results_df


def predict_race_stage1(model, drivers):
    """Stage 1: same prediction for everyone."""
    probs_start = model.predict_proba(START)
    results = []
    for did, cid, name in drivers:
        results.append((did, name, cid, prob_top3(probs_start), probs_start.copy()))
    return results


def predict_race_stage2(model, drivers):
    """Stage 2: driver-specific."""
    results = []
    for did, cid, name in drivers:
        if did in model.driver_counts_:
            probs = model.predict_proba(did, START)
        else:
            probs = model.predict_proba_new_driver(START)
        results.append((did, name, cid, prob_top3(probs), probs.copy()))
    return results


def predict_race_stage3(model, drivers):
    """Stage 3: constructor + driver."""
    results = []
    for did, cid, name in drivers:
        if did in model.driver_constructor_counts_:
            probs = model.predict_proba(did, START, constructor_id=cid)
        else:
            probs = model.predict_proba_new_driver(START, constructor_id=cid)
        results.append((did, name, cid, prob_top3(probs), probs.copy()))
    return results


def predict_race_stage4(model, drivers):
    """Stage 4: recency + constructor + driver."""
    results = []
    for did, cid, name in drivers:
        if did in model.driver_constructor_counts_:
            probs = model.predict_proba(did, START, constructor_id=cid)
        else:
            probs = model.predict_proba_new_driver(START, constructor_id=cid)
        results.append((did, name, cid, prob_top3(probs), probs.copy()))
    return results


def predict_race_stage5(model, drivers, circuit_id):
    """Stage 5: constructor + circuit + driver."""
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


def evaluate_predictions(predictions, actual_results):
    """
    Evaluate predictions for a single race.

    predictions: list of (did, name, cid, p_top3, probs_vector)
    actual_results: DataFrame with driverId, position_mapped columns
    """
    # Rank by P(top3)
    ranked = sorted(predictions, key=lambda x: x[3], reverse=True)
    pred_top3_ids = [r[0] for r in ranked[:3]]

    # Actual top 3
    actual_sorted = actual_results.sort_values("positionOrder")
    actual_top3 = actual_sorted.head(3)
    actual_top3_ids = actual_top3["driverId"].tolist()

    exact, in_top3 = score_top3(pred_top3_ids, actual_top3_ids)

    # Calibration: sum of P(actual position) for top-3 finishers
    pred_by_driver = {r[0]: r[4] for r in predictions}
    calibration = 0.0
    for _, row in actual_top3.iterrows():
        did = row["driverId"]
        actual_pos = int(row["position_mapped"])
        if did in pred_by_driver and actual_pos < N_OUTCOMES:
            calibration += pred_by_driver[did][actual_pos]

    # Log-likelihood across all drivers
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


def main():
    print("=" * 70)
    print("FULL 2025 SEASON EVALUATION — 5 Model Stages")
    print("Training: 2020-2024 | Evaluation: 24 races of 2025")
    print("=" * 70)

    # Load 2025 race data
    races_2025, results_2025 = load_2025_races()
    print(f"\n2025 races: {len(races_2025)}")
    print(f"2025 results: {len(results_2025)} entries")

    # Get driver names from results
    driver_names = (
        results_2025[["driverId", "driver_name"]]
        .drop_duplicates()
        .set_index("driverId")["driver_name"]
        .to_dict()
    )

    # =====================================================================
    # Train all models on 2020-2024
    # =====================================================================
    print("\nTraining models on 2020-2024 data...")

    # Stage 1
    loader1 = S1Loader(DATA_DIR)
    df1 = loader1.load_merged(min_year=2020, max_year=2024)
    prev1, next1, meta1 = s1_prepare(df1)
    model1 = DirichletMultinomialF1(prior_alpha=1.0)
    model1.fit(prev1, next1)
    print(f"  Stage 1: {len(prev1)} transitions")

    # Stage 2
    loader2 = S2Loader(DATA_DIR)
    df2 = loader2.load_merged(min_year=2020, max_year=2024)
    prev2, next2, meta2 = s2_prepare(df2)
    model2 = PartialPooledDirichletF1(
        prior_alpha_global=1.0, kappa_init=10.0, kappa_bounds=(0.1, 500.0),
    )
    model2.fit(prev2, next2, meta2)
    print(f"  Stage 2: kappa={model2.kappa_:.2f}")

    # Stage 3
    loader3 = S3Loader(DATA_DIR)
    df3 = loader3.load_merged(min_year=2020, max_year=2024)
    prev3, next3, meta3 = s3_prepare(df3)
    model3 = ConstructorPooledDirichletF1(
        prior_alpha_global=1.0, prior_alpha_constructor=1.0,
        kappa_init=(10.0, 10.0), kappa_bounds=((0.1, 500.0), (0.01, 500.0)),
    )
    model3.fit(prev3, next3, meta3)
    print(f"  Stage 3: kappa_g={model3.kappa_g_:.2f}, kappa_c={model3.kappa_c_:.2f}")

    # Stage 4
    loader4 = S4Loader(DATA_DIR)
    df4 = loader4.load_merged(min_year=2020, max_year=2024)
    prev4, next4, meta4 = s4_prepare(df4)
    model4 = RecencyGridDirichletF1(
        prior_alpha_global=1.0, prior_alpha_constructor=1.0,
        kappa_init=(5.0, 10.0), kappa_bounds=((0.01, 500.0), (0.01, 500.0)),
        lambda_init=0.15, lambda_bounds=(0.01, 0.7),
    )
    model4.fit(prev4, next4, meta4)
    print(f"  Stage 4: kappa_g={model4.kappa_g_:.2f}, kappa_c={model4.kappa_c_:.2f}, "
          f"lambda={model4.lambda_:.4f}")

    # Stage 5
    loader5 = S5Loader(DATA_DIR)
    df5 = loader5.load_merged(min_year=2020, max_year=2024)
    prev5, next5, meta5 = s5_prepare(df5)
    model5 = CircuitDirichletF1(
        prior_alpha_global=1.0, prior_alpha_constructor=1.0,
        prior_alpha_circuit=1.0,
        kappa_init=(10.0, 10.0, 1.0),
        kappa_bounds=((0.1, 500.0), (0.01, 500.0), (0.01, 200.0)),
    )
    model5.fit(prev5, next5, meta5)
    print(f"  Stage 5: kappa_g={model5.kappa_g_:.2f}, kappa_c={model5.kappa_c_:.2f}, "
          f"kappa_k={model5.kappa_k_:.2f}")

    # =====================================================================
    # Evaluate each race
    # =====================================================================
    stage_names = ["Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5"]
    all_metrics = {name: [] for name in stage_names}

    print("\n" + "=" * 70)
    print("PER-RACE RESULTS")
    print("=" * 70)

    for _, race in races_2025.iterrows():
        race_id = race["raceId"]
        circuit_id = int(race["circuitId"])
        race_name = race["name"]
        race_round = race["round"]

        # Get results for this race
        race_results = results_2025[results_2025["raceId"] == race_id]
        if len(race_results) == 0:
            continue

        # Build driver list: (driverId, constructorId, name)
        drivers = []
        for _, r in race_results.iterrows():
            did = r["driverId"]
            cid = r["constructorId"]
            name = r.get("driver_name", str(did))
            drivers.append((did, cid, name))

        # Predict with each model
        preds = {
            "Stage 1": predict_race_stage1(model1, drivers),
            "Stage 2": predict_race_stage2(model2, drivers),
            "Stage 3": predict_race_stage3(model3, drivers),
            "Stage 4": predict_race_stage4(model4, drivers),
            "Stage 5": predict_race_stage5(model5, drivers, circuit_id),
        }

        # Evaluate
        actual_sorted = race_results.sort_values("positionOrder")
        actual_top3_ids = actual_sorted.head(3)["driverId"].tolist()
        actual_top3_names = [driver_names.get(d, str(d)) for d in actual_top3_ids]

        print(f"\n  R{race_round:02d} {race_name:<30} "
              f"Actual: {', '.join(actual_top3_names)}")

        for stage_name in stage_names:
            metrics = evaluate_predictions(preds[stage_name], race_results)
            all_metrics[stage_name].append(metrics)

            pred_names = [driver_names.get(d, str(d)) for d in metrics["pred_top3"]]
            print(f"    {stage_name:<10} [{metrics['correct_in_top3']}/3] "
                  f"cal={metrics['calibration']:.3f}  "
                  f"Pred: {', '.join(pred_names)}")

    # =====================================================================
    # Aggregate Results
    # =====================================================================
    print("\n" + "=" * 70)
    print("SEASON SUMMARY")
    print("=" * 70)

    print(f"\n{'Model':<12} {'Avg T3':>7} {'Avg Exact':>10} {'Avg Cal':>8} "
          f"{'Total LL':>10} {'Perfect':>8}")
    print("-" * 60)

    summary = {}
    for stage_name in stage_names:
        metrics = all_metrics[stage_name]
        n_races = len(metrics)

        avg_in_top3 = np.mean([m["correct_in_top3"] for m in metrics])
        avg_exact = np.mean([m["exact_matches"] for m in metrics])
        avg_cal = np.mean([m["calibration"] for m in metrics])
        total_ll = sum(m["log_likelihood"] for m in metrics)
        perfect = sum(1 for m in metrics if m["correct_in_top3"] == 3)

        summary[stage_name] = {
            "avg_in_top3": avg_in_top3,
            "avg_exact": avg_exact,
            "avg_cal": avg_cal,
            "total_ll": total_ll,
            "perfect": perfect,
        }

        print(f"{stage_name:<12} {avg_in_top3:>7.2f} {avg_exact:>10.2f} "
              f"{avg_cal:>8.4f} {total_ll:>10.1f} {perfect:>8}")

    # Best model
    print(f"\n--- Best Model ---")
    best = max(summary.items(),
               key=lambda x: (x[1]["avg_in_top3"], x[1]["avg_cal"]))
    print(f"  {best[0]}: avg {best[1]['avg_in_top3']:.2f}/3 correct in top 3, "
          f"calibration {best[1]['avg_cal']:.4f}")

    # Per-race breakdown for best model
    print(f"\n--- {best[0]} Per-Race Detail ---")
    best_metrics = all_metrics[best[0]]
    for i, (_, race) in enumerate(races_2025.iterrows()):
        if i >= len(best_metrics):
            break
        m = best_metrics[i]
        race_name = race["name"]
        pred_names = [driver_names.get(d, str(d)) for d in m["pred_top3"]]
        actual_names = [driver_names.get(d, str(d)) for d in m["actual_top3"]]
        marker = " OK" if m["correct_in_top3"] == 3 else ""
        print(f"  R{race['round']:02d} [{m['correct_in_top3']}/3]{marker} "
              f"Pred: {', '.join(pred_names)}")


if __name__ == "__main__":
    main()
