"""
Evaluate 4 Markov model stages on predicting the 2025 Australian GP top 3.

Trains each model on pre-2025 data, then predicts the first race of 2025
(all drivers start from START=21) and ranks by probability of finishing
in positions 1-3.

Actual 2025 Australian GP top 3:
  P1: Lando Norris (McLaren)
  P2: Max Verstappen (Red Bull)
  P3: George Russell (Mercedes)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add models dir to path
sys.path.insert(0, str(Path(__file__).parent / "models"))

from stage1_global_transition import (
    F1DataLoader as S1Loader,
    DirichletMultinomialF1,
    prepare_transitions as s1_prepare,
    START, N_OUTCOMES, OUTCOME_LABELS,
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


DATA_DIR = Path(__file__).parent / "data"

# 2025 Australian GP actual top 3 (driverId -> finishing position)
ACTUAL_TOP3 = {
    846: 1,   # Norris
    830: 2,   # Verstappen
    847: 3,   # Russell
}

# All 2025 R1 drivers: driverId -> (constructorId, name, grid_position)
DRIVERS_2025_R1 = {
    846: (1, "Lando Norris", 1),        # McLaren, pole
    857: (1, "Oscar Piastri", 2),       # McLaren
    830: (9, "Max Verstappen", 3),      # Red Bull
    847: (131, "George Russell", 4),    # Mercedes
    852: (215, "Yuki Tsunoda", 5),      # Racing Bulls
    848: (3, "Alex Albon", 6),          # Williams
    844: (6, "Charles Leclerc", 7),     # Ferrari
    1: (6, "Lewis Hamilton", 8),        # Ferrari
    842: (214, "Pierre Gasly", 9),      # Alpine
    832: (3, "Carlos Sainz", 10),       # Williams
    864: (215, "Isack Hadjar", 11),     # Racing Bulls
    4: (117, "Fernando Alonso", 12),    # Aston Martin
    840: (117, "Lance Stroll", 13),     # Aston Martin
    862: (214, "Jack Doohan", 14),      # Alpine
    865: (15, "Gabriel Bortoleto", 15), # Kick Sauber
    863: (131, "Kimi Antonelli", 16),   # Mercedes
    807: (15, "Nico Hulkenberg", 17),   # Kick Sauber
    859: (9, "Liam Lawson", 18),        # Red Bull
    839: (210, "Esteban Ocon", 19),     # Haas
    860: (210, "Oliver Bearman", 20),   # Haas
}


def score_top3(predicted_top3_ids, actual_top3_ids):
    """Score how well predicted top 3 matches actual top 3."""
    pred_set = set(predicted_top3_ids)
    actual_set = set(actual_top3_ids)

    exact = sum(1 for i, d in enumerate(predicted_top3_ids)
                if i < len(actual_top3_ids) and d == actual_top3_ids[i])
    in_top3 = len(pred_set & actual_set)

    return exact, in_top3


def expected_position(probs):
    """E[position] from probability vector (excluding DNF)."""
    return sum(j * probs[j] for j in range(1, N_OUTCOMES))


def prob_top3(probs):
    """P(finish in P1, P2, or P3)."""
    return probs[1] + probs[2] + probs[3]


def main():
    print("=" * 70)
    print("Evaluating 4 Markov Models on 2025 Australian GP Top 3 Prediction")
    print("=" * 70)

    actual_top3_ids = [846, 830, 847]  # Norris, Verstappen, Russell
    print(f"\nActual top 3: {', '.join(DRIVERS_2025_R1[d][1] for d in actual_top3_ids)}")

    # =========================================================================
    # Stage 1: Global Dirichlet-Multinomial
    # =========================================================================
    print("\n" + "=" * 70)
    print("STAGE 1: Global Transition Matrix")
    print("=" * 70)

    loader1 = S1Loader(DATA_DIR)
    df1 = loader1.load_merged(min_year=2020, max_year=2024)
    prev1, next1, meta1 = s1_prepare(df1)

    model1 = DirichletMultinomialF1(prior_alpha=1.0)
    model1.fit(prev1, next1)

    probs_start_s1 = model1.predict_proba(START)

    print(f"Training data: {len(prev1)} transitions, {df1['year'].nunique()} seasons")
    print(f"\nGlobal P(next | prev=START):")
    print(f"  P(P1) = {probs_start_s1[1]:.4f}")
    print(f"  P(P2) = {probs_start_s1[2]:.4f}")
    print(f"  P(P3) = {probs_start_s1[3]:.4f}")
    print(f"  P(top3) = {prob_top3(probs_start_s1):.4f}")
    print(f"\n  NOTE: Stage 1 gives identical predictions for all drivers.")

    # =========================================================================
    # Stage 2: Driver-specific Partial Pooling
    # =========================================================================
    print("\n" + "=" * 70)
    print("STAGE 2: Driver-Specific Partial Pooling")
    print("=" * 70)

    loader2 = S2Loader(DATA_DIR)
    df2 = loader2.load_merged(min_year=2020, max_year=2024)
    prev2, next2, meta2 = s2_prepare(df2)

    model2 = PartialPooledDirichletF1(
        prior_alpha_global=1.0,
        kappa_init=10.0,
        kappa_bounds=(0.1, 500.0),
    )
    model2.fit(prev2, next2, meta2)

    print(f"Training data: {len(prev2)} transitions")
    print(f"Optimal kappa: {model2.kappa_:.2f}")

    s2_results = []
    for driver_id, (cid, name, grid) in DRIVERS_2025_R1.items():
        if driver_id in model2.driver_counts_:
            probs = model2.predict_proba(driver_id, START)
            n_obs = model2.driver_counts_[driver_id].sum()
            pf = model2.pooling_factor(driver_id)
        else:
            probs = model2.predict_proba_new_driver(START)
            n_obs = 0
            pf = 1.0

        s2_results.append({
            "driver_id": driver_id,
            "name": name,
            "P(top3)": prob_top3(probs),
            "P(P1)": probs[1],
            "P(P2)": probs[2],
            "P(P3)": probs[3],
            "E[pos]": expected_position(probs),
            "n_obs": n_obs,
            "pooling": pf,
        })

    s2_df = pd.DataFrame(s2_results).sort_values("P(top3)", ascending=False)

    print(f"\nPredicted rankings by P(top 3 finish):")
    print(f"{'Rank':<5} {'Driver':<22} {'P(top3)':>8} {'P(P1)':>7} "
          f"{'P(P2)':>7} {'P(P3)':>7} {'E[pos]':>7} {'n_obs':>6} {'pool':>5}")
    print("-" * 82)
    for rank, (_, row) in enumerate(s2_df.iterrows(), 1):
        marker = " ***" if row["driver_id"] in ACTUAL_TOP3 else ""
        print(f"{rank:<5} {row['name']:<22} {row['P(top3)']:>8.4f} "
              f"{row['P(P1)']:>7.4f} {row['P(P2)']:>7.4f} "
              f"{row['P(P3)']:>7.4f} {row['E[pos]']:>7.2f} "
              f"{row['n_obs']:>6.0f} {row['pooling']:>5.2f}{marker}")

    pred_top3_s2 = s2_df["driver_id"].head(3).tolist()
    exact2, in_top3_2 = score_top3(pred_top3_s2, actual_top3_ids)
    print(f"\nPredicted top 3: {', '.join(DRIVERS_2025_R1[d][1] for d in pred_top3_s2)}")
    print(f"Exact position matches: {exact2}/3")
    print(f"Correct drivers in top 3: {in_top3_2}/3")

    # =========================================================================
    # Stage 3: Constructor Effect
    # =========================================================================
    print("\n" + "=" * 70)
    print("STAGE 3: Constructor + Driver Pooling")
    print("=" * 70)

    loader3 = S3Loader(DATA_DIR)
    df3 = loader3.load_merged(min_year=2020, max_year=2024)
    prev3, next3, meta3 = s3_prepare(df3)

    model3 = ConstructorPooledDirichletF1(
        prior_alpha_global=1.0,
        prior_alpha_constructor=1.0,
        kappa_init=(10.0, 10.0),
        kappa_bounds=((0.1, 500.0), (0.01, 500.0)),
    )
    model3.fit(prev3, next3, meta3)

    print(f"Training data: {len(prev3)} transitions")
    print(f"Optimal kappa_g: {model3.kappa_g_:.2f}, kappa_c: {model3.kappa_c_:.2f}")
    kg_share = model3.kappa_g_ / (model3.kappa_g_ + model3.kappa_c_)
    print(f"Prior blend: {kg_share:.0%} global / {1-kg_share:.0%} constructor")

    constructor_names = (
        df3[["constructorId", "constructor_name"]]
        .drop_duplicates()
        .set_index("constructorId")["constructor_name"]
        .to_dict()
    )

    s3_results = []
    for driver_id, (cid_2025, name, grid) in DRIVERS_2025_R1.items():
        if driver_id in model3.driver_constructor_counts_:
            probs = model3.predict_proba(driver_id, START, constructor_id=cid_2025)
            dc = model3.driver_constructor_counts_.get(driver_id, {})
            n_obs = sum(m.sum() for m in dc.values())
        else:
            probs = model3.predict_proba_new_driver(START, constructor_id=cid_2025)
            n_obs = 0

        s3_results.append({
            "driver_id": driver_id,
            "name": name,
            "constructor_id": cid_2025,
            "P(top3)": prob_top3(probs),
            "P(P1)": probs[1],
            "P(P2)": probs[2],
            "P(P3)": probs[3],
            "E[pos]": expected_position(probs),
            "n_obs": n_obs,
        })

    s3_df = pd.DataFrame(s3_results).sort_values("P(top3)", ascending=False)

    print(f"\nPredicted rankings by P(top 3 finish):")
    print(f"{'Rank':<5} {'Driver':<22} {'Team':<15} {'P(top3)':>8} "
          f"{'P(P1)':>7} {'P(P2)':>7} {'P(P3)':>7} {'E[pos]':>7}")
    print("-" * 90)
    for rank, (_, row) in enumerate(s3_df.iterrows(), 1):
        marker = " ***" if row["driver_id"] in ACTUAL_TOP3 else ""
        team = constructor_names.get(row["constructor_id"], str(row["constructor_id"]))
        print(f"{rank:<5} {row['name']:<22} {team:<15} "
              f"{row['P(top3)']:>8.4f} {row['P(P1)']:>7.4f} "
              f"{row['P(P2)']:>7.4f} {row['P(P3)']:>7.4f} "
              f"{row['E[pos]']:>7.2f}{marker}")

    pred_top3_s3 = s3_df["driver_id"].head(3).tolist()
    exact3, in_top3_3 = score_top3(pred_top3_s3, actual_top3_ids)
    print(f"\nPredicted top 3: {', '.join(DRIVERS_2025_R1[d][1] for d in pred_top3_s3)}")
    print(f"Exact position matches: {exact3}/3")
    print(f"Correct drivers in top 3: {in_top3_3}/3")

    # =========================================================================
    # Stage 4: Recency + Grid Position
    # =========================================================================
    print("\n" + "=" * 70)
    print("STAGE 4: Recency-Weighted Constructor + Driver")
    print("=" * 70)

    loader4 = S4Loader(DATA_DIR)
    df4 = loader4.load_merged(min_year=2020, max_year=2024)
    prev4, next4, meta4 = s4_prepare(df4)

    model4 = RecencyGridDirichletF1(
        prior_alpha_global=1.0,
        prior_alpha_constructor=1.0,
        kappa_init=(5.0, 10.0),
        kappa_bounds=((0.01, 500.0), (0.01, 500.0)),
        lambda_init=0.15,
        lambda_bounds=(0.01, 0.7),
    )
    model4.fit(prev4, next4, meta4)

    print(f"Training data: {len(prev4)} transitions")
    print(f"Optimal kappa_g: {model4.kappa_g_:.2f}, "
          f"kappa_c: {model4.kappa_c_:.2f}")
    print(f"Recency lambda: {model4.lambda_:.4f} "
          f"(half-life: {np.log(2)/max(model4.lambda_, 1e-10):.1f} years)")
    total_k = model4.kappa_g_ + model4.kappa_c_
    print(f"Prior blend: {model4.kappa_g_/total_k:.0%} global / "
          f"{model4.kappa_c_/total_k:.0%} constructor")

    s4_results = []
    for driver_id, (cid_2025, name, grid_pos) in DRIVERS_2025_R1.items():
        if driver_id in model4.driver_constructor_counts_:
            probs = model4.predict_proba(
                driver_id, START, constructor_id=cid_2025
            )
            dc = model4.driver_constructor_counts_.get(driver_id, {})
            n_obs = sum(m.sum() for m in dc.values())
        else:
            probs = model4.predict_proba_new_driver(
                START, constructor_id=cid_2025
            )
            n_obs = 0

        s4_results.append({
            "driver_id": driver_id,
            "name": name,
            "constructor_id": cid_2025,
            "P(top3)": prob_top3(probs),
            "P(P1)": probs[1],
            "P(P2)": probs[2],
            "P(P3)": probs[3],
            "E[pos]": expected_position(probs),
            "n_obs": n_obs,
        })

    s4_df = pd.DataFrame(s4_results).sort_values("P(top3)", ascending=False)

    print(f"\nPredicted rankings by P(top 3 finish):")
    print(f"{'Rank':<5} {'Driver':<22} {'Team':<15} "
          f"{'P(top3)':>8} {'P(P1)':>7} {'P(P2)':>7} {'P(P3)':>7} {'E[pos]':>7}")
    print("-" * 90)
    for rank, (_, row) in enumerate(s4_df.iterrows(), 1):
        marker = " ***" if row["driver_id"] in ACTUAL_TOP3 else ""
        team = constructor_names.get(row["constructor_id"], str(row["constructor_id"]))
        print(f"{rank:<5} {row['name']:<22} {team:<15} "
              f"{row['P(top3)']:>8.4f} {row['P(P1)']:>7.4f} "
              f"{row['P(P2)']:>7.4f} {row['P(P3)']:>7.4f} "
              f"{row['E[pos]']:>7.2f}{marker}")

    pred_top3_s4 = s4_df["driver_id"].head(3).tolist()
    exact4, in_top3_4 = score_top3(pred_top3_s4, actual_top3_ids)
    print(f"\nPredicted top 3: {', '.join(DRIVERS_2025_R1[d][1] for d in pred_top3_s4)}")
    print(f"Exact position matches: {exact4}/3")
    print(f"Correct drivers in top 3: {in_top3_4}/3")

    # =========================================================================
    # Comparison Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\nActual 2025 Australian GP Top 3:")
    print(f"  P1: Lando Norris (McLaren)")
    print(f"  P2: Max Verstappen (Red Bull)")
    print(f"  P3: George Russell (Mercedes)")

    print(f"\n{'Model':<40} {'Predicted Top 3':<45} {'Exact':>6} {'In T3':>6}")
    print("-" * 100)

    print(f"{'Stage 1 (Global)':<40} {'(cannot differentiate drivers)':<45} {'N/A':>6} {'N/A':>6}")

    s2_names = ', '.join(DRIVERS_2025_R1[d][1] for d in pred_top3_s2)
    print(f"{'Stage 2 (Driver Pooling)':<40} {s2_names:<45} {exact2:>6}/3 {in_top3_2:>6}/3")

    s3_names = ', '.join(DRIVERS_2025_R1[d][1] for d in pred_top3_s3)
    print(f"{'Stage 3 (Constructor+Driver)':<40} {s3_names:<45} {exact3:>6}/3 {in_top3_3:>6}/3")

    s4_names = ', '.join(DRIVERS_2025_R1[d][1] for d in pred_top3_s4)
    print(f"{'Stage 4 (Recency+Constr+Driver)':<40} {s4_names:<45} {exact4:>6}/3 {in_top3_4:>6}/3")

    # --- Calibration ---
    print(f"\n--- Calibration: P(actual outcome) assigned to actual top 3 ---")

    s1_cal = sum(probs_start_s1[pos] for pos in [1, 2, 3])
    print(f"  Stage 1: sum P(P1)+P(P2)+P(P3) = {s1_cal:.4f} (same for all drivers)")

    # Stage 2
    s2_cal = 0
    for did, actual_pos in ACTUAL_TOP3.items():
        if did in model2.driver_counts_:
            p = model2.predict_proba(did, START)[actual_pos]
        else:
            p = model2.predict_proba_new_driver(START)[actual_pos]
        s2_cal += p
        print(f"    {DRIVERS_2025_R1[did][1]}: P(P{actual_pos}) = {p:.4f}")
    print(f"  Stage 2: sum = {s2_cal:.4f}")

    # Stage 3
    s3_cal = 0
    for did, actual_pos in ACTUAL_TOP3.items():
        cid = DRIVERS_2025_R1[did][0]
        if did in model3.driver_constructor_counts_:
            p = model3.predict_proba(did, START, constructor_id=cid)[actual_pos]
        else:
            p = model3.predict_proba_new_driver(START, constructor_id=cid)[actual_pos]
        s3_cal += p
        print(f"    {DRIVERS_2025_R1[did][1]}: P(P{actual_pos}) = {p:.4f}")
    print(f"  Stage 3: sum = {s3_cal:.4f}")

    # Stage 4
    s4_cal = 0
    for did, actual_pos in ACTUAL_TOP3.items():
        cid, name, grid_pos = DRIVERS_2025_R1[did]
        if did in model4.driver_constructor_counts_:
            p = model4.predict_proba(did, START, constructor_id=cid)[actual_pos]
        else:
            p = model4.predict_proba_new_driver(START, constructor_id=cid)[actual_pos]
        s4_cal += p
        print(f"    {name}: P(P{actual_pos}) = {p:.4f}")
    print(f"  Stage 4: sum = {s4_cal:.4f}")

    # Winner
    print(f"\n--- Best Model ---")
    scores = {
        "Stage 2": (in_top3_2, exact2, s2_cal),
        "Stage 3": (in_top3_3, exact3, s3_cal),
        "Stage 4": (in_top3_4, exact4, s4_cal),
    }
    best = max(scores.items(), key=lambda x: (x[1][0], x[1][1], x[1][2]))
    print(f"  {best[0]} wins with {best[1][0]}/3 correct drivers in top 3, "
          f"{best[1][1]}/3 exact matches, calibration score {best[1][2]:.4f}")


if __name__ == "__main__":
    main()
