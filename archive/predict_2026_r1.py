"""
Predict the 2026 Australian GP (Melbourne, March 8 2026) using Markov models
trained on 2010-2025 data.

Predictions:
  - Top 3 finishing positions
  - Pole position (P1 finish probability proxy)
  - Big positive surprise
  - Big negative surprise
  - One crazy prediction
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

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


DATA_DIR = Path(__file__).parent / "data"

# 2026 Australian GP drivers: driverId -> (constructorId, name)
# New constructor IDs: Cadillac = 216 (new team, no history)
# Audi = 15 (rebranded Sauber, same entity)
DRIVERS_2026_R1 = {
    # McLaren (1)
    846: (1, "Lando Norris"),
    857: (1, "Oscar Piastri"),
    # Ferrari (6)
    1:   (6, "Lewis Hamilton"),
    844: (6, "Charles Leclerc"),
    # Mercedes (131)
    847: (131, "George Russell"),
    863: (131, "Kimi Antonelli"),
    # Red Bull (9)
    830: (9, "Max Verstappen"),
    864: (9, "Isack Hadjar"),
    # Aston Martin (117)
    4:   (117, "Fernando Alonso"),
    840: (117, "Lance Stroll"),
    # Williams (3)
    832: (3, "Carlos Sainz"),
    848: (3, "Alex Albon"),
    # Alpine (214)
    861: (214, "Franco Colapinto"),
    842: (214, "Pierre Gasly"),
    # Haas (210)
    839: (210, "Esteban Ocon"),
    860: (210, "Oliver Bearman"),
    # Audi / Sauber (15)
    807: (15, "Nico Hulkenberg"),
    865: (15, "Gabriel Bortoleto"),
    # Racing Bulls (215)
    859: (215, "Liam Lawson"),
    866: (215, "Arvid Lindblad"),      # new rookie, no history
    # Cadillac (216) — brand new team
    822: (216, "Valtteri Bottas"),
    815: (216, "Sergio Perez"),
}


def prob_top3(probs):
    return probs[1] + probs[2] + probs[3]


def expected_position(probs):
    return sum(j * probs[j] for j in range(1, N_OUTCOMES))


def prob_dnf(probs):
    return probs[0]


def main():
    print("=" * 70)
    print("2026 Australian GP Predictions (Melbourne, March 8 2026)")
    print("Training on 2010-2025 seasons")
    print("=" * 70)

    # =====================================================================
    # Stage 2: Driver-Specific Partial Pooling
    # =====================================================================
    print("\n--- Training Stage 2 (Driver Partial Pooling) ---")
    loader2 = S2Loader(DATA_DIR)
    df2 = loader2.load_merged(min_year=2010, max_year=2025)
    prev2, next2, meta2 = s2_prepare(df2)

    model2 = PartialPooledDirichletF1(
        prior_alpha_global=1.0,
        kappa_init=10.0,
        kappa_bounds=(0.1, 500.0),
    )
    model2.fit(prev2, next2, meta2)
    print(f"Transitions: {len(prev2)}, Kappa: {model2.kappa_:.2f}")

    # =====================================================================
    # Stage 3: Constructor + Driver Pooling
    # =====================================================================
    print("\n--- Training Stage 3 (Constructor + Driver) ---")
    loader3 = S3Loader(DATA_DIR)
    df3 = loader3.load_merged(min_year=2010, max_year=2025)
    prev3, next3, meta3 = s3_prepare(df3)

    model3 = ConstructorPooledDirichletF1(
        prior_alpha_global=1.0,
        prior_alpha_constructor=1.0,
        kappa_init=(10.0, 10.0),
        kappa_bounds=((0.1, 500.0), (0.01, 500.0)),
    )
    model3.fit(prev3, next3, meta3)
    print(f"Transitions: {len(prev3)}, Kappa_g: {model3.kappa_g_:.2f}, "
          f"Kappa_c: {model3.kappa_c_:.2f}")

    constructor_names = (
        df3[["constructorId", "constructor_name"]]
        .drop_duplicates()
        .set_index("constructorId")["constructor_name"]
        .to_dict()
    )
    # Add new constructors not in training data
    constructor_names[216] = "Cadillac"
    constructor_names[15] = "Audi"

    # =====================================================================
    # Predictions
    # =====================================================================
    print("\n" + "=" * 70)
    print("STAGE 2 PREDICTIONS (Driver History)")
    print("=" * 70)

    s2_results = []
    for driver_id, (cid, name) in DRIVERS_2026_R1.items():
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
            "constructor_id": cid,
            "P(top3)": prob_top3(probs),
            "P(P1)": probs[1],
            "P(P2)": probs[2],
            "P(P3)": probs[3],
            "E[pos]": expected_position(probs),
            "P(DNF)": prob_dnf(probs),
            "n_obs": n_obs,
            "pooling": pf,
        })

    s2_df = pd.DataFrame(s2_results).sort_values("P(top3)", ascending=False)

    print(f"\n{'Rank':<5} {'Driver':<22} {'Team':<15} {'P(top3)':>8} "
          f"{'P(P1)':>7} {'P(P2)':>7} {'P(P3)':>7} {'E[pos]':>7} {'n_obs':>6}")
    print("-" * 95)
    for rank, (_, row) in enumerate(s2_df.iterrows(), 1):
        team = constructor_names.get(row["constructor_id"], str(row["constructor_id"]))
        print(f"{rank:<5} {row['name']:<22} {team:<15} "
              f"{row['P(top3)']:>8.4f} {row['P(P1)']:>7.4f} "
              f"{row['P(P2)']:>7.4f} {row['P(P3)']:>7.4f} "
              f"{row['E[pos]']:>7.2f} {row['n_obs']:>6.0f}")

    print("\n" + "=" * 70)
    print("STAGE 3 PREDICTIONS (Constructor + Driver)")
    print("=" * 70)

    s3_results = []
    for driver_id, (cid, name) in DRIVERS_2026_R1.items():
        if driver_id in model3.driver_constructor_counts_:
            probs = model3.predict_proba(driver_id, START, constructor_id=cid)
            dc = model3.driver_constructor_counts_.get(driver_id, {})
            n_obs = sum(m.sum() for m in dc.values())
        else:
            probs = model3.predict_proba_new_driver(START, constructor_id=cid)
            n_obs = 0

        s3_results.append({
            "driver_id": driver_id,
            "name": name,
            "constructor_id": cid,
            "P(top3)": prob_top3(probs),
            "P(P1)": probs[1],
            "P(P2)": probs[2],
            "P(P3)": probs[3],
            "E[pos]": expected_position(probs),
            "P(DNF)": prob_dnf(probs),
            "n_obs": n_obs,
        })

    s3_df = pd.DataFrame(s3_results).sort_values("P(top3)", ascending=False)

    print(f"\n{'Rank':<5} {'Driver':<22} {'Team':<15} {'P(top3)':>8} "
          f"{'P(P1)':>7} {'P(P2)':>7} {'P(P3)':>7} {'E[pos]':>7} {'n_obs':>6}")
    print("-" * 95)
    for rank, (_, row) in enumerate(s3_df.iterrows(), 1):
        team = constructor_names.get(row["constructor_id"], str(row["constructor_id"]))
        print(f"{rank:<5} {row['name']:<22} {team:<15} "
              f"{row['P(top3)']:>8.4f} {row['P(P1)']:>7.4f} "
              f"{row['P(P2)']:>7.4f} {row['P(P3)']:>7.4f} "
              f"{row['E[pos]']:>7.2f} {row['n_obs']:>6.0f}")

    # =====================================================================
    # FINAL PREDICTIONS (blending Stage 2 and Stage 3 insights)
    # =====================================================================
    print("\n" + "=" * 70)
    print("FINAL PREDICTIONS — 2026 AUSTRALIAN GP")
    print("=" * 70)

    # Use Stage 3 as primary (constructor-aware) but note Stage 2 rankings
    # for driver-level signal

    # Expected position range for top contenders
    print("\n--- Top contender probabilities ---")
    top_contenders = s3_df.head(10)["driver_id"].tolist()
    for did in top_contenders:
        cid, name = DRIVERS_2026_R1[did]
        if did in model3.driver_constructor_counts_:
            probs = model3.predict_proba(did, START, constructor_id=cid)
        else:
            probs = model3.predict_proba_new_driver(START, constructor_id=cid)
        team = constructor_names.get(cid, str(cid))
        epos = expected_position(probs)
        ptop3 = prob_top3(probs)
        print(f"  {name:<22} {team:<15}  P(top3)={ptop3:.1%}, E[pos]={epos:.1f}")

    # --- Top 3 Prediction ---
    print("\n" + "-" * 50)
    print("TOP 3 PREDICTION")
    print("-" * 50)
    pred_top3 = s3_df.head(3)
    for rank, (_, row) in enumerate(pred_top3.iterrows(), 1):
        team = constructor_names.get(row["constructor_id"], str(row["constructor_id"]))
        print(f"  P{rank}: {row['name']} ({team}) — "
              f"P(top3)={row['P(top3)']:.1%}, P(P{rank})={row[f'P(P{rank})']:.1%}")

    # --- Pole Position ---
    print("\n" + "-" * 50)
    print("POLE POSITION (highest P(P1))")
    print("-" * 50)
    pole_s3 = s3_df.sort_values("P(P1)", ascending=False).head(3)
    for rank, (_, row) in enumerate(pole_s3.iterrows(), 1):
        team = constructor_names.get(row["constructor_id"], str(row["constructor_id"]))
        print(f"  {rank}. {row['name']} ({team}) — P(P1)={row['P(P1)']:.1%}")

    # --- Biggest Positive Surprise ---
    print("\n" + "-" * 50)
    print("BIG POSITIVE SURPRISE")
    print("-" * 50)
    # Driver ranked low in general expectations but with non-trivial P(top3)
    # Merge Stage 2 and Stage 3 to find disagreements
    merged = s3_df.copy()
    merged["s3_rank"] = range(1, len(merged) + 1)
    s2_ranks = {row["driver_id"]: rank for rank, (_, row)
                in enumerate(s2_df.iterrows(), 1)}
    merged["s2_rank"] = merged["driver_id"].map(s2_ranks)
    merged["surprise_up"] = merged["s2_rank"] - merged["s3_rank"]  # positive = higher in S3

    # Drivers who could overperform: mid-grid constructor but strong personal history
    # or constructor boost from team change
    # Focus on drivers ranked 5-15 in Stage 3 with decent P(top3)
    mid_grid = merged[(merged["s3_rank"] >= 4) & (merged["s3_rank"] <= 15)]
    mid_grid = mid_grid.sort_values("P(top3)", ascending=False)

    if len(mid_grid) > 0:
        surprise = mid_grid.iloc[0]
        team = constructor_names.get(surprise["constructor_id"],
                                     str(surprise["constructor_id"]))
        print(f"  {surprise['name']} ({team})")
        print(f"  Stage 3 rank: {surprise['s3_rank']}, P(top3)={surprise['P(top3)']:.1%}")
        print(f"  Reasoning: Strong historical results could combine with team upgrades")

    # --- Biggest Negative Surprise ---
    print("\n" + "-" * 50)
    print("BIG NEGATIVE SURPRISE")
    print("-" * 50)
    # Top-ranked driver most likely to underperform
    top6 = merged[merged["s3_rank"] <= 6].copy()
    top6["dnf_risk"] = top6["driver_id"].apply(
        lambda d: s3_df[s3_df["driver_id"] == d]["P(DNF)"].values[0]
    )
    top6 = top6.sort_values("dnf_risk", ascending=False)
    if len(top6) > 0:
        bust = top6.iloc[0]
        team = constructor_names.get(bust["constructor_id"],
                                     str(bust["constructor_id"]))
        print(f"  {bust['name']} ({team})")
        print(f"  Stage 3 rank: {bust['s3_rank']}, P(DNF)={bust['dnf_risk']:.1%}")
        print(f"  Expected to contend but highest DNF risk among top contenders")

    # --- Crazy Prediction ---
    print("\n" + "-" * 50)
    print("CRAZY PREDICTION")
    print("-" * 50)
    # Find a backmarker with the best P(top3) — the long shot
    backmarkers = merged[merged["s3_rank"] >= 15]
    if len(backmarkers) > 0:
        crazy = backmarkers.sort_values("P(top3)", ascending=False).iloc[0]
        team = constructor_names.get(crazy["constructor_id"],
                                     str(crazy["constructor_id"]))
        # Calculate odds
        p = crazy["P(top3)"]
        odds = (1 - p) / max(p, 0.001)
        print(f"  {crazy['name']} finishes on the podium!")
        print(f"  ({team}) — P(top3)={p:.1%} ({odds:.0f}-to-1 against)")
        print(f"  A chaotic wet Melbourne race could make this happen")

    # --- Season-level narrative ---
    print("\n" + "-" * 50)
    print("KEY STORYLINES FOR THE MODEL")
    print("-" * 50)

    # Hamilton at Ferrari
    ham_s3 = s3_df[s3_df["driver_id"] == 1].iloc[0]
    team_ham = constructor_names.get(ham_s3["constructor_id"], "")
    print(f"  Hamilton at {team_ham}: P(top3)={ham_s3['P(top3)']:.1%}, "
          f"P(P1)={ham_s3['P(P1)']:.1%}")

    # Verstappen
    ver_s3 = s3_df[s3_df["driver_id"] == 830].iloc[0]
    team_ver = constructor_names.get(ver_s3["constructor_id"], "")
    print(f"  Verstappen at {team_ver}: P(top3)={ver_s3['P(top3)']:.1%}, "
          f"P(P1)={ver_s3['P(P1)']:.1%}")

    # Cadillac (new team)
    cadillac_drivers = merged[merged["constructor_id"] == 216]
    if len(cadillac_drivers) > 0:
        best_cad = cadillac_drivers.sort_values("P(top3)", ascending=False).iloc[0]
        print(f"  Cadillac's best bet: {best_cad['name']} — "
              f"P(top3)={best_cad['P(top3)']:.1%} (new team, no constructor history)")


if __name__ == "__main__":
    main()
