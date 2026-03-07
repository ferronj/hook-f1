#!/usr/bin/env python3
"""
Standalone NUTS training script for Stage 9 Bayesian State-Space.

Usage:
  micromamba run -n f1-markov python3 run_stage9_nuts.py \
    --min-year 2020 --max-year 2024 \
    --draws 1000 --tune 2000 --chains 4 --cores 4 \
    --output stage9_nuts_2025.nc

  # Quick test (< 5 min):
  micromamba run -n f1-markov python3 run_stage9_nuts.py \
    --min-year 2023 --max-year 2024 \
    --draws 50 --tune 100 --chains 1 --cores 1 \
    --output test_trace.nc

Steps:
  1. Load data, prepare transitions
  2. Run MAP CV (existing) to select sigma_d, sigma_c
  3. Run MAP on full data for NUTS warm start
  4. Build PyMC model and run NUTS with checkpointing
  5. Save trace (.nc) + metadata (.pkl)
  6. Print convergence diagnostics
"""

import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "models"))

from stage3_constructor import (
    F1DataLoader,
    prepare_transitions,
)
from stage9_bayesian_ss import BayesianStateSpaceF1


def main():
    parser = argparse.ArgumentParser(
        description="Train Stage 9 BSS with NUTS sampling"
    )
    parser.add_argument("--min-year", type=int, default=2020,
                        help="First training year (default: 2020)")
    parser.add_argument("--max-year", type=int, default=2024,
                        help="Last training year (default: 2024)")
    parser.add_argument("--draws", type=int, default=1000,
                        help="NUTS draws per chain (default: 1000)")
    parser.add_argument("--tune", type=int, default=2000,
                        help="NUTS tuning steps per chain (default: 2000)")
    parser.add_argument("--chains", type=int, default=4,
                        help="Number of chains (default: 4)")
    parser.add_argument("--cores", type=int, default=4,
                        help="Number of CPU cores (default: 4)")
    parser.add_argument("--target-accept", type=float, default=0.9,
                        help="Target acceptance rate (default: 0.9)")
    parser.add_argument("--max-treedepth", type=int, default=12,
                        help="Maximum tree depth for NUTS (default: 12)")
    parser.add_argument("--output", type=str, default="stage9_nuts_trace.nc",
                        help="Output trace file path (default: stage9_nuts_trace.nc)")
    parser.add_argument("--checkpoint-every", type=int, default=200,
                        help="Checkpoint every N draws (default: 200)")
    parser.add_argument("--n-mc-samples", type=int, default=3000,
                        help="MC samples for position marginals (default: 3000)")

    args = parser.parse_args()

    data_dir = Path(__file__).parent / "data"

    print("=" * 70)
    print("STAGE 9 NUTS TRAINING")
    print("=" * 70)
    print(f"  Training data: {args.min_year}-{args.max_year}")
    print(f"  NUTS config: {args.chains} chains × ({args.tune} tune + {args.draws} draws)")
    print(f"  Target accept: {args.target_accept}")
    print(f"  Output: {args.output}")
    print(f"  Checkpoint every: {args.checkpoint_every} draws")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    loader = F1DataLoader(data_dir)
    df = loader.load_merged(min_year=args.min_year, max_year=args.max_year)
    prev_pos, next_pos, meta = prepare_transitions(df)
    print(f"  {len(df)} results, {df['year'].nunique()} seasons, "
          f"{len(prev_pos)} transitions")

    # Create model
    model = BayesianStateSpaceF1(
        sigma_d_candidates=(0.02, 0.05, 0.1, 0.2),
        sigma_c_candidates=(0.02, 0.05, 0.1, 0.2),
        n_mc_samples=args.n_mc_samples,
    )

    # Fit with NUTS
    t0 = time.time()
    model.fit_nuts(
        prev_pos, next_pos, meta,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        cores=args.cores,
        target_accept=args.target_accept,
        max_treedepth=args.max_treedepth,
        output_path=args.output,
        checkpoint_every=args.checkpoint_every,
    )
    elapsed = time.time() - t0

    print(f"\n{'=' * 70}")
    print(f"COMPLETE in {elapsed/60:.1f} minutes")
    print(f"  Trace: {args.output}")
    print(f"  Metadata: {Path(args.output).with_suffix('.pkl')}")
    print(f"{'=' * 70}")

    # Quick prediction sanity check
    from stage9_bayesian_ss import START, N_OUTCOMES
    print("\nSanity check — predict_proba_bayesian from START:")
    driver_names = (
        df[["driverId", "driver_name"]]
        .drop_duplicates()
        .set_index("driverId")["driver_name"]
        .to_dict()
    )
    test_drivers = sorted(
        model.driver_strengths_.items(),
        key=lambda x: x[1], reverse=True
    )[:5]
    for did, _ in test_drivers:
        cid = meta[meta["driver"] == did]["constructor"].iloc[-1]
        probs_map = model.predict_proba(did, START, constructor_id=cid)
        probs_bayes = model.predict_proba_bayesian(
            did, START, constructor_id=cid, n_posterior_draws=200
        )
        name = driver_names.get(did, str(did))
        e_map = sum(j * probs_map[j] for j in range(1, N_OUTCOMES))
        e_bay = sum(j * probs_bayes[j] for j in range(1, N_OUTCOMES))
        print(f"  {name:25s}: MAP E[pos]={e_map:.1f}, Bayes E[pos]={e_bay:.1f}")


if __name__ == "__main__":
    main()
