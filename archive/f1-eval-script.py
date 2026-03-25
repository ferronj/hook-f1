#!/usr/bin/env python
"""
F1 Dirichlet-Multinomial Model Evaluation Script
==================================================
Reads a YAML config, builds models and split strategies, runs the
evaluation framework, and writes results to disk.

Usage:
    python evaluate_models.py --config config.yaml
    python evaluate_models.py --config config.yaml --dry-run
"""

import argparse
import sys
import time
from pathlib import Path
from itertools import combinations

import yaml
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project imports — adjust these if your module filenames differ
# ---------------------------------------------------------------------------
from stage1_global_transition import (
    F1DataLoader,
)
from stage3_constructor import (
    prepare_transitions,
)
from evaluation_framework import (
    # Adapters
    Stage1Adapter,
    Stage2Adapter,
    Stage3Adapter,
    MODEL_REGISTRY,
    # Splits
    TemporalHoldout,
    RollingOrigin,
    LeaveOneSeasonOut,
    LeaveOneRoundOut,
    # Evaluation
    Evaluator,
    EvaluationReport,
    Scorer,
    # Analysis
    calibration_table,
    per_driver_breakdown,
    per_prev_position_breakdown,
)


# ===========================================================================
# Config Parsing
# ===========================================================================
def load_config(path: str) -> dict:
    """Load and validate YAML config."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with open(p) as f:
        cfg = yaml.safe_load(f)

    # Apply defaults
    cfg.setdefault("data", {})
    cfg["data"].setdefault("dir", "data/f1")
    cfg["data"].setdefault("min_year", None)
    cfg["data"].setdefault("max_year", None)

    cfg.setdefault("models", [])
    cfg.setdefault("splits", [])

    cfg.setdefault("output", {})
    cfg["output"].setdefault("dir", "results")
    cfg["output"].setdefault("prefix", "eval")
    cfg["output"].setdefault("save_per_observation", True)
    cfg["output"].setdefault("save_fold_results", True)
    cfg["output"].setdefault("save_head_to_head", True)

    cfg.setdefault("display", {})
    cfg["display"].setdefault("verbose", True)
    cfg["display"].setdefault("top_n_drivers", 10)
    cfg["display"].setdefault("show_calibration", True)
    cfg["display"].setdefault("show_prev_position_breakdown", True)
    cfg["display"].setdefault("show_per_driver_breakdown", True)

    return cfg


# ===========================================================================
# Factory Functions
# ===========================================================================
ADAPTER_FACTORY = {
    "stage1": lambda p: Stage1Adapter(
        prior_alpha=p.get("prior_alpha", 1.0),
    ),
    "stage2": lambda p: Stage2Adapter(
        prior_alpha_global=p.get("prior_alpha_global", 1.0),
        kappa_init=p.get("kappa_init", 10.0),
        kappa_bounds=tuple(p.get("kappa_bounds", [0.1, 500.0])),
    ),
    "stage3": lambda p: Stage3Adapter(
        prior_alpha_global=p.get("prior_alpha_global", 1.0),
        prior_alpha_constructor=p.get("prior_alpha_constructor", 1.0),
        kappa_init=tuple(p.get("kappa_init", [10.0, 10.0])),
        kappa_bounds=tuple(
            tuple(b) for b in p.get(
                "kappa_bounds", [[0.1, 500.0], [0.01, 500.0]]
            )
        ),
    ),
}

SPLIT_FACTORY = {
    "temporal_holdout": lambda p: TemporalHoldout(
        cutoff_years=p.get("cutoff_years", [2024]),
    ),
    "rolling_origin": lambda p: RollingOrigin(
        min_train_seasons=p.get("min_train_seasons", 3),
    ),
    "leave_one_season_out": lambda p: LeaveOneSeasonOut(),
    "leave_one_round_out": lambda p: LeaveOneRoundOut(
        seasons=p.get("seasons"),
    ),
}


def build_models(cfg: dict) -> list:
    """Instantiate enabled model adapters from config."""
    models = []
    for entry in cfg["models"]:
        if not entry.get("enabled", True):
            continue
        name = entry["name"]
        params = entry.get("params", {})
        if name not in ADAPTER_FACTORY:
            raise ValueError(
                f"Unknown model '{name}'. "
                f"Available: {list(ADAPTER_FACTORY.keys())}"
            )
        models.append(ADAPTER_FACTORY[name](params))
    return models


def build_splits(cfg: dict) -> list:
    """Instantiate enabled split strategies from config."""
    splits = []
    for entry in cfg["splits"]:
        if not entry.get("enabled", True):
            continue
        name = entry["name"]
        params = entry.get("params", {})
        if name not in SPLIT_FACTORY:
            raise ValueError(
                f"Unknown split '{name}'. "
                f"Available: {list(SPLIT_FACTORY.keys())}"
            )
        splits.append(SPLIT_FACTORY[name](params))
    return splits


# ===========================================================================
# Data Loading
# ===========================================================================
def load_data(cfg: dict) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, dict, dict]:
    """
    Load Kaggle data and prepare transitions.

    Returns: (raw_df, prev, next_, meta, driver_names, constructor_names)
    """
    data_cfg = cfg["data"]
    data_dir = Path(data_cfg["dir"])

    if not (data_dir / "results.csv").exists():
        print(f"ERROR: Kaggle data not found at {data_dir}")
        print(
            "Download from: https://www.kaggle.com/datasets/"
            "rohanrao/formula-1-world-championship-1950-2020"
        )
        print(f"Extract CSVs into {data_dir}/")
        sys.exit(1)

    loader = F1DataLoader(data_dir)
    df = loader.load_merged(
        min_year=data_cfg.get("min_year"),
        max_year=data_cfg.get("max_year"),
    )
    prev, next_, meta = prepare_transitions(df)

    driver_names = (
        df[["driverId", "driver_name"]]
        .drop_duplicates()
        .set_index("driverId")["driver_name"]
        .to_dict()
    )
    constructor_names = (
        df[["constructorId", "constructor_name"]]
        .drop_duplicates()
        .set_index("constructorId")["constructor_name"]
        .to_dict()
    )

    return df, prev, next_, meta, driver_names, constructor_names


# ===========================================================================
# Output
# ===========================================================================
def save_results(
    cfg: dict,
    report: EvaluationReport,
    models: list,
    driver_names: dict,
):
    """Write evaluation results to CSV files."""
    out_cfg = cfg["output"]
    out_dir = Path(out_cfg["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_cfg["prefix"]
    saved = []

    # Model summary (always saved)
    ms = report.model_summary()
    p = out_dir / f"{prefix}_model_summary.csv"
    ms.to_csv(p, index=False)
    saved.append(p)

    # Fold results
    if out_cfg.get("save_fold_results"):
        fs = report.summary()
        p = out_dir / f"{prefix}_fold_results.csv"
        fs.to_csv(p, index=False)
        saved.append(p)

    # Per-observation predictions
    if out_cfg.get("save_per_observation"):
        obs = report.per_observation_df()
        if not obs.empty:
            p = out_dir / f"{prefix}_per_observation.csv"
            obs.to_csv(p, index=False)
            saved.append(p)

    # Head-to-head for all model pairs
    if out_cfg.get("save_head_to_head") and len(models) >= 2:
        model_names = [m.name for m in models]
        h2h_rows = []
        for a, b in combinations(model_names, 2):
            h2h = report.head_to_head(a, b, "log_loss")
            if not h2h.empty:
                h2h["model_a"] = a
                h2h["model_b"] = b
                h2h_rows.append(h2h)
        if h2h_rows:
            h2h_all = pd.concat(h2h_rows, ignore_index=True)
            p = out_dir / f"{prefix}_head_to_head.csv"
            h2h_all.to_csv(p, index=False)
            saved.append(p)

    # Per-driver breakdown
    pdb = per_driver_breakdown(report, driver_name_map=driver_names)
    if not pdb.empty:
        p = out_dir / f"{prefix}_per_driver.csv"
        pdb.to_csv(p, index=False)
        saved.append(p)

    # Per-prev-position breakdown
    ppb = per_prev_position_breakdown(report)
    if not ppb.empty:
        p = out_dir / f"{prefix}_per_prev_position.csv"
        ppb.to_csv(p, index=False)
        saved.append(p)

    return saved


# ===========================================================================
# Display
# ===========================================================================
def print_report(
    cfg: dict,
    report: EvaluationReport,
    models: list,
    driver_names: dict,
):
    """Print formatted evaluation results to stdout."""
    disp = cfg["display"]

    # --- Model summary ---
    print("\n" + "=" * 70)
    print("MODEL COMPARISON (averaged across folds)")
    print("=" * 70)
    ms = report.model_summary()
    print(ms.to_string(index=False))

    # --- Head-to-head for all pairs ---
    model_names = [m.name for m in models]
    if len(model_names) >= 2:
        print("\n" + "-" * 70)
        print("HEAD-TO-HEAD COMPARISONS (log_loss, lower = better)")
        print("-" * 70)
        for a, b in combinations(model_names, 2):
            h2h = report.head_to_head(a, b, "log_loss")
            if h2h.empty:
                continue
            wins_a = (h2h["diff (A-B)"] < 0).sum()
            wins_b = (h2h["diff (A-B)"] > 0).sum()
            mean_diff = h2h["diff (A-B)"].mean()
            print(
                f"\n  {a}\n  vs {b}"
            )
            print(
                f"  → Mean diff: {mean_diff:+.4f}  "
                f"(A wins {wins_a}/{len(h2h)}, "
                f"B wins {wins_b}/{len(h2h)})"
            )

    # --- Per-prev-position breakdown ---
    if disp.get("show_prev_position_breakdown"):
        print("\n" + "-" * 70)
        print("ACCURACY BY PREVIOUS POSITION (best model)")
        print("-" * 70)
        ppb = per_prev_position_breakdown(report)
        if not ppb.empty:
            best_model = ms.iloc[0]["model"]
            subset = ppb[ppb["model"] == best_model]
            cols = [
                "prev_label", "n_obs", "log_loss",
                "accuracy_top1", "accuracy_top3", "position_error",
            ]
            print(f"  Model: {best_model}\n")
            print(
                subset[cols]
                .round(3)
                .to_string(index=False)
            )

    # --- Per-driver breakdown ---
    if disp.get("show_per_driver_breakdown"):
        n = disp.get("top_n_drivers", 10)
        print("\n" + "-" * 70)
        print(f"PER-DRIVER BREAKDOWN (best model, top {n} hardest)")
        print("-" * 70)
        pdb = per_driver_breakdown(report, driver_name_map=driver_names)
        if not pdb.empty:
            best_model = ms.iloc[0]["model"]
            subset = (
                pdb[pdb["model"] == best_model]
                .sort_values("log_loss", ascending=False)
                .head(n)
            )
            cols = [
                c for c in
                ["driver_name", "n_obs", "log_loss",
                 "accuracy_top1", "position_error"]
                if c in subset.columns
            ]
            print(f"  Model: {best_model}\n")
            print(subset[cols].round(3).to_string(index=False))

    # --- Calibration ---
    if disp.get("show_calibration"):
        print("\n" + "-" * 70)
        print("CALIBRATION (best model)")
        print("-" * 70)
        best_model = ms.iloc[0]["model"]
        cal = calibration_table(report, best_model, n_bins=5)
        if not cal.empty:
            print(f"  Model: {best_model}\n")
            print(cal.round(4).to_string(index=False))


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="F1 Dirichlet-Multinomial Model Evaluation"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Load config and data, show plan, but don't run evaluation",
    )
    args = parser.parse_args()

    # --- Load config ---
    cfg = load_config(args.config)
    print("=" * 70)
    print("F1 DIRICHLET-MULTINOMIAL MODEL EVALUATION")
    print("=" * 70)
    print(f"Config: {args.config}")

    # --- Load data ---
    t0 = time.time()
    df, prev, next_, meta, driver_names, constructor_names = load_data(cfg)
    load_time = time.time() - t0

    seasons = sorted(meta["season"].unique())
    print(f"\nData loaded in {load_time:.1f}s")
    print(
        f"  Transitions: {len(prev):,}")
    print(
        f"  Seasons:     {len(seasons)} "
        f"({seasons[0]}–{seasons[-1]})"
    )
    print(f"  Drivers:     {meta['driver'].nunique()}")
    print(f"  Constructors: {meta['constructor'].nunique()}")

    # --- Build models ---
    models = build_models(cfg)
    if not models:
        print("\nERROR: No models enabled in config.")
        sys.exit(1)
    print(f"\nModels ({len(models)}):")
    for m in models:
        print(f"  • {m.name}")

    # --- Build splits ---
    splits = build_splits(cfg)
    if not splits:
        print("\nERROR: No split strategies enabled in config.")
        sys.exit(1)
    print(f"\nSplit strategies ({len(splits)}):")
    for s in splits:
        n_folds = len(s.generate_splits(meta))
        print(f"  • {s.name}  ({n_folds} folds)")

    total_evals = sum(
        len(s.generate_splits(meta)) for s in splits
    ) * len(models)
    print(f"\nTotal evaluations: {total_evals} (models × folds)")

    # --- Dry run ---
    if args.dry_run:
        print("\n[DRY RUN] Exiting before evaluation.")
        sys.exit(0)

    # --- Run ---
    t0 = time.time()
    evaluator = Evaluator(
        models=models,
        splits=splits,
        verbose=cfg["display"]["verbose"],
    )
    report = evaluator.run(prev, next_, meta)
    eval_time = time.time() - t0
    print(f"\nEvaluation completed in {eval_time:.1f}s")

    # --- Display ---
    print_report(cfg, report, models, driver_names)

    # --- Save ---
    saved = save_results(cfg, report, models, driver_names)
    print("\n" + "-" * 70)
    print(f"RESULTS SAVED ({len(saved)} files)")
    print("-" * 70)
    for p in saved:
        print(f"  {p}")

    print(f"\nDone. Total wall time: {load_time + eval_time:.1f}s")


if __name__ == "__main__":
    main()
