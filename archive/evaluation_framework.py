"""
F1 Dirichlet-Multinomial Model Evaluation Framework
=====================================================
Pluggable framework for cross-validation, temporal holdout, and
model comparison across all model stages.

Architecture:
  - ModelAdapter (Protocol): uniform fit/predict interface wrapping any stage
  - SplitStrategy: generates train/test index splits (temporal, k-fold, LOO)
  - Scorer: computes metrics on predicted probability vectors vs. actuals
  - Evaluator: orchestrates splits → fit → predict → score → aggregate

Extending:
  - New model:  subclass ModelAdapter, register in MODEL_REGISTRY
  - New metric: add function to Scorer or subclass it
  - New split:  subclass SplitStrategy
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable
from collections import defaultdict
import warnings
import time

# ---------------------------------------------------------------------------
# Shared constants (must match model modules)
# ---------------------------------------------------------------------------
DNF = 0
START = 21
N_PREV_STATES = 22
N_OUTCOMES = 21
OUTCOME_LABELS = ["DNF"] + [f"P{i}" for i in range(1, 21)]
PREV_STATE_LABELS = ["START"] + [f"P{i}" for i in range(1, 21)]

# ===========================================================================
# 1. MODEL ADAPTER PROTOCOL
# ===========================================================================
@runtime_checkable
class ModelAdapter(Protocol):
    """
    Uniform interface that wraps any model stage for evaluation.

    Implementations must handle their own data prep internally.
    The Evaluator only passes raw DataFrames and transition arrays.
    """

    @property
    def name(self) -> str:
        """Human-readable model name for reports."""
        ...

    def fit(
        self,
        prev: np.ndarray,
        next_: np.ndarray,
        meta: pd.DataFrame,
    ) -> None:
        """Fit the model on training data."""
        ...

    def predict_proba(
        self,
        prev_position: int,
        driver: Optional = None,
        constructor: Optional = None,
    ) -> np.ndarray:
        """
        Return P(next | prev, driver?, constructor?) as a (N_OUTCOMES,) array.
        Models that don't use driver/constructor ignore those args.
        """
        ...


# ---------------------------------------------------------------------------
# Adapters for Stages 1–3
# ---------------------------------------------------------------------------
class Stage1Adapter:
    """Wraps DirichletMultinomialF1 (global model)."""

    def __init__(self, prior_alpha: float = 1.0):
        self.prior_alpha = prior_alpha
        self._model = None

    @property
    def name(self) -> str:
        return f"Stage1_Global(α={self.prior_alpha})"

    def fit(self, prev, next_, meta):
        from stage1_global_transition import DirichletMultinomialF1
        self._model = DirichletMultinomialF1(
            prior_alpha=self.prior_alpha
        )
        self._model.fit(prev, next_)

    def predict_proba(self, prev_position, driver=None, constructor=None):
        return self._model.predict_proba(prev_position)


class Stage2Adapter:
    """Wraps PartialPooledDirichletF1 (driver partial pooling)."""

    def __init__(
        self,
        prior_alpha_global: float = 1.0,
        kappa_init: float = 10.0,
        kappa_bounds: tuple = (0.1, 500.0),
    ):
        self.prior_alpha_global = prior_alpha_global
        self.kappa_init = kappa_init
        self.kappa_bounds = kappa_bounds
        self._model = None

    @property
    def name(self) -> str:
        return f"Stage2_DriverPooled(α={self.prior_alpha_global})"

    def fit(self, prev, next_, meta):
        from stage2_driver_pooling import PartialPooledDirichletF1
        self._model = PartialPooledDirichletF1(
            prior_alpha_global=self.prior_alpha_global,
            kappa_init=self.kappa_init,
            kappa_bounds=self.kappa_bounds,
        )
        self._model.fit(prev, next_, meta)

    def predict_proba(self, prev_position, driver=None, constructor=None):
        if driver is not None and driver in self._model.driver_ids_:
            return self._model.predict_proba(driver, prev_position)
        return self._model.predict_proba_new_driver(prev_position)


class Stage3Adapter:
    """Wraps ConstructorPooledDirichletF1 (driver + constructor)."""

    def __init__(
        self,
        prior_alpha_global: float = 1.0,
        prior_alpha_constructor: float = 1.0,
        kappa_init: tuple = (10.0, 10.0),
        kappa_bounds: tuple = ((0.1, 500.0), (0.01, 500.0)),
    ):
        self.prior_alpha_global = prior_alpha_global
        self.prior_alpha_constructor = prior_alpha_constructor
        self.kappa_init = kappa_init
        self.kappa_bounds = kappa_bounds
        self._model = None

    @property
    def name(self) -> str:
        return (
            f"Stage3_Constructor"
            f"(αg={self.prior_alpha_global},"
            f"αc={self.prior_alpha_constructor})"
        )

    def fit(self, prev, next_, meta):
        from stage3_constructor import ConstructorPooledDirichletF1
        self._model = ConstructorPooledDirichletF1(
            prior_alpha_global=self.prior_alpha_global,
            prior_alpha_constructor=self.prior_alpha_constructor,
            kappa_init=self.kappa_init,
            kappa_bounds=self.kappa_bounds,
        )
        self._model.fit(prev, next_, meta)

    def predict_proba(self, prev_position, driver=None, constructor=None):
        if driver is not None and driver in self._model.driver_ids_:
            return self._model.predict_proba(
                driver, prev_position, constructor
            )
        return self._model.predict_proba_new_driver(
            prev_position, constructor
        )


# Default registry for convenience
MODEL_REGISTRY: dict[str, type] = {
    "stage1": Stage1Adapter,
    "stage2": Stage2Adapter,
    "stage3": Stage3Adapter,
}


# ===========================================================================
# 2. SPLIT STRATEGIES
# ===========================================================================
@dataclass
class SplitResult:
    """One train/test split with index arrays and metadata."""
    train_idx: np.ndarray
    test_idx: np.ndarray
    fold_id: str            # e.g. "fold_3", "holdout_2024"
    description: str = ""


class SplitStrategy(ABC):
    """Base class for generating train/test splits."""

    @abstractmethod
    def generate_splits(
        self, meta_df: pd.DataFrame
    ) -> list[SplitResult]:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class TemporalHoldout(SplitStrategy):
    """
    Train on seasons < cutoff, test on seasons >= cutoff.

    Supports multiple cutoff years for rolling-origin evaluation.
    E.g. cutoff_years=[2022, 2023, 2024] produces 3 splits:
      train<2022 / test==2022, train<2023 / test==2023, etc.
    """

    def __init__(
        self,
        cutoff_years: list[int],
        season_col: str = "season",
    ):
        self.cutoff_years = sorted(cutoff_years)
        self.season_col = season_col

    @property
    def name(self) -> str:
        return f"TemporalHoldout({self.cutoff_years})"

    def generate_splits(self, meta_df):
        splits = []
        for year in self.cutoff_years:
            train_mask = meta_df[self.season_col] < year
            test_mask = meta_df[self.season_col] == year
            if train_mask.sum() == 0 or test_mask.sum() == 0:
                warnings.warn(
                    f"Skipping cutoff {year}: empty train or test set"
                )
                continue
            splits.append(SplitResult(
                train_idx=np.where(train_mask)[0],
                test_idx=np.where(test_mask)[0],
                fold_id=f"holdout_{year}",
                description=(
                    f"Train: <{year} ({train_mask.sum()} obs), "
                    f"Test: {year} ({test_mask.sum()} obs)"
                ),
            ))
        return splits


class RollingOrigin(SplitStrategy):
    """
    Expanding-window temporal CV: for each test season, train on
    all prior seasons (optionally with a minimum training window).

    More rigorous than single holdout for time-series evaluation.
    """

    def __init__(
        self,
        min_train_seasons: int = 3,
        season_col: str = "season",
    ):
        self.min_train_seasons = min_train_seasons
        self.season_col = season_col

    @property
    def name(self) -> str:
        return f"RollingOrigin(min_train={self.min_train_seasons})"

    def generate_splits(self, meta_df):
        seasons = sorted(meta_df[self.season_col].unique())
        splits = []
        for i in range(self.min_train_seasons, len(seasons)):
            train_seasons = seasons[:i]
            test_season = seasons[i]
            train_mask = meta_df[self.season_col].isin(train_seasons)
            test_mask = meta_df[self.season_col] == test_season
            splits.append(SplitResult(
                train_idx=np.where(train_mask)[0],
                test_idx=np.where(test_mask)[0],
                fold_id=f"rolling_{test_season}",
                description=(
                    f"Train: {train_seasons[0]}–{train_seasons[-1]} "
                    f"({train_mask.sum()}), "
                    f"Test: {test_season} ({test_mask.sum()})"
                ),
            ))
        return splits


class LeaveOneSeasonOut(SplitStrategy):
    """
    Leave-one-season-out CV. Each season is held out once;
    training uses all other seasons.

    Note: this violates temporal ordering (future data leaks into
    training). Use for model diagnostics, not for forecasting claims.
    """

    def __init__(self, season_col: str = "season"):
        self.season_col = season_col

    @property
    def name(self) -> str:
        return "LeaveOneSeasonOut"

    def generate_splits(self, meta_df):
        seasons = sorted(meta_df[self.season_col].unique())
        splits = []
        for s in seasons:
            train_mask = meta_df[self.season_col] != s
            test_mask = meta_df[self.season_col] == s
            splits.append(SplitResult(
                train_idx=np.where(train_mask)[0],
                test_idx=np.where(test_mask)[0],
                fold_id=f"loso_{s}",
                description=(
                    f"Hold out {s} ({test_mask.sum()} obs), "
                    f"train on rest ({train_mask.sum()} obs)"
                ),
            ))
        return splits


class LeaveOneRoundOut(SplitStrategy):
    """
    Leave-one-race-round-out within each season. Tests generalization
    to individual race weekends. Can be filtered to specific seasons.
    """

    def __init__(
        self,
        seasons: Optional[list[int]] = None,
        season_col: str = "season",
        round_col: str = "race_order",
    ):
        self.seasons = seasons
        self.season_col = season_col
        self.round_col = round_col

    @property
    def name(self) -> str:
        return f"LeaveOneRoundOut(seasons={self.seasons})"

    def generate_splits(self, meta_df):
        if self.seasons:
            df = meta_df[meta_df[self.season_col].isin(self.seasons)]
        else:
            df = meta_df

        splits = []
        for (season, rnd), grp in df.groupby(
            [self.season_col, self.round_col]
        ):
            test_mask = (
                (meta_df[self.season_col] == season)
                & (meta_df[self.round_col] == rnd)
            )
            train_mask = ~test_mask
            splits.append(SplitResult(
                train_idx=np.where(train_mask)[0],
                test_idx=np.where(test_mask)[0],
                fold_id=f"loro_S{season}_R{rnd}",
                description=(
                    f"Hold out Season {season} Round {rnd} "
                    f"({test_mask.sum()} obs)"
                ),
            ))
        return splits


# ===========================================================================
# 3. SCORING METRICS
# ===========================================================================
class Scorer:
    """
    Computes evaluation metrics from predicted probability vectors
    and actual outcomes.

    All metric methods are static so they can be used standalone.
    The `score_all()` method returns a dict of all metrics at once.
    """

    @staticmethod
    def log_loss(proba: np.ndarray, actual: int) -> float:
        """
        Negative log probability of the actual outcome.
        Lower is better. Also called "surprisal" or "log score".
        """
        return -np.log(proba[actual] + 1e-300)

    @staticmethod
    def accuracy_top_k(proba: np.ndarray, actual: int, k: int = 1) -> float:
        """1.0 if actual outcome is in the top-k predicted, else 0.0."""
        top_k = np.argsort(proba)[::-1][:k]
        return 1.0 if actual in top_k else 0.0

    @staticmethod
    def ranked_probability_score(proba: np.ndarray, actual: int) -> float:
        """
        Ranked Probability Score (RPS): measures calibration for ordinal
        outcomes. Penalizes predictions that place mass far from the
        actual outcome.

        RPS = (1/(K-1)) * sum_{k=1}^{K} (CDF_pred(k) - CDF_actual(k))^2

        Lower is better. Ranges [0, 1].
        """
        K = len(proba)
        cdf_pred = np.cumsum(proba)
        cdf_actual = np.cumsum(np.eye(K)[actual])
        return np.sum((cdf_pred - cdf_actual) ** 2) / (K - 1)

    @staticmethod
    def brier_score(proba: np.ndarray, actual: int) -> float:
        """
        Multi-class Brier score: mean squared error of predicted probs
        vs. one-hot actual. Lower is better.
        """
        one_hot = np.zeros(len(proba))
        one_hot[actual] = 1.0
        return np.mean((proba - one_hot) ** 2)

    @staticmethod
    def expected_position(proba: np.ndarray) -> float:
        """E[position] under the predicted distribution (DNF treated as 21)."""
        weights = np.array(
            [21] + list(range(1, N_OUTCOMES))  # DNF=21, P1=1, ..., P20=20
        )
        return np.dot(proba, weights)

    @staticmethod
    def position_error(proba: np.ndarray, actual: int) -> float:
        """
        |E[position] - actual position|. Simple interpretable error.
        DNF (0) is mapped to 21 for distance purposes.
        """
        actual_pos = 21 if actual == DNF else actual
        pred_pos = Scorer.expected_position(proba)
        return abs(pred_pos - actual_pos)

    @classmethod
    def score_all(cls, proba: np.ndarray, actual: int) -> dict:
        """Compute all metrics for a single prediction."""
        return {
            "log_loss": cls.log_loss(proba, actual),
            "accuracy_top1": cls.accuracy_top_k(proba, actual, k=1),
            "accuracy_top3": cls.accuracy_top_k(proba, actual, k=3),
            "accuracy_top5": cls.accuracy_top_k(proba, actual, k=5),
            "rps": cls.ranked_probability_score(proba, actual),
            "brier": cls.brier_score(proba, actual),
            "position_error": cls.position_error(proba, actual),
            "expected_position": cls.expected_position(proba),
        }


# ===========================================================================
# 4. EVALUATOR
# ===========================================================================
@dataclass
class FoldResult:
    """Results from one model on one split fold."""
    model_name: str
    fold_id: str
    fold_description: str
    n_test: int
    fit_time_s: float
    metrics: pd.DataFrame        # one row per test observation
    aggregate: dict              # mean metrics across test set

    def __repr__(self):
        agg = ", ".join(
            f"{k}={v:.4f}" for k, v in self.aggregate.items()
            if k != "expected_position"
        )
        return (
            f"FoldResult({self.model_name}, {self.fold_id}, "
            f"n={self.n_test}, {agg})"
        )


@dataclass
class EvaluationReport:
    """Full evaluation: all models × all folds."""
    fold_results: list[FoldResult] = field(default_factory=list)

    def summary(self) -> pd.DataFrame:
        """
        Aggregate table: one row per (model, fold) with mean metrics.
        """
        rows = []
        for fr in self.fold_results:
            row = {
                "model": fr.model_name,
                "fold": fr.fold_id,
                "n_test": fr.n_test,
                "fit_time_s": round(fr.fit_time_s, 2),
            }
            row.update(fr.aggregate)
            rows.append(row)
        return pd.DataFrame(rows)

    def model_summary(self) -> pd.DataFrame:
        """
        Aggregate across folds: one row per model with mean ± std of
        each metric.
        """
        df = self.summary()
        metrics = [
            "log_loss", "accuracy_top1", "accuracy_top3",
            "accuracy_top5", "rps", "brier", "position_error",
        ]
        rows = []
        for model, grp in df.groupby("model"):
            row = {"model": model, "n_folds": len(grp)}
            for m in metrics:
                if m in grp.columns:
                    row[f"{m}_mean"] = round(grp[m].mean(), 4)
                    row[f"{m}_std"] = round(grp[m].std(), 4)
            row["total_fit_time_s"] = round(grp["fit_time_s"].sum(), 1)
            rows.append(row)
        return pd.DataFrame(rows).sort_values("log_loss_mean")

    def per_observation_df(self) -> pd.DataFrame:
        """
        All per-observation predictions across all models and folds.
        Useful for deep dives: calibration plots, per-driver analysis, etc.
        """
        dfs = []
        for fr in self.fold_results:
            obs = fr.metrics.copy()
            obs["model"] = fr.model_name
            obs["fold"] = fr.fold_id
            dfs.append(obs)
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame()

    def head_to_head(
        self, model_a: str, model_b: str, metric: str = "log_loss"
    ) -> pd.DataFrame:
        """
        Direct comparison of two models on matching folds.
        Shows metric difference (A - B) per fold.
        """
        df = self.summary()
        a = df[df["model"] == model_a].set_index("fold")
        b = df[df["model"] == model_b].set_index("fold")
        common = a.index.intersection(b.index)
        return pd.DataFrame({
            "fold": common,
            f"{model_a}": a.loc[common, metric].values,
            f"{model_b}": b.loc[common, metric].values,
            "diff (A-B)": (
                a.loc[common, metric].values
                - b.loc[common, metric].values
            ),
        })


class Evaluator:
    """
    Orchestrates model evaluation across split strategies.

    Usage:
        evaluator = Evaluator(
            models=[Stage1Adapter(), Stage2Adapter(), Stage3Adapter()],
            splits=[TemporalHoldout([2023, 2024])],
        )
        report = evaluator.run(prev, next_, meta)
        print(report.model_summary())
    """

    def __init__(
        self,
        models: list[ModelAdapter],
        splits: list[SplitStrategy],
        scorer: Optional[Scorer] = None,
        verbose: bool = True,
    ):
        self.models = models
        self.splits = splits
        self.scorer = scorer or Scorer()
        self.verbose = verbose

    def run(
        self,
        prev_positions: np.ndarray,
        next_positions: np.ndarray,
        meta_df: pd.DataFrame,
    ) -> EvaluationReport:
        """
        Execute full evaluation pipeline.

        For each split × model:
          1. Partition data into train/test
          2. Fit model on training data
          3. Generate predictions for each test observation
          4. Score predictions
          5. Aggregate results
        """
        report = EvaluationReport()

        for strategy in self.splits:
            splits = strategy.generate_splits(meta_df)
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Split Strategy: {strategy.name}")
                print(f"  {len(splits)} fold(s)")
                print(f"{'='*60}")

            for split in splits:
                if self.verbose:
                    print(f"\n  --- {split.fold_id}: {split.description}")

                # Partition
                train_prev = prev_positions[split.train_idx]
                train_next = next_positions[split.train_idx]
                train_meta = meta_df.iloc[split.train_idx].reset_index(
                    drop=True
                )
                test_meta = meta_df.iloc[split.test_idx].reset_index(
                    drop=True
                )

                for model in self.models:
                    result = self._evaluate_model_on_fold(
                        model,
                        train_prev, train_next, train_meta,
                        test_meta, split,
                    )
                    report.fold_results.append(result)

                    if self.verbose:
                        agg = result.aggregate
                        print(
                            f"    {model.name:40s}  "
                            f"LL={agg['log_loss']:.3f}  "
                            f"Top1={agg['accuracy_top1']:.3f}  "
                            f"Top3={agg['accuracy_top3']:.3f}  "
                            f"RPS={agg['rps']:.4f}  "
                            f"PosErr={agg['position_error']:.2f}  "
                            f"({result.fit_time_s:.1f}s)"
                        )

        return report

    def _evaluate_model_on_fold(
        self,
        model: ModelAdapter,
        train_prev, train_next, train_meta,
        test_meta, split: SplitResult,
    ) -> FoldResult:
        """Fit model, predict on test, score."""

        # Fit
        t0 = time.time()
        model.fit(train_prev, train_next, train_meta)
        fit_time = time.time() - t0

        # Predict + score each test observation
        obs_rows = []
        for _, row in test_meta.iterrows():
            prev = int(row["prev_position"])
            actual = int(row["next_position"])
            driver = row.get("driver")
            constructor = row.get("constructor")

            proba = model.predict_proba(
                prev, driver=driver, constructor=constructor
            )

            scores = self.scorer.score_all(proba, actual)
            scores["prev_position"] = prev
            scores["actual"] = actual
            scores["predicted_mode"] = int(np.argmax(proba))
            scores["driver"] = driver
            scores["constructor"] = constructor
            scores["season"] = row.get("season")
            scores["race_order"] = row.get("race_order")
            obs_rows.append(scores)

        metrics_df = pd.DataFrame(obs_rows)

        # Aggregate
        metric_cols = [
            "log_loss", "accuracy_top1", "accuracy_top3",
            "accuracy_top5", "rps", "brier", "position_error",
        ]
        aggregate = {
            col: metrics_df[col].mean()
            for col in metric_cols
            if col in metrics_df.columns
        }

        return FoldResult(
            model_name=model.name,
            fold_id=split.fold_id,
            fold_description=split.description,
            n_test=len(test_meta),
            fit_time_s=fit_time,
            metrics=metrics_df,
            aggregate=aggregate,
        )


# ===========================================================================
# 5. ANALYSIS HELPERS
# ===========================================================================
def calibration_table(
    report: EvaluationReport,
    model_name: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Calibration analysis: bin predicted probabilities of the actual outcome
    and compare to observed frequency.

    Perfect calibration: predicted_prob ≈ observed_freq in each bin.
    """
    obs = report.per_observation_df()
    obs = obs[obs["model"] == model_name].copy()
    if obs.empty:
        return pd.DataFrame()

    # Get P(actual) for each observation
    # We stored log_loss = -log(p), so p = exp(-log_loss)
    obs["p_actual"] = np.exp(-obs["log_loss"])
    obs["bin"] = pd.cut(obs["p_actual"], bins=n_bins)

    cal = (
        obs.groupby("bin", observed=False)
        .agg(
            n=("p_actual", "size"),
            mean_predicted=("p_actual", "mean"),
            # "accuracy" = fraction where the top-1 prediction was correct
            observed_top1=("accuracy_top1", "mean"),
        )
        .reset_index()
    )
    return cal


def per_driver_breakdown(
    report: EvaluationReport,
    driver_name_map: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Break down metrics by driver across all models.
    Useful for finding which drivers are hardest to predict.
    """
    obs = report.per_observation_df()
    if obs.empty:
        return pd.DataFrame()

    metrics = ["log_loss", "accuracy_top1", "rps", "position_error"]
    agg = {m: "mean" for m in metrics}
    agg["actual"] = "count"

    result = (
        obs.groupby(["model", "driver"])
        .agg(agg)
        .rename(columns={"actual": "n_obs"})
        .reset_index()
    )
    if driver_name_map:
        result["driver_name"] = result["driver"].map(driver_name_map)
    return result.sort_values(["model", "log_loss"])


def per_prev_position_breakdown(
    report: EvaluationReport,
) -> pd.DataFrame:
    """
    Break down metrics by previous position. Shows which conditioning
    states are hardest to predict from.
    """
    obs = report.per_observation_df()
    if obs.empty:
        return pd.DataFrame()

    metrics = ["log_loss", "accuracy_top1", "rps", "position_error"]
    agg = {m: "mean" for m in metrics}
    agg["actual"] = "count"

    result = (
        obs.groupby(["model", "prev_position"])
        .agg(agg)
        .rename(columns={"actual": "n_obs"})
        .reset_index()
    )
    result["prev_label"] = result["prev_position"].map(
        lambda x: PREV_STATE_LABELS[x] if x < len(PREV_STATE_LABELS) else "?"
    )
    return result.sort_values(["model", "prev_position"])


# ===========================================================================
# 6. DEMO / MAIN
# ===========================================================================
if __name__ == "__main__":
    import sys

    print("=" * 65)
    print("F1 Model Evaluation Framework")
    print("=" * 65)

    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/f1")

    # --- Load data ---
    if (data_dir / "results.csv").exists():
        from stage3_constructor import (
            F1DataLoader, prepare_transitions
        )
        print(f"\nLoading Kaggle data from {data_dir} ...")
        loader = F1DataLoader(data_dir)
        df = loader.load_merged(min_year=2014)
        prev, next_, meta = prepare_transitions(df)

        driver_names = (
            df[["driverId", "driver_name"]]
            .drop_duplicates()
            .set_index("driverId")["driver_name"]
            .to_dict()
        )
        print(
            f"  {len(prev)} transitions, "
            f"{df['year'].nunique()} seasons "
            f"({df['year'].min()}–{df['year'].max()}), "
            f"{df['driverId'].nunique()} drivers"
        )
    else:
        print(
            f"\nKaggle data not found at {data_dir}. "
            "Using synthetic data for demo.\n"
        )
        rng = np.random.default_rng(42)
        n_drivers, n_constructors = 20, 5
        rows = []
        for d in range(1, n_drivers + 1):
            skill = d
            cid = (d - 1) % n_constructors + 1
            for season in range(2018, 2025):
                for rnd in range(1, 23):
                    boost = (3 - cid) * 1.5
                    pos = int(np.clip(
                        rng.normal(skill - boost, 3), 1, 20
                    ))
                    if rng.random() < 0.05:
                        pos = DNF
                    rows.append({
                        "driver": d, "constructor": cid,
                        "season": season, "race_order": rnd,
                        "next_position": pos,
                    })
        meta = pd.DataFrame(rows)
        # Set prev_position
        meta["prev_position"] = START
        for d, grp in meta.groupby("driver"):
            idxs = grp.index.tolist()
            for i, idx in enumerate(idxs):
                if (
                    i > 0
                    and meta.loc[idx, "season"]
                    == meta.loc[idxs[i - 1], "season"]
                ):
                    meta.loc[idx, "prev_position"] = (
                        meta.loc[idxs[i - 1], "next_position"]
                    )
        prev = meta["prev_position"].values.astype(int)
        next_ = meta["next_position"].values.astype(int)
        driver_names = {d: f"Driver_{d:02d}" for d in range(1, n_drivers + 1)}

    # --- Define models ---
    models = [
        Stage1Adapter(prior_alpha=1.0),
        Stage2Adapter(prior_alpha_global=1.0),
        Stage3Adapter(
            prior_alpha_global=1.0,
            prior_alpha_constructor=1.0,
        ),
    ]

    # --- Define split strategies ---
    max_year = int(meta["season"].max())
    splits = [
        # Primary: train on all but last year, test on last year
        TemporalHoldout(cutoff_years=[max_year]),
        # Rolling origin: expanding window
        RollingOrigin(min_train_seasons=3),
    ]

    # --- Run evaluation ---
    evaluator = Evaluator(
        models=models,
        splits=splits,
        verbose=True,
    )
    report = evaluator.run(prev, next_, meta)

    # --- Print summaries ---
    print("\n" + "=" * 65)
    print("MODEL COMPARISON (averaged across folds)")
    print("=" * 65)
    ms = report.model_summary()
    print(ms.to_string(index=False))

    # --- Head-to-head ---
    print("\n--- Head-to-Head: Stage 1 vs Stage 3 (log_loss) ---")
    s1_name = models[0].name
    s3_name = models[2].name
    h2h = report.head_to_head(s1_name, s3_name, "log_loss")
    if not h2h.empty:
        print(h2h.to_string(index=False))
        wins = (h2h["diff (A-B)"] > 0).sum()
        losses = (h2h["diff (A-B)"] < 0).sum()
        print(
            f"\n  Stage 3 wins {losses}/{len(h2h)} folds "
            f"(lower log_loss = better)"
        )

    # --- Per-previous-position breakdown ---
    print("\n--- Accuracy by Previous Position (last fold) ---")
    ppb = per_prev_position_breakdown(report)
    if not ppb.empty:
        # Show for Stage 3 only
        s3ppb = ppb[ppb["model"] == s3_name][
            ["prev_label", "n_obs", "log_loss",
             "accuracy_top1", "accuracy_top3", "position_error"]
        ]
        if not s3ppb.empty:
            print(s3ppb.round(3).to_string(index=False))

    # --- Per-driver breakdown (top 5 hardest to predict) ---
    print("\n--- Hardest Drivers to Predict (Stage 3, by log_loss) ---")
    pdb = per_driver_breakdown(report, driver_name_map=driver_names)
    if not pdb.empty:
        s3pdb = pdb[pdb["model"] == s3_name].tail(5)
        show = ["driver_name" if "driver_name" in s3pdb else "driver",
                "n_obs", "log_loss", "accuracy_top1", "position_error"]
        show = [c for c in show if c in s3pdb.columns]
        print(s3pdb[show].round(3).to_string(index=False))
