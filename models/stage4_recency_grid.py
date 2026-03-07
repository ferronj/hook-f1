"""
F1 Dirichlet-Multinomial Markov Model
======================================
Stage 4: Recency-weighted constructor + driver pooling

Extends Stage 3 with exponential recency decay applied to the *prior-level*
transition matrices (global, constructor) so they reflect recent performance
more than distant history. Driver-level counts are kept unweighted to
preserve evidence strength.

    w(y) = exp(-lambda * (ref_year - y))

Model:
    alpha_{i,C,s} = kappa_g * pi_s^{global,recency}
                  + kappa_c * pi_s^{(C),recency}
                  + n_{i,s}  (unweighted driver counts)

Parameters (kappa_g, kappa_c, lambda) are jointly optimized via Empirical
Bayes (maximize marginal likelihood).

Data source: Kaggle F1 World Championship dataset
  https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.special import gammaln
from pathlib import Path
from typing import Optional
from collections import defaultdict


# ---------------------------------------------------------------------------
# Constants (shared across all stages)
# ---------------------------------------------------------------------------
DNF = 0
POSITIONS = list(range(1, 21))
START = 21

N_PREV_STATES = 22
N_OUTCOMES = 21
OUTCOME_LABELS = ["DNF"] + [f"P{i}" for i in range(1, 21)]
PREV_STATE_LABELS = ["DNF"] + [f"P{i}" for i in range(1, 21)] + ["START"]

CLASSIFIED_STATUS_KEYWORDS = {"finished", "lap", "laps"}


# ---------------------------------------------------------------------------
# Data Loading (shared)
# ---------------------------------------------------------------------------
class F1DataLoader:
    """
    Loads and joins the Kaggle F1 dataset CSVs into a unified race results
    DataFrame suitable for transition modeling.
    """

    REQUIRED_FILES = [
        "results.csv", "races.csv", "drivers.csv",
        "constructors.csv", "status.csv",
    ]

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self._validate_files()
        self._raw: dict[str, pd.DataFrame] = {}

    def _validate_files(self):
        missing = [
            f for f in self.REQUIRED_FILES
            if not (self.data_dir / f).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing files in {self.data_dir}: {missing}"
            )

    def load_raw(self) -> dict[str, pd.DataFrame]:
        if not self._raw:
            for f in self.REQUIRED_FILES:
                name = f.replace(".csv", "")
                self._raw[name] = pd.read_csv(
                    self.data_dir / f, na_values=["\\N", ""]
                )
            circuits_path = self.data_dir / "circuits.csv"
            if circuits_path.exists():
                self._raw["circuits"] = pd.read_csv(
                    circuits_path, na_values=["\\N", ""]
                )
        return self._raw

    def load_merged(
        self,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
    ) -> pd.DataFrame:
        raw = self.load_raw()

        results = raw["results"][
            ["resultId", "raceId", "driverId", "constructorId",
             "grid", "positionOrder", "statusId"]
        ].copy()
        results["positionOrder"] = pd.to_numeric(
            results["positionOrder"], errors="coerce"
        )
        results["grid"] = pd.to_numeric(
            results["grid"], errors="coerce"
        ).fillna(0).astype(int)

        races = raw["races"][
            ["raceId", "year", "round", "circuitId", "name", "date"]
        ].rename(columns={"name": "race_name", "date": "race_date"})

        drivers = raw["drivers"][["driverId", "forename", "surname"]].copy()
        drivers["driver_name"] = (
            drivers["forename"] + " " + drivers["surname"]
        )

        constructors = raw["constructors"][
            ["constructorId", "name"]
        ].rename(columns={"name": "constructor_name"})

        status = raw["status"].rename(columns={"status": "status_text"})

        df = (
            results
            .merge(races, on="raceId", how="left")
            .merge(drivers[["driverId", "driver_name"]], on="driverId", how="left")
            .merge(constructors, on="constructorId", how="left")
            .merge(status, on="statusId", how="left")
        )

        if min_year is not None:
            df = df[df["year"] >= min_year]
        if max_year is not None:
            df = df[df["year"] <= max_year]

        df["position_mapped"] = df.apply(self._map_position, axis=1)

        df = df.sort_values(
            ["driverId", "year", "round"]
        ).reset_index(drop=True)
        return df

    @staticmethod
    def _map_position(row) -> int:
        status = str(row.get("status_text", "")).lower().strip()
        is_classified = any(
            kw in status for kw in CLASSIFIED_STATUS_KEYWORDS
        )
        if is_classified:
            pos = row.get("positionOrder", np.nan)
            if pd.isna(pos):
                return DNF
            return int(min(pos, 20))
        return DNF


# ---------------------------------------------------------------------------
# Transition Preparation
# ---------------------------------------------------------------------------
def prepare_transitions(
    df: pd.DataFrame,
    driver_col: str = "driverId",
    season_col: str = "year",
    race_order_col: str = "round",
    position_col: str = "position_mapped",
    constructor_col: str = "constructorId",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Convert race results into (prev_position, next_position) arrays.
    """
    df = df.sort_values(
        [driver_col, season_col, race_order_col]
    ).copy()
    prev_list, next_list, meta_rows = [], [], []

    for driver, grp in df.groupby(driver_col):
        positions = grp[position_col].values
        seasons = grp[season_col].values
        race_orders = grp[race_order_col].values
        constructors = grp[constructor_col].values

        for i in range(len(positions)):
            if i == 0 or seasons[i] != seasons[i - 1]:
                prev = START
            else:
                prev = int(positions[i - 1])

            prev_list.append(prev)
            next_list.append(int(positions[i]))
            meta_rows.append({
                "driver": driver,
                "constructor": constructors[i],
                "season": seasons[i],
                "race_order": race_orders[i],
                "prev_position": prev,
                "next_position": int(positions[i]),
            })

    return (
        np.array(prev_list, dtype=int),
        np.array(next_list, dtype=int),
        pd.DataFrame(meta_rows),
    )


# ---------------------------------------------------------------------------
# Count Matrix Construction
# ---------------------------------------------------------------------------
def compute_recency_weights(
    seasons: np.ndarray,
    decay_lambda: float,
    ref_year: Optional[int] = None,
) -> np.ndarray:
    """
    Compute exponential recency weights: w(y) = exp(-lambda * (ref - y)).
    """
    if ref_year is None:
        ref_year = int(seasons.max()) + 1
    ages = ref_year - seasons.astype(float)
    return np.exp(-decay_lambda * ages)


def build_weighted_prior_counts(
    meta_df: pd.DataFrame,
    decay_lambda: float,
    ref_year: Optional[int] = None,
) -> tuple[np.ndarray, dict]:
    """
    Build recency-weighted count matrices for PRIOR components only
    (global, constructor).

    Returns
    -------
    global_counts : (N_PREV_STATES, N_OUTCOMES) weighted
    constructor_counts : dict[constructor_id -> array]
    """
    weights = compute_recency_weights(
        meta_df["season"].values, decay_lambda, ref_year
    )

    global_counts = np.zeros((N_PREV_STATES, N_OUTCOMES), dtype=float)
    constructor_counts: dict = defaultdict(
        lambda: np.zeros((N_PREV_STATES, N_OUTCOMES), dtype=float)
    )

    for idx, (_, row) in enumerate(meta_df.iterrows()):
        s = int(row["prev_position"])
        j = int(row["next_position"])
        w = weights[idx]
        global_counts[s, j] += w
        constructor_counts[row["constructor"]][s, j] += w

    return (
        global_counts,
        dict(constructor_counts),
    )


def build_unweighted_driver_counts(
    meta_df: pd.DataFrame,
) -> tuple[dict, dict]:
    """
    Build UNWEIGHTED per-driver count matrices (preserves evidence strength).

    Returns
    -------
    driver_counts : dict[driver_id -> array]
    driver_constructor_counts : dict[driver -> dict[constructor -> array]]
    """
    driver_counts: dict = defaultdict(
        lambda: np.zeros((N_PREV_STATES, N_OUTCOMES), dtype=int)
    )
    driver_constructor_counts: dict = defaultdict(
        lambda: defaultdict(
            lambda: np.zeros((N_PREV_STATES, N_OUTCOMES), dtype=int)
        )
    )

    for _, row in meta_df.iterrows():
        s = int(row["prev_position"])
        j = int(row["next_position"])
        driver_counts[row["driver"]][s, j] += 1
        driver_constructor_counts[row["driver"]][row["constructor"]][s, j] += 1

    return (
        dict(driver_counts),
        {d: dict(c_map) for d, c_map in driver_constructor_counts.items()},
    )


def build_driver_constructor_map(meta_df: pd.DataFrame) -> dict:
    """Most recent constructor assignment per driver."""
    latest = (
        meta_df
        .sort_values(["season", "race_order"])
        .groupby("driver")["constructor"]
        .last()
    )
    return latest.to_dict()


# ---------------------------------------------------------------------------
# Marginal Likelihood
# ---------------------------------------------------------------------------
def _dm_log_ml(alpha: np.ndarray, counts: np.ndarray) -> float:
    """Log marginal likelihood for one Dirichlet-Multinomial row."""
    N = counts.sum()
    if N < 1e-10:
        return 0.0
    A = alpha.sum()
    return (
        gammaln(A) - gammaln(A + N)
        + np.sum(gammaln(alpha + counts) - gammaln(alpha))
    )


def total_log_marginal_likelihood(
    kappa_g: float,
    kappa_c: float,
    global_pi: np.ndarray,
    constructor_pis: dict,
    driver_constructor_counts: dict,
) -> float:
    """
    Total log marginal likelihood across all drivers/states.

    Prior components (global_pi, constructor_pis) are recency-weighted.
    Driver counts are unweighted (integer).
    """
    lml = 0.0
    for driver, constr_map in driver_constructor_counts.items():
        for constr, counts in constr_map.items():
            c_pi = constructor_pis.get(constr, global_pi)
            for s in range(N_PREV_STATES):
                n_s = counts[s]
                if n_s.sum() < 1e-10:
                    continue
                alpha_s = (
                    kappa_g * global_pi[s]
                    + kappa_c * c_pi[s]
                )
                alpha_s = np.maximum(alpha_s, 1e-10)
                lml += _dm_log_ml(alpha_s, n_s)
    return lml


# ---------------------------------------------------------------------------
# Stage 4 Model
# ---------------------------------------------------------------------------
class RecencyGridDirichletF1:
    """
    Stage 4: Recency-weighted constructor + driver pooling.

    Model:
        p_{i,C,s} ~ Dirichlet(kappa_g * pi_s^{global,recency}
                              + kappa_c * pi_s^{(C),recency})

    Recency weighting is applied to the prior-level transition matrices
    (global, constructor) but NOT to driver-level counts. This ensures
    the priors reflect recent team performance while preserving the
    full evidence from each driver's race history.

    Parameters (kappa_g, kappa_c, lambda) are jointly optimized
    via Empirical Bayes (maximize marginal likelihood).
    """

    def __init__(
        self,
        prior_alpha_global: float = 1.0,
        prior_alpha_constructor: float = 1.0,
        kappa_init: tuple[float, float] = (5.0, 10.0),
        kappa_bounds: tuple = ((0.01, 500.0), (0.01, 500.0)),
        lambda_init: float = 0.15,
        lambda_bounds: tuple[float, float] = (0.01, 0.7),
    ):
        self.prior_alpha_global = prior_alpha_global
        self.prior_alpha_constructor = prior_alpha_constructor
        self.kappa_init = kappa_init
        self.kappa_bounds = kappa_bounds
        self.lambda_init = lambda_init
        self.lambda_bounds = lambda_bounds

        # Fitted attributes
        self.global_pi_: Optional[np.ndarray] = None
        self.global_counts_: Optional[np.ndarray] = None
        self.constructor_pis_: Optional[dict] = None
        self.constructor_counts_: Optional[dict] = None
        self.driver_counts_: Optional[dict] = None
        self.driver_constructor_counts_: Optional[dict] = None
        self.driver_latest_constructor_: Optional[dict] = None
        self.kappa_g_: Optional[float] = None
        self.kappa_c_: Optional[float] = None
        self.lambda_: Optional[float] = None
        self.driver_ids_: Optional[list] = None
        self.constructor_ids_: Optional[list] = None
        self.meta_df_: Optional[pd.DataFrame] = None
        self.opt_result_: Optional[optimize.OptimizeResult] = None

    @property
    def is_fitted(self) -> bool:
        return self.kappa_g_ is not None

    def fit(
        self,
        prev_positions: np.ndarray,
        next_positions: np.ndarray,
        meta_df: pd.DataFrame,
    ) -> "RecencyGridDirichletF1":
        """
        Fit the model:
          1. Build unweighted driver counts (fixed throughout optimization).
          2. Jointly optimize (kappa_g, kappa_c, lambda) by rebuilding
             recency-weighted prior counts at each lambda and computing
             marginal likelihood against unweighted driver counts.
        """
        self.meta_df_ = meta_df

        # Driver counts are fixed (unweighted) throughout optimization
        driver_counts, driver_constructor_counts = (
            build_unweighted_driver_counts(meta_df)
        )

        def neg_lml(params):
            log_kg, log_kc, lam = params
            kg = np.exp(log_kg)
            kc = np.exp(log_kc)

            # Rebuild recency-weighted prior counts at this lambda
            g_counts, c_counts = (
                build_weighted_prior_counts(meta_df, lam)
            )

            # Compute prior transition matrices from weighted counts
            g_alpha = self.prior_alpha_global + g_counts
            g_pi = g_alpha / g_alpha.sum(axis=1, keepdims=True)

            c_pis = {}
            for cid, cc in c_counts.items():
                c_alpha = self.prior_alpha_constructor + cc
                c_pis[cid] = c_alpha / c_alpha.sum(axis=1, keepdims=True)

            # Marginal likelihood uses unweighted driver counts
            return -total_log_marginal_likelihood(
                kg, kc,
                g_pi, c_pis,
                driver_constructor_counts,
            )

        x0 = [
            np.log(self.kappa_init[0]),
            np.log(self.kappa_init[1]),
            self.lambda_init,
        ]
        bounds = [
            (np.log(self.kappa_bounds[0][0]), np.log(self.kappa_bounds[0][1])),
            (np.log(self.kappa_bounds[1][0]), np.log(self.kappa_bounds[1][1])),
            self.lambda_bounds,
        ]

        result = optimize.minimize(
            neg_lml,
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 300, "ftol": 1e-8},
        )

        self.kappa_g_ = np.exp(result.x[0])
        self.kappa_c_ = np.exp(result.x[1])
        self.lambda_ = result.x[2]
        self.opt_result_ = result

        # Store driver counts (unweighted)
        self.driver_counts_ = driver_counts
        self.driver_constructor_counts_ = driver_constructor_counts

        # Rebuild prior matrices at optimal lambda
        self.global_counts_, self.constructor_counts_ = (
            build_weighted_prior_counts(meta_df, self.lambda_)
        )

        # Global pi
        g_alpha = self.prior_alpha_global + self.global_counts_
        self.global_pi_ = g_alpha / g_alpha.sum(axis=1, keepdims=True)

        # Constructor pis
        self.constructor_pis_ = {}
        for cid, cc in self.constructor_counts_.items():
            c_alpha = self.prior_alpha_constructor + cc
            self.constructor_pis_[cid] = (
                c_alpha / c_alpha.sum(axis=1, keepdims=True)
            )
        self.constructor_ids_ = sorted(self.constructor_pis_.keys())

        self.driver_ids_ = sorted(self.driver_constructor_counts_.keys())
        self.driver_latest_constructor_ = (
            build_driver_constructor_map(meta_df)
        )

        return self

    # --- Prior / Posterior ---

    def _driver_prior_alpha(
        self,
        constructor_id,
    ) -> np.ndarray:
        """
        Prior alpha for a driver at a given constructor:
            kappa_g * pi^{global} + kappa_c * pi^{(C)}
        """
        c_pi = self.constructor_pis_.get(constructor_id, self.global_pi_)
        return (
            self.kappa_g_ * self.global_pi_
            + self.kappa_c_ * c_pi
        )

    def driver_posterior_alpha(
        self,
        driver_id,
        constructor_id: Optional = None,
    ) -> np.ndarray:
        """
        Posterior Dirichlet alpha for a driver.

        If constructor_id is None, uses driver's most recent constructor.
        """
        self._check_fitted()
        if constructor_id is None:
            constructor_id = self.driver_latest_constructor_.get(driver_id)

        prior = self._driver_prior_alpha(constructor_id)

        dc = self.driver_constructor_counts_.get(driver_id, {})
        counts = dc.get(
            constructor_id,
            np.zeros((N_PREV_STATES, N_OUTCOMES))
        )
        return prior + counts

    def driver_transition_matrix(
        self,
        driver_id,
        constructor_id: Optional = None,
    ) -> np.ndarray:
        """Posterior mean transition matrix."""
        alpha = self.driver_posterior_alpha(driver_id, constructor_id)
        return alpha / alpha.sum(axis=1, keepdims=True)

    # --- Prediction ---

    def predict_proba(
        self,
        driver_id,
        prev_position: int,
        constructor_id: Optional = None,
        **kwargs,  # accept and ignore grid_bin for API compat
    ) -> np.ndarray:
        """P(next | prev, driver, constructor)."""
        return self.driver_transition_matrix(
            driver_id, constructor_id
        )[prev_position]

    def predict_proba_new_driver(
        self,
        prev_position: int,
        constructor_id: Optional = None,
        **kwargs,  # accept and ignore grid_bin for API compat
    ) -> np.ndarray:
        """Prediction for an unseen driver using priors only."""
        self._check_fitted()
        if constructor_id is not None:
            alpha = self._driver_prior_alpha(constructor_id)
        else:
            alpha = self.kappa_g_ * self.global_pi_
        row = alpha[prev_position]
        return row / row.sum()

    def predict_distribution(
        self,
        driver_id,
        prev_position: int,
        constructor_id: Optional = None,
    ) -> stats.dirichlet:
        """Full posterior Dirichlet."""
        alpha = self.driver_posterior_alpha(
            driver_id, constructor_id
        )[prev_position]
        return stats.dirichlet(alpha)

    def credible_interval(
        self,
        driver_id,
        prev_position: int,
        outcome: int,
        constructor_id: Optional = None,
        alpha: float = 0.05,
        n_samples: int = 10000,
    ) -> tuple[float, float]:
        """(1-alpha) credible interval for P(outcome | prev, driver, C)."""
        post_alpha = self.driver_posterior_alpha(
            driver_id, constructor_id
        )[prev_position]
        p_samples = stats.dirichlet.rvs(post_alpha, size=n_samples)
        probs = p_samples[:, outcome]
        return (
            np.percentile(probs, 100 * alpha / 2),
            np.percentile(probs, 100 * (1 - alpha / 2)),
        )

    # --- Diagnostics ---

    def pooling_factors(
        self, driver_id, constructor_id=None
    ) -> dict:
        """Decompose effective weight of each prior component."""
        self._check_fitted()
        if constructor_id is None:
            constructor_id = self.driver_latest_constructor_.get(driver_id)

        w_g = self.kappa_g_ * N_OUTCOMES
        w_c = self.kappa_c_ * N_OUTCOMES

        dc = self.driver_constructor_counts_.get(driver_id, {})
        n = dc.get(
            constructor_id,
            np.zeros((N_PREV_STATES, N_OUTCOMES))
        ).sum()
        n_per_row = n / max(1, N_PREV_STATES)

        total = w_g + w_c + n_per_row
        return {
            "global_weight": round(w_g / total, 3),
            "constructor_weight": round(w_c / total, 3),
            "data_weight": round(n_per_row / total, 3),
        }

    def log_marginal_likelihood(self) -> float:
        """Total log marginal likelihood at fitted parameters."""
        self._check_fitted()
        return total_log_marginal_likelihood(
            self.kappa_g_, self.kappa_c_,
            self.global_pi_, self.constructor_pis_,
            self.driver_constructor_counts_,
        )

    def summary(
        self,
        driver_id=None,
        constructor_id=None,
    ) -> pd.DataFrame:
        """Transition matrix as a labeled DataFrame."""
        self._check_fitted()
        if driver_id is not None:
            tm = self.driver_transition_matrix(driver_id, constructor_id)
            label = f"Driver {driver_id}"
        elif constructor_id is not None:
            tm = self.constructor_pis_.get(constructor_id, self.global_pi_)
            label = f"Constructor {constructor_id}"
        else:
            tm = self.global_pi_
            label = "Global"
        df = pd.DataFrame(
            tm, index=PREV_STATE_LABELS, columns=OUTCOME_LABELS
        )
        df.index.name = f"Previous Position ({label})"
        return df

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    print("=" * 65)
    print("F1 Dirichlet-Multinomial — Stage 4: Recency-Weighted Constructor")
    print("=" * 65)

    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/f1")

    if (data_dir / "results.csv").exists():
        print(f"\nLoading data from {data_dir} ...")
        loader = F1DataLoader(data_dir)
        df = loader.load_merged(min_year=2014)
        print(
            f"  Rows: {len(df)}, Seasons: {df['year'].nunique()}, "
            f"Drivers: {df['driverId'].nunique()}"
        )
        prev_pos, next_pos, meta = prepare_transitions(df)
    else:
        print(f"\nData not found at {data_dir}.")
        sys.exit(1)

    print(f"Total transitions: {len(prev_pos)}")

    model = RecencyGridDirichletF1()
    model.fit(prev_pos, next_pos, meta)

    print(f"\n--- Fitted Parameters ---")
    print(f"  kappa_g:      {model.kappa_g_:.2f}")
    print(f"  kappa_c:      {model.kappa_c_:.2f}")
    print(f"  lambda:       {model.lambda_:.4f}")
    print(f"  Optimizer converged: {model.opt_result_.success}")

    total_k = model.kappa_g_ + model.kappa_c_
    print(f"  Prior blend: {model.kappa_g_/total_k:.0%} global / "
          f"{model.kappa_c_/total_k:.0%} constructor")
    print(f"  Recency half-life: {np.log(2)/max(model.lambda_, 1e-10):.1f} years")
