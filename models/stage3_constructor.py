"""
F1 Dirichlet-Multinomial Markov Model
======================================
Stage 3: Constructor effect via additive pseudo-count priors

Extends Stage 2 by adding a constructor-specific transition matrix as a
second source of prior information. Drivers change constructors across
(and sometimes within) seasons, so the constructor prior is tied to each
*transition*, not to the driver globally.

Model (for driver i at constructor C, previous state s):

    F_{i,r} | F_{i,r-1}=s  ~  Categorical(p_{i,C,s})

    p_{i,C,s}  ~  Dirichlet(alpha_{i,C,s})

    alpha_{i,C,s} = kappa_g * pi_s^{global}
                  + kappa_c * pi_s^{(C)}
                  + n_{i,s}

Where:
    pi_s^{global}   = global posterior mean transition probs (from all data)
    pi_s^{(C)}      = constructor C's posterior mean transition probs
    kappa_g, kappa_c = concentration parameters (learned via Empirical Bayes)
    n_{i,s}          = driver i's observed count vector for state s

The constructor matrix pi^{(C)} is estimated from ALL results where any
driver raced for constructor C, using a mildly informative Dirichlet prior
(the global pi itself, weighted by a small alpha_c_prior).

Joint optimization of (kappa_g, kappa_c) maximizes the total log marginal
likelihood across all drivers, preserving conjugacy for per-driver posteriors.

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

        races = raw["races"][
            ["raceId", "year", "round", "circuitId", "name", "date"]
        ].rename(columns={"name": "race_name", "date": "race_date"})

        drivers = raw["drivers"][["driverId", "forename", "surname"]].copy()
        drivers["driver_name"] = drivers["forename"] + " " + drivers["surname"]

        constructors = raw["constructors"][
            ["constructorId", "name"]
        ].rename(columns={"name": "constructor_name"})

        status = raw["status"].rename(columns={"status": "status_text"})

        df = (
            results
            .merge(races, on="raceId", how="left")
            .merge(
                drivers[["driverId", "driver_name"]], on="driverId", how="left"
            )
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

    Stage 3 extension: also captures the constructor for each transition
    (the constructor the driver raced for in the *current* race).

    Returns
    -------
    prev_positions, next_positions : np.ndarray
    meta_df : DataFrame with [driver, constructor, season, race_order,
              prev_position, next_position]
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


def build_count_matrices(
    meta_df: pd.DataFrame,
) -> tuple[np.ndarray, dict, dict]:
    """
    Build count matrices from the transition meta DataFrame.

    Returns
    -------
    global_counts : (N_PREV_STATES, N_OUTCOMES) total count matrix
    driver_counts : dict[driver_id -> (N_PREV_STATES, N_OUTCOMES)]
    constructor_counts : dict[constructor_id -> (N_PREV_STATES, N_OUTCOMES)]
    """
    global_counts = np.zeros((N_PREV_STATES, N_OUTCOMES), dtype=int)
    driver_counts: dict = defaultdict(
        lambda: np.zeros((N_PREV_STATES, N_OUTCOMES), dtype=int)
    )
    constructor_counts: dict = defaultdict(
        lambda: np.zeros((N_PREV_STATES, N_OUTCOMES), dtype=int)
    )

    for _, row in meta_df.iterrows():
        s = int(row["prev_position"])
        j = int(row["next_position"])
        global_counts[s, j] += 1
        driver_counts[row["driver"]][s, j] += 1
        constructor_counts[row["constructor"]][s, j] += 1

    # Convert defaultdicts to regular dicts (freeze)
    return (
        global_counts,
        dict(driver_counts),
        dict(constructor_counts),
    )


def build_driver_constructor_map(
    meta_df: pd.DataFrame,
) -> dict:
    """
    For each driver, identify their *most recent* constructor assignment.
    Used for prediction when we need a default constructor.

    Returns dict[driver_id -> constructor_id]
    """
    latest = (
        meta_df
        .sort_values(["season", "race_order"])
        .groupby("driver")["constructor"]
        .last()
    )
    return latest.to_dict()


# ---------------------------------------------------------------------------
# Marginal Likelihood Utilities
# ---------------------------------------------------------------------------
def _dm_log_ml(alpha: np.ndarray, counts: np.ndarray) -> float:
    """
    Log marginal likelihood for one Dirichlet-Multinomial row.

    log P(n | alpha) = logGamma(A) - logGamma(A+N)
                     + sum_j [logGamma(a_j + n_j) - logGamma(a_j)]
    """
    N = counts.sum()
    if N == 0:
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
    driver_counts: dict,
    driver_constructor_history: dict,
) -> float:
    """
    Total log marginal likelihood across all drivers/states.

    For each driver i and previous state s:
        alpha_{i,s} = kappa_g * pi_s^{global} + kappa_c * pi_s^{(C_i)}

    where C_i is determined by driver_constructor_history.

    In practice, a driver may race for multiple constructors across
    the dataset. We handle this by splitting their counts by constructor
    and applying the appropriate constructor prior to each segment.

    Parameters
    ----------
    driver_constructor_history : dict[driver -> dict[constructor -> counts]]
        Per-driver, per-constructor count matrices.
    """
    lml = 0.0
    for driver, constr_map in driver_constructor_history.items():
        for constr, counts in constr_map.items():
            c_pi = constructor_pis.get(constr, global_pi)
            for s in range(N_PREV_STATES):
                n_s = counts[s]
                if n_s.sum() == 0:
                    continue
                alpha_s = (
                    kappa_g * global_pi[s] + kappa_c * c_pi[s]
                )
                alpha_s = np.maximum(alpha_s, 1e-10)
                lml += _dm_log_ml(alpha_s, n_s)
    return lml


def build_driver_constructor_counts(
    meta_df: pd.DataFrame,
) -> dict:
    """
    Build per-driver, per-constructor count matrices.

    Returns dict[driver -> dict[constructor -> (N_PREV_STATES, N_OUTCOMES)]]

    This correctly handles drivers who switch constructors by keeping
    separate counts for each constructor stint.
    """
    result: dict = defaultdict(
        lambda: defaultdict(
            lambda: np.zeros((N_PREV_STATES, N_OUTCOMES), dtype=int)
        )
    )
    for _, row in meta_df.iterrows():
        s = int(row["prev_position"])
        j = int(row["next_position"])
        result[row["driver"]][row["constructor"]][s, j] += 1

    # Freeze
    return {
        d: dict(c_map) for d, c_map in result.items()
    }


# ---------------------------------------------------------------------------
# Stage 3 Model
# ---------------------------------------------------------------------------
class ConstructorPooledDirichletF1:
    """
    Stage 3: Driver-specific transitions with global + constructor priors.

    Model:
        p_{i,C,s} ~ Dirichlet(kappa_g * pi_s^{global} + kappa_c * pi_s^{(C)})
        F_{i,r} | F_{i,r-1}=s ~ Categorical(p_{i,C,s})

    kappa_g and kappa_c are jointly optimized via Empirical Bayes.

    Parameters
    ----------
    prior_alpha_global : float
        Symmetric Dirichlet prior for the global model.
    prior_alpha_constructor : float
        Prior for constructor-level transition matrices. The constructor
        matrices are estimated with Dirichlet(prior_alpha_constructor)
        before being used as components in the driver prior.
    kappa_init : tuple[float, float]
        Initial (kappa_g, kappa_c) for optimization.
    kappa_bounds : tuple[tuple, tuple]
        ((kg_min, kg_max), (kc_min, kc_max)) bounds.
    """

    def __init__(
        self,
        prior_alpha_global: float = 1.0,
        prior_alpha_constructor: float = 1.0,
        kappa_init: tuple[float, float] = (10.0, 10.0),
        kappa_bounds: tuple[tuple, tuple] = ((0.1, 500.0), (0.01, 500.0)),
    ):
        self.prior_alpha_global = prior_alpha_global
        self.prior_alpha_constructor = prior_alpha_constructor
        self.kappa_init = kappa_init
        self.kappa_bounds = kappa_bounds

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
        self.driver_ids_: Optional[list] = None
        self.constructor_ids_: Optional[list] = None
        self.opt_result_: Optional[optimize.OptimizeResult] = None

    @property
    def is_fitted(self) -> bool:
        return self.kappa_g_ is not None

    def fit(
        self,
        prev_positions: np.ndarray,
        next_positions: np.ndarray,
        meta_df: pd.DataFrame,
    ) -> "ConstructorPooledDirichletF1":
        """
        Fit the model:
          1. Compute global transition matrix.
          2. Compute per-constructor transition matrices.
          3. Compute per-driver (split by constructor) count matrices.
          4. Jointly optimize (kappa_g, kappa_c) via marginal likelihood.

        Parameters
        ----------
        prev_positions, next_positions : arrays from prepare_transitions()
        meta_df : DataFrame with 'driver' and 'constructor' columns
        """
        # Step 1: Global
        global_counts, driver_counts, constructor_counts = (
            build_count_matrices(meta_df)
        )
        self.global_counts_ = global_counts
        self.driver_counts_ = driver_counts
        self.constructor_counts_ = constructor_counts

        g_alpha = self.prior_alpha_global + global_counts
        self.global_pi_ = g_alpha / g_alpha.sum(axis=1, keepdims=True)

        # Step 2: Per-constructor transition matrices
        # Each constructor's pi is estimated from its own data + a prior
        # informed by the global pi (mild regularization).
        self.constructor_pis_ = {}
        for cid, c_counts in constructor_counts.items():
            c_alpha = self.prior_alpha_constructor + c_counts
            self.constructor_pis_[cid] = (
                c_alpha / c_alpha.sum(axis=1, keepdims=True)
            )
        self.constructor_ids_ = sorted(self.constructor_pis_.keys())

        # Step 3: Per-driver, per-constructor counts
        self.driver_constructor_counts_ = (
            build_driver_constructor_counts(meta_df)
        )
        self.driver_ids_ = sorted(self.driver_constructor_counts_.keys())
        self.driver_latest_constructor_ = (
            build_driver_constructor_map(meta_df)
        )

        # Step 4: Optimize kappa_g, kappa_c jointly
        def neg_lml(log_kappas):
            kg, kc = np.exp(log_kappas)
            return -total_log_marginal_likelihood(
                kg, kc,
                self.global_pi_, self.constructor_pis_,
                self.driver_counts_,
                self.driver_constructor_counts_,
            )

        log_k0 = np.log(self.kappa_init)
        log_bounds = [
            (np.log(b[0]), np.log(b[1])) for b in self.kappa_bounds
        ]

        result = optimize.minimize(
            neg_lml,
            x0=log_k0,
            method="L-BFGS-B",
            bounds=log_bounds,
            options={"maxiter": 300, "ftol": 1e-8},
        )
        self.kappa_g_, self.kappa_c_ = np.exp(result.x)
        self.opt_result_ = result

        return self

    # --- Prior / Posterior computation ---

    def _driver_prior_alpha_for_constructor(
        self, constructor_id
    ) -> np.ndarray:
        """
        Prior alpha matrix for a driver racing with a given constructor:
            kappa_g * pi^{global} + kappa_c * pi^{(C)}
        """
        c_pi = self.constructor_pis_.get(constructor_id, self.global_pi_)
        return self.kappa_g_ * self.global_pi_ + self.kappa_c_ * c_pi

    def driver_posterior_alpha(
        self,
        driver_id,
        constructor_id: Optional = None,
    ) -> np.ndarray:
        """
        Posterior Dirichlet alpha for a driver.

        If constructor_id is given, uses that constructor's prior and only
        the driver's counts from that constructor stint.

        If constructor_id is None, uses the driver's most recent constructor
        and their full count history with that constructor.
        """
        self._check_fitted()
        if constructor_id is None:
            constructor_id = self.driver_latest_constructor_.get(driver_id)

        prior = self._driver_prior_alpha_for_constructor(constructor_id)

        # Get driver's counts for this specific constructor
        dc_counts = self.driver_constructor_counts_.get(driver_id, {})
        counts = dc_counts.get(
            constructor_id, np.zeros((N_PREV_STATES, N_OUTCOMES))
        )
        return prior + counts

    def driver_transition_matrix(
        self, driver_id, constructor_id: Optional = None
    ) -> np.ndarray:
        """Posterior mean transition matrix for a driver (+ constructor)."""
        alpha = self.driver_posterior_alpha(driver_id, constructor_id)
        return alpha / alpha.sum(axis=1, keepdims=True)

    # --- Prediction ---

    def predict_proba(
        self,
        driver_id,
        prev_position: int,
        constructor_id: Optional = None,
    ) -> np.ndarray:
        """P(next | prev, driver, constructor)."""
        return self.driver_transition_matrix(
            driver_id, constructor_id
        )[prev_position]

    def predict_distribution(
        self,
        driver_id,
        prev_position: int,
        constructor_id: Optional = None,
    ) -> stats.dirichlet:
        """Full posterior Dirichlet for a driver/prev_position."""
        alpha = self.driver_posterior_alpha(
            driver_id, constructor_id
        )[prev_position]
        return stats.dirichlet(alpha)

    def sample_predictive(
        self,
        driver_id,
        prev_position: int,
        constructor_id: Optional = None,
        n_samples: int = 1000,
    ) -> np.ndarray:
        """Posterior predictive samples for a driver."""
        alpha = self.driver_posterior_alpha(
            driver_id, constructor_id
        )[prev_position]
        p_samples = stats.dirichlet.rvs(alpha, size=n_samples)
        return np.array([
            np.random.choice(N_OUTCOMES, p=p) for p in p_samples
        ])

    def predict_proba_new_driver(
        self,
        prev_position: int,
        constructor_id: Optional = None,
    ) -> np.ndarray:
        """
        Prediction for an unseen driver. If constructor_id is given,
        uses the blended global + constructor prior. Otherwise global only.
        """
        self._check_fitted()
        if constructor_id is not None:
            alpha = self._driver_prior_alpha_for_constructor(constructor_id)
        else:
            alpha = self.kappa_g_ * self.global_pi_
        row = alpha[prev_position]
        return row / row.sum()

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

    def pooling_factors(self, driver_id, constructor_id=None) -> dict:
        """
        Decompose the effective weight of each prior component for a driver.

        Returns dict with keys:
            'global_weight': effective fraction from global prior
            'constructor_weight': effective fraction from constructor prior
            'data_weight': effective fraction from driver's own data
        """
        self._check_fitted()
        if constructor_id is None:
            constructor_id = self.driver_latest_constructor_.get(driver_id)

        w_g = self.kappa_g_ * N_OUTCOMES  # sum of global prior per row
        w_c = self.kappa_c_ * N_OUTCOMES  # sum of constructor prior per row

        dc = self.driver_constructor_counts_.get(driver_id, {})
        n = dc.get(
            constructor_id, np.zeros((N_PREV_STATES, N_OUTCOMES))
        ).sum()
        # Average per active row
        n_per_row = n / max(1, N_PREV_STATES)

        total = w_g + w_c + n_per_row
        return {
            "global_weight": round(w_g / total, 3),
            "constructor_weight": round(w_c / total, 3),
            "data_weight": round(n_per_row / total, 3),
        }

    def log_marginal_likelihood(self) -> float:
        """Total log marginal likelihood at fitted kappas."""
        self._check_fitted()
        return total_log_marginal_likelihood(
            self.kappa_g_, self.kappa_c_,
            self.global_pi_, self.constructor_pis_,
            self.driver_counts_,
            self.driver_constructor_counts_,
        )

    def log_likelihood(
        self,
        prev_positions: np.ndarray,
        next_positions: np.ndarray,
        meta_df: pd.DataFrame,
    ) -> float:
        """
        Point-estimate log-likelihood using per-driver posterior means.
        """
        self._check_fitted()
        ll = 0.0
        # Cache transition matrices
        tm_cache = {}
        for _, row in meta_df.iterrows():
            d, c = row["driver"], row["constructor"]
            key = (d, c)
            if key not in tm_cache:
                tm_cache[key] = self.driver_transition_matrix(d, c)
            s = int(row["prev_position"])
            j = int(row["next_position"])
            ll += np.log(tm_cache[key][s, j] + 1e-300)
        return ll

    # --- Summary ---

    def summary(
        self,
        driver_id=None,
        constructor_id=None,
    ) -> pd.DataFrame:
        """
        Transition matrix as a labeled DataFrame.

        - No args: global pi
        - constructor_id only: constructor pi
        - driver_id (+ optional constructor): driver posterior
        """
        self._check_fitted()
        if driver_id is not None:
            tm = self.driver_transition_matrix(driver_id, constructor_id)
            label = f"Driver {driver_id}"
            if constructor_id:
                label += f" @ Constructor {constructor_id}"
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

    def driver_summary_table(
        self,
        name_map: Optional[dict] = None,
        constructor_name_map: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Overview table: per-driver stats, pooling decomposition, and
        most probable outcomes from key states.
        """
        self._check_fitted()
        rows = []
        for d in self.driver_ids_:
            c = self.driver_latest_constructor_.get(d)
            dc = self.driver_constructor_counts_.get(d, {})
            n_total = sum(m.sum() for m in dc.values())
            n_at_c = dc.get(
                c, np.zeros((N_PREV_STATES, N_OUTCOMES))
            ).sum()
            pf = self.pooling_factors(d, c)
            tm = self.driver_transition_matrix(d, c)
            rows.append({
                "driver": d,
                "driver_name": (
                    name_map.get(d, d) if name_map else d
                ),
                "constructor": c,
                "constructor_name": (
                    constructor_name_map.get(c, c)
                    if constructor_name_map else c
                ),
                "n_total": n_total,
                "n_at_constructor": n_at_c,
                "w_global": pf["global_weight"],
                "w_constructor": pf["constructor_weight"],
                "w_data": pf["data_weight"],
                "mode_from_P1": OUTCOME_LABELS[np.argmax(tm[1])],
                "mode_from_P10": OUTCOME_LABELS[np.argmax(tm[10])],
                "mode_from_START": OUTCOME_LABELS[np.argmax(tm[START])],
            })
        return (
            pd.DataFrame(rows)
            .sort_values("n_total", ascending=False)
            .reset_index(drop=True)
        )

    def constructor_summary_table(
        self,
        name_map: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Summary of constructor-level transition matrices."""
        self._check_fitted()
        rows = []
        for cid in self.constructor_ids_:
            n = self.constructor_counts_[cid].sum()
            tm = self.constructor_pis_[cid]
            rows.append({
                "constructor": cid,
                "name": name_map.get(cid, cid) if name_map else cid,
                "n_transitions": n,
                "P(P1|prev=P1)": round(tm[1, 1], 3),
                "P(DNF|any)": round(tm[:, 0].mean(), 3),
                "mode_from_P5": OUTCOME_LABELS[np.argmax(tm[5])],
                "mode_from_P10": OUTCOME_LABELS[np.argmax(tm[10])],
                "mode_from_START": OUTCOME_LABELS[np.argmax(tm[START])],
            })
        return (
            pd.DataFrame(rows)
            .sort_values("n_transitions", ascending=False)
            .reset_index(drop=True)
        )

    def kappa_profile_2d(
        self,
        kg_range: Optional[np.ndarray] = None,
        kc_range: Optional[np.ndarray] = None,
        n_points: int = 20,
    ) -> pd.DataFrame:
        """
        2D profile of log marginal likelihood over (kappa_g, kappa_c).
        Returns a long-form DataFrame for heatmap visualization.
        """
        self._check_fitted()
        if kg_range is None:
            kg_range = np.logspace(-1, np.log10(300), n_points)
        if kc_range is None:
            kc_range = np.logspace(-1, np.log10(300), n_points)

        rows = []
        for kg in kg_range:
            for kc in kc_range:
                lml = total_log_marginal_likelihood(
                    kg, kc,
                    self.global_pi_, self.constructor_pis_,
                    self.driver_counts_,
                    self.driver_constructor_counts_,
                )
                rows.append({
                    "kappa_g": kg, "kappa_c": kc,
                    "log_marginal_likelihood": lml,
                })
        return pd.DataFrame(rows)

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")


# ---------------------------------------------------------------------------
# Scenario Analysis Helpers
# ---------------------------------------------------------------------------
def compare_constructor_effect(
    model: ConstructorPooledDirichletF1,
    driver_id,
    constructors: list,
    prev_position: int = 5,
    constructor_name_map: Optional[dict] = None,
) -> pd.DataFrame:
    """
    What-if analysis: how would a driver's predictions change
    if they raced for different constructors?

    Returns a DataFrame with one row per constructor showing the
    predicted probability distribution from prev_position.
    """
    rows = []
    for cid in constructors:
        probs = model.predict_proba(driver_id, prev_position, cid)
        cname = (
            constructor_name_map.get(cid, cid)
            if constructor_name_map else cid
        )
        row = {"constructor": cname}
        for j, label in enumerate(OUTCOME_LABELS):
            row[label] = round(probs[j], 4)
        row["E[position]"] = round(
            sum(j * probs[j] for j in range(1, N_OUTCOMES)), 2
        )
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    print("=" * 65)
    print("F1 Dirichlet-Multinomial — Stage 3: Constructor Effect")
    print("=" * 65)

    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/f1")

    if (data_dir / "results.csv").exists():
        print(f"\nLoading Kaggle data from {data_dir} ...")
        loader = F1DataLoader(data_dir)
        df = loader.load_merged(min_year=2014)
        print(
            f"  Rows: {len(df)}, Seasons: {df['year'].nunique()}, "
            f"Drivers: {df['driverId'].nunique()}, "
            f"Constructors: {df['constructorId'].nunique()}"
        )
        prev_pos, next_pos, meta = prepare_transitions(df)

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
    else:
        print(f"\nKaggle data not found; using synthetic data.")
        rng = np.random.default_rng(42)
        n_drivers, n_races = 20, 60
        constructors_pool = [1, 2, 3, 4, 5]
        rows = []
        for d in range(1, n_drivers + 1):
            skill = d
            cid = constructors_pool[(d - 1) % len(constructors_pool)]
            for r in range(1, n_races + 1):
                # Constructor boost: lower cid = better car
                boost = (3 - cid) * 1.5
                pos = int(np.clip(
                    rng.normal(skill - boost, 3), 1, 20
                ))
                if rng.random() < 0.05:
                    pos = DNF
                s = 2020 + r // 23
                rows.append({
                    "driver": d, "constructor": cid,
                    "season": s, "race_order": r % 23 + 1,
                    "prev_position": START, "next_position": pos,
                })
        meta = pd.DataFrame(rows)
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
        prev_pos = meta["prev_position"].values.astype(int)
        next_pos = meta["next_position"].values.astype(int)
        driver_names = {d: f"Driver_{d:02d}" for d in range(1, n_drivers + 1)}
        constructor_names = {
            c: f"Team_{c}" for c in constructors_pool
        }

    print(f"Total transitions: {len(prev_pos)}")

    # --- Fit ---
    model = ConstructorPooledDirichletF1(
        prior_alpha_global=1.0,
        prior_alpha_constructor=1.0,
        kappa_init=(10.0, 10.0),
        kappa_bounds=((0.1, 500.0), (0.01, 500.0)),
    )
    model.fit(prev_pos, next_pos, meta)

    print(f"\n--- Empirical Bayes Results ---")
    print(f"  kappa_global:      {model.kappa_g_:.2f}  "
          f"(~{model.kappa_g_:.0f} pseudo-obs from global prior)")
    print(f"  kappa_constructor: {model.kappa_c_:.2f}  "
          f"(~{model.kappa_c_:.0f} pseudo-obs from constructor prior)")
    print(f"  Log marginal lik:  {model.log_marginal_likelihood():.1f}")
    print(f"  Optimizer converged: {model.opt_result_.success}")

    kg_share = model.kappa_g_ / (model.kappa_g_ + model.kappa_c_)
    print(f"  Prior blend: {kg_share:.0%} global / "
          f"{1-kg_share:.0%} constructor")

    # --- Constructor summary ---
    print("\n--- Constructor Summary (top 10) ---")
    cs = model.constructor_summary_table(name_map=constructor_names)
    print(cs.head(10).to_string(index=False))

    # --- Driver summary ---
    print("\n--- Driver Summary (top 10 by data volume) ---")
    ds = model.driver_summary_table(
        name_map=driver_names,
        constructor_name_map=constructor_names,
    )
    cols = [
        "driver_name", "constructor_name", "n_total", "n_at_constructor",
        "w_global", "w_constructor", "w_data",
        "mode_from_P1", "mode_from_START",
    ]
    print(ds[cols].head(10).to_string(index=False))

    # --- Constructor what-if ---
    if len(model.driver_ids_) > 0 and len(model.constructor_ids_) >= 2:
        test_driver = model.driver_ids_[0]
        test_name = driver_names.get(test_driver, test_driver)
        top_constructors = (
            cs.sort_values("n_transitions", ascending=False)
            ["constructor"].head(4).tolist()
        )
        print(f"\n--- What-If: {test_name} at different constructors "
              f"(prev=P5) ---")
        wif = compare_constructor_effect(
            model, test_driver, top_constructors,
            prev_position=5,
            constructor_name_map=constructor_names,
        )
        show_cols = [
            "constructor", "DNF", "P1", "P2", "P3", "P5",
            "P10", "P15", "E[position]",
        ]
        print(wif[show_cols].to_string(index=False))

    # --- Posterior predictive ---
    if model.driver_ids_:
        ex_d = model.driver_ids_[0]
        ex_name = driver_names.get(ex_d, ex_d)
        print(f"\n--- Posterior Predictive: {ex_name}, prev=P3 ---")
        samples = model.sample_predictive(ex_d, prev_position=3)
        unique, counts = np.unique(samples, return_counts=True)
        top5 = np.argsort(counts)[::-1][:5]
        for idx in top5:
            lo, hi = model.credible_interval(ex_d, 3, unique[idx])
            print(
                f"  {OUTCOME_LABELS[unique[idx]]}: "
                f"{counts[idx]/len(samples):.3f}  "
                f"[95% CI: {lo:.3f}–{hi:.3f}]"
            )
