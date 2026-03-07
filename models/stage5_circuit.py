"""
F1 Dirichlet-Multinomial Markov Model
======================================
Stage 5: Constructor + Circuit + Driver

Extends Stage 3 by adding a circuit-specific transition matrix as a third
source of prior information. Different tracks produce different outcome
distributions (e.g., Monaco favors front-runners, Monza allows more
overtaking). Circuit effects also implicitly capture weather patterns
associated with certain venues.

Model (for driver i at constructor C, circuit K, previous state s):

    F_{i,r} | F_{i,r-1}=s  ~  Categorical(p_{i,C,K,s})

    p_{i,C,K,s}  ~  Dirichlet(alpha_{i,C,K,s})

    alpha_{i,C,K,s} = kappa_g * pi_s^{global}
                    + kappa_c * pi_s^{(C)}
                    + kappa_k * pi_s^{(K)}
                    + n_{i,C,s}

Where:
    pi_s^{global}   = global posterior mean transition probs (from all data)
    pi_s^{(C)}      = constructor C's posterior mean transition probs
    pi_s^{(K)}      = circuit K's posterior mean transition probs
    kappa_g, kappa_c, kappa_k = concentration parameters (learned via EB)
    n_{i,C,s}       = driver i's observed count vector for state s at constructor C

Because the circuit prior varies per race, the marginal likelihood cannot
aggregate counts across races for a (driver, constructor) pair. Instead,
we use a per-race log posterior-predictive score.

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
    circuit_col: str = "circuitId",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Convert race results into (prev_position, next_position) arrays.

    Stage 5: captures constructor and circuit for each transition.

    Returns
    -------
    prev_positions, next_positions : np.ndarray
    meta_df : DataFrame with [driver, constructor, circuit, season,
              race_order, prev_position, next_position]
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
        circuits = grp[circuit_col].values

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
                "circuit": int(circuits[i]),
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
    circuit_counts : dict[circuit_id -> (N_PREV_STATES, N_OUTCOMES)]
    """
    global_counts = np.zeros((N_PREV_STATES, N_OUTCOMES), dtype=int)
    driver_counts: dict = defaultdict(
        lambda: np.zeros((N_PREV_STATES, N_OUTCOMES), dtype=int)
    )
    constructor_counts: dict = defaultdict(
        lambda: np.zeros((N_PREV_STATES, N_OUTCOMES), dtype=int)
    )
    circuit_counts: dict = defaultdict(
        lambda: np.zeros((N_PREV_STATES, N_OUTCOMES), dtype=int)
    )

    for _, row in meta_df.iterrows():
        s = int(row["prev_position"])
        j = int(row["next_position"])
        global_counts[s, j] += 1
        driver_counts[row["driver"]][s, j] += 1
        constructor_counts[row["constructor"]][s, j] += 1
        circuit_counts[row["circuit"]][s, j] += 1

    return (
        global_counts,
        dict(driver_counts),
        dict(constructor_counts),
        dict(circuit_counts),
    )


def build_driver_constructor_map(
    meta_df: pd.DataFrame,
) -> dict:
    """
    For each driver, identify their *most recent* constructor assignment.
    Returns dict[driver_id -> constructor_id]
    """
    latest = (
        meta_df
        .sort_values(["season", "race_order"])
        .groupby("driver")["constructor"]
        .last()
    )
    return latest.to_dict()


def build_driver_constructor_counts(
    meta_df: pd.DataFrame,
) -> dict:
    """
    Build per-driver, per-constructor count matrices.

    Returns dict[driver -> dict[constructor -> (N_PREV_STATES, N_OUTCOMES)]]
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

    return {
        d: dict(c_map) for d, c_map in result.items()
    }


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


def build_driver_constructor_circuit_counts(
    meta_df: pd.DataFrame,
) -> dict:
    """
    Build per-driver, per-constructor, per-circuit count matrices.

    Returns dict[driver -> dict[constructor -> dict[circuit -> (N_PREV_STATES, N_OUTCOMES)]]]
    """
    result: dict = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: np.zeros((N_PREV_STATES, N_OUTCOMES), dtype=int)
            )
        )
    )
    for _, row in meta_df.iterrows():
        s = int(row["prev_position"])
        j = int(row["next_position"])
        result[row["driver"]][row["constructor"]][row["circuit"]][s, j] += 1

    return {
        d: {c: dict(k_map) for c, k_map in c_map.items()}
        for d, c_map in result.items()
    }


def total_log_marginal_likelihood(
    kappa_g: float,
    kappa_c: float,
    kappa_k: float,
    global_pi: np.ndarray,
    constructor_pis: dict,
    circuit_pis: dict,
    driver_constructor_circuit_counts: dict,
) -> float:
    """
    Total log marginal likelihood across all drivers/states.

    For each (driver, constructor, circuit, prev_state) group:
        alpha_s = kappa_g * pi^{global}_s + kappa_c * pi^{(C)}_s + kappa_k * pi^{(K)}_s

    Uses the Dirichlet-Multinomial marginal likelihood to integrate out
    the transition probabilities.
    """
    lml = 0.0
    for driver, constr_map in driver_constructor_circuit_counts.items():
        for constr, circuit_map in constr_map.items():
            c_pi = constructor_pis.get(constr, global_pi)
            for circuit, counts in circuit_map.items():
                k_pi = circuit_pis.get(circuit, global_pi)
                for s in range(N_PREV_STATES):
                    n_s = counts[s]
                    if n_s.sum() == 0:
                        continue
                    alpha_s = (
                        kappa_g * global_pi[s]
                        + kappa_c * c_pi[s]
                        + kappa_k * k_pi[s]
                    )
                    alpha_s = np.maximum(alpha_s, 1e-10)
                    lml += _dm_log_ml(alpha_s, n_s)
    return lml


# ---------------------------------------------------------------------------
# Stage 5 Model
# ---------------------------------------------------------------------------
class CircuitDirichletF1:
    """
    Stage 5: Driver-specific transitions with global + constructor + circuit priors.

    Model:
        p_{i,C,K,s} ~ Dirichlet(kappa_g * pi^{global} + kappa_c * pi^{(C)}
                                 + kappa_k * pi^{(K)})
        F_{i,r} | F_{i,r-1}=s ~ Categorical(p_{i,C,K,s})

    kappa_g, kappa_c, kappa_k are jointly optimized via Empirical Bayes.
    """

    def __init__(
        self,
        prior_alpha_global: float = 1.0,
        prior_alpha_constructor: float = 1.0,
        prior_alpha_circuit: float = 1.0,
        kappa_init: tuple[float, float, float] = (10.0, 10.0, 1.0),
        kappa_bounds: tuple[tuple, tuple, tuple] = (
            (0.1, 500.0), (0.01, 500.0), (0.01, 200.0)
        ),
    ):
        self.prior_alpha_global = prior_alpha_global
        self.prior_alpha_constructor = prior_alpha_constructor
        self.prior_alpha_circuit = prior_alpha_circuit
        self.kappa_init = kappa_init
        self.kappa_bounds = kappa_bounds

        # Fitted attributes
        self.global_pi_: Optional[np.ndarray] = None
        self.global_counts_: Optional[np.ndarray] = None
        self.constructor_pis_: Optional[dict] = None
        self.constructor_counts_: Optional[dict] = None
        self.circuit_pis_: Optional[dict] = None
        self.circuit_counts_: Optional[dict] = None
        self.driver_counts_: Optional[dict] = None
        self.driver_constructor_counts_: Optional[dict] = None
        self.driver_latest_constructor_: Optional[dict] = None
        self.kappa_g_: Optional[float] = None
        self.kappa_c_: Optional[float] = None
        self.kappa_k_: Optional[float] = None
        self.driver_ids_: Optional[list] = None
        self.constructor_ids_: Optional[list] = None
        self.circuit_ids_: Optional[list] = None
        self.opt_result_: Optional[optimize.OptimizeResult] = None

    @property
    def is_fitted(self) -> bool:
        return self.kappa_g_ is not None

    def fit(
        self,
        prev_positions: np.ndarray,
        next_positions: np.ndarray,
        meta_df: pd.DataFrame,
    ) -> "CircuitDirichletF1":
        """
        Fit the model:
          1. Compute global, constructor, and circuit transition matrices.
          2. Compute per-driver (split by constructor) count matrices.
          3. Jointly optimize (kappa_g, kappa_c, kappa_k) via log predictive.
        """
        # Step 1: Count matrices
        global_counts, driver_counts, constructor_counts, circuit_counts = (
            build_count_matrices(meta_df)
        )
        self.global_counts_ = global_counts
        self.driver_counts_ = driver_counts
        self.constructor_counts_ = constructor_counts
        self.circuit_counts_ = circuit_counts

        # Global pi
        g_alpha = self.prior_alpha_global + global_counts
        self.global_pi_ = g_alpha / g_alpha.sum(axis=1, keepdims=True)

        # Per-constructor pi
        self.constructor_pis_ = {}
        for cid, c_counts in constructor_counts.items():
            c_alpha = self.prior_alpha_constructor + c_counts
            self.constructor_pis_[cid] = (
                c_alpha / c_alpha.sum(axis=1, keepdims=True)
            )
        self.constructor_ids_ = sorted(self.constructor_pis_.keys())

        # Per-circuit pi
        self.circuit_pis_ = {}
        for kid, k_counts in circuit_counts.items():
            k_alpha = self.prior_alpha_circuit + k_counts
            self.circuit_pis_[kid] = (
                k_alpha / k_alpha.sum(axis=1, keepdims=True)
            )
        self.circuit_ids_ = sorted(self.circuit_pis_.keys())

        # Per-driver, per-constructor counts (for prediction)
        self.driver_constructor_counts_ = (
            build_driver_constructor_counts(meta_df)
        )
        self.driver_ids_ = sorted(self.driver_constructor_counts_.keys())
        self.driver_latest_constructor_ = (
            build_driver_constructor_map(meta_df)
        )

        # Per-driver, per-constructor, per-circuit counts (for optimization)
        driver_constructor_circuit_counts = (
            build_driver_constructor_circuit_counts(meta_df)
        )

        # Step 2: Optimize kappa_g, kappa_c, kappa_k
        def neg_lml(log_kappas):
            kg, kc, kk = np.exp(log_kappas)
            return -total_log_marginal_likelihood(
                kg, kc, kk,
                self.global_pi_, self.constructor_pis_, self.circuit_pis_,
                driver_constructor_circuit_counts,
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
        self.kappa_g_, self.kappa_c_, self.kappa_k_ = np.exp(result.x)
        self.opt_result_ = result

        return self

    # --- Prior / Posterior computation ---

    def _driver_prior_alpha(
        self, constructor_id, circuit_id=None,
    ) -> np.ndarray:
        """
        Prior alpha matrix for a driver racing with a given constructor
        at a given circuit:
            kappa_g * pi^{global} + kappa_c * pi^{(C)} + kappa_k * pi^{(K)}
        """
        c_pi = self.constructor_pis_.get(constructor_id, self.global_pi_)
        k_pi = (
            self.circuit_pis_.get(circuit_id, self.global_pi_)
            if circuit_id is not None else self.global_pi_
        )
        return (
            self.kappa_g_ * self.global_pi_
            + self.kappa_c_ * c_pi
            + self.kappa_k_ * k_pi
        )

    def driver_posterior_alpha(
        self,
        driver_id,
        constructor_id: Optional = None,
        circuit_id: Optional = None,
    ) -> np.ndarray:
        """
        Posterior Dirichlet alpha for a driver.

        Uses the driver's counts from the specified constructor stint.
        Circuit affects the prior but not the driver counts.
        """
        self._check_fitted()
        if constructor_id is None:
            constructor_id = self.driver_latest_constructor_.get(driver_id)

        prior = self._driver_prior_alpha(constructor_id, circuit_id)

        dc_counts = self.driver_constructor_counts_.get(driver_id, {})
        counts = dc_counts.get(
            constructor_id, np.zeros((N_PREV_STATES, N_OUTCOMES))
        )
        return prior + counts

    def driver_transition_matrix(
        self, driver_id, constructor_id: Optional = None,
        circuit_id: Optional = None,
    ) -> np.ndarray:
        """Posterior mean transition matrix for a driver."""
        alpha = self.driver_posterior_alpha(
            driver_id, constructor_id, circuit_id
        )
        return alpha / alpha.sum(axis=1, keepdims=True)

    # --- Prediction ---

    def predict_proba(
        self,
        driver_id,
        prev_position: int,
        constructor_id: Optional = None,
        circuit_id: Optional = None,
        **kwargs,
    ) -> np.ndarray:
        """P(next | prev, driver, constructor, circuit)."""
        return self.driver_transition_matrix(
            driver_id, constructor_id, circuit_id
        )[prev_position]

    def predict_proba_new_driver(
        self,
        prev_position: int,
        constructor_id: Optional = None,
        circuit_id: Optional = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Prediction for an unseen driver. Uses blended priors only.
        """
        self._check_fitted()
        if constructor_id is not None:
            alpha = self._driver_prior_alpha(constructor_id, circuit_id)
        else:
            alpha = self.kappa_g_ * self.global_pi_
            if circuit_id is not None:
                k_pi = self.circuit_pis_.get(circuit_id, self.global_pi_)
                alpha += self.kappa_k_ * k_pi
        row = alpha[prev_position]
        return row / row.sum()

    def predict_distribution(
        self,
        driver_id,
        prev_position: int,
        constructor_id: Optional = None,
        circuit_id: Optional = None,
    ) -> stats.dirichlet:
        """Full posterior Dirichlet for a driver/prev_position."""
        alpha = self.driver_posterior_alpha(
            driver_id, constructor_id, circuit_id
        )[prev_position]
        return stats.dirichlet(alpha)

    def credible_interval(
        self,
        driver_id,
        prev_position: int,
        outcome: int,
        constructor_id: Optional = None,
        circuit_id: Optional = None,
        alpha: float = 0.05,
        n_samples: int = 10000,
    ) -> tuple[float, float]:
        """(1-alpha) credible interval for P(outcome | prev, driver, C, K)."""
        post_alpha = self.driver_posterior_alpha(
            driver_id, constructor_id, circuit_id
        )[prev_position]
        p_samples = stats.dirichlet.rvs(post_alpha, size=n_samples)
        probs = p_samples[:, outcome]
        return (
            np.percentile(probs, 100 * alpha / 2),
            np.percentile(probs, 100 * (1 - alpha / 2)),
        )

    # --- Diagnostics ---

    def log_marginal_likelihood(self, meta_df: pd.DataFrame) -> float:
        """Total log marginal likelihood at fitted kappas."""
        self._check_fitted()
        dcc_counts = build_driver_constructor_circuit_counts(meta_df)
        return total_log_marginal_likelihood(
            self.kappa_g_, self.kappa_c_, self.kappa_k_,
            self.global_pi_, self.constructor_pis_, self.circuit_pis_,
            dcc_counts,
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
        tm_cache = {}
        for _, row in meta_df.iterrows():
            d, c = row["driver"], row["constructor"]
            k = row["circuit"]
            key = (d, c, k)
            if key not in tm_cache:
                tm_cache[key] = self.driver_transition_matrix(d, c, k)
            s = int(row["prev_position"])
            j = int(row["next_position"])
            ll += np.log(tm_cache[key][s, j] + 1e-300)
        return ll

    # --- Summary ---

    def summary(
        self,
        driver_id=None,
        constructor_id=None,
        circuit_id=None,
    ) -> pd.DataFrame:
        """
        Transition matrix as a labeled DataFrame.

        - No args: global pi
        - constructor_id only: constructor pi
        - circuit_id only: circuit pi
        - driver_id (+ optional constructor/circuit): driver posterior
        """
        self._check_fitted()
        if driver_id is not None:
            tm = self.driver_transition_matrix(
                driver_id, constructor_id, circuit_id
            )
            label = f"Driver {driver_id}"
        elif constructor_id is not None:
            tm = self.constructor_pis_.get(constructor_id, self.global_pi_)
            label = f"Constructor {constructor_id}"
        elif circuit_id is not None:
            tm = self.circuit_pis_.get(circuit_id, self.global_pi_)
            label = f"Circuit {circuit_id}"
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
    print("F1 Dirichlet-Multinomial — Stage 5: Circuit Effect")
    print("=" * 65)

    data_dir = Path(__file__).parent.parent / "data"

    print(f"\nLoading data from {data_dir} ...")
    loader = F1DataLoader(data_dir)
    df = loader.load_merged(min_year=2020, max_year=2024)
    print(
        f"  Rows: {len(df)}, Seasons: {df['year'].nunique()}, "
        f"Drivers: {df['driverId'].nunique()}, "
        f"Constructors: {df['constructorId'].nunique()}, "
        f"Circuits: {df['circuitId'].nunique()}"
    )
    prev_pos, next_pos, meta = prepare_transitions(df)
    print(f"  Total transitions: {len(prev_pos)}")

    # --- Fit ---
    model = CircuitDirichletF1(
        prior_alpha_global=1.0,
        prior_alpha_constructor=1.0,
        prior_alpha_circuit=1.0,
        kappa_init=(10.0, 10.0, 1.0),
        kappa_bounds=((0.1, 500.0), (0.01, 500.0), (0.01, 200.0)),
    )
    model.fit(prev_pos, next_pos, meta)

    print(f"\n--- Empirical Bayes Results ---")
    print(f"  kappa_global:      {model.kappa_g_:.2f}")
    print(f"  kappa_constructor: {model.kappa_c_:.2f}")
    print(f"  kappa_circuit:     {model.kappa_k_:.2f}")
    print(f"  Optimizer converged: {model.opt_result_.success}")

    total_k = model.kappa_g_ + model.kappa_c_ + model.kappa_k_
    print(f"  Prior blend: {model.kappa_g_/total_k:.0%} global / "
          f"{model.kappa_c_/total_k:.0%} constructor / "
          f"{model.kappa_k_/total_k:.0%} circuit")

    print(f"\n  Log marginal likelihood: {model.log_marginal_likelihood(meta):.1f}")
    print(f"  Circuits: {len(model.circuit_ids_)}")
