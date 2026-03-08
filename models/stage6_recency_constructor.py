"""
F1 Dirichlet-Multinomial Markov Model
======================================
Stage 6: Year-weighted constructor priors (+ optional circuit prior)

Extends Stage 3 by applying geometric recency decay to constructor count
matrices before normalizing them into prior transition probabilities.
This captures team-level non-stationarity (e.g., McLaren's 2024 breakthrough)
without affecting the stable global baseline or raw driver evidence.

Model (for driver i at constructor C, previous state s):

    F_{i,r} | F_{i,r-1}=s  ~  Categorical(p_{i,C,s})

    p_{i,C,s}  ~  Dirichlet(alpha_{i,C,s})

    alpha_{i,C,s} = kappa_g * pi_s^{global}
                  + kappa_c * pi_s^{(C,w)}
                  + n_{i,C,s}

When circuit data is available (meta_df has 'circuit' column):

    alpha_{i,C,K,s} = kappa_g * pi_s^{global}
                    + kappa_c * pi_s^{(C,w)}
                    + kappa_k * pi_s^{(K)}
                    + n_{i,C,s}

Where:
    pi_s^{global}   = global posterior mean (unweighted, stable baseline)
    pi_s^{(C,w)}    = constructor C's year-weighted posterior mean
    pi_s^{(K)}      = circuit K's posterior mean (static, unweighted)
    kappa_g, kappa_c = concentration parameters (learned via Empirical Bayes)
    kappa_k          = circuit concentration parameter (optional)
    w               = geometric decay factor in (0, 1)
    n_{i,C,s}       = driver i's observed counts (unweighted)

Constructor weighting:
    Each observation in season y gets weight w^(ref_year - y), where
    ref_year = max(training years) + 1. With w=0.5 and ref=2025:
        2024 → w^1=0.5, 2023 → w^2=0.25, 2022 → 0.125, ...

Key difference from Stage 4 (recency on both priors):
    - Only constructor prior gets recency-weighted
    - Global prior stays unweighted (stable baseline)
    - Circuit prior stays unweighted (tracks don't change physically)
    - Driver counts stay unweighted (preserve evidence)
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.special import gammaln
from pathlib import Path
from typing import Optional
from collections import defaultdict

# Import shared components from Stage 3
from stage3_constructor import (
    F1DataLoader,
    prepare_transitions,
    build_count_matrices,
    build_driver_constructor_counts,
    build_driver_constructor_map,
    _dm_log_ml,
    DNF, POSITIONS, START,
    N_PREV_STATES, N_OUTCOMES,
    OUTCOME_LABELS, PREV_STATE_LABELS,
)


# ---------------------------------------------------------------------------
# Year-Weighted Constructor Counts
# ---------------------------------------------------------------------------
def build_weighted_constructor_counts(
    meta_df: pd.DataFrame,
    w: float,
    ref_year: Optional[int] = None,
) -> dict:
    """
    Build constructor count matrices with geometric year weighting.

    Each observation in season y gets weight w^(ref_year - y).
    More recent seasons contribute more to the constructor profile.

    Parameters
    ----------
    meta_df : DataFrame with 'constructor', 'season', 'prev_position',
              'next_position' columns
    w : float in (0, 1), geometric decay factor (1 = no decay)
    ref_year : reference year (default: max season + 1)

    Returns
    -------
    dict[constructor_id -> (N_PREV_STATES, N_OUTCOMES)] float arrays
    """
    if ref_year is None:
        ref_year = int(meta_df["season"].max()) + 1

    constructor_counts: dict = defaultdict(
        lambda: np.zeros((N_PREV_STATES, N_OUTCOMES), dtype=float)
    )

    seasons = meta_df["season"].values
    weights = np.power(w, ref_year - seasons.astype(float))

    for idx, (_, row) in enumerate(meta_df.iterrows()):
        s = int(row["prev_position"])
        j = int(row["next_position"])
        constructor_counts[row["constructor"]][s, j] += weights[idx]

    return dict(constructor_counts)


# ---------------------------------------------------------------------------
# Circuit Count Matrices
# ---------------------------------------------------------------------------
def build_circuit_counts(meta_df: pd.DataFrame) -> dict:
    """
    Build per-circuit count matrices (static, unweighted).

    Returns dict[circuit_id -> (N_PREV_STATES, N_OUTCOMES)] int arrays.
    """
    circuit_counts: dict = defaultdict(
        lambda: np.zeros((N_PREV_STATES, N_OUTCOMES), dtype=int)
    )
    for _, row in meta_df.iterrows():
        s = int(row["prev_position"])
        j = int(row["next_position"])
        circuit_counts[row["circuit"]][s, j] += 1
    return dict(circuit_counts)


def build_driver_constructor_circuit_counts(meta_df: pd.DataFrame) -> dict:
    """
    Build per-driver, per-constructor, per-circuit count matrices.

    Needed for marginal likelihood when circuit prior varies per race.

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


# ---------------------------------------------------------------------------
# Marginal Likelihood
# ---------------------------------------------------------------------------
def total_log_marginal_likelihood(
    kappa_g: float,
    kappa_c: float,
    global_pi: np.ndarray,
    constructor_pis: dict,
    driver_constructor_history: dict,
) -> float:
    """
    Total log marginal likelihood across all drivers/states.

    Same structure as Stage 3, but constructor_pis are year-weighted.

    Parameters
    ----------
    driver_constructor_history : dict[driver -> dict[constructor -> counts]]
        Per-driver, per-constructor count matrices (unweighted integers).
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


def total_log_marginal_likelihood_with_circuit(
    kappa_g: float,
    kappa_c: float,
    kappa_k: float,
    global_pi: np.ndarray,
    constructor_pis: dict,
    circuit_pis: dict,
    driver_constructor_circuit_history: dict,
) -> float:
    """
    Total log marginal likelihood with circuit prior.

    Iterates over (driver, constructor, circuit, prev_state) tuples since
    the alpha vector varies per circuit.

    Parameters
    ----------
    driver_constructor_circuit_history :
        dict[driver -> dict[constructor -> dict[circuit -> counts]]]
    """
    lml = 0.0
    for driver, constr_map in driver_constructor_circuit_history.items():
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
# Stage 6 Model
# ---------------------------------------------------------------------------
class RecencyConstructorDirichletF1:
    """
    Stage 6: Driver-specific transitions with global + year-weighted
    constructor priors (+ optional circuit prior).

    Model (without circuit):
        p_{i,C,s} ~ Dirichlet(kappa_g * pi_s^{global}
                             + kappa_c * pi_s^{(C,w)})

    Model (with circuit):
        p_{i,C,K,s} ~ Dirichlet(kappa_g * pi_s^{global}
                               + kappa_c * pi_s^{(C,w)}
                               + kappa_k * pi_s^{(K)})

    Circuit mode is activated automatically when meta_df has a 'circuit'
    column. When disabled, behavior is identical to the original Stage 6.

    Parameters
    ----------
    prior_alpha_global : float
        Symmetric Dirichlet prior for the global model.
    prior_alpha_constructor : float
        Prior for constructor-level transition matrices.
    prior_alpha_circuit : float
        Prior for circuit-level transition matrices. Only used in circuit mode.
    kappa_init : tuple[float, float]
        Initial (kappa_g, kappa_c) for optimization.
    kappa_bounds : tuple[tuple, tuple]
        ((kg_min, kg_max), (kc_min, kc_max)) bounds.
    kappa_k_init : float
        Initial kappa_k for circuit prior. Only used in circuit mode.
    kappa_k_bounds : tuple[float, float]
        (kk_min, kk_max) bounds. Only used in circuit mode.
    w_candidates : tuple[float, ...]
        Grid of w values to evaluate. Selected via leave-last-year-out CV.
    """

    def __init__(
        self,
        prior_alpha_global: float = 1.0,
        prior_alpha_constructor: float = 1.0,
        prior_alpha_circuit: float = 1.0,
        kappa_init: tuple[float, float] = (10.0, 10.0),
        kappa_bounds: tuple[tuple, tuple] = ((0.1, 500.0), (0.01, 500.0)),
        kappa_k_init: float = 1.0,
        kappa_k_bounds: tuple[float, float] = (0.01, 200.0),
        w_candidates: tuple[float, ...] = (0.3, 0.5, 0.7, 0.85, 1.0),
    ):
        self.prior_alpha_global = prior_alpha_global
        self.prior_alpha_constructor = prior_alpha_constructor
        self.prior_alpha_circuit = prior_alpha_circuit
        self.kappa_init = kappa_init
        self.kappa_bounds = kappa_bounds
        self.kappa_k_init = kappa_k_init
        self.kappa_k_bounds = kappa_k_bounds
        self.w_candidates = w_candidates

        # Fitted attributes
        self.global_pi_: Optional[np.ndarray] = None
        self.global_counts_: Optional[np.ndarray] = None
        self.constructor_pis_: Optional[dict] = None
        self.driver_counts_: Optional[dict] = None
        self.driver_constructor_counts_: Optional[dict] = None
        self.driver_latest_constructor_: Optional[dict] = None
        self.kappa_g_: Optional[float] = None
        self.kappa_c_: Optional[float] = None
        self.kappa_k_: Optional[float] = None
        self.w_: Optional[float] = None
        self.driver_ids_: Optional[list] = None
        self.constructor_ids_: Optional[list] = None
        self.circuit_pis_: Optional[dict] = None
        self.circuit_counts_: Optional[dict] = None
        self.circuit_ids_: Optional[list] = None
        self.opt_result_: Optional[optimize.OptimizeResult] = None
        self._meta_df: Optional[pd.DataFrame] = None
        self._circuit_mode: bool = False

    @property
    def is_fitted(self) -> bool:
        return self.kappa_g_ is not None

    def fit(
        self,
        prev_positions: np.ndarray,
        next_positions: np.ndarray,
        meta_df: pd.DataFrame,
    ) -> "RecencyConstructorDirichletF1":
        """
        Fit the model using leave-last-year-out cross-validation for w:

          1. Split data: years 1..N-1 = train, year N = validation
          2. For each w candidate:
             a. Build weighted constructor pis from train years
             b. Optimize kappa_g, kappa_c (+ kappa_k if circuit) on train
             c. Evaluate DM log-likelihood on validation year
          3. Pick w with best validation log-likelihood
          4. Retrain on ALL data with selected w

        If meta_df has a 'circuit' column, circuit mode is activated and
        kappa_k is optimized jointly with kappa_g, kappa_c.
        """
        self._meta_df = meta_df
        self._circuit_mode = "circuit" in meta_df.columns

        # Step 1: Global (unweighted — stable baseline from ALL data)
        global_counts, driver_counts, _ = build_count_matrices(meta_df)
        self.global_counts_ = global_counts
        self.driver_counts_ = driver_counts

        g_alpha = self.prior_alpha_global + global_counts
        self.global_pi_ = g_alpha / g_alpha.sum(axis=1, keepdims=True)

        # Per-driver, per-constructor counts (unweighted, ALL data)
        self.driver_constructor_counts_ = (
            build_driver_constructor_counts(meta_df)
        )
        self.driver_ids_ = sorted(self.driver_constructor_counts_.keys())
        self.driver_latest_constructor_ = (
            build_driver_constructor_map(meta_df)
        )

        # Circuit mode: build circuit pis from ALL data
        if self._circuit_mode:
            all_circuit_counts = build_circuit_counts(meta_df)
            k_pis_all = {}
            for kid, kc in all_circuit_counts.items():
                k_alpha = self.prior_alpha_circuit + kc
                k_pis_all[kid] = k_alpha / k_alpha.sum(
                    axis=1, keepdims=True
                )
            self.circuit_counts_ = all_circuit_counts
            self.circuit_ids_ = sorted(all_circuit_counts.keys())

        # Step 2: Leave-last-year-out CV for w selection
        all_seasons = sorted(meta_df["season"].unique())
        if len(all_seasons) < 2:
            best_w = 1.0
        else:
            val_year = all_seasons[-1]
            train_meta = meta_df[meta_df["season"] < val_year]
            val_meta = meta_df[meta_df["season"] == val_year]

            # Train-set global pi (unweighted)
            train_global, _, _ = build_count_matrices(train_meta)
            train_g_alpha = self.prior_alpha_global + train_global
            train_g_pi = train_g_alpha / train_g_alpha.sum(
                axis=1, keepdims=True
            )

            # Circuit pis from train data (static, unweighted)
            if self._circuit_mode:
                train_circuit_counts = build_circuit_counts(train_meta)
                train_k_pis = {}
                for kid, kc in train_circuit_counts.items():
                    k_alpha = self.prior_alpha_circuit + kc
                    train_k_pis[kid] = k_alpha / k_alpha.sum(
                        axis=1, keepdims=True
                    )
                train_dcc_counts = (
                    build_driver_constructor_circuit_counts(train_meta)
                )
                val_dcc_counts = (
                    build_driver_constructor_circuit_counts(val_meta)
                )
            else:
                train_dc_counts = build_driver_constructor_counts(train_meta)
                val_dc_counts = build_driver_constructor_counts(val_meta)

            best_val_ll = -np.inf
            best_w = 1.0

            for w in self.w_candidates:
                # Weighted constructor pis from TRAIN data only
                weighted_c = build_weighted_constructor_counts(
                    train_meta, w
                )
                c_pis_w = {}
                for cid, wc in weighted_c.items():
                    c_alpha = self.prior_alpha_constructor + wc
                    c_pis_w[cid] = c_alpha / c_alpha.sum(
                        axis=1, keepdims=True
                    )

                if self._circuit_mode:
                    # 3-kappa optimization
                    def neg_lml(log_kappas, _c_pis=c_pis_w):
                        kg, kc, kk = np.exp(log_kappas)
                        return -total_log_marginal_likelihood_with_circuit(
                            kg, kc, kk,
                            train_g_pi, _c_pis, train_k_pis,
                            train_dcc_counts,
                        )

                    log_k0 = np.log([
                        self.kappa_init[0], self.kappa_init[1],
                        self.kappa_k_init,
                    ])
                    log_bounds = [
                        (np.log(b[0]), np.log(b[1]))
                        for b in self.kappa_bounds
                    ] + [
                        (np.log(self.kappa_k_bounds[0]),
                         np.log(self.kappa_k_bounds[1]))
                    ]
                else:
                    # 2-kappa optimization (original behavior)
                    def neg_lml(log_kappas, _c_pis=c_pis_w):
                        kg, kc = np.exp(log_kappas)
                        return -total_log_marginal_likelihood(
                            kg, kc,
                            train_g_pi, _c_pis,
                            train_dc_counts,
                        )

                    log_k0 = np.log(self.kappa_init)
                    log_bounds = [
                        (np.log(b[0]), np.log(b[1]))
                        for b in self.kappa_bounds
                    ]

                result = optimize.minimize(
                    neg_lml,
                    x0=log_k0,
                    method="L-BFGS-B",
                    bounds=log_bounds,
                    options={"maxiter": 300, "ftol": 1e-8},
                )

                if self._circuit_mode:
                    kg, kc, kk = np.exp(result.x)
                    val_ll = total_log_marginal_likelihood_with_circuit(
                        kg, kc, kk,
                        train_g_pi, c_pis_w, train_k_pis,
                        val_dcc_counts,
                    )
                else:
                    kg, kc = np.exp(result.x)
                    val_ll = total_log_marginal_likelihood(
                        kg, kc,
                        train_g_pi, c_pis_w,
                        val_dc_counts,
                    )

                if val_ll > best_val_ll:
                    best_val_ll = val_ll
                    best_w = w

        # Step 3: Retrain on ALL data with selected w
        weighted_c_final = build_weighted_constructor_counts(
            meta_df, best_w
        )
        c_pis_final = {}
        for cid, wc in weighted_c_final.items():
            c_alpha = self.prior_alpha_constructor + wc
            c_pis_final[cid] = c_alpha / c_alpha.sum(
                axis=1, keepdims=True
            )

        if self._circuit_mode:
            all_dcc_counts = (
                build_driver_constructor_circuit_counts(meta_df)
            )

            def neg_lml_final(log_kappas):
                kg, kc, kk = np.exp(log_kappas)
                return -total_log_marginal_likelihood_with_circuit(
                    kg, kc, kk,
                    self.global_pi_, c_pis_final, k_pis_all,
                    all_dcc_counts,
                )

            log_k0 = np.log([
                self.kappa_init[0], self.kappa_init[1],
                self.kappa_k_init,
            ])
            log_bounds = [
                (np.log(b[0]), np.log(b[1]))
                for b in self.kappa_bounds
            ] + [
                (np.log(self.kappa_k_bounds[0]),
                 np.log(self.kappa_k_bounds[1]))
            ]
        else:
            def neg_lml_final(log_kappas):
                kg, kc = np.exp(log_kappas)
                return -total_log_marginal_likelihood(
                    kg, kc,
                    self.global_pi_, c_pis_final,
                    self.driver_constructor_counts_,
                )

            log_k0 = np.log(self.kappa_init)
            log_bounds = [
                (np.log(b[0]), np.log(b[1])) for b in self.kappa_bounds
            ]

        result = optimize.minimize(
            neg_lml_final,
            x0=log_k0,
            method="L-BFGS-B",
            bounds=log_bounds,
            options={"maxiter": 300, "ftol": 1e-8},
        )

        if self._circuit_mode:
            self.kappa_g_, self.kappa_c_, self.kappa_k_ = np.exp(result.x)
            self.circuit_pis_ = k_pis_all
        else:
            self.kappa_g_, self.kappa_c_ = np.exp(result.x)

        self.w_ = best_w
        self.opt_result_ = result

        # Step 4: Store final constructor pis and weighted counts
        self.constructor_pis_ = c_pis_final
        self._weighted_constructor_counts_ = weighted_c_final
        self.constructor_ids_ = sorted(self.constructor_pis_.keys())
        # Store ref_year for incorporate_race (one year past training data)
        self._ref_year_ = int(meta_df["season"].max()) + 1

        return self

    # --- Prior / Posterior computation ---

    def _driver_prior_alpha_for_constructor(
        self, constructor_id, circuit_id=None,
    ) -> np.ndarray:
        """
        Prior alpha matrix for a driver racing with a given constructor
        (and optionally at a specific circuit):
            kappa_g * pi^{global} + kappa_c * pi^{(C,w)} [+ kappa_k * pi^{(K)}]
        """
        c_pi = self.constructor_pis_.get(constructor_id, self.global_pi_)
        alpha = self.kappa_g_ * self.global_pi_ + self.kappa_c_ * c_pi
        if circuit_id is not None and self._circuit_mode:
            k_pi = self.circuit_pis_.get(circuit_id, self.global_pi_)
            alpha = alpha + self.kappa_k_ * k_pi
        return alpha

    def driver_posterior_alpha(
        self,
        driver_id,
        constructor_id: Optional = None,
        circuit_id=None,
    ) -> np.ndarray:
        """Posterior Dirichlet alpha for a driver at a constructor."""
        self._check_fitted()
        if constructor_id is None:
            constructor_id = self.driver_latest_constructor_.get(driver_id)

        prior = self._driver_prior_alpha_for_constructor(
            constructor_id, circuit_id
        )

        dc_counts = self.driver_constructor_counts_.get(driver_id, {})
        counts = dc_counts.get(
            constructor_id, np.zeros((N_PREV_STATES, N_OUTCOMES))
        )
        return prior + counts

    def driver_transition_matrix(
        self, driver_id, constructor_id: Optional = None,
        circuit_id=None,
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
        circuit_id=None,
        **kwargs,
    ) -> np.ndarray:
        """P(next | prev, driver, constructor [, circuit])."""
        return self.driver_transition_matrix(
            driver_id, constructor_id, circuit_id
        )[prev_position]

    def predict_proba_new_driver(
        self,
        prev_position: int,
        constructor_id: Optional = None,
        circuit_id=None,
        **kwargs,
    ) -> np.ndarray:
        """Prediction for an unseen driver."""
        self._check_fitted()
        if constructor_id is not None:
            alpha = self._driver_prior_alpha_for_constructor(
                constructor_id, circuit_id
            )
        else:
            alpha = self.kappa_g_ * self.global_pi_
            if circuit_id is not None and self._circuit_mode:
                k_pi = self.circuit_pis_.get(circuit_id, self.global_pi_)
                alpha = alpha + self.kappa_k_ * k_pi
        row = alpha[prev_position]
        return row / row.sum()

    # --- In-season update ---

    def incorporate_race(self, race_results, season_year):
        """
        Update model state with observed race results (no hyperparameter refit).

        Adds observations to global counts, driver-constructor counts,
        and weighted constructor counts. Recomputes the affected pis.

        Parameters
        ----------
        race_results : list of (driver_id, constructor_id, prev_position, finish_position)
        season_year : int
            Year of the observed race (e.g. 2026).
        """
        self._check_fitted()
        weight = self.w_ ** (self._ref_year_ - season_year)

        for did, cid, prev_pos, finish_pos in race_results:
            # Clamp positions to valid range
            prev_pos = min(prev_pos, N_PREV_STATES - 1)
            finish_pos = min(finish_pos, N_OUTCOMES - 1)

            # Update global counts and pi
            self.global_counts_[prev_pos, finish_pos] += 1

            # Update driver-constructor counts
            if did not in self.driver_constructor_counts_:
                self.driver_constructor_counts_[did] = {}
            if cid not in self.driver_constructor_counts_[did]:
                self.driver_constructor_counts_[did][cid] = np.zeros(
                    (N_PREV_STATES, N_OUTCOMES), dtype=int
                )
            self.driver_constructor_counts_[did][cid][prev_pos, finish_pos] += 1

            # Update weighted constructor counts
            if cid not in self._weighted_constructor_counts_:
                self._weighted_constructor_counts_[cid] = np.zeros(
                    (N_PREV_STATES, N_OUTCOMES), dtype=float
                )
            self._weighted_constructor_counts_[cid][prev_pos, finish_pos] += weight

            # Track latest constructor
            self.driver_latest_constructor_[did] = cid

        # Recompute global pi
        g_alpha = self.prior_alpha_global + self.global_counts_
        self.global_pi_ = g_alpha / g_alpha.sum(axis=1, keepdims=True)

        # Recompute affected constructor pis
        affected_cids = set(cid for _, cid, _, _ in race_results)
        for cid in affected_cids:
            wc = self._weighted_constructor_counts_[cid]
            c_alpha = self.prior_alpha_constructor + wc
            self.constructor_pis_[cid] = c_alpha / c_alpha.sum(
                axis=1, keepdims=True
            )

    # --- Diagnostics ---

    def pooling_factors(self, driver_id, constructor_id=None) -> dict:
        """Decompose effective weight of each prior component."""
        self._check_fitted()
        if constructor_id is None:
            constructor_id = self.driver_latest_constructor_.get(driver_id)

        w_g = self.kappa_g_ * N_OUTCOMES
        w_c = self.kappa_c_ * N_OUTCOMES
        w_k = (self.kappa_k_ * N_OUTCOMES) if self._circuit_mode else 0.0

        dc = self.driver_constructor_counts_.get(driver_id, {})
        n = dc.get(
            constructor_id, np.zeros((N_PREV_STATES, N_OUTCOMES))
        ).sum()
        n_per_row = n / max(1, N_PREV_STATES)

        total = w_g + w_c + w_k + n_per_row
        result = {
            "global_weight": round(w_g / total, 3),
            "constructor_weight": round(w_c / total, 3),
            "data_weight": round(n_per_row / total, 3),
        }
        if self._circuit_mode:
            result["circuit_weight"] = round(w_k / total, 3)
        return result

    def log_marginal_likelihood(self) -> float:
        """Total log marginal likelihood at fitted parameters."""
        self._check_fitted()
        if self._circuit_mode:
            dcc_counts = build_driver_constructor_circuit_counts(
                self._meta_df
            )
            return total_log_marginal_likelihood_with_circuit(
                self.kappa_g_, self.kappa_c_, self.kappa_k_,
                self.global_pi_, self.constructor_pis_, self.circuit_pis_,
                dcc_counts,
            )
        return total_log_marginal_likelihood(
            self.kappa_g_, self.kappa_c_,
            self.global_pi_, self.constructor_pis_,
            self.driver_constructor_counts_,
        )

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")
