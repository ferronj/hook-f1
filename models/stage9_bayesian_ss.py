"""
F1 Bayesian State-Space Model
==============================
Stage 9: Random walk on driver + constructor latent strengths with PyMC

Architecture:
    - Driver strength: mu_i(t) ~ N(mu_i(t-1), sigma_driver^2)
    - Constructor effect: beta_C(t) ~ N(beta_C(t-1), sigma_constr^2)
    - Composite: log_lambda_i(t) = mu_i(t) + beta_C(t)
    - Observation model: Plackett-Luce ranking among finishers
    - DNF modeled separately (empirical with shrinkage, same as Stage 8)
    - Inference: MAP via L-BFGS-B with analytic gradients over full trajectory

This model generalizes Stage 8's exponential smoothing to a proper
state-space formulation. Key differences from Stage 8:
    - Joint optimization over all time steps (not greedy sequential)
    - Innovation variance (sigma_d, sigma_c) learned from data via CV
    - Gap-aware random walk (missed races scale the variance)
    - Better calibration expected (random walk prior prevents overconcentration)

Prediction interface matches Stages 1-8:
    predict_proba(driver_id, prev_position, constructor_id) -> (21,) array

Data source: Same as Stages 1-8 (Kaggle F1 World Championship dataset)
"""

import numpy as np
import pandas as pd
import pickle
from scipy import optimize
from scipy.special import logsumexp
from typing import Optional
from collections import defaultdict
from pathlib import Path

from stage3_constructor import (
    F1DataLoader,
    prepare_transitions,
    build_count_matrices,
    START, N_OUTCOMES, N_PREV_STATES,
    DNF, OUTCOME_LABELS, PREV_STATE_LABELS,
)
from stage8_plackett_luce import (
    prepare_race_rankings,
    pl_log_likelihood,
    pl_sample_ranking,
)


# ---------------------------------------------------------------------------
# Plackett-Luce log-likelihood with analytic gradient
# ---------------------------------------------------------------------------
def pl_log_lik_and_grad(z, ranking_indices):
    """
    PL log-likelihood and gradient w.r.t. log-strengths for one race.

    Parameters
    ----------
    z : (n_participants,) log-strengths for all participants in this race
    ranking_indices : list of indices into z, in finishing order (1st to last)

    Returns
    -------
    ll : float
    grad : (n_participants,) gradient of LL w.r.t. z
    """
    n = len(ranking_indices)
    grad = np.zeros_like(z)
    ll = 0.0

    # Precompute suffix log-sum-exps for numerical stability
    # suffix_lse[j] = log(sum_{k=j..n-1} exp(z[ranking[k]]))
    suffix_lse = np.full(n + 1, -np.inf)
    for j in range(n - 1, -1, -1):
        suffix_lse[j] = np.logaddexp(suffix_lse[j + 1], z[ranking_indices[j]])

    # Forward pass
    for j in range(n):
        idx = ranking_indices[j]
        ll += z[idx] - suffix_lse[j]

        # Gradient: +1 for the chosen driver at position j
        grad[idx] += 1.0

    # Subtract softmax contributions for denominator terms
    for j in range(n):
        denom = suffix_lse[j]
        for k in range(j, n):
            idx_k = ranking_indices[k]
            grad[idx_k] -= np.exp(z[idx_k] - denom)

    return ll, grad


# ---------------------------------------------------------------------------
# Parameter indexing for the flat optimization vector
# ---------------------------------------------------------------------------
class ParamIndex:
    """Maps (entity, race) pairs to indices in the flat parameter vector."""

    def __init__(self, races):
        """
        Build index from race-level data.

        Parameters
        ----------
        races : list of dicts from prepare_race_rankings()
        """
        self.n_params = 0

        # driver_id -> list of (global_race_idx, param_offset)
        self.driver_indices = defaultdict(list)
        # constructor_id -> list of (global_race_idx, param_offset)
        self.constructor_indices = defaultdict(list)
        # global_race_idx -> list of (driver_id, cid, d_offset, c_offset, finish_pos, is_dnf)
        self.race_info = {}

        # Assign race indices in chronological order
        for race_idx, race in enumerate(races):
            drivers_in_race = set()
            constructors_in_race = set()
            for did, cid, pos, is_dnf in race["entries"]:
                drivers_in_race.add(did)
                constructors_in_race.add(cid)

            # Allocate driver params for this race
            driver_offsets = {}
            for did in sorted(drivers_in_race):
                driver_offsets[did] = self.n_params
                self.driver_indices[did].append((race_idx, self.n_params))
                self.n_params += 1

            # Allocate constructor params for this race
            constructor_offsets = {}
            for cid in sorted(constructors_in_race):
                constructor_offsets[cid] = self.n_params
                self.constructor_indices[cid].append((race_idx, self.n_params))
                self.n_params += 1

            # Store race info
            entries = []
            for did, cid, pos, is_dnf in race["entries"]:
                entries.append((
                    did, cid,
                    driver_offsets[did],
                    constructor_offsets[cid],
                    pos, is_dnf,
                ))
            self.race_info[race_idx] = entries

        self.n_races = len(races)
        self.driver_ids = sorted(self.driver_indices.keys())
        self.constructor_ids = sorted(self.constructor_indices.keys())


# ---------------------------------------------------------------------------
# Negative log-posterior (objective for L-BFGS-B)
# ---------------------------------------------------------------------------
def neg_log_posterior(theta, pidx, sigma_d, sigma_c, sigma_0, center_penalty):
    """
    Compute -log P(theta | data) and its gradient.

    Components:
        1. PL log-likelihood for each race (among finishers)
        2. Random walk prior on driver strengths
        3. Random walk prior on constructor strengths
        4. Initial prior N(0, sigma_0^2)
        5. Centering penalty for identifiability

    Returns
    -------
    neg_lp : float
    neg_grad : (n_params,) gradient
    """
    neg_lp = 0.0
    neg_grad = np.zeros_like(theta)

    # --- 1. PL observation log-likelihood ---
    for race_idx in range(pidx.n_races):
        entries = pidx.race_info[race_idx]

        # Extract finishers (non-DNF) in finishing order
        finishers = [
            (did, cid, d_off, c_off, pos)
            for did, cid, d_off, c_off, pos, is_dnf in entries
            if not is_dnf
        ]
        if len(finishers) < 2:
            continue

        finishers.sort(key=lambda x: x[4])  # sort by finish position

        # Build composite log-strengths: z_i = mu_i + beta_C
        n_fin = len(finishers)
        z = np.zeros(n_fin)
        d_offsets = []
        c_offsets = []
        for k, (did, cid, d_off, c_off, pos) in enumerate(finishers):
            z[k] = theta[d_off] + theta[c_off]
            d_offsets.append(d_off)
            c_offsets.append(c_off)

        # PL log-lik and gradient w.r.t. z
        ranking = list(range(n_fin))  # already sorted by position
        ll, grad_z = pl_log_lik_and_grad(z, ranking)
        neg_lp -= ll

        # Chain rule: dLL/d(mu_i) = dLL/dz_i, dLL/d(beta_C) = dLL/dz_i
        for k in range(n_fin):
            neg_grad[d_offsets[k]] -= grad_z[k]
            neg_grad[c_offsets[k]] -= grad_z[k]

    # --- 2 & 3. Random walk priors ---
    inv_var_d = 1.0 / (sigma_d ** 2)
    inv_var_c = 1.0 / (sigma_c ** 2)
    inv_var_0 = 1.0 / (sigma_0 ** 2)

    # Driver random walk
    for did in pidx.driver_ids:
        appearances = pidx.driver_indices[did]  # (race_idx, param_offset)
        # Initial prior
        _, off0 = appearances[0]
        neg_lp += 0.5 * inv_var_0 * theta[off0] ** 2
        neg_grad[off0] += inv_var_0 * theta[off0]

        # Transitions
        for t in range(1, len(appearances)):
            race_prev, off_prev = appearances[t - 1]
            race_curr, off_curr = appearances[t]
            gap = race_curr - race_prev  # number of races between appearances
            diff = theta[off_curr] - theta[off_prev]
            inv_var_step = inv_var_d / gap
            neg_lp += 0.5 * inv_var_step * diff ** 2
            neg_grad[off_curr] += inv_var_step * diff
            neg_grad[off_prev] -= inv_var_step * diff

    # Constructor random walk
    for cid in pidx.constructor_ids:
        appearances = pidx.constructor_indices[cid]
        _, off0 = appearances[0]
        neg_lp += 0.5 * inv_var_0 * theta[off0] ** 2
        neg_grad[off0] += inv_var_0 * theta[off0]

        for t in range(1, len(appearances)):
            race_prev, off_prev = appearances[t - 1]
            race_curr, off_curr = appearances[t]
            gap = race_curr - race_prev
            diff = theta[off_curr] - theta[off_prev]
            inv_var_step = inv_var_c / gap
            neg_lp += 0.5 * inv_var_step * diff ** 2
            neg_grad[off_curr] += inv_var_step * diff
            neg_grad[off_prev] -= inv_var_step * diff

    # --- 5. Centering penalty for identifiability ---
    # Penalize the mean of driver log-strengths at each race
    if center_penalty > 0:
        for race_idx in range(pidx.n_races):
            entries = pidx.race_info[race_idx]
            d_offs = [d_off for _, _, d_off, _, _, _ in entries]
            if len(d_offs) == 0:
                continue
            mean_mu = sum(theta[o] for o in d_offs) / len(d_offs)
            neg_lp += 0.5 * center_penalty * mean_mu ** 2
            grad_mean = center_penalty * mean_mu / len(d_offs)
            for o in d_offs:
                neg_grad[o] += grad_mean

    return neg_lp, neg_grad


# ---------------------------------------------------------------------------
# Stage 9 Model
# ---------------------------------------------------------------------------
class BayesianStateSpaceF1:
    """
    Stage 9: Bayesian state-space model with MAP inference.

    Random walk on driver and constructor log-strengths, observed via
    Plackett-Luce rankings. MAP estimation via L-BFGS-B with analytic
    gradients over the full trajectory.

    Parameters
    ----------
    sigma_d_candidates : tuple of floats
        Innovation SD candidates for driver random walk (CV-selected).
    sigma_c_candidates : tuple of floats
        Innovation SD candidates for constructor random walk (CV-selected).
    sigma_0 : float
        Initial prior SD for log-strengths.
    dnf_shrinkage : float
        Shrinkage strength for DNF rate estimation.
    n_mc_samples : int
        MC samples for position marginal computation.
    center_penalty : float
        Soft centering penalty for identifiability.
    maxiter : int
        Maximum L-BFGS-B iterations.
    """

    def __init__(
        self,
        sigma_d_candidates: tuple = (0.02, 0.05, 0.1, 0.2),
        sigma_c_candidates: tuple = (0.02, 0.05, 0.1, 0.2),
        sigma_0: float = 1.0,
        dnf_shrinkage: float = 20.0,
        n_mc_samples: int = 3000,
        center_penalty: float = 0.01,
        maxiter: int = 500,
    ):
        self.sigma_d_candidates = sigma_d_candidates
        self.sigma_c_candidates = sigma_c_candidates
        self.sigma_0 = sigma_0
        self.dnf_shrinkage = dnf_shrinkage
        self.n_mc_samples = n_mc_samples
        self.center_penalty = center_penalty
        self.maxiter = maxiter

        # Fitted attributes
        self.driver_strengths_: Optional[dict] = None  # did -> float (exp-space)
        self.constructor_strengths_: Optional[dict] = None  # cid -> float
        self.dnf_rates_: Optional[dict] = None
        self.global_dnf_rate_: Optional[float] = None
        self.sigma_d_: Optional[float] = None
        self.sigma_c_: Optional[float] = None
        self.global_correction_: Optional[np.ndarray] = None  # (22, 21)
        self.driver_ids_: Optional[list] = None
        self.constructor_ids_: Optional[list] = None

    @property
    def is_fitted(self) -> bool:
        return self.driver_strengths_ is not None

    def fit(
        self,
        prev_positions: np.ndarray,
        next_positions: np.ndarray,
        meta_df: pd.DataFrame,
    ) -> "BayesianStateSpaceF1":
        """
        Fit the Bayesian state-space model:
          1. Build race-level rankings
          2. Select (sigma_d, sigma_c) via leave-last-year-out CV
          3. Fit MAP on full data with best hyperparameters
          4. Extract final strengths
          5. Compute DNF rates and Markov correction
        """
        races = prepare_race_rankings(meta_df)
        years = sorted(meta_df["season"].unique())

        self.driver_ids_ = sorted(meta_df["driver"].unique().tolist())
        self.constructor_ids_ = sorted(meta_df["constructor"].unique().tolist())

        # Global transition matrix for Markov correction
        global_counts = np.zeros((N_PREV_STATES, N_OUTCOMES), dtype=int)
        for s, j in zip(prev_positions, next_positions):
            global_counts[s, j] += 1
        g_alpha = 0.5 + global_counts
        global_pi = g_alpha / g_alpha.sum(axis=1, keepdims=True)

        # DNF rates
        self._compute_dnf_rates(meta_df)

        # Leave-last-year-out CV for (sigma_d, sigma_c)
        best_sigma_d, best_sigma_c = self.sigma_d_candidates[1], self.sigma_c_candidates[1]
        if len(years) >= 2:
            val_year = years[-1]
            train_races = [r for r in races if r["season"] < val_year]
            val_races = [r for r in races if r["season"] == val_year]

            print(f"  Stage 9 (BSS): CV with {len(train_races)} train races, "
                  f"{len(val_races)} val races (year {val_year})")

            best_val_ll = -np.inf
            for sigma_d in self.sigma_d_candidates:
                for sigma_c in self.sigma_c_candidates:
                    # Fit MAP on training data
                    theta, pidx = self._fit_map(train_races, sigma_d, sigma_c)

                    # Evaluate on validation races using final train strengths
                    val_ll = self._eval_validation(
                        theta, pidx, train_races, val_races
                    )

                    if val_ll > best_val_ll:
                        best_val_ll = val_ll
                        best_sigma_d = sigma_d
                        best_sigma_c = sigma_c

            print(f"  Selected sigma_d={best_sigma_d:.3f}, "
                  f"sigma_c={best_sigma_c:.3f} (val LL={best_val_ll:.1f})")

        self.sigma_d_ = best_sigma_d
        self.sigma_c_ = best_sigma_c

        # Final fit on all data
        theta_final, pidx_final = self._fit_map(
            races, self.sigma_d_, self.sigma_c_
        )

        # Store trajectory state for incorporate_race()
        self._theta_ = theta_final
        self._pidx_ = pidx_final
        self._races_ = races

        # Extract final log-strengths (last appearance of each entity)
        self.driver_strengths_ = {}
        for did in pidx_final.driver_ids:
            appearances = pidx_final.driver_indices[did]
            _, last_off = appearances[-1]
            self.driver_strengths_[did] = np.exp(theta_final[last_off])

        self.constructor_strengths_ = {}
        for cid in pidx_final.constructor_ids:
            appearances = pidx_final.constructor_indices[cid]
            _, last_off = appearances[-1]
            self.constructor_strengths_[cid] = np.exp(theta_final[last_off])

        # Markov correction
        marginal = global_pi.mean(axis=0)
        self.global_correction_ = np.zeros((N_PREV_STATES, N_OUTCOMES))
        for s in range(N_PREV_STATES):
            for k in range(N_OUTCOMES):
                if marginal[k] > 1e-10:
                    self.global_correction_[s, k] = global_pi[s, k] / marginal[k]
                else:
                    self.global_correction_[s, k] = 1.0

        # Print summary
        print(f"  Stage 9 (BSS): fitted {len(self.driver_strengths_)} drivers, "
              f"{len(self.constructor_strengths_)} constructors")
        sorted_drivers = sorted(
            self.driver_strengths_.items(),
            key=lambda x: x[1], reverse=True
        )[:5]
        for did, strength in sorted_drivers:
            print(f"    Driver {did}: strength = {strength:.3f}")

        return self

    def _fit_map(self, races, sigma_d, sigma_c, theta0=None):
        """Run MAP optimization via L-BFGS-B."""
        pidx = ParamIndex(races)
        if theta0 is None:
            theta0 = np.zeros(pidx.n_params)

        result = optimize.minimize(
            neg_log_posterior,
            theta0,
            args=(pidx, sigma_d, sigma_c, self.sigma_0, self.center_penalty),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": self.maxiter, "ftol": 1e-10, "gtol": 1e-6},
        )
        return result.x, pidx

    def _eval_validation(self, theta, pidx, train_races, val_races):
        """Evaluate PL log-likelihood on validation races using final MAP
        strengths from training."""
        # Extract final log-strengths from training
        driver_log_str = {}
        for did in pidx.driver_ids:
            appearances = pidx.driver_indices[did]
            _, last_off = appearances[-1]
            driver_log_str[did] = theta[last_off]

        constructor_log_str = {}
        for cid in pidx.constructor_ids:
            appearances = pidx.constructor_indices[cid]
            _, last_off = appearances[-1]
            constructor_log_str[cid] = theta[last_off]

        val_ll = 0.0
        for race in val_races:
            finishers = [
                (did, cid, pos)
                for did, cid, pos, is_dnf in race["entries"]
                if not is_dnf
            ]
            if len(finishers) < 2:
                continue

            finishers.sort(key=lambda x: x[2])
            ranking = [f[0] for f in finishers]

            # Build composite strengths in exp-space for pl_log_likelihood
            race_strengths = {}
            for did, cid, pos in finishers:
                log_d = driver_log_str.get(did, 0.0)
                log_c = constructor_log_str.get(cid, 0.0)
                race_strengths[did] = np.exp(log_d + log_c)

            val_ll += pl_log_likelihood(race_strengths, ranking)

        return val_ll

    def _compute_dnf_rates(self, meta_df):
        """Compute per-driver DNF rates with shrinkage (same as Stage 8)."""
        total_races = defaultdict(int)
        total_dnfs = defaultdict(int)
        global_races = 0
        global_dnfs = 0

        for _, row in meta_df.iterrows():
            did = row["driver"]
            total_races[did] += 1
            global_races += 1
            if int(row["next_position"]) == DNF:
                total_dnfs[did] += 1
                global_dnfs += 1

        self.global_dnf_rate_ = global_dnfs / max(global_races, 1)
        tau = self.dnf_shrinkage

        # Store raw counters for incremental updates
        self._dnf_total_races_ = dict(total_races)
        self._dnf_total_dnfs_ = dict(total_dnfs)
        self._dnf_global_races_ = global_races
        self._dnf_global_dnfs_ = global_dnfs

        self.dnf_rates_ = {}
        for did in total_races:
            n = total_races[did]
            empirical = total_dnfs[did] / max(n, 1)
            weight = n / (n + tau)
            self.dnf_rates_[did] = (
                weight * empirical + (1 - weight) * self.global_dnf_rate_
            )

    # --- In-season update ---

    def incorporate_race(self, race_results):
        """
        Extend trajectory with observed race, re-run MAP with fixed sigma_d/c.

        Warm-starts from previous MAP solution for fast convergence.

        Parameters
        ----------
        race_results : list of (driver_id, constructor_id, finish_position, is_dnf)
        """
        self._check_fitted()

        # Convert to race dict format
        max_race_order = max(r["race_order"] for r in self._races_) + 1
        new_race = {
            "season": 2026,
            "race_order": max_race_order,
            "entries": [
                (did, cid, pos, is_dnf)
                for did, cid, pos, is_dnf in race_results
            ],
        }
        self._races_.append(new_race)

        # Build new ParamIndex
        new_pidx = ParamIndex(self._races_)

        # Build warm-start theta: map old params to new layout
        theta0 = np.zeros(new_pidx.n_params)
        old_pidx = self._pidx_

        # Copy driver params from old solution
        for did in new_pidx.driver_ids:
            new_appearances = new_pidx.driver_indices[did]
            if did in old_pidx.driver_indices:
                old_appearances = old_pidx.driver_indices[did]
                # Map overlapping race indices
                old_by_race = {ri: off for ri, off in old_appearances}
                last_old_val = self._theta_[old_appearances[-1][1]]
                for ri, new_off in new_appearances:
                    if ri in old_by_race:
                        theta0[new_off] = self._theta_[old_by_race[ri]]
                    else:
                        # New race: init from last known value
                        theta0[new_off] = last_old_val
            # else: new driver, leave at 0.0

        # Copy constructor params
        for cid in new_pidx.constructor_ids:
            new_appearances = new_pidx.constructor_indices[cid]
            if cid in old_pidx.constructor_indices:
                old_appearances = old_pidx.constructor_indices[cid]
                old_by_race = {ri: off for ri, off in old_appearances}
                last_old_val = self._theta_[old_appearances[-1][1]]
                for ri, new_off in new_appearances:
                    if ri in old_by_race:
                        theta0[new_off] = self._theta_[old_by_race[ri]]
                    else:
                        theta0[new_off] = last_old_val
            # else: new constructor, leave at 0.0

        # Re-run MAP with warm start
        theta_new, pidx_new = self._fit_map(
            self._races_, self.sigma_d_, self.sigma_c_, theta0=theta0
        )

        # Update stored state
        self._theta_ = theta_new
        self._pidx_ = pidx_new

        # Extract updated final strengths
        self.driver_strengths_ = {}
        for did in pidx_new.driver_ids:
            appearances = pidx_new.driver_indices[did]
            _, last_off = appearances[-1]
            self.driver_strengths_[did] = np.exp(theta_new[last_off])

        self.constructor_strengths_ = {}
        for cid in pidx_new.constructor_ids:
            appearances = pidx_new.constructor_indices[cid]
            _, last_off = appearances[-1]
            self.constructor_strengths_[cid] = np.exp(theta_new[last_off])

        # Update DNF rates incrementally
        for did, cid, pos, is_dnf in race_results:
            n_prev = self._dnf_total_races_.get(did, 0)
            n_dnf_prev = self._dnf_total_dnfs_.get(did, 0)
            self._dnf_total_races_[did] = n_prev + 1
            if is_dnf:
                self._dnf_total_dnfs_[did] = n_dnf_prev + 1
                self._dnf_global_dnfs_ += 1
            self._dnf_global_races_ += 1

        self.global_dnf_rate_ = (
            self._dnf_global_dnfs_ / max(self._dnf_global_races_, 1)
        )
        tau = self.dnf_shrinkage
        for did in self._dnf_total_races_:
            n = self._dnf_total_races_[did]
            empirical = self._dnf_total_dnfs_.get(did, 0) / max(n, 1)
            weight = n / (n + tau)
            self.dnf_rates_[did] = (
                weight * empirical + (1 - weight) * self.global_dnf_rate_
            )

    def predict_proba(
        self,
        driver_id,
        prev_position: int,
        constructor_id: Optional = None,
        **kwargs,
    ) -> np.ndarray:
        """P(next | prev, driver, constructor)."""
        self._check_fitted()

        d_str = self.driver_strengths_.get(driver_id, 1.0)
        c_str = 1.0
        if constructor_id is not None:
            c_str = self.constructor_strengths_.get(constructor_id, 1.0)
        composite = d_str * c_str

        p_dnf = self.dnf_rates_.get(driver_id, self.global_dnf_rate_)
        return self._compute_position_probs(
            driver_id, composite, p_dnf, prev_position
        )

    def predict_proba_new_driver(
        self,
        prev_position: int,
        constructor_id: Optional = None,
        **kwargs,
    ) -> np.ndarray:
        """Prediction for unseen driver."""
        self._check_fitted()

        d_str = 1.0
        c_str = 1.0
        if constructor_id is not None:
            c_str = self.constructor_strengths_.get(constructor_id, 1.0)
        composite = d_str * c_str

        p_dnf = self.global_dnf_rate_
        return self._compute_position_probs(
            None, composite, p_dnf, prev_position
        )

    def _compute_position_probs(
        self, driver_id, composite_strength, p_dnf, prev_position
    ):
        """
        Compute (21,) probability array via MC sampling of PL rankings.
        Same approach as Stage 8.
        """
        n_field = 20
        all_strengths = list(self.driver_strengths_.values())

        if len(all_strengths) == 0:
            all_strengths = [1.0] * n_field

        sorted_strengths = sorted(all_strengths, reverse=True)
        field_strengths = sorted_strengths[:n_field - 1]
        while len(field_strengths) < n_field - 1:
            field_strengths.append(np.median(all_strengths))

        all_field = np.array([composite_strength] + field_strengths)
        all_field = np.maximum(all_field, 1e-10)

        rng = np.random.default_rng(
            hash(str(driver_id)) % (2**31) if driver_id else 42
        )
        pos_probs_matrix = pl_sample_ranking(all_field, rng, self.n_mc_samples)
        finish_probs = pos_probs_matrix[0]

        probs = np.zeros(N_OUTCOMES)
        probs[0] = p_dnf

        remaining_prob = 1.0 - p_dnf
        for j in range(min(len(finish_probs), 20)):
            probs[j + 1] = remaining_prob * finish_probs[j]

        if self.global_correction_ is not None:
            correction = self.global_correction_[prev_position]
            probs *= correction
            probs = np.maximum(probs, 1e-10)
            probs /= probs.sum()

        return probs

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")

    # ===================================================================
    # NUTS (Full Bayesian) methods
    # ===================================================================

    def _precompute_race_arrays(self, pidx):
        """
        Pre-compute padded index arrays for vectorized pytensor PL log-lik.

        Returns arrays of shape (n_races, max_finishers):
            d_offset_arr, c_offset_arr, n_finishers_arr
        """
        max_fin = 20
        n_races = pidx.n_races

        d_offset_arr = np.zeros((n_races, max_fin), dtype=np.int64)
        c_offset_arr = np.zeros((n_races, max_fin), dtype=np.int64)
        n_finishers_arr = np.zeros(n_races, dtype=np.int64)

        for race_idx in range(n_races):
            entries = pidx.race_info[race_idx]
            finishers = [
                (d_off, c_off, pos)
                for _, _, d_off, c_off, pos, is_dnf in entries
                if not is_dnf
            ]
            if len(finishers) < 2:
                n_finishers_arr[race_idx] = 0
                continue

            finishers.sort(key=lambda x: x[2])
            n_fin = min(len(finishers), max_fin)
            n_finishers_arr[race_idx] = n_fin

            for k in range(n_fin):
                d_offset_arr[race_idx, k] = finishers[k][0]
                c_offset_arr[race_idx, k] = finishers[k][1]

        return d_offset_arr, c_offset_arr, n_finishers_arr

    def _precompute_rw_arrays_vectorized(self, pidx):
        """
        Pre-compute flat arrays for gap-aware random walk priors.

        Returns dict with:
            init_offsets_d, init_offsets_c : (n_entities,) initial param offsets
            rw_prev_d, rw_curr_d, rw_inv_gap_d : (n_transitions,) flat arrays
            rw_prev_c, rw_curr_c, rw_inv_gap_c : (n_transitions,) flat arrays
        """
        # Driver initial offsets and transitions
        init_d = []
        prev_d, curr_d, inv_gap_d = [], [], []
        for did in pidx.driver_ids:
            appearances = pidx.driver_indices[did]
            _, off0 = appearances[0]
            init_d.append(off0)
            for t in range(1, len(appearances)):
                race_prev, off_prev = appearances[t - 1]
                race_curr, off_curr = appearances[t]
                gap = race_curr - race_prev
                prev_d.append(off_prev)
                curr_d.append(off_curr)
                inv_gap_d.append(1.0 / gap)

        # Constructor initial offsets and transitions
        init_c = []
        prev_c, curr_c, inv_gap_c = [], [], []
        for cid in pidx.constructor_ids:
            appearances = pidx.constructor_indices[cid]
            _, off0 = appearances[0]
            init_c.append(off0)
            for t in range(1, len(appearances)):
                race_prev, off_prev = appearances[t - 1]
                race_curr, off_curr = appearances[t]
                gap = race_curr - race_prev
                prev_c.append(off_prev)
                curr_c.append(off_curr)
                inv_gap_c.append(1.0 / gap)

        return {
            'init_d': np.array(init_d, dtype=np.int64),
            'init_c': np.array(init_c, dtype=np.int64),
            'rw_prev_d': np.array(prev_d, dtype=np.int64),
            'rw_curr_d': np.array(curr_d, dtype=np.int64),
            'rw_inv_gap_d': np.array(inv_gap_d, dtype=np.float64),
            'rw_prev_c': np.array(prev_c, dtype=np.int64),
            'rw_curr_c': np.array(curr_c, dtype=np.int64),
            'rw_inv_gap_c': np.array(inv_gap_c, dtype=np.float64),
        }

    def _precompute_center_arrays_flat(self, pidx):
        """
        Pre-compute flat arrays for centering penalty.

        Returns:
            race_ids : (total_entries,) race index for each driver offset
            d_offs : (total_entries,) driver param offsets
            n_drivers_per_race : (n_races,) driver count per race
        """
        race_ids_list, d_offs_list = [], []
        n_drivers_per_race = np.zeros(pidx.n_races, dtype=np.int64)
        for race_idx in range(pidx.n_races):
            entries = pidx.race_info[race_idx]
            offs = [d_off for _, _, d_off, _, _, _ in entries]
            n_drivers_per_race[race_idx] = len(offs)
            for off in offs:
                race_ids_list.append(race_idx)
                d_offs_list.append(off)

        return {
            'race_ids': np.array(race_ids_list, dtype=np.int64),
            'd_offs': np.array(d_offs_list, dtype=np.int64),
            'n_drivers': n_drivers_per_race,
        }

    def _build_pymc_model(self, pidx, sigma_d, sigma_c, theta_init):
        """
        Build a PyMC model matching the MAP objective.

        Uses fully vectorized pytensor operations (no Python for-loops in
        the computation graph) to avoid slow graph compilation.

        The model uses a flat Normal prior + Potentials for:
          - PL observation log-likelihood (vectorized per-race)
          - Gap-aware random walk prior (vectorized)
          - Centering penalty (vectorized)

        Parameters
        ----------
        pidx : ParamIndex
        sigma_d, sigma_c : float (selected via CV)
        theta_init : (n_params,) MAP solution for warm start

        Returns
        -------
        pm.Model
        """
        import pymc as pm
        import pytensor
        import pytensor.tensor as pt

        # Pre-compute all index arrays as numpy
        d_off_arr, c_off_arr, n_fin_arr = self._precompute_race_arrays(pidx)
        rw = self._precompute_rw_arrays_vectorized(pidx)
        center = self._precompute_center_arrays_flat(pidx)

        inv_var_d = 1.0 / (sigma_d ** 2)
        inv_var_c = 1.0 / (sigma_c ** 2)
        inv_var_0 = 1.0 / (self.sigma_0 ** 2)

        # Filter to valid races (>=2 finishers) and build flat PL arrays
        valid_mask = n_fin_arr >= 2
        valid_races = np.where(valid_mask)[0]

        # Build flat arrays for PL: for each valid race, store
        # the driver+constructor offsets in finish order
        # We'll compute PL per race using scan over valid races
        max_fin = 20
        n_valid = len(valid_races)
        pl_d_offsets = np.zeros((n_valid, max_fin), dtype=np.int64)
        pl_c_offsets = np.zeros((n_valid, max_fin), dtype=np.int64)
        pl_n_fin = np.zeros(n_valid, dtype=np.int64)
        pl_mask = np.zeros((n_valid, max_fin), dtype=np.float64)

        for i, ri in enumerate(valid_races):
            nf = n_fin_arr[ri]
            pl_d_offsets[i, :nf] = d_off_arr[ri, :nf]
            pl_c_offsets[i, :nf] = c_off_arr[ri, :nf]
            pl_n_fin[i] = nf
            pl_mask[i, :nf] = 1.0

        print(f"    {n_valid} valid races (≥2 finishers), "
              f"max field size {pl_n_fin.max()}")
        print(f"    {len(rw['rw_prev_d'])} driver RW transitions, "
              f"{len(rw['rw_prev_c'])} constructor RW transitions")

        with pm.Model() as model:
            # Flat parameter vector — same layout as MAP
            theta = pm.Flat('theta', shape=pidx.n_params,
                            initval=theta_init)

            # --- PL observation log-likelihood (vectorized with scan) ---
            # Shared data (constant during sampling)
            pl_d_sh = pt.as_tensor_variable(pl_d_offsets)
            pl_c_sh = pt.as_tensor_variable(pl_c_offsets)
            pl_mask_sh = pt.as_tensor_variable(pl_mask)
            pl_nfin_sh = pt.as_tensor_variable(pl_n_fin)

            def pl_one_race(d_offs, c_offs, mask, n_fin, theta_):
                """PL log-likelihood for one race (padded to max_fin)."""
                # Composite log-strengths (padded positions get 0 but masked)
                z = theta_[d_offs] + theta_[c_offs]
                # Mask: set padded entries to -inf so they don't contribute
                z_masked = pt.switch(pt.eq(mask, 1.0), z,
                                     pt.constant(-1e30))

                # Suffix logsumexp: for position k, lse = log(sum(exp(z[k:])))
                # Compute via reversed cumsum
                z_rev = z_masked[::-1]
                max_z = pt.max(z_rev)
                shifted = z_rev - max_z
                cumsum_exp = pt.cumsum(pt.exp(shifted))
                suffix_lse_rev = pt.log(cumsum_exp) + max_z
                suffix_lse = suffix_lse_rev[::-1]

                # PL log-lik = sum over finishers of (z[k] - suffix_lse[k])
                # Use pt.switch to avoid inf*0=nan for padded entries
                contrib = pt.switch(pt.eq(mask, 1.0),
                                    z_masked - suffix_lse,
                                    pt.constant(0.0))
                return pt.sum(contrib)

            race_lls, _ = pytensor.scan(
                fn=pl_one_race,
                sequences=[pl_d_sh, pl_c_sh, pl_mask_sh, pl_nfin_sh],
                non_sequences=[theta],
                n_steps=n_valid,
            )
            total_ll = pt.sum(race_lls)
            pm.Potential('pl_obs', total_ll)

            # --- Random walk priors (fully vectorized) ---

            # Driver initial: -0.5 * inv_var_0 * theta[init_d]^2
            theta_init_d = theta[rw['init_d']]
            rw_lp_d_init = -0.5 * inv_var_0 * pt.sum(theta_init_d ** 2)

            # Driver transitions: -0.5 * inv_var_d * inv_gap * (curr - prev)^2
            if len(rw['rw_prev_d']) > 0:
                diffs_d = theta[rw['rw_curr_d']] - theta[rw['rw_prev_d']]
                rw_lp_d_trans = -0.5 * inv_var_d * pt.sum(
                    pt.as_tensor_variable(rw['rw_inv_gap_d']) * diffs_d ** 2
                )
            else:
                rw_lp_d_trans = pt.constant(0.0)

            # Constructor initial
            theta_init_c = theta[rw['init_c']]
            rw_lp_c_init = -0.5 * inv_var_0 * pt.sum(theta_init_c ** 2)

            # Constructor transitions
            if len(rw['rw_prev_c']) > 0:
                diffs_c = theta[rw['rw_curr_c']] - theta[rw['rw_prev_c']]
                rw_lp_c_trans = -0.5 * inv_var_c * pt.sum(
                    pt.as_tensor_variable(rw['rw_inv_gap_c']) * diffs_c ** 2
                )
            else:
                rw_lp_c_trans = pt.constant(0.0)

            pm.Potential('rw_prior',
                         rw_lp_d_init + rw_lp_d_trans +
                         rw_lp_c_init + rw_lp_c_trans)

            # --- Centering penalty (vectorized) ---
            if self.center_penalty > 0 and len(center['d_offs']) > 0:
                # Compute per-race mean of driver strengths, then penalize
                # Use segment-based reduction
                driver_vals = theta[center['d_offs']]
                race_ids = pt.as_tensor_variable(center['race_ids'])
                n_races_total = pidx.n_races

                # Sum per race using advanced indexing
                race_sums = pt.zeros(n_races_total)
                race_sums = pt.inc_subtensor(
                    race_sums[race_ids], driver_vals
                )
                n_drv = pt.as_tensor_variable(
                    center['n_drivers'].astype(np.float64)
                )
                # Avoid div by zero for races with 0 drivers
                n_drv_safe = pt.maximum(n_drv, 1.0)
                race_means = race_sums / n_drv_safe

                # Only penalize races that have drivers
                has_drivers = pt.cast(n_drv > 0, 'float64')
                center_lp = -0.5 * self.center_penalty * pt.sum(
                    has_drivers * race_means ** 2
                )
                pm.Potential('center', center_lp)

        return model

    def fit_nuts(
        self,
        prev_positions: np.ndarray,
        next_positions: np.ndarray,
        meta_df: pd.DataFrame,
        draws: int = 1000,
        tune: int = 2000,
        chains: int = 4,
        cores: int = 4,
        target_accept: float = 0.9,
        max_treedepth: int = 12,
        output_path: str = "stage9_nuts_trace.nc",
        checkpoint_every: int = 200,
    ) -> "BayesianStateSpaceF1":
        """
        Fit Stage 9 with NUTS sampling on the final model.

        Steps:
          1. Run MAP CV to select sigma_d, sigma_c (reuses existing code)
          2. Run MAP on full data for warm start
          3. Build PyMC model
          4. Run NUTS with checkpointing
          5. Save trace + metadata
          6. Extract posterior strengths
        """
        import pymc as pm
        import arviz as az

        races = prepare_race_rankings(meta_df)
        years = sorted(meta_df["season"].unique())

        self.driver_ids_ = sorted(meta_df["driver"].unique().tolist())
        self.constructor_ids_ = sorted(meta_df["constructor"].unique().tolist())

        # Global transition matrix for Markov correction
        global_counts = np.zeros((N_PREV_STATES, N_OUTCOMES), dtype=int)
        for s, j in zip(prev_positions, next_positions):
            global_counts[s, j] += 1
        g_alpha = 0.5 + global_counts
        global_pi = g_alpha / g_alpha.sum(axis=1, keepdims=True)

        # DNF rates
        self._compute_dnf_rates(meta_df)

        # --- Step 1: MAP CV for sigma_d, sigma_c ---
        print("Step 1: MAP CV for hyperparameter selection...")
        best_sigma_d = self.sigma_d_candidates[1]
        best_sigma_c = self.sigma_c_candidates[1]
        if len(years) >= 2:
            val_year = years[-1]
            train_races = [r for r in races if r["season"] < val_year]
            val_races = [r for r in races if r["season"] == val_year]

            print(f"  CV: {len(train_races)} train races, "
                  f"{len(val_races)} val races (year {val_year})")

            best_val_ll = -np.inf
            for sigma_d in self.sigma_d_candidates:
                for sigma_c in self.sigma_c_candidates:
                    theta, pidx = self._fit_map(train_races, sigma_d, sigma_c)
                    val_ll = self._eval_validation(theta, pidx, train_races, val_races)
                    print(f"    sigma_d={sigma_d:.3f}, sigma_c={sigma_c:.3f} -> val LL={val_ll:.1f}")
                    if val_ll > best_val_ll:
                        best_val_ll = val_ll
                        best_sigma_d = sigma_d
                        best_sigma_c = sigma_c

            print(f"  Selected: sigma_d={best_sigma_d:.3f}, sigma_c={best_sigma_c:.3f}")

        self.sigma_d_ = best_sigma_d
        self.sigma_c_ = best_sigma_c

        # --- Step 2: MAP on full data for warm start ---
        print("\nStep 2: MAP fit on full data for NUTS initialization...")
        theta_map, pidx_full = self._fit_map(races, self.sigma_d_, self.sigma_c_)
        print(f"  MAP fit complete: {pidx_full.n_params} parameters, "
              f"{pidx_full.n_races} races")

        # --- Step 3: Build PyMC model ---
        print("\nStep 3: Building PyMC model...")
        pymc_model = self._build_pymc_model(
            pidx_full, self.sigma_d_, self.sigma_c_, theta_map
        )
        print("  Model built successfully")

        # --- Step 4: NUTS sampling with checkpointing ---
        print(f"\nStep 4: NUTS sampling ({chains} chains, {tune} tune, {draws} draws)...")
        output_dir = Path(output_path).parent
        output_stem = Path(output_path).stem

        checkpoint_path = output_dir / f"{output_stem}_checkpoint.nc"

        _draw_count = [0]  # mutable counter in closure

        def checkpoint_callback(trace, draw):
            """Save checkpoint every N draws."""
            _draw_count[0] += 1
            if _draw_count[0] > 0 and _draw_count[0] % checkpoint_every == 0:
                try:
                    # During sampling, trace may not be InferenceData yet
                    # Just print progress; final save happens after completion
                    print(f"    Progress: {_draw_count[0]} draws completed")
                except Exception as e:
                    print(f"    Checkpoint progress error: {e}")

        with pymc_model:
            step = pm.NUTS(
                target_accept=target_accept,
                max_treedepth=max_treedepth,
            )
            trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                step=step,
                initvals={"theta": theta_map},
                callback=checkpoint_callback,
                return_inferencedata=True,
                progressbar=True,
            )

        # --- Step 5: Save trace + metadata ---
        print(f"\nStep 5: Saving trace to {output_path}...")
        az.to_netcdf(trace, str(output_path))

        # Save metadata sidecar (pidx mapping, DNF rates, Markov correction, etc.)
        metadata_path = Path(output_path).with_suffix('.pkl')
        metadata = {
            'sigma_d': self.sigma_d_,
            'sigma_c': self.sigma_c_,
            'sigma_0': self.sigma_0,
            'center_penalty': self.center_penalty,
            'n_mc_samples': self.n_mc_samples,
            'dnf_shrinkage': self.dnf_shrinkage,
            'driver_ids': self.driver_ids_,
            'constructor_ids': self.constructor_ids_,
            'dnf_rates': dict(self.dnf_rates_),
            'global_dnf_rate': self.global_dnf_rate_,
            'global_pi': global_pi,
            # ParamIndex info for extracting strengths
            'driver_last_offsets': {
                did: pidx_full.driver_indices[did][-1][1]
                for did in pidx_full.driver_ids
            },
            'constructor_last_offsets': {
                cid: pidx_full.constructor_indices[cid][-1][1]
                for cid in pidx_full.constructor_ids
            },
            'n_params': pidx_full.n_params,
            'theta_map': theta_map,
        }
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"  Metadata saved to {metadata_path}")

        # --- Step 6: Extract posterior strengths ---
        self._extract_posterior_strengths(trace, metadata, global_pi)

        # Print convergence diagnostics
        self._print_diagnostics(trace, pidx_full)

        return self

    def _extract_posterior_strengths(self, trace, metadata, global_pi):
        """Extract driver/constructor strengths from posterior samples."""
        import arviz as az

        theta_samples = trace.posterior['theta'].values  # (chains, draws, n_params)
        n_chains, n_draws, n_params = theta_samples.shape

        # Flatten chains x draws
        theta_flat = theta_samples.reshape(-1, n_params)  # (n_samples, n_params)
        n_samples = theta_flat.shape[0]

        # Store posterior samples of final strengths (exp-space)
        self.driver_strengths_samples_ = {}
        for did, off in metadata['driver_last_offsets'].items():
            self.driver_strengths_samples_[did] = np.exp(theta_flat[:, off])

        self.constructor_strengths_samples_ = {}
        for cid, off in metadata['constructor_last_offsets'].items():
            self.constructor_strengths_samples_[cid] = np.exp(theta_flat[:, off])

        # Point estimates = posterior mean (for backward-compatible predict_proba)
        self.driver_strengths_ = {
            did: samples.mean()
            for did, samples in self.driver_strengths_samples_.items()
        }
        self.constructor_strengths_ = {
            cid: samples.mean()
            for cid, samples in self.constructor_strengths_samples_.items()
        }

        # Markov correction
        marginal = global_pi.mean(axis=0)
        self.global_correction_ = np.zeros((N_PREV_STATES, N_OUTCOMES))
        for s in range(N_PREV_STATES):
            for k in range(N_OUTCOMES):
                if marginal[k] > 1e-10:
                    self.global_correction_[s, k] = global_pi[s, k] / marginal[k]
                else:
                    self.global_correction_[s, k] = 1.0

        # Store number of posterior samples for predict_proba_bayesian
        self.n_posterior_samples_ = n_samples

    def _print_diagnostics(self, trace, pidx):
        """Print NUTS convergence diagnostics."""
        import arviz as az

        print("\n" + "=" * 60)
        print("CONVERGENCE DIAGNOSTICS")
        print("=" * 60)

        # R-hat and ESS for the theta vector
        summary = az.summary(trace, var_names=['theta'],
                            filter_vars='like',
                            kind='diagnostics')

        rhat_vals = summary['r_hat'].values
        ess_bulk = summary['ess_bulk'].values
        ess_tail = summary['ess_tail'].values

        print(f"  Parameters: {len(rhat_vals)}")
        print(f"  R-hat: min={np.nanmin(rhat_vals):.4f}, "
              f"max={np.nanmax(rhat_vals):.4f}, "
              f"mean={np.nanmean(rhat_vals):.4f}")
        n_bad_rhat = np.sum(rhat_vals > 1.05)
        print(f"  R-hat > 1.05: {n_bad_rhat} / {len(rhat_vals)} "
              f"({100*n_bad_rhat/len(rhat_vals):.1f}%)")
        print(f"  ESS bulk: min={np.nanmin(ess_bulk):.0f}, "
              f"median={np.nanmedian(ess_bulk):.0f}")
        print(f"  ESS tail: min={np.nanmin(ess_tail):.0f}, "
              f"median={np.nanmedian(ess_tail):.0f}")

        # Divergences
        if hasattr(trace, 'sample_stats') and 'diverging' in trace.sample_stats:
            divs = trace.sample_stats['diverging'].values.sum()
            total = trace.sample_stats['diverging'].values.size
            print(f"  Divergences: {divs} / {total} ({100*divs/total:.1f}%)")

        # Top-5 driver strengths (posterior mean ± std)
        print(f"\n  Top-5 driver strengths (posterior mean ± std):")
        sorted_drivers = sorted(
            self.driver_strengths_.items(),
            key=lambda x: x[1], reverse=True
        )[:5]
        for did, mean_str in sorted_drivers:
            std_str = self.driver_strengths_samples_[did].std()
            print(f"    Driver {did}: {mean_str:.3f} ± {std_str:.3f}")

        print(f"\n  Constructor strengths (posterior mean ± std):")
        sorted_constrs = sorted(
            self.constructor_strengths_.items(),
            key=lambda x: x[1], reverse=True
        )
        for cid, mean_str in sorted_constrs:
            std_str = self.constructor_strengths_samples_[cid].std()
            print(f"    Constructor {cid}: {mean_str:.3f} ± {std_str:.3f}")

    @classmethod
    def load_nuts_trace(cls, trace_path, n_mc_samples=3000):
        """
        Load a saved NUTS trace and metadata, returning a fitted model.

        Parameters
        ----------
        trace_path : str or Path
            Path to the .nc trace file (metadata .pkl expected alongside)
        n_mc_samples : int
            MC samples for position marginal computation

        Returns
        -------
        BayesianStateSpaceF1 with fitted attributes set from posterior
        """
        import arviz as az

        trace_path = Path(trace_path)
        metadata_path = trace_path.with_suffix('.pkl')

        print(f"Loading trace from {trace_path}...")
        trace = az.from_netcdf(str(trace_path))

        print(f"Loading metadata from {metadata_path}...")
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        # Create model instance
        model = cls(
            sigma_d_candidates=(metadata['sigma_d'],),
            sigma_c_candidates=(metadata['sigma_c'],),
            sigma_0=metadata['sigma_0'],
            dnf_shrinkage=metadata['dnf_shrinkage'],
            n_mc_samples=n_mc_samples,
            center_penalty=metadata['center_penalty'],
        )

        model.sigma_d_ = metadata['sigma_d']
        model.sigma_c_ = metadata['sigma_c']
        model.driver_ids_ = metadata['driver_ids']
        model.constructor_ids_ = metadata['constructor_ids']
        model.dnf_rates_ = metadata['dnf_rates']
        model.global_dnf_rate_ = metadata['global_dnf_rate']

        # Extract posterior strengths
        model._extract_posterior_strengths(
            trace, metadata, metadata['global_pi']
        )

        print(f"  Loaded {model.n_posterior_samples_} posterior samples "
              f"({len(model.driver_strengths_)} drivers, "
              f"{len(model.constructor_strengths_)} constructors)")

        return model

    def predict_proba_bayesian(
        self,
        driver_id,
        prev_position: int,
        constructor_id: Optional = None,
        n_posterior_draws: int = 100,
        **kwargs,
    ) -> np.ndarray:
        """
        Posterior-averaged P(next | prev, driver, constructor).

        Averages predictions over n_posterior_draws posterior samples,
        each giving different strength values. This naturally widens
        prediction intervals and improves calibration.

        Parameters
        ----------
        driver_id : int
        prev_position : int
        constructor_id : int, optional
        n_posterior_draws : int
            Number of posterior draws to average over (default 100).
            Higher = more accurate but slower.

        Returns
        -------
        (21,) array of probabilities
        """
        self._check_fitted()

        if not hasattr(self, 'driver_strengths_samples_') or \
           self.driver_strengths_samples_ is None:
            # Fall back to MAP-based prediction
            return self.predict_proba(driver_id, prev_position,
                                     constructor_id=constructor_id)

        n_total = self.n_posterior_samples_
        draw_indices = np.linspace(0, n_total - 1,
                                   min(n_posterior_draws, n_total),
                                   dtype=int)

        # Get driver/constructor sample arrays
        d_samples = self.driver_strengths_samples_.get(driver_id, None)
        c_samples = None
        if constructor_id is not None:
            c_samples = self.constructor_strengths_samples_.get(
                constructor_id, None
            )

        p_dnf = self.dnf_rates_.get(driver_id, self.global_dnf_rate_)

        # Average predictions over posterior draws
        avg_probs = np.zeros(N_OUTCOMES)
        for idx in draw_indices:
            d_str = d_samples[idx] if d_samples is not None else 1.0
            c_str = c_samples[idx] if c_samples is not None else 1.0
            composite = d_str * c_str

            probs = self._compute_position_probs(
                driver_id, composite, p_dnf, prev_position
            )
            avg_probs += probs

        avg_probs /= len(draw_indices)

        # Renormalize
        avg_probs = np.maximum(avg_probs, 1e-10)
        avg_probs /= avg_probs.sum()

        return avg_probs


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path

    print("=" * 65)
    print("F1 Bayesian State-Space — Stage 9")
    print("=" * 65)

    data_dir = Path(__file__).parent.parent / "data"

    loader = F1DataLoader(data_dir)
    df = loader.load_merged(min_year=2020, max_year=2024)
    print(f"\nLoaded {len(df)} results, "
          f"{df['year'].nunique()} seasons")

    prev_pos, next_pos, meta = prepare_transitions(df)
    print(f"Total transitions: {len(prev_pos)}")

    model = BayesianStateSpaceF1(
        sigma_d_candidates=(0.02, 0.05, 0.1, 0.2),
        sigma_c_candidates=(0.02, 0.05, 0.1, 0.2),
        n_mc_samples=3000,
    )
    model.fit(prev_pos, next_pos, meta)

    print(f"\nConstructor strengths:")
    constructor_names = (
        df[["constructorId", "constructor_name"]]
        .drop_duplicates()
        .set_index("constructorId")["constructor_name"]
        .to_dict()
    )
    for cid in sorted(model.constructor_strengths_,
                      key=lambda c: model.constructor_strengths_[c],
                      reverse=True):
        name = constructor_names.get(cid, str(cid))
        print(f"  {name:20s}: {model.constructor_strengths_[cid]:.3f}")

    print(f"\nSample predictions from START:")
    driver_names = (
        df[["driverId", "driver_name"]]
        .drop_duplicates()
        .set_index("driverId")["driver_name"]
        .to_dict()
    )
    test_drivers = list(model.driver_strengths_.keys())[:8]
    for did in test_drivers:
        cid = meta[meta["driver"] == did]["constructor"].iloc[-1]
        probs = model.predict_proba(did, START, constructor_id=cid)
        e_pos = sum(j * probs[j] for j in range(1, N_OUTCOMES))
        name = driver_names.get(did, str(did))
        cname = constructor_names.get(cid, str(cid))
        print(f"  {name:25s} ({cname:15s}): "
              f"P(P1)={probs[1]:.3f}, P(top3)={sum(probs[1:4]):.3f}, "
              f"E[pos]={e_pos:.1f}")
