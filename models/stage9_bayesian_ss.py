"""
F1 Bayesian State-Space Model
==============================
Stage 9: Random walk on driver + constructor latent strengths

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
