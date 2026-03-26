"""
F1 Hidden Markov Model
======================
Stage 7: Constructor-tier HMM with driver-modulated emissions

Models each constructor as having a latent competitiveness tier that evolves
over time. The tiers are:
    0 = DOMINANT    (expected finishes ~P1-P3)
    1 = FRONTRUNNER (expected finishes ~P3-P7)
    2 = MIDFIELD    (expected finishes ~P8-P14)
    3 = BACKMARKER  (expected finishes ~P15-P20)

Architecture:
    - Per-constructor HMM: transition matrix A (4x4), initial distribution pi_0
    - Shared tier emissions: global emission structure shifted per-tier
    - Driver offset: additive log-odds shift, shrunk toward zero for sparse drivers
    - Prediction: forward algorithm -> final belief -> mixture of tier emissions

Fitting uses Baum-Welch EM with K-means initialization and random restarts.
"""

import numpy as np
import pandas as pd
from scipy.special import softmax
from pathlib import Path
from typing import Optional
from collections import defaultdict

from stage3_constructor import (
    F1DataLoader,
    prepare_transitions,
    build_count_matrices,
    START, N_OUTCOMES, N_PREV_STATES,
    DNF, OUTCOME_LABELS, PREV_STATE_LABELS,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_TIERS = 4
TIER_NAMES = ["dominant", "frontrunner", "midfield", "backmarker"]
# Modal positions for each tier (used for initialization)
TIER_MODAL_POSITIONS = [2, 5, 11, 17]  # P2, P5, P11, P17


# ---------------------------------------------------------------------------
# HMM Utilities
# ---------------------------------------------------------------------------
def _log_sum_exp(a: np.ndarray) -> float:
    """Numerically stable log-sum-exp."""
    a_max = np.max(a)
    if np.isinf(a_max):
        return -np.inf
    return a_max + np.log(np.sum(np.exp(a - a_max)))


def _forward(log_pi0, log_A, log_B_seq):
    """
    Forward algorithm in log space.

    Parameters
    ----------
    log_pi0 : (K,) log initial state distribution
    log_A : (K, K) log transition matrix
    log_B_seq : list of (K,) arrays, log emission probabilities per timestep

    Returns
    -------
    log_alpha : (T, K) forward log-probabilities
    log_lik : float, total log-likelihood
    """
    T = len(log_B_seq)
    K = len(log_pi0)
    log_alpha = np.full((T, K), -np.inf)

    # t=0
    log_alpha[0] = log_pi0 + log_B_seq[0]

    for t in range(1, T):
        for j in range(K):
            log_alpha[t, j] = (
                _log_sum_exp(log_alpha[t - 1] + log_A[:, j])
                + log_B_seq[t][j]
            )

    log_lik = _log_sum_exp(log_alpha[-1])
    return log_alpha, log_lik


def _backward(log_A, log_B_seq):
    """Backward algorithm in log space."""
    T = len(log_B_seq)
    K = log_A.shape[0]
    log_beta = np.full((T, K), -np.inf)
    log_beta[-1] = 0.0  # log(1)

    for t in range(T - 2, -1, -1):
        for i in range(K):
            log_beta[t, i] = _log_sum_exp(
                log_A[i, :] + log_B_seq[t + 1] + log_beta[t + 1]
            )

    return log_beta


def _forward_backward(log_pi0, log_A, log_B_seq):
    """
    Forward-backward algorithm.

    Returns
    -------
    gamma : (T, K) posterior state probabilities
    xi : (T-1, K, K) posterior transition probabilities
    log_lik : float
    """
    log_alpha, log_lik = _forward(log_pi0, log_A, log_B_seq)
    log_beta = _backward(log_A, log_B_seq)

    T, K = log_alpha.shape

    # Gamma: P(z_t = k | observations)
    log_gamma = log_alpha + log_beta
    log_gamma -= _log_sum_exp(log_gamma[0])  # not quite right for each row
    # Normalize each row
    gamma = np.zeros((T, K))
    for t in range(T):
        log_row = log_alpha[t] + log_beta[t]
        log_norm = _log_sum_exp(log_row)
        gamma[t] = np.exp(log_row - log_norm)

    # Xi: P(z_t = i, z_{t+1} = j | observations)
    xi = np.zeros((T - 1, K, K))
    for t in range(T - 1):
        for i in range(K):
            for j in range(K):
                xi[t, i, j] = (
                    log_alpha[t, i] + log_A[i, j]
                    + log_B_seq[t + 1][j] + log_beta[t + 1, j]
                )
        log_norm = _log_sum_exp(xi[t].ravel())
        xi[t] = np.exp(xi[t] - log_norm)

    return gamma, xi, log_lik


# ---------------------------------------------------------------------------
# Tier Emission Model
# ---------------------------------------------------------------------------
def _build_tier_emission(global_counts, prior_alpha=0.5):
    """
    Build tier emission distributions by shifting the global emission.

    Each tier's emission for a given prev_position is the global emission
    shifted in position space toward the tier's modal position.

    Returns
    -------
    tier_emissions : (N_TIERS, N_PREV_STATES, N_OUTCOMES) probability arrays
    """
    # Global empirical distribution (smoothed)
    global_alpha = prior_alpha + global_counts
    global_pi = global_alpha / global_alpha.sum(axis=1, keepdims=True)

    tier_emissions = np.zeros((N_TIERS, N_PREV_STATES, N_OUTCOMES))

    for tier_idx in range(N_TIERS):
        modal = TIER_MODAL_POSITIONS[tier_idx]
        for s in range(N_PREV_STATES):
            # Start from global distribution
            base = global_pi[s].copy()

            # Shift distribution toward tier's modal position
            # Use a soft reweighting: multiply by exp(-|pos - modal|/sigma)
            # Then renormalize
            sigma = 4.0 + tier_idx * 1.0  # wider spread for worse tiers
            weights = np.ones(N_OUTCOMES)
            for j in range(1, N_OUTCOMES):  # positions P1-P20
                weights[j] = np.exp(-abs(j - modal) / sigma)
            # DNF weight: higher for backmarkers
            weights[0] = 0.5 + 0.15 * tier_idx

            shifted = base * weights
            shifted = np.maximum(shifted, 1e-10)
            shifted /= shifted.sum()
            tier_emissions[tier_idx, s] = shifted

    return tier_emissions


def _init_tier_emissions_from_data(meta_df, tier_assignments, prior_alpha=0.5):
    """
    Build tier emissions from data using tier assignments.

    Parameters
    ----------
    meta_df : DataFrame with prev_position, next_position columns
    tier_assignments : dict[constructor_id -> int] mapping constructors to tiers

    Returns
    -------
    tier_emissions : (N_TIERS, N_PREV_STATES, N_OUTCOMES)
    """
    tier_counts = np.zeros((N_TIERS, N_PREV_STATES, N_OUTCOMES))

    for _, row in meta_df.iterrows():
        cid = row["constructor"]
        tier = tier_assignments.get(cid, 2)  # default midfield
        s = int(row["prev_position"])
        j = int(row["next_position"])
        tier_counts[tier, s, j] += 1

    # Smooth and normalize
    tier_emissions = np.zeros((N_TIERS, N_PREV_STATES, N_OUTCOMES))
    for tier in range(N_TIERS):
        alpha = prior_alpha + tier_counts[tier]
        tier_emissions[tier] = alpha / alpha.sum(axis=1, keepdims=True)

    return tier_emissions


# ---------------------------------------------------------------------------
# Stage 7 Model
# ---------------------------------------------------------------------------
class HiddenMarkovF1:
    """
    Stage 7: Constructor-tier HMM with driver-modulated emissions.

    Each constructor has a latent competitiveness tier (hidden state) that
    evolves over time. Emissions are finishing position distributions
    conditioned on the tier and previous position.

    Parameters
    ----------
    n_tiers : int
        Number of hidden competitiveness tiers (default: 4).
    em_iters : int
        Maximum EM iterations per restart (default: 50).
    em_tol : float
        EM convergence tolerance on log-likelihood (default: 1e-3).
    n_restarts : int
        Number of random restarts for Baum-Welch (default: 5).
    driver_shrinkage : float
        Shrinkage denominator for driver offsets (default: 20.0).
    emission_prior : float
        Dirichlet smoothing for emission distributions (default: 0.5).
    """

    def __init__(
        self,
        n_tiers: int = N_TIERS,
        em_iters: int = 50,
        em_tol: float = 1e-3,
        n_restarts: int = 5,
        driver_shrinkage: float = 20.0,
        emission_prior: float = 0.5,
    ):
        self.n_tiers = n_tiers
        self.em_iters = em_iters
        self.em_tol = em_tol
        self.n_restarts = n_restarts
        self.driver_shrinkage = driver_shrinkage
        self.emission_prior = emission_prior

        # Fitted attributes
        self.tier_emissions_: Optional[np.ndarray] = None  # (K, 22, 21)
        self.constructor_A_: Optional[dict] = None  # cid -> (K, K)
        self.constructor_pi0_: Optional[dict] = None  # cid -> (K,)
        self.constructor_beliefs_: Optional[dict] = None  # cid -> (K,)
        self.driver_offsets_: Optional[dict] = None  # did -> (22, 21)
        self.driver_ids_: Optional[list] = None
        self.constructor_ids_: Optional[list] = None
        self.global_pi_: Optional[np.ndarray] = None
        self.total_ll_: Optional[float] = None

    @property
    def is_fitted(self) -> bool:
        return self.tier_emissions_ is not None

    def fit(
        self,
        prev_positions: np.ndarray,
        next_positions: np.ndarray,
        meta_df: pd.DataFrame,
    ) -> "HiddenMarkovF1":
        """
        Fit the HMM:
          1. Build constructor observation sequences
          2. Initialize tiers via K-means on mean finishing positions
          3. Run Baum-Welch EM with multiple restarts
          4. Compute driver offsets

        Parameters
        ----------
        prev_positions, next_positions : arrays from prepare_transitions()
        meta_df : DataFrame with driver, constructor, season, race_order,
                  prev_position, next_position columns
        """
        # Build global counts for fallback
        global_counts = np.zeros((N_PREV_STATES, N_OUTCOMES), dtype=int)
        for s, j in zip(prev_positions, next_positions):
            global_counts[s, j] += 1
        g_alpha = self.emission_prior + global_counts
        self.global_pi_ = g_alpha / g_alpha.sum(axis=1, keepdims=True)

        # Build observation sequences per constructor
        # Each observation is (prev_position, next_position)
        constructor_sequences = self._build_constructor_sequences(meta_df)
        self.constructor_ids_ = sorted(constructor_sequences.keys())

        # Initialize tiers via K-means on mean finishing positions
        tier_assignments = self._kmeans_init(meta_df)

        # Run Baum-Welch with multiple restarts
        best_ll = -np.inf
        best_params = None
        rng = np.random.default_rng(42)

        for restart in range(self.n_restarts + 1):
            if restart == 0:
                # Use K-means initialization
                init_emissions = _init_tier_emissions_from_data(
                    meta_df, tier_assignments, self.emission_prior
                )
            else:
                # Random initialization
                init_emissions = self._random_emission_init(
                    global_counts, rng
                )

            params = self._run_em(
                constructor_sequences, init_emissions, rng
            )
            if params is not None and params["total_ll"] > best_ll:
                best_ll = params["total_ll"]
                best_params = params

        if best_params is None:
            raise RuntimeError("All EM restarts failed")

        # Store fitted parameters
        self.tier_emissions_ = best_params["tier_emissions"]
        self.constructor_A_ = best_params["constructor_A"]
        self.constructor_pi0_ = best_params["constructor_pi0"]
        self.total_ll_ = best_params["total_ll"]

        # Compute final beliefs for each constructor
        self.constructor_beliefs_ = {}
        for cid in self.constructor_ids_:
            seq = constructor_sequences[cid]
            if len(seq) == 0:
                self.constructor_beliefs_[cid] = np.ones(self.n_tiers) / self.n_tiers
                continue
            log_pi0 = np.log(np.maximum(self.constructor_pi0_[cid], 1e-300))
            log_A = np.log(np.maximum(self.constructor_A_[cid], 1e-300))
            log_B_seq = self._compute_log_emissions(seq, self.tier_emissions_)
            log_alpha, _ = _forward(log_pi0, log_A, log_B_seq)
            # Final belief = normalized forward at T-1
            log_belief = log_alpha[-1]
            belief = np.exp(log_belief - _log_sum_exp(log_belief))
            self.constructor_beliefs_[cid] = belief

        # Compute driver offsets
        self.driver_ids_ = sorted(
            meta_df["driver"].unique().tolist()
        )
        self.driver_offsets_ = self._compute_driver_offsets(meta_df)

        print(f"  Stage 7 (HMM): {self.n_tiers} tiers, "
              f"total LL = {self.total_ll_:.1f}")
        for cid in self.constructor_ids_[:5]:
            belief = self.constructor_beliefs_[cid]
            tier = TIER_NAMES[np.argmax(belief)]
            print(f"    Constructor {cid}: {tier} "
                  f"(belief: {belief.round(2)})")

        return self

    def _build_constructor_sequences(self, meta_df):
        """
        Build chronological observation sequences per constructor.

        Each element is a list of (prev_position, next_position) tuples,
        one per race entry (aggregated across all drivers at that constructor
        in that race).

        For the HMM, we aggregate per constructor per race:
        each "observation" is the collection of all driver results for that
        constructor in that race.
        """
        sequences = defaultdict(list)

        # Group by constructor, season, race_order
        grouped = meta_df.sort_values(
            ["constructor", "season", "race_order"]
        ).groupby(["constructor", "season", "race_order"])

        for (cid, season, race_order), grp in grouped:
            # Store all (prev, next) pairs for this constructor in this race
            obs = []
            for _, row in grp.iterrows():
                obs.append((int(row["prev_position"]),
                           int(row["next_position"])))
            sequences[cid].append(obs)

        return dict(sequences)

    def _compute_log_emissions(self, sequence, tier_emissions):
        """
        Compute log emission probabilities for a constructor's sequence.

        For each race (timestep), the emission probability for tier k is
        the product of P(next_j | prev_j, tier_k) across all drivers
        in that race for that constructor.

        Parameters
        ----------
        sequence : list of list of (prev, next) tuples
        tier_emissions : (K, 22, 21) array

        Returns
        -------
        log_B_seq : list of (K,) arrays
        """
        K = self.n_tiers
        log_B_seq = []

        for race_obs in sequence:
            log_b = np.zeros(K)
            for prev, nxt in race_obs:
                for k in range(K):
                    log_b[k] += np.log(
                        max(tier_emissions[k, prev, nxt], 1e-300)
                    )
            log_B_seq.append(log_b)

        return log_B_seq

    def _kmeans_init(self, meta_df):
        """
        Initialize tier assignments via K-means on mean finishing positions.
        """
        # Compute mean finishing position per constructor
        constructor_means = {}
        for cid, grp in meta_df.groupby("constructor"):
            positions = grp["next_position"].values
            # Exclude DNFs for mean position calculation
            classified = positions[positions > 0]
            if len(classified) > 0:
                constructor_means[cid] = classified.mean()
            else:
                constructor_means[cid] = 15.0  # default high

        # Sort constructors by mean position
        sorted_cids = sorted(constructor_means.keys(),
                            key=lambda c: constructor_means[c])

        # Assign to tiers based on quartiles
        n = len(sorted_cids)
        tier_assignments = {}
        for i, cid in enumerate(sorted_cids):
            tier = min(int(i / n * self.n_tiers), self.n_tiers - 1)
            tier_assignments[cid] = tier

        return tier_assignments

    def _random_emission_init(self, global_counts, rng):
        """Random initialization of tier emissions."""
        emissions = np.zeros((self.n_tiers, N_PREV_STATES, N_OUTCOMES))
        for k in range(self.n_tiers):
            modal = TIER_MODAL_POSITIONS[k]
            noise = rng.dirichlet(
                np.ones(N_OUTCOMES) * 0.5,
                size=N_PREV_STATES
            )
            # Blend with shifted global
            g_alpha = self.emission_prior + global_counts
            g_pi = g_alpha / g_alpha.sum(axis=1, keepdims=True)
            for s in range(N_PREV_STATES):
                sigma = 3.0 + k * 1.5 + rng.uniform(-1, 1)
                weights = np.ones(N_OUTCOMES)
                for j in range(1, N_OUTCOMES):
                    weights[j] = np.exp(-abs(j - modal) / sigma)
                weights[0] = 0.5 + 0.15 * k
                shifted = g_pi[s] * weights
                shifted = np.maximum(shifted, 1e-10)
                shifted /= shifted.sum()
                emissions[k, s] = 0.7 * shifted + 0.3 * noise[s]
                emissions[k, s] /= emissions[k, s].sum()
        return emissions

    def _run_em(self, constructor_sequences, init_emissions, rng):
        """
        Run Baum-Welch EM for all constructors jointly.

        Tier emissions are shared across constructors.
        Each constructor has its own A and pi0.
        """
        K = self.n_tiers
        tier_emissions = init_emissions.copy()

        # Initialize per-constructor HMM params
        constructor_A = {}
        constructor_pi0 = {}
        for cid in constructor_sequences:
            # Slightly sticky diagonal initialization
            A = np.full((K, K), 0.05 / (K - 1))
            np.fill_diagonal(A, 0.95)
            A += rng.uniform(0, 0.02, (K, K))
            A /= A.sum(axis=1, keepdims=True)
            constructor_A[cid] = A
            constructor_pi0[cid] = np.ones(K) / K

        prev_ll = -np.inf

        for iteration in range(self.em_iters):
            # E-step: forward-backward for each constructor
            all_gamma = {}
            all_xi = {}
            total_ll = 0.0

            for cid, seq in constructor_sequences.items():
                if len(seq) == 0:
                    continue
                log_pi0 = np.log(np.maximum(constructor_pi0[cid], 1e-300))
                log_A = np.log(np.maximum(constructor_A[cid], 1e-300))
                log_B_seq = self._compute_log_emissions(seq, tier_emissions)

                gamma, xi, ll = _forward_backward(log_pi0, log_A, log_B_seq)
                all_gamma[cid] = gamma
                all_xi[cid] = xi
                total_ll += ll

            # Check convergence
            if abs(total_ll - prev_ll) < self.em_tol:
                break
            prev_ll = total_ll

            # M-step

            # Update per-constructor A and pi0
            for cid in constructor_sequences:
                if cid not in all_gamma:
                    continue
                gamma = all_gamma[cid]
                xi = all_xi[cid]

                # pi0
                constructor_pi0[cid] = gamma[0] + 1e-10
                constructor_pi0[cid] /= constructor_pi0[cid].sum()

                # A
                if len(xi) > 0:
                    A_new = xi.sum(axis=0) + 1e-10  # (K, K)
                    A_new /= A_new.sum(axis=1, keepdims=True)
                    constructor_A[cid] = A_new

            # Update shared tier emissions
            tier_counts = np.zeros((K, N_PREV_STATES, N_OUTCOMES))
            for cid, seq in constructor_sequences.items():
                if cid not in all_gamma:
                    continue
                gamma = all_gamma[cid]
                for t, race_obs in enumerate(seq):
                    for prev, nxt in race_obs:
                        for k in range(K):
                            tier_counts[k, prev, nxt] += gamma[t, k]

            for k in range(K):
                alpha = self.emission_prior + tier_counts[k]
                tier_emissions[k] = alpha / alpha.sum(axis=1, keepdims=True)

        return {
            "tier_emissions": tier_emissions,
            "constructor_A": constructor_A,
            "constructor_pi0": constructor_pi0,
            "total_ll": total_ll,
        }

    def _compute_driver_offsets(self, meta_df):
        """
        Compute per-driver log-odds offsets from their constructor's
        tier-predicted distribution.

        For each driver, compare their empirical finishing distribution
        to what the model predicts for their constructor. The offset
        is the difference in log-odds, shrunk toward zero.
        """
        offsets = {}
        tau = self.driver_shrinkage

        for driver_id in self.driver_ids_:
            driver_data = meta_df[meta_df["driver"] == driver_id]
            n_obs = len(driver_data)
            shrink_weight = n_obs / (n_obs + tau)

            # Driver's empirical counts
            driver_counts = np.zeros((N_PREV_STATES, N_OUTCOMES))
            for _, row in driver_data.iterrows():
                s = int(row["prev_position"])
                j = int(row["next_position"])
                driver_counts[s, j] += 1

            # Constructor's predicted distribution (weighted by belief)
            # Use the driver's most common constructor
            cid_counts = driver_data["constructor"].value_counts()
            primary_cid = cid_counts.index[0]
            belief = self.constructor_beliefs_.get(
                primary_cid, np.ones(self.n_tiers) / self.n_tiers
            )

            predicted = np.zeros((N_PREV_STATES, N_OUTCOMES))
            for k in range(self.n_tiers):
                predicted += belief[k] * self.tier_emissions_[k]

            # Compute log-odds offset
            offset = np.zeros((N_PREV_STATES, N_OUTCOMES))
            for s in range(N_PREV_STATES):
                if driver_counts[s].sum() < 2:
                    continue  # too few observations, keep offset at 0
                driver_emp = (driver_counts[s] + 0.5)
                driver_emp /= driver_emp.sum()
                # Log-odds difference
                log_odds_driver = np.log(driver_emp / (1 - driver_emp + 1e-10))
                log_odds_pred = np.log(predicted[s] / (1 - predicted[s] + 1e-10))
                raw_offset = log_odds_driver - log_odds_pred
                offset[s] = shrink_weight * raw_offset

            offsets[driver_id] = offset

        return offsets

    # --- Prediction ---

    def predict_proba(
        self,
        driver_id,
        prev_position: int,
        constructor_id: Optional = None,
        **kwargs,
    ) -> np.ndarray:
        """
        P(next | prev, driver, constructor).

        1. Get constructor belief over tiers
        2. Mixture of tier emissions
        3. Apply driver offset
        """
        self._check_fitted()

        # Get constructor belief
        if constructor_id is not None:
            belief = self.constructor_beliefs_.get(
                constructor_id,
                np.ones(self.n_tiers) / self.n_tiers
            )
        else:
            belief = np.ones(self.n_tiers) / self.n_tiers

        # Mixture of tier emissions
        probs = np.zeros(N_OUTCOMES)
        for k in range(self.n_tiers):
            probs += belief[k] * self.tier_emissions_[k, prev_position]

        # Apply driver offset if available
        if driver_id in self.driver_offsets_:
            offset = self.driver_offsets_[driver_id][prev_position]
            # Convert to log-odds, add offset, convert back
            log_odds = np.log(probs / (1 - probs + 1e-10))
            log_odds += offset
            # Softmax to convert back to probabilities
            probs = softmax(log_odds)

        # Ensure valid distribution
        probs = np.maximum(probs, 1e-10)
        probs /= probs.sum()
        return probs

    def predict_proba_new_driver(
        self,
        prev_position: int,
        constructor_id: Optional = None,
        **kwargs,
    ) -> np.ndarray:
        """Prediction for unseen driver (no driver offset)."""
        self._check_fitted()

        if constructor_id is not None:
            belief = self.constructor_beliefs_.get(
                constructor_id,
                np.ones(self.n_tiers) / self.n_tiers
            )
        else:
            belief = np.ones(self.n_tiers) / self.n_tiers

        probs = np.zeros(N_OUTCOMES)
        for k in range(self.n_tiers):
            probs += belief[k] * self.tier_emissions_[k, prev_position]

        probs = np.maximum(probs, 1e-10)
        probs /= probs.sum()
        return probs

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    print("=" * 65)
    print("F1 Hidden Markov Model — Stage 7: Constructor-Tier HMM")
    print("=" * 65)

    data_dir = Path(__file__).parent.parent / "data"

    loader = F1DataLoader(data_dir)
    df = loader.load_merged(min_year=2020, max_year=2024)
    print(f"\nLoaded {len(df)} results, "
          f"{df['year'].nunique()} seasons, "
          f"{df['constructorId'].nunique()} constructors")

    prev_pos, next_pos, meta = prepare_transitions(df)
    print(f"Total transitions: {len(prev_pos)}")

    model = HiddenMarkovF1(n_tiers=4, em_iters=50, n_restarts=5)
    model.fit(prev_pos, next_pos, meta)

    print(f"\nConstructor beliefs:")
    constructor_names = (
        df[["constructorId", "constructor_name"]]
        .drop_duplicates()
        .set_index("constructorId")["constructor_name"]
        .to_dict()
    )
    for cid in model.constructor_ids_:
        belief = model.constructor_beliefs_[cid]
        tier = TIER_NAMES[np.argmax(belief)]
        name = constructor_names.get(cid, str(cid))
        print(f"  {name:20s}: {tier:12s} (belief: {belief.round(3)})")

    # Test prediction
    print(f"\nSample predictions from START:")
    driver_names = (
        df[["driverId", "driver_name"]]
        .drop_duplicates()
        .set_index("driverId")["driver_name"]
        .to_dict()
    )
    for did in list(model.driver_ids_)[:5]:
        cid = meta[meta["driver"] == did]["constructor"].iloc[-1]
        probs = model.predict_proba(did, START, constructor_id=cid)
        e_pos = sum(j * probs[j] for j in range(1, N_OUTCOMES))
        name = driver_names.get(did, str(did))
        cname = constructor_names.get(cid, str(cid))
        print(f"  {name:25s} ({cname:15s}): "
              f"P(P1)={probs[1]:.3f}, P(top3)={sum(probs[1:4]):.3f}, "
              f"E[pos]={e_pos:.1f}")
