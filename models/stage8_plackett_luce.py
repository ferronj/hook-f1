"""
F1 Time-Varying Plackett-Luce Model
====================================
Stage 8: Time-varying driver/constructor strengths with Plackett-Luce ranking model

Unlike the DM-Markov models (Stages 1-6) which treat each driver independently,
the Plackett-Luce model captures the key constraint that finishing positions are
mutually exclusive: if driver A finishes P1, no one else can.

Architecture:
    - Each driver has a latent log-strength mu_i(t) that evolves over time
    - Constructor effect is additive: log_lambda_i(t) = mu_i(t) + beta_C(t)
    - DNF modeled separately: P(DNF) estimated empirically with shrinkage
    - Among finishers, ranking follows PL: P(ranking) = prod lambda_k / sum remaining
    - Time-varying via exponential smoothing: lambda <- alpha * lambda + (1-alpha) * update
    - Alpha selected via leave-last-year-out CV

Prediction interface matches Stages 1-6:
    predict_proba(driver_id, prev_position, constructor_id) -> (21,) array
    where the Markov dependency on prev_position is injected via a correction
    factor learned from the global transition matrix.
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
# Plackett-Luce Utilities
# ---------------------------------------------------------------------------
def pl_log_likelihood(strengths, ranking):
    """
    Log-likelihood of a single race ranking under Plackett-Luce.

    Parameters
    ----------
    strengths : dict[driver_id -> float], positive strengths (lambda)
    ranking : list of driver_ids in finishing order (1st, 2nd, ..., last)

    Returns
    -------
    ll : float
    """
    ll = 0.0
    remaining = set(ranking)
    for driver in ranking:
        denom = sum(strengths[d] for d in remaining)
        if denom <= 0:
            break
        ll += np.log(strengths[driver]) - np.log(denom)
        remaining.discard(driver)
    return ll


def pl_mm_update(strengths, rankings):
    """
    Minorization-Maximization update for Plackett-Luce (Hunter 2004).

    One step of the MM algorithm. For each driver i:
        lambda_i_new = w_i / sum_r sum_{j: rank(i,r) <= j} [1 / sum_{k: rank(k,r)>=j} lambda_k]

    where w_i = number of times driver i appears in any ranking.

    Parameters
    ----------
    strengths : dict[driver_id -> float]
    rankings : list of list of driver_ids (each ranking is finisher-ordered)

    Returns
    -------
    new_strengths : dict[driver_id -> float]
    """
    numerator = defaultdict(float)
    denominator = defaultdict(float)

    for ranking in rankings:
        n = len(ranking)
        # Precompute suffix sums of strengths
        suffix_sum = np.zeros(n + 1)
        for j in range(n - 1, -1, -1):
            suffix_sum[j] = suffix_sum[j + 1] + strengths[ranking[j]]

        for i, driver in enumerate(ranking):
            numerator[driver] += 1.0  # driver appeared once in this ranking
            # Denominator contribution
            for j in range(i + 1):  # all stages where driver was still in
                if suffix_sum[j] > 0:
                    denominator[driver] += 1.0 / suffix_sum[j]

    new_strengths = {}
    for driver in strengths:
        if denominator[driver] > 0:
            new_strengths[driver] = numerator[driver] / denominator[driver]
        else:
            new_strengths[driver] = strengths[driver]

    return new_strengths


def pl_sample_ranking(strengths_array, rng, n_samples=5000):
    """
    Monte Carlo sampling of PL rankings.

    Parameters
    ----------
    strengths_array : (n_drivers,) array of positive strengths
    rng : numpy random generator
    n_samples : int

    Returns
    -------
    position_probs : (n_drivers, n_drivers) array
        position_probs[i, j] = P(driver i finishes in position j+1)
    """
    n = len(strengths_array)
    counts = np.zeros((n, n), dtype=int)

    for _ in range(n_samples):
        remaining = list(range(n))
        remaining_strengths = strengths_array.copy()

        for pos in range(n):
            # Normalize to get probabilities
            active_strengths = remaining_strengths[remaining]
            total = active_strengths.sum()
            if total <= 0:
                # Assign remaining positions uniformly
                for k, idx in enumerate(remaining):
                    counts[idx, pos + k] += 1
                break
            probs = active_strengths / total
            # Sample winner
            chosen_idx = rng.choice(len(remaining), p=probs)
            chosen_driver = remaining[chosen_idx]
            counts[chosen_driver, pos] += 1
            remaining.pop(chosen_idx)

    return counts / n_samples


# ---------------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------------
def prepare_race_rankings(meta_df):
    """
    Convert transition-level data back to race-level complete rankings.

    Parameters
    ----------
    meta_df : DataFrame with driver, constructor, season, race_order,
              prev_position, next_position columns

    Returns
    -------
    races : list of dicts with keys:
        season, race_order,
        entries: list of (driver_id, constructor_id, finish_pos, is_dnf)
    """
    races = []
    grouped = meta_df.groupby(["season", "race_order"])

    for (season, race_order), grp in sorted(grouped):
        entries = []
        for _, row in grp.iterrows():
            did = row["driver"]
            cid = row["constructor"]
            finish = int(row["next_position"])
            is_dnf = finish == DNF
            entries.append((did, cid, finish, is_dnf))

        races.append({
            "season": season,
            "race_order": race_order,
            "entries": entries,
        })

    return races


# ---------------------------------------------------------------------------
# Stage 8 Model
# ---------------------------------------------------------------------------
class TimeVaryingPlackettLuceF1:
    """
    Stage 8: Time-varying Plackett-Luce with constructor effects.

    Parameters
    ----------
    alpha_candidates : tuple of floats
        Smoothing parameter candidates for leave-last-year-out CV.
    dnf_shrinkage : float
        Shrinkage strength for DNF rate estimation.
    mm_iters : int
        MM algorithm iterations per race for strength estimation.
    n_mc_samples : int
        Number of MC samples for position marginal computation.
    """

    def __init__(
        self,
        alpha_candidates: tuple = (0.85, 0.9, 0.95, 0.99),
        dnf_shrinkage: float = 20.0,
        mm_iters: int = 10,
        n_mc_samples: int = 5000,
    ):
        self.alpha_candidates = alpha_candidates
        self.dnf_shrinkage = dnf_shrinkage
        self.mm_iters = mm_iters
        self.n_mc_samples = n_mc_samples

        # Fitted attributes
        self.driver_strengths_: Optional[dict] = None  # did -> float
        self.constructor_strengths_: Optional[dict] = None  # cid -> float
        self.dnf_rates_: Optional[dict] = None  # did -> float
        self.global_dnf_rate_: Optional[float] = None
        self.alpha_: Optional[float] = None
        self.global_correction_: Optional[np.ndarray] = None  # (22, 21)
        self.driver_ids_: Optional[list] = None
        self.constructor_ids_: Optional[list] = None
        self._all_driver_strengths: Optional[dict] = None  # final strengths

    @property
    def is_fitted(self) -> bool:
        return self.driver_strengths_ is not None

    def fit(
        self,
        prev_positions: np.ndarray,
        next_positions: np.ndarray,
        meta_df: pd.DataFrame,
    ) -> "TimeVaryingPlackettLuceF1":
        """
        Fit the time-varying PL model:
          1. Build race-level rankings from meta_df
          2. Select alpha via leave-last-year-out CV
          3. Fit on full data with best alpha
          4. Build Markov correction matrix
          5. Compute DNF rates
        """
        races = prepare_race_rankings(meta_df)
        years = sorted(meta_df["season"].unique())

        self.driver_ids_ = sorted(meta_df["driver"].unique().tolist())
        self.constructor_ids_ = sorted(meta_df["constructor"].unique().tolist())

        # Compute global transition matrix for Markov correction
        global_counts = np.zeros((N_PREV_STATES, N_OUTCOMES), dtype=int)
        for s, j in zip(prev_positions, next_positions):
            global_counts[s, j] += 1
        g_alpha = 0.5 + global_counts
        global_pi = g_alpha / g_alpha.sum(axis=1, keepdims=True)

        # Compute DNF rates
        self._compute_dnf_rates(meta_df)

        # Leave-last-year-out CV for alpha selection
        if len(years) >= 2:
            best_alpha = None
            best_val_ll = -np.inf

            val_year = years[-1]
            train_races = [r for r in races if r["season"] < val_year]
            val_races = [r for r in races if r["season"] == val_year]

            print(f"  Stage 8 (PL): CV with {len(train_races)} train races, "
                  f"{len(val_races)} val races (year {val_year})")

            for alpha in self.alpha_candidates:
                # Fit on train data
                strengths = self._fit_sequential(train_races, alpha)

                # Evaluate on validation data
                val_ll = 0.0
                for race in val_races:
                    finishers = [
                        (did, cid, pos) for did, cid, pos, is_dnf
                        in race["entries"] if not is_dnf
                    ]
                    if len(finishers) < 2:
                        continue
                    finishers.sort(key=lambda x: x[2])
                    ranking = [f[0] for f in finishers]

                    race_strengths = {}
                    for did in ranking:
                        d_str = strengths.get(did, 1.0)
                        cid = next(
                            (c for d, c, _, _ in race["entries"] if d == did),
                            None
                        )
                        c_str = strengths.get(f"C_{cid}", 1.0)
                        race_strengths[did] = d_str * c_str

                    val_ll += pl_log_likelihood(race_strengths, ranking)

                print(f"    alpha={alpha:.2f}: val LL = {val_ll:.1f}")
                if val_ll > best_val_ll:
                    best_val_ll = val_ll
                    best_alpha = alpha

            self.alpha_ = best_alpha
        else:
            self.alpha_ = 0.95  # default

        print(f"  Selected alpha = {self.alpha_}")

        # Fit on full data with best alpha
        all_strengths = self._fit_sequential(races, self.alpha_)

        # Separate driver and constructor strengths
        self.driver_strengths_ = {}
        self.constructor_strengths_ = {}
        for key, val in all_strengths.items():
            if isinstance(key, str) and key.startswith("C_"):
                cid = int(key[2:])
                self.constructor_strengths_[cid] = val
            else:
                self.driver_strengths_[key] = val

        self._all_driver_strengths = all_strengths

        # Build Markov correction matrix
        # correction[s, k] = P_global(k|s) / P_global(k)
        # where P_global(k) = sum_s P_global(k|s) * P(s) [marginal]
        marginal = global_pi.mean(axis=0)  # approximate marginal
        self.global_correction_ = np.zeros((N_PREV_STATES, N_OUTCOMES))
        for s in range(N_PREV_STATES):
            for k in range(N_OUTCOMES):
                if marginal[k] > 1e-10:
                    self.global_correction_[s, k] = global_pi[s, k] / marginal[k]
                else:
                    self.global_correction_[s, k] = 1.0

        print(f"  Stage 8 (PL): fitted {len(self.driver_strengths_)} drivers, "
              f"{len(self.constructor_strengths_)} constructors")

        # Print top strengths
        sorted_drivers = sorted(
            self.driver_strengths_.items(),
            key=lambda x: x[1], reverse=True
        )[:5]
        for did, strength in sorted_drivers:
            print(f"    Driver {did}: strength = {strength:.3f}")

        return self

    def _fit_sequential(self, races, alpha):
        """
        Fit PL strengths sequentially with exponential smoothing in log space.

        Works with log-strengths to prevent unbounded growth:
            log_lambda_i(t) = alpha * log_lambda_i(t-1) + (1-alpha) * log_update_i

        Driver and constructor effects are additive in log space:
            log_lambda_composite = log_mu_driver + log_beta_constructor

        Returns
        -------
        strengths : dict mapping driver_id and "C_{cid}" to float strengths
        """
        log_driver = {}  # did -> float (log-strength)
        log_constructor = {}  # "C_{cid}" -> float

        for race in races:
            # Extract finisher ranking
            finishers = [
                (did, cid, pos) for did, cid, pos, is_dnf
                in race["entries"] if not is_dnf
            ]
            if len(finishers) < 2:
                continue

            finishers.sort(key=lambda x: x[2])
            ranking = [f[0] for f in finishers]
            driver_constructors = {f[0]: f[1] for f in finishers}

            # Current composite strengths (exp space) for MM algorithm
            race_strengths = {}
            for did in ranking:
                log_d = log_driver.get(did, 0.0)
                cid = driver_constructors[did]
                log_c = log_constructor.get(f"C_{cid}", 0.0)
                race_strengths[did] = np.exp(log_d + log_c)

            # MM updates to get race-level strength estimates
            updated = race_strengths.copy()
            for _ in range(self.mm_iters):
                updated = pl_mm_update(updated, [ranking])

            # Update log-strengths via exponential smoothing

            # First compute log of MM-updated composite per driver
            log_updated = {
                did: np.log(max(updated[did], 1e-300)) for did in ranking
            }

            # Decompose: updated composite = driver_new * constructor_old
            # So log_driver_new = log_composite_updated - log_constructor_old
            for did in ranking:
                cid = driver_constructors[did]
                log_c = log_constructor.get(f"C_{cid}", 0.0)
                log_d_new = log_updated[did] - log_c

                if did in log_driver:
                    log_driver[did] = (
                        alpha * log_driver[did] + (1 - alpha) * log_d_new
                    )
                else:
                    log_driver[did] = log_d_new

            # Update constructor log-strengths
            # Constructor signal = avg composite - avg driver (in log space)
            for cid in set(driver_constructors.values()):
                c_key = f"C_{cid}"
                team_drivers = [
                    d for d, c in driver_constructors.items() if c == cid
                ]
                if len(team_drivers) > 0:
                    avg_log_composite = np.mean([
                        log_updated[d] for d in team_drivers
                    ])
                    avg_log_driver = np.mean([
                        log_driver.get(d, 0.0) for d in team_drivers
                    ])
                    log_c_new = avg_log_composite - avg_log_driver

                    if c_key in log_constructor:
                        log_constructor[c_key] = (
                            alpha * log_constructor[c_key]
                            + (1 - alpha) * log_c_new
                        )
                    else:
                        log_constructor[c_key] = log_c_new

            # Normalize: center driver log-strengths at 0
            if log_driver:
                mean_log = np.mean(list(log_driver.values()))
                for did in log_driver:
                    log_driver[did] -= mean_log

        # Convert to exp space
        strengths = {}
        for did, log_s in log_driver.items():
            strengths[did] = np.exp(log_s)
        for c_key, log_s in log_constructor.items():
            strengths[c_key] = np.exp(log_s)

        return strengths

    def _compute_dnf_rates(self, meta_df):
        """Compute per-driver DNF rates with shrinkage."""
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

        self.dnf_rates_ = {}
        for did in total_races:
            n = total_races[did]
            empirical = total_dnfs[did] / max(n, 1)
            # Shrinkage toward global
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
        """
        P(next | prev, driver, constructor).

        1. Compute base PL position distribution via MC sampling
        2. Incorporate DNF probability
        3. Apply Markov correction for prev_position
        """
        self._check_fitted()

        # Get driver and constructor strength
        d_str = self.driver_strengths_.get(driver_id, 1.0)
        c_str = 1.0
        if constructor_id is not None:
            c_str = self.constructor_strengths_.get(constructor_id, 1.0)
        composite = d_str * c_str

        # Get DNF rate
        p_dnf = self.dnf_rates_.get(driver_id, self.global_dnf_rate_)

        # Build position distribution
        probs = self._compute_position_probs(
            driver_id, composite, p_dnf, prev_position
        )

        return probs

    def predict_proba_new_driver(
        self,
        prev_position: int,
        constructor_id: Optional = None,
        **kwargs,
    ) -> np.ndarray:
        """Prediction for unseen driver."""
        self._check_fitted()

        d_str = 1.0  # average driver
        c_str = 1.0
        if constructor_id is not None:
            c_str = self.constructor_strengths_.get(constructor_id, 1.0)
        composite = d_str * c_str

        p_dnf = self.global_dnf_rate_

        probs = self._compute_position_probs(
            None, composite, p_dnf, prev_position
        )

        return probs

    def _compute_position_probs(
        self, driver_id, composite_strength, p_dnf, prev_position
    ):
        """
        Compute (21,) probability array for a driver.

        Uses the driver's strength relative to all other known drivers
        to compute position marginals via the PL model.
        """
        # Build strength array for all known drivers
        # We approximate by using the average "field" strength
        n_field = 20  # typical F1 field size
        all_strengths = []

        # Collect all composite strengths
        for did in self.driver_strengths_:
            d_s = self.driver_strengths_[did]
            # Use their most recent constructor
            # (approximate - we don't track this perfectly)
            all_strengths.append(d_s)

        if len(all_strengths) == 0:
            all_strengths = [1.0] * n_field

        # Build a representative field: driver of interest + n_field-1 others
        # Use the distribution of strengths from training
        sorted_strengths = sorted(all_strengths, reverse=True)
        # Take up to n_field-1 other drivers (representative sample)
        field_strengths = sorted_strengths[:n_field - 1]
        while len(field_strengths) < n_field - 1:
            field_strengths.append(np.median(all_strengths))

        # Our driver is index 0
        all_field = np.array([composite_strength] + field_strengths)
        all_field = np.maximum(all_field, 1e-10)

        # MC sample rankings
        rng = np.random.default_rng(
            hash(str(driver_id)) % (2**31) if driver_id else 42
        )
        pos_probs_matrix = pl_sample_ranking(all_field, rng, self.n_mc_samples)

        # Our driver's position distribution (among finishers)
        finish_probs = pos_probs_matrix[0]  # (n_field,) = P(pos j+1 | finish)

        # Map to (21,) output: DNF + P1-P20
        probs = np.zeros(N_OUTCOMES)
        probs[0] = p_dnf

        # Distribute finishing probabilities across positions 1-20
        remaining_prob = 1.0 - p_dnf
        for j in range(min(len(finish_probs), 20)):
            probs[j + 1] = remaining_prob * finish_probs[j]

        # Apply Markov correction
        if self.global_correction_ is not None:
            correction = self.global_correction_[prev_position]
            probs *= correction
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
    print("F1 Time-Varying Plackett-Luce — Stage 8")
    print("=" * 65)

    data_dir = Path(__file__).parent.parent / "data"

    loader = F1DataLoader(data_dir)
    df = loader.load_merged(min_year=2020, max_year=2024)
    print(f"\nLoaded {len(df)} results, "
          f"{df['year'].nunique()} seasons")

    prev_pos, next_pos, meta = prepare_transitions(df)
    print(f"Total transitions: {len(prev_pos)}")

    model = TimeVaryingPlackettLuceF1(
        alpha_candidates=(0.85, 0.9, 0.95, 0.99),
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

    # Test prediction
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
