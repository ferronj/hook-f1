"""
F1 Dirichlet-Multinomial Markov Model
======================================
Stage 2: Driver-specific partial pooling via Empirical Bayes

Building on Stage 1 (global conjugate model), this module adds per-driver
transition matrices that are regularized toward the global mean through a
shared concentration parameter kappa.

Model (for driver i, previous state s):
    F_{i,r} | F_{i,r-1}=s  ~  Multinomial(1, p_{i,s})
    p_{i,s}  ~  Dirichlet(alpha_{i,s})
    alpha_{i,s} = kappa * pi_s^{global} + n_{i,s}^{obs}    [pseudo-count form]

But more precisely, the partial-pooling prior IS the Dirichlet:
    alpha_{i,s} = kappa * pi_s^{global}

And the posterior after observing driver i's transitions is:
    alpha_{i,s}^{post} = kappa * pi_s^{global} + n_{i,s}

Where:
    pi_s^{global} = posterior mean transition probs from Stage 1 (global model)
    kappa          = concentration parameter controlling pooling strength
    n_{i,s}        = driver i's observed count vector for previous state s

Partial pooling behavior:
    kappa -> 0:    No pooling (each driver estimated independently)
    kappa -> inf:  Complete pooling (all drivers = global mean)
    kappa ~ O(n):  Partial pooling (global prior has weight ~kappa vs ~n obs)

kappa is learned via Empirical Bayes (maximize marginal likelihood across
all drivers), preserving full conjugacy for the per-driver posteriors.

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

# Import shared constants and data loading from Stage 1
# In practice these would be imported; here we redefine for self-containment.

# ---------------------------------------------------------------------------
# Constants (shared with Stage 1)
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
# Data Loading (shared with Stage 1)
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
            .merge(drivers[["driverId", "driver_name"]], on="driverId", how="left")
            .merge(constructors, on="constructorId", how="left")
            .merge(status, on="statusId", how="left")
        )

        if min_year is not None:
            df = df[df["year"] >= min_year]
        if max_year is not None:
            df = df[df["year"] <= max_year]

        df["position_mapped"] = df.apply(self._map_position, axis=1)
        df = df.sort_values(["driverId", "year", "round"]).reset_index(drop=True)
        return df

    @staticmethod
    def _map_position(row) -> int:
        status = str(row.get("status_text", "")).lower().strip()
        is_classified = any(kw in status for kw in CLASSIFIED_STATUS_KEYWORDS)
        if is_classified:
            pos = row.get("positionOrder", np.nan)
            if pd.isna(pos):
                return DNF
            return int(min(pos, 20))
        return DNF


# ---------------------------------------------------------------------------
# Transition Preparation (shared with Stage 1)
# ---------------------------------------------------------------------------
def prepare_transitions(
    df: pd.DataFrame,
    driver_col: str = "driverId",
    season_col: str = "year",
    race_order_col: str = "round",
    position_col: str = "position_mapped",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Convert race results into (prev_position, next_position) arrays.
    First race of each season uses START (21) as previous position.
    """
    df = df.sort_values([driver_col, season_col, race_order_col]).copy()
    prev_list, next_list, meta_rows = [], [], []

    for driver, grp in df.groupby(driver_col):
        positions = grp[position_col].values
        seasons = grp[season_col].values
        race_orders = grp[race_order_col].values

        for i in range(len(positions)):
            if i == 0 or seasons[i] != seasons[i - 1]:
                prev = START
            else:
                prev = int(positions[i - 1])

            prev_list.append(prev)
            next_list.append(int(positions[i]))
            meta_rows.append({
                "driver": driver,
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


def prepare_driver_transitions(
    meta_df: pd.DataFrame,
) -> dict[int | str, np.ndarray]:
    """
    From the meta DataFrame, build per-driver count matrices.

    Returns
    -------
    driver_counts : dict mapping driver_id -> (N_PREV_STATES, N_OUTCOMES) array
    """
    driver_counts = {}
    for driver, grp in meta_df.groupby("driver"):
        counts = np.zeros((N_PREV_STATES, N_OUTCOMES), dtype=int)
        for _, row in grp.iterrows():
            s = int(row["prev_position"])
            j = int(row["next_position"])
            counts[s, j] += 1
        driver_counts[driver] = counts
    return driver_counts


# ---------------------------------------------------------------------------
# Marginal Likelihood Utilities
# ---------------------------------------------------------------------------
def _dirichlet_multinomial_log_ml(
    alpha: np.ndarray, counts: np.ndarray
) -> float:
    """
    Log marginal likelihood for a single Dirichlet-Multinomial:

        log P(n | alpha) = log Gamma(sum alpha) - log Gamma(sum alpha + N)
                         + sum_j [log Gamma(alpha_j + n_j) - log Gamma(alpha_j)]

    Parameters
    ----------
    alpha : (K,) array, Dirichlet concentration parameters
    counts : (K,) array, observed counts
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
    kappa: float,
    global_pi: np.ndarray,
    driver_counts: dict,
) -> float:
    """
    Total log marginal likelihood across all drivers and all previous states.

    alpha_{i,s} = kappa * pi_s^{global}  (the prior for driver i, state s)

    We compute:
        sum_i sum_s  log P(n_{i,s} | kappa * pi_s)
    """
    lml = 0.0
    for driver, counts in driver_counts.items():
        for s in range(N_PREV_STATES):
            n_s = counts[s]
            if n_s.sum() == 0:
                continue
            alpha_s = kappa * global_pi[s]
            # Guard against zero alpha entries
            alpha_s = np.maximum(alpha_s, 1e-10)
            lml += _dirichlet_multinomial_log_ml(alpha_s, n_s)
    return lml


# ---------------------------------------------------------------------------
# Stage 2 Model
# ---------------------------------------------------------------------------
class PartialPooledDirichletF1:
    """
    Stage 2: Driver-specific transition matrices with partial pooling
    toward a global mean, controlled by a shared concentration kappa.

    Model:
        p_{i,s} ~ Dirichlet(kappa * pi_s^{global})
        F_{i,r} | F_{i,r-1}=s ~ Categorical(p_{i,s})

    kappa is estimated via Empirical Bayes (maximizing marginal likelihood).

    Parameters
    ----------
    prior_alpha_global : float
        Symmetric Dirichlet prior for the Stage 1 global model.
    kappa_init : float
        Initial guess for kappa optimization.
    kappa_bounds : tuple
        (min, max) bounds for kappa search.
    """

    def __init__(
        self,
        prior_alpha_global: float = 1.0,
        kappa_init: float = 10.0,
        kappa_bounds: tuple[float, float] = (0.1, 500.0),
    ):
        self.prior_alpha_global = prior_alpha_global
        self.kappa_init = kappa_init
        self.kappa_bounds = kappa_bounds

        # Fitted attributes (populated by .fit())
        self.global_pi_: Optional[np.ndarray] = None       # (22, 21)
        self.global_counts_: Optional[np.ndarray] = None    # (22, 21)
        self.kappa_: Optional[float] = None
        self.driver_counts_: Optional[dict] = None
        self.driver_ids_: Optional[list] = None
        self.opt_result_: Optional[optimize.OptimizeResult] = None

    @property
    def is_fitted(self) -> bool:
        return self.kappa_ is not None

    def fit(
        self,
        prev_positions: np.ndarray,
        next_positions: np.ndarray,
        meta_df: pd.DataFrame,
    ) -> "PartialPooledDirichletF1":
        """
        Fit the model:
          1. Compute global transition matrix (Stage 1 posterior mean).
          2. Compute per-driver count matrices.
          3. Optimize kappa via marginal likelihood.

        Parameters
        ----------
        prev_positions, next_positions : arrays from prepare_transitions()
        meta_df : DataFrame from prepare_transitions() with 'driver' column
        """
        # Step 1: Global model (Stage 1)
        global_counts = np.zeros((N_PREV_STATES, N_OUTCOMES), dtype=int)
        for s, j in zip(prev_positions, next_positions):
            global_counts[s, j] += 1

        self.global_counts_ = global_counts
        global_alpha = self.prior_alpha_global + global_counts
        self.global_pi_ = global_alpha / global_alpha.sum(axis=1, keepdims=True)

        # Step 2: Per-driver counts
        self.driver_counts_ = prepare_driver_transitions(meta_df)
        self.driver_ids_ = sorted(self.driver_counts_.keys())

        # Step 3: Optimize kappa
        def neg_lml(log_kappa):
            kappa = np.exp(log_kappa)
            return -total_log_marginal_likelihood(
                kappa, self.global_pi_, self.driver_counts_
            )

        # Optimize in log-space for stability
        log_k0 = np.log(self.kappa_init)
        log_bounds = (np.log(self.kappa_bounds[0]), np.log(self.kappa_bounds[1]))

        result = optimize.minimize_scalar(
            neg_lml,
            bounds=log_bounds,
            method="bounded",
            options={"xatol": 1e-4, "maxiter": 200},
        )
        self.kappa_ = np.exp(result.x)
        self.opt_result_ = result

        return self

    def driver_prior_alpha(self, driver_id) -> np.ndarray:
        """
        Prior Dirichlet alpha for a specific driver:
            alpha_{i,s} = kappa * pi_s^{global}

        Shape: (N_PREV_STATES, N_OUTCOMES)
        """
        self._check_fitted()
        return self.kappa_ * self.global_pi_

    def driver_posterior_alpha(self, driver_id) -> np.ndarray:
        """
        Posterior Dirichlet alpha for a specific driver:
            alpha_{i,s}^{post} = kappa * pi_s^{global} + n_{i,s}

        Shape: (N_PREV_STATES, N_OUTCOMES)
        """
        self._check_fitted()
        prior = self.driver_prior_alpha(driver_id)
        counts = self.driver_counts_.get(driver_id, np.zeros_like(prior))
        return prior + counts

    def driver_transition_matrix(self, driver_id) -> np.ndarray:
        """Posterior mean transition matrix for a specific driver."""
        alpha = self.driver_posterior_alpha(driver_id)
        return alpha / alpha.sum(axis=1, keepdims=True)

    def predict_proba(self, driver_id, prev_position: int) -> np.ndarray:
        """P(next | prev=prev_position) for a specific driver."""
        return self.driver_transition_matrix(driver_id)[prev_position]

    def predict_distribution(
        self, driver_id, prev_position: int
    ) -> stats.dirichlet:
        """Full posterior Dirichlet for driver/prev_position."""
        return stats.dirichlet(
            self.driver_posterior_alpha(driver_id)[prev_position]
        )

    def sample_predictive(
        self, driver_id, prev_position: int, n_samples: int = 1000
    ) -> np.ndarray:
        """Posterior predictive samples for a specific driver."""
        alpha = self.driver_posterior_alpha(driver_id)[prev_position]
        p_samples = stats.dirichlet.rvs(alpha, size=n_samples)
        return np.array([
            np.random.choice(N_OUTCOMES, p=p) for p in p_samples
        ])

    def predict_proba_new_driver(self, prev_position: int) -> np.ndarray:
        """
        Prediction for an unseen driver (uses global prior only).
        Equivalent to kappa * pi_s / sum(kappa * pi_s) = pi_s.
        """
        self._check_fitted()
        return self.global_pi_[prev_position]

    def credible_interval(
        self, driver_id, prev_position: int, outcome: int,
        alpha: float = 0.05, n_samples: int = 10000,
    ) -> tuple[float, float]:
        """(1-alpha) credible interval for P(outcome | prev, driver)."""
        post_alpha = self.driver_posterior_alpha(driver_id)[prev_position]
        p_samples = stats.dirichlet.rvs(post_alpha, size=n_samples)
        probs = p_samples[:, outcome]
        return (
            np.percentile(probs, 100 * alpha / 2),
            np.percentile(probs, 100 * (1 - alpha / 2)),
        )

    def pooling_factor(self, driver_id) -> float:
        """
        How much this driver's posterior is dominated by the global prior
        vs. their own data. Returns kappa / (kappa + n_driver) averaged
        over active states.

        Values near 1 = mostly global (sparse driver data).
        Values near 0 = mostly driver-specific (lots of data).
        """
        self._check_fitted()
        counts = self.driver_counts_.get(
            driver_id, np.zeros((N_PREV_STATES, N_OUTCOMES))
        )
        n_total = counts.sum()
        if n_total == 0:
            return 1.0
        # Average pooling across states weighted by observation count
        total_prior = self.kappa_ * N_OUTCOMES  # sum of prior alpha per row
        return total_prior / (total_prior + n_total / N_PREV_STATES)

    def log_marginal_likelihood(self) -> float:
        """Total log marginal likelihood at the fitted kappa."""
        self._check_fitted()
        return total_log_marginal_likelihood(
            self.kappa_, self.global_pi_, self.driver_counts_
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
        for _, row in meta_df.iterrows():
            driver = row["driver"]
            s = int(row["prev_position"])
            j = int(row["next_position"])
            tm = self.driver_transition_matrix(driver)
            ll += np.log(tm[s, j] + 1e-300)
        return ll

    def summary(self, driver_id=None) -> pd.DataFrame:
        """
        Transition matrix summary. If driver_id is given, shows that
        driver's posterior. Otherwise shows the global prior.
        """
        self._check_fitted()
        if driver_id is not None:
            tm = self.driver_transition_matrix(driver_id)
            label = f"Driver {driver_id}"
        else:
            tm = self.global_pi_
            label = "Global"
        df = pd.DataFrame(tm, index=PREV_STATE_LABELS, columns=OUTCOME_LABELS)
        df.index.name = f"Previous Position ({label})"
        return df

    def driver_summary_table(self) -> pd.DataFrame:
        """
        Summary across all drivers: total observations, pooling factor,
        and most probable outcome from each common previous state.
        """
        self._check_fitted()
        rows = []
        for d in self.driver_ids_:
            counts = self.driver_counts_[d]
            n_obs = counts.sum()
            pf = self.pooling_factor(d)
            # Most likely next position from P1, P10, START
            tm = self.driver_transition_matrix(d)
            rows.append({
                "driver": d,
                "n_transitions": n_obs,
                "pooling_factor": round(pf, 3),
                "mode_from_P1": OUTCOME_LABELS[np.argmax(tm[1])],
                "mode_from_P10": OUTCOME_LABELS[np.argmax(tm[10])],
                "mode_from_START": OUTCOME_LABELS[np.argmax(tm[START])],
            })
        return pd.DataFrame(rows).sort_values("n_transitions", ascending=False)

    def kappa_profile(
        self,
        kappa_range: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Profile log marginal likelihood over a range of kappa values.
        Useful for visualizing the optimization landscape.
        """
        self._check_fitted()
        if kappa_range is None:
            kappa_range = np.logspace(-1, np.log10(500), 50)
        rows = []
        for k in kappa_range:
            lml = total_log_marginal_likelihood(
                k, self.global_pi_, self.driver_counts_
            )
            rows.append({"kappa": k, "log_marginal_likelihood": lml})
        return pd.DataFrame(rows)

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    print("=" * 65)
    print("F1 Dirichlet-Multinomial — Stage 2: Partial Pooling")
    print("=" * 65)

    # --- Load data ---
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/f1")

    if (data_dir / "results.csv").exists():
        print(f"\nLoading Kaggle data from {data_dir} ...")
        loader = F1DataLoader(data_dir)
        df = loader.load_merged(min_year=2014)
        print(f"  Rows: {len(df)}, Seasons: {df['year'].nunique()}, "
              f"Drivers: {df['driverId'].nunique()}")
        prev_pos, next_pos, meta = prepare_transitions(df)
        # Attach driver names for display
        driver_names = (
            df[["driverId", "driver_name"]]
            .drop_duplicates()
            .set_index("driverId")["driver_name"]
            .to_dict()
        )
    else:
        print(f"\nKaggle data not found at {data_dir}; using synthetic data.")
        rng = np.random.default_rng(42)
        n_drivers, n_races = 20, 50
        rows = []
        for d in range(1, n_drivers + 1):
            skill = d  # lower = better
            for r in range(1, n_races + 1):
                pos = int(np.clip(rng.normal(skill, 3), 1, 20))
                if rng.random() < 0.05:
                    pos = DNF
                rows.append({
                    "driver": d, "season": 2020 + r // 23,
                    "race_order": r % 23 + 1,
                    "prev_position": START, "next_position": pos,
                })
        meta = pd.DataFrame(rows)
        # Recompute prev properly
        for d, grp in meta.groupby("driver"):
            idxs = grp.index.tolist()
            for i, idx in enumerate(idxs):
                if i > 0 and meta.loc[idxs[i], "season"] == meta.loc[idxs[i-1], "season"]:
                    meta.loc[idx, "prev_position"] = meta.loc[idxs[i-1], "next_position"]
        prev_pos = meta["prev_position"].values.astype(int)
        next_pos = meta["next_position"].values.astype(int)
        driver_names = {d: f"Driver_{d:02d}" for d in range(1, n_drivers + 1)}

    print(f"Total transitions: {len(prev_pos)}")
    n_drivers = meta["driver"].nunique()
    print(f"Unique drivers: {n_drivers}")

    # --- Fit Stage 2 ---
    model = PartialPooledDirichletF1(
        prior_alpha_global=1.0,
        kappa_init=10.0,
        kappa_bounds=(0.1, 500.0),
    )
    model.fit(prev_pos, next_pos, meta)

    print(f"\n--- Empirical Bayes Results ---")
    print(f"  Optimal kappa: {model.kappa_:.2f}")
    print(f"  Log marginal likelihood: {model.log_marginal_likelihood():.1f}")
    print(f"  Interpretation: global prior has effective weight of "
          f"~{model.kappa_:.0f} pseudo-observations per row")

    # --- Kappa profile ---
    print("\n--- Kappa Profile (log marginal likelihood) ---")
    profile = model.kappa_profile()
    peak = profile.loc[profile["log_marginal_likelihood"].idxmax()]
    for _, row in profile.iloc[::10].iterrows():
        marker = " <-- optimal" if abs(row["kappa"] - peak["kappa"]) < 1 else ""
        print(f"  kappa={row['kappa']:7.1f}  "
              f"LML={row['log_marginal_likelihood']:12.1f}{marker}")

    # --- Driver summary ---
    print("\n--- Driver Summary (top 10 by data volume) ---")
    ds = model.driver_summary_table().head(10)
    if driver_names:
        ds["driver_name"] = ds["driver"].map(driver_names).fillna("?")
        cols = ["driver_name", "n_transitions", "pooling_factor",
                "mode_from_P1", "mode_from_P10", "mode_from_START"]
    else:
        cols = ds.columns.tolist()
    print(ds[cols].to_string(index=False))

    # --- Compare pooling levels ---
    if len(model.driver_ids_) >= 2:
        # Find driver with most data and one with least
        counts_by_driver = {
            d: model.driver_counts_[d].sum() for d in model.driver_ids_
        }
        top_driver = max(counts_by_driver, key=counts_by_driver.get)
        bot_driver = min(counts_by_driver, key=counts_by_driver.get)
        top_name = driver_names.get(top_driver, top_driver)
        bot_name = driver_names.get(bot_driver, bot_driver)

        print(f"\n--- Pooling Comparison ---")
        print(f"  Most data:  {top_name} "
              f"({counts_by_driver[top_driver]} obs, "
              f"pooling={model.pooling_factor(top_driver):.3f})")
        print(f"  Least data: {bot_name} "
              f"({counts_by_driver[bot_driver]} obs, "
              f"pooling={model.pooling_factor(bot_driver):.3f})")

        # Compare predictions from P5
        print(f"\n  P(next | prev=P5) comparison:")
        p_top = model.predict_proba(top_driver, prev_position=5)
        p_bot = model.predict_proba(bot_driver, prev_position=5)
        p_glo = model.predict_proba_new_driver(prev_position=5)

        print(f"  {'Outcome':<8} {'Global':>8} {top_name[:12]:>12} {bot_name[:12]:>12}")
        for idx in np.argsort(p_glo)[::-1][:7]:
            print(f"  {OUTCOME_LABELS[idx]:<8} {p_glo[idx]:8.3f} "
                  f"{p_top[idx]:12.3f} {p_bot[idx]:12.3f}")

    # --- Posterior predictive for a specific driver ---
    if model.driver_ids_:
        example_driver = model.driver_ids_[0]
        ex_name = driver_names.get(example_driver, example_driver)
        print(f"\n--- Posterior Predictive: {ex_name}, prev=P3 ---")
        samples = model.sample_predictive(
            example_driver, prev_position=3, n_samples=2000
        )
        unique, counts = np.unique(samples, return_counts=True)
        top5 = np.argsort(counts)[::-1][:5]
        for idx in top5:
            lo, hi = model.credible_interval(
                example_driver, 3, unique[idx]
            )
            print(f"  {OUTCOME_LABELS[unique[idx]]}: "
                  f"{counts[idx]/2000:.3f}  [95% CI: {lo:.3f}–{hi:.3f}]")

    # --- Model comparison: Stage 1 vs Stage 2 ---
    print("\n--- Model Comparison ---")
    # Stage 1 baseline (complete pooling = global model for everyone)
    from stage1_global_transition import DirichletMultinomialF1  # noqa
    try:
        s1 = DirichletMultinomialF1(prior_alpha=1.0).fit(prev_pos, next_pos)
        ll_s1 = s1.log_likelihood(prev_pos, next_pos)
        print(f"  Stage 1 (complete pooling) LL: {ll_s1:.1f}")
    except Exception:
        print("  (Stage 1 import unavailable; skipping comparison)")

    ll_s2 = model.log_likelihood(prev_pos, next_pos, meta)
    print(f"  Stage 2 (partial pooling)  LL: {ll_s2:.1f}")
    print(f"  Stage 2 log marginal lik:     {model.log_marginal_likelihood():.1f}")
