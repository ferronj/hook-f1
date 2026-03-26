"""
F1 Dirichlet-Multinomial Markov Model
======================================
Stage 1: Global transition matrix (conjugate update, no MCMC needed)

State space:
  - 0 = DNF (Did Not Finish / retired / disqualified)
  - 1-20 = Finishing positions (capped at 20)
  - 21 = START (only valid as a "previous" state, first race of season)

Transition matrix shape: (22 previous states) x (21 outcomes)
  - Rows: previous finishing position (0-21, including START)
  - Cols: next finishing position (0-20, excluding START)

Data source: Kaggle F1 World Championship dataset
  https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DNF = 0
POSITIONS = list(range(1, 21))  # 1-20
START = 21

N_PREV_STATES = 22   # 0-21 (DNF, P1-P20, START)
N_OUTCOMES = 21       # 0-20 (DNF, P1-P20); START is never an outcome
OUTCOME_LABELS = ["DNF"] + [f"P{i}" for i in range(1, 21)]
PREV_STATE_LABELS = ["DNF"] + [f"P{i}" for i in range(1, 21)] + ["START"]

# Status IDs in the Kaggle dataset that indicate a classified finish
# (driver completed the race or was classified despite not finishing all laps)
CLASSIFIED_STATUS_KEYWORDS = {"finished", "lap", "laps"}


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
class F1DataLoader:
    """
    Loads and joins the Kaggle F1 dataset CSVs into a unified race results
    DataFrame suitable for transition modeling.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing the CSV files (results.csv, races.csv,
        drivers.csv, constructors.csv, status.csv, circuits.csv).
    """

    REQUIRED_FILES = [
        "results.csv", "races.csv", "drivers.csv",
        "constructors.csv", "status.csv",
    ]

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self._validate_files()
        self._raw: dict[str, pd.DataFrame] = {}
        self._merged: Optional[pd.DataFrame] = None

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
        """Load all raw CSVs into a dict keyed by table name."""
        if not self._raw:
            for f in self.REQUIRED_FILES:
                name = f.replace(".csv", "")
                self._raw[name] = pd.read_csv(
                    self.data_dir / f, na_values=["\\N", ""]
                )
            # Optional: circuits
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
        """
        Load and merge all tables into a single race-results DataFrame.

        Returns DataFrame with columns:
            resultId, raceId, driverId, constructorId, grid, positionOrder,
            statusId, year, round, circuitId, race_name, race_date,
            driver_name, constructor_name, status_text, position_mapped
        """
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
        drivers["driver_name"] = (
            drivers["forename"] + " " + drivers["surname"]
        )

        constructors = raw["constructors"][
            ["constructorId", "name"]
        ].rename(columns={"name": "constructor_name"})

        status = raw["status"].rename(columns={"status": "status_text"})

        # Merge
        df = (
            results
            .merge(races, on="raceId", how="left")
            .merge(drivers[["driverId", "driver_name"]], on="driverId", how="left")
            .merge(constructors, on="constructorId", how="left")
            .merge(status, on="statusId", how="left")
        )

        # Filter years
        if min_year is not None:
            df = df[df["year"] >= min_year]
        if max_year is not None:
            df = df[df["year"] <= max_year]

        # Map positions to our state space
        df["position_mapped"] = df.apply(self._map_position, axis=1)

        df = df.sort_values(["driverId", "year", "round"]).reset_index(drop=True)
        self._merged = df
        return df

    @staticmethod
    def _map_position(row) -> int:
        """
        Map a result row to our state space {0=DNF, 1-20=positions}.

        Logic:
          - If status_text indicates classification (contains "Finished"
            or "+N Lap(s)"), use positionOrder capped at 20.
          - Otherwise, DNF (0).
        """
        status = str(row.get("status_text", "")).lower().strip()
        is_classified = any(kw in status for kw in CLASSIFIED_STATUS_KEYWORDS)

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
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Convert a race results DataFrame into transition arrays.

    First race of each season for each driver uses START (21) as prev.

    Returns
    -------
    prev_positions : np.ndarray of int
    next_positions : np.ndarray of int
    meta_df : pd.DataFrame with columns [driver, season, race_order,
              prev_position, next_position] plus any extra ID columns
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


# ---------------------------------------------------------------------------
# Core Model
# ---------------------------------------------------------------------------
class DirichletMultinomialF1:
    """
    Stage 1: Global Dirichlet-Multinomial transition model.

    For each previous state s, we maintain:
        p_s ~ Dirichlet(alpha_s)
        F_next | F_prev=s ~ Multinomial(1, p_s)

    Conjugate update:
        alpha_s_posterior = alpha_s_prior + counts_s

    Parameters
    ----------
    prior_alpha : float or np.ndarray
        Prior concentration. Float gives symmetric Dirichlet.
        Array of shape (N_OUTCOMES,) or (N_PREV_STATES, N_OUTCOMES).
    """

    def __init__(self, prior_alpha: float | np.ndarray = 1.0):
        if isinstance(prior_alpha, (int, float)):
            self.prior_alpha = np.full((N_PREV_STATES, N_OUTCOMES), prior_alpha)
        elif isinstance(prior_alpha, np.ndarray):
            if prior_alpha.shape == (N_OUTCOMES,):
                self.prior_alpha = np.tile(prior_alpha, (N_PREV_STATES, 1))
            elif prior_alpha.shape == (N_PREV_STATES, N_OUTCOMES):
                self.prior_alpha = prior_alpha.copy()
            else:
                raise ValueError(
                    f"prior_alpha shape {prior_alpha.shape} not supported."
                )
        else:
            self.prior_alpha = np.full((N_PREV_STATES, N_OUTCOMES), float(prior_alpha))

        self.counts = np.zeros((N_PREV_STATES, N_OUTCOMES), dtype=int)

    @property
    def posterior_alpha(self) -> np.ndarray:
        """Posterior Dirichlet parameters: prior + counts."""
        return self.prior_alpha + self.counts

    @property
    def transition_matrix(self) -> np.ndarray:
        """Posterior mean transition probs: E[p_s] = alpha_s / sum(alpha_s)."""
        alpha = self.posterior_alpha
        return alpha / alpha.sum(axis=1, keepdims=True)

    def fit(
        self,
        prev_positions: np.ndarray,
        next_positions: np.ndarray,
    ) -> "DirichletMultinomialF1":
        """Update counts from observed (prev, next) transition pairs."""
        assert len(prev_positions) == len(next_positions)
        for s, j in zip(prev_positions, next_positions):
            self.counts[s, j] += 1
        return self

    def predict_proba(self, prev_position: int) -> np.ndarray:
        """Posterior mean P(next | prev=prev_position)."""
        return self.transition_matrix[prev_position]

    def predict_distribution(self, prev_position: int) -> stats.dirichlet:
        """Full posterior Dirichlet for row `prev_position`."""
        return stats.dirichlet(self.posterior_alpha[prev_position])

    def sample_predictive(
        self, prev_position: int, n_samples: int = 1000
    ) -> np.ndarray:
        """
        Posterior predictive samples integrating over p uncertainty.
        Returns array of sampled outcome indices in {0, ..., N_OUTCOMES-1}.
        """
        p_samples = stats.dirichlet.rvs(
            self.posterior_alpha[prev_position], size=n_samples
        )
        return np.array([
            np.random.choice(N_OUTCOMES, p=p) for p in p_samples
        ])

    def log_marginal_likelihood(self) -> float:
        """
        Log marginal likelihood (Dirichlet-Multinomial):
        
        log P(data | alpha_prior) = sum over rows s of:
            log Gamma(sum alpha_s) - log Gamma(sum alpha_s + N_s)
            + sum_j [log Gamma(alpha_sj + n_sj) - log Gamma(alpha_sj)]
        """
        lml = 0.0
        for s in range(N_PREV_STATES):
            a = self.prior_alpha[s]
            n = self.counts[s]
            N_s = n.sum()
            if N_s == 0:
                continue
            lml += gammaln(a.sum()) - gammaln(a.sum() + N_s)
            lml += np.sum(gammaln(a + n) - gammaln(a))
        return lml

    def log_likelihood(
        self,
        prev_positions: np.ndarray,
        next_positions: np.ndarray,
    ) -> float:
        """Point-estimate log-likelihood under posterior mean probs."""
        tm = self.transition_matrix
        return sum(
            np.log(tm[s, j] + 1e-300)
            for s, j in zip(prev_positions, next_positions)
        )

    def credible_interval(
        self, prev_position: int, outcome: int,
        alpha: float = 0.05, n_samples: int = 10000,
    ) -> tuple[float, float]:
        """(1-alpha) credible interval for P(outcome | prev_position)."""
        p_samples = stats.dirichlet.rvs(
            self.posterior_alpha[prev_position], size=n_samples
        )
        probs = p_samples[:, outcome]
        return (
            np.percentile(probs, 100 * alpha / 2),
            np.percentile(probs, 100 * (1 - alpha / 2)),
        )

    def summary(self) -> pd.DataFrame:
        """Posterior mean transition matrix as a labeled DataFrame."""
        return pd.DataFrame(
            self.transition_matrix,
            index=PREV_STATE_LABELS,
            columns=OUTCOME_LABELS,
        ).rename_axis("Previous Position")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    print("=" * 65)
    print("F1 Dirichlet-Multinomial Model — Stage 1: Global Prior")
    print("=" * 65)

    # --- Try to load real Kaggle data; fall back to synthetic ---
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/f1")

    if (data_dir / "results.csv").exists():
        print(f"\nLoading Kaggle data from {data_dir} ...")
        loader = F1DataLoader(data_dir)
        df = loader.load_merged(min_year=2010)
        print(f"  Rows: {len(df)}, Seasons: {df['year'].nunique()}, "
              f"Drivers: {df['driverId'].nunique()}")
        prev_pos, next_pos, meta = prepare_transitions(df)
    else:
        print(f"\nKaggle data not found at {data_dir}; using synthetic data.")
        rng = np.random.default_rng(42)
        n = 2000
        prev_pos = rng.choice(N_PREV_STATES, size=n)
        next_pos = np.clip(
            prev_pos[:n] + rng.integers(-3, 4, size=n), 0, 20
        ).astype(int)
        next_pos[prev_pos == START] = rng.integers(1, 21, size=(prev_pos == START).sum())

    print(f"Total transitions: {len(prev_pos)}")

    # --- Fit ---
    model = DirichletMultinomialF1(prior_alpha=1.0)
    model.fit(prev_pos, next_pos)

    # --- Summary (subset) ---
    print("\n--- Posterior Mean Transition Probabilities (subset) ---")
    summary = model.summary()
    show_r = ["P1", "P5", "P10", "P15", "P20", "DNF", "START"]
    show_c = ["DNF", "P1", "P2", "P3", "P5", "P10", "P15", "P20"]
    print(summary.loc[show_r, show_c].round(3).to_string())

    # --- Prediction example ---
    print("\n--- Prediction: previous finish = P3 ---")
    probs = model.predict_proba(3)
    for idx in np.argsort(probs)[::-1][:5]:
        lo, hi = model.credible_interval(3, idx)
        print(f"  {OUTCOME_LABELS[idx]}: {probs[idx]:.3f}  "
              f"[95% CI: {lo:.3f}–{hi:.3f}]")

    # --- Marginal likelihood for prior sensitivity ---
    print("\n--- Prior Sensitivity (log marginal likelihood) ---")
    for a in [0.1, 0.5, 1.0, 2.0, 5.0]:
        m = DirichletMultinomialF1(prior_alpha=a)
        m.fit(prev_pos, next_pos)
        print(f"  alpha={a:4.1f}  log P(data|alpha)={m.log_marginal_likelihood():.1f}")
