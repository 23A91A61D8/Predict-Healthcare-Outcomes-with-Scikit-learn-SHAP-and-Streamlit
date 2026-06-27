"""
src/data_prep.py
Reusable data ingestion, cleaning, and imputation pipelines.
Heart Disease UCI Dataset preprocessing.
"""

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ── Column name mapping from UCI raw format ──────────────────────────────────
UCI_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

NUMERIC_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
TARGET_COL = "target"


# ── 1. Load ──────────────────────────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the UCI Heart Disease CSV (with or without headers).
    Handles the classic UCI format where '?' represents missing values.
    """
    df = pd.read_csv(
        filepath,
        header=None,
        names=UCI_COLUMNS,
        na_values=["?", " ?", "? ", "", " "]
    )
    return df


# ── 2. Clean ─────────────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Contract requirement: clean_data(df) -> DataFrame with:
      - No duplicated rows
      - Correctly formatted column types
      - Target variable binarised (0 = no disease, 1 = disease)
    """
    df = df.copy()

    # Remove exact duplicate rows
    df = df.drop_duplicates()

    # Ensure UCI column names present; if loading a different format, rename
    if "target" not in df.columns and df.shape[1] == 14:
        df.columns = UCI_COLUMNS

    # Convert numeric cols to float
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert categorical cols to int (they are already int-like)
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Binarise target: original UCI target is 0-4; 0=no disease, 1-4=disease
    if TARGET_COL in df.columns:
        df[TARGET_COL] = (df[TARGET_COL] > 0).astype(int)

    # Physiologically impossible outlier capping
    df = _cap_outliers(df)

    return df


def _cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Clip physiologically impossible values using domain-knowledge bounds."""
    bounds = {
        "age":      (18, 110),
        "trestbps": (60, 250),
        "chol":     (100, 600),
        "thalach":  (50, 220),
        "oldpeak":  (0.0, 10.0),
        "ca":       (0, 4),
    }
    for col, (lo, hi) in bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lo, upper=hi)
    return df


# ── 3. Imputation ─────────────────────────────────────────────────────────────
def impute_missing_values(df: pd.DataFrame, strategy: str = "mice") -> pd.DataFrame:
    """
    Contract requirement: impute_missing_values(df) -> DataFrame with zero NaNs.

    Parameters
    ----------
    df       : cleaned DataFrame (may contain NaN)
    strategy : 'mice' (IterativeImputer) | 'knn' (KNNImputer)
    """
    df = df.copy()

    # Split target off so it is not used in imputation
    target = None
    if TARGET_COL in df.columns:
        target = df[TARGET_COL].copy()
        feature_df = df.drop(columns=[TARGET_COL])
    else:
        feature_df = df.copy()

    if strategy == "mice":
        imputer = IterativeImputer(
            max_iter=10,
            random_state=42,
            initial_strategy="mean"
        )
    elif strategy == "knn":
        imputer = KNNImputer(n_neighbors=5)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose 'mice' or 'knn'.")

    imputed_array = imputer.fit_transform(feature_df)
    imputed_df = pd.DataFrame(imputed_array, columns=feature_df.columns, index=feature_df.index)

    # Round categorical columns back to integers
    for col in CATEGORICAL_COLS:
        if col in imputed_df.columns:
            imputed_df[col] = imputed_df[col].round().astype(int)

    # Re-attach target
    if target is not None:
        imputed_df[TARGET_COL] = target.values

    assert imputed_df.isnull().sum().sum() == 0, "Imputation failed: NaNs remain."
    return imputed_df


# ── 4. Convenience wrapper ────────────────────────────────────────────────────
def load_and_prepare(filepath: str, strategy: str = "mice") -> pd.DataFrame:
    """
    Full pipeline: load → clean → impute.
    Returns a ready-to-use DataFrame.
    """
    df = load_data(filepath)
    df = clean_data(df)
    df = impute_missing_values(df, strategy=strategy)
    return df


# ── 5. Generate synthetic data (for demo / testing when UCI file absent) ─────
def generate_synthetic_data(n: int = 303, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic dataset that mirrors the UCI Heart Disease schema.
    Used for testing and demonstration when the real dataset is unavailable.
    """
    rng = np.random.default_rng(random_state)

    n_pos = n // 2
    n_neg = n - n_pos

    def make_group(size, disease):
        age_mean = 57 if disease else 52
        return {
            "age":      rng.normal(age_mean, 9, size).clip(29, 77).astype(int),
            "sex":      rng.integers(0, 2, size),
            "cp":       rng.integers(0, 4, size),
            "trestbps": rng.normal(134 if disease else 129, 18, size).clip(94, 200).astype(int),
            "chol":     rng.normal(251 if disease else 242, 51, size).clip(126, 564).astype(int),
            "fbs":      rng.integers(0, 2, size),
            "restecg":  rng.integers(0, 3, size),
            "thalach":  rng.normal(139 if disease else 158, 23, size).clip(71, 202).astype(int),
            "exang":    rng.integers(0, 2, size),
            "oldpeak":  rng.exponential(1.6 if disease else 0.8, size).clip(0, 6.2).round(1),
            "slope":    rng.integers(0, 3, size),
            "ca":       rng.integers(0, 4, size),
            "thal":     rng.choice([1, 2, 3], size),
            "target":   np.full(size, int(disease)),
        }

    pos_data = make_group(n_pos, True)
    neg_data = make_group(n_neg, False)

    df = pd.concat([
        pd.DataFrame(pos_data),
        pd.DataFrame(neg_data),
    ], ignore_index=True)

    # Inject ~5 % missingness in two columns to simulate real-world data
    miss_idx_ca  = rng.choice(len(df), size=int(0.05 * len(df)), replace=False)
    miss_idx_thal = rng.choice(len(df), size=int(0.05 * len(df)), replace=False)
    df.loc[miss_idx_ca, "ca"] = np.nan
    df.loc[miss_idx_thal, "thal"] = np.nan

    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df
