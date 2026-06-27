"""
src/features.py
Feature creation logic for the Heart Disease predictive pipeline.
Creates >= 5 novel engineered features as required by the project contract.
"""

import pandas as pd
import numpy as np
from typing import List
import warnings
warnings.filterwarnings("ignore")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Contract requirement: engineer_features(df) -> DataFrame with at least
    5 newly named columns derived mathematically or conditionally from
    existing columns.

    New features created
    --------------------
    1.  age_thalach_ratio      – Age divided by max heart rate; higher = worse
    2.  bp_chol_product        – Interaction: blood pressure × cholesterol
    3.  high_risk_flag         – Binary: age>55 AND trestbps>140 AND chol>240
    4.  st_depression_flag     – Binary: oldpeak > 2 (significant ST depression)
    5.  vessel_thal_score      – Composite: (ca + 1) × thal (vessel × thal type)
    6.  age_risk_bin           – Categorical risk tier based on age
    7.  exang_oldpeak_interact – Exercise-induced angina × ST depression interaction
    8.  chol_age_ratio         – Cholesterol per year of age
    """
    df = df.copy()

    # ── Feature 1: Age-to-MaxHeartRate ratio ─────────────────────────────────
    # Higher ratio → older patient achieving less max HR → worse cardiovascular
    thalach = df["thalach"].replace(0, np.nan).fillna(df["thalach"].median())
    df["age_thalach_ratio"] = df["age"] / thalach

    # ── Feature 2: Blood-pressure × Cholesterol product ──────────────────────
    # Combined vascular stress indicator
    df["bp_chol_product"] = df["trestbps"] * df["chol"]

    # ── Feature 3: High-risk composite flag ──────────────────────────────────
    # Clinical rule: elderly + hypertensive + hypercholesterolaemic
    df["high_risk_flag"] = (
        (df["age"] > 55) &
        (df["trestbps"] > 140) &
        (df["chol"] > 240)
    ).astype(int)

    # ── Feature 4: Significant ST-depression flag ─────────────────────────────
    # oldpeak > 2 mm ST depression is clinically significant
    df["st_depression_flag"] = (df["oldpeak"] > 2.0).astype(int)

    # ── Feature 5: Vessel-Thal composite score ────────────────────────────────
    # Number of major vessels coloured (ca) interacted with thalassemia type (thal)
    df["vessel_thal_score"] = (df["ca"].fillna(0) + 1) * df["thal"].fillna(df["thal"].median())

    # ── Feature 6: Age risk bin ───────────────────────────────────────────────
    # Categorical risk stratification by age decade
    bins   = [0, 40, 50, 60, 70, 120]
    labels = [0, 1, 2, 3, 4]          # ordinal encoding
    df["age_risk_bin"] = pd.cut(df["age"], bins=bins, labels=labels, right=True).astype(float).fillna(2)

    # ── Feature 7: Exercise-angina × ST-depression interaction ───────────────
    df["exang_oldpeak_interact"] = df["exang"] * df["oldpeak"]

    # ── Feature 8: Cholesterol-per-year-of-age ────────────────────────────────
    age_safe = df["age"].replace(0, np.nan).fillna(df["age"].median())
    df["chol_age_ratio"] = df["chol"] / age_safe

    return df


def get_feature_names(df_before: pd.DataFrame, df_after: pd.DataFrame) -> List[str]:
    """Return the list of newly created feature names."""
    return list(set(df_after.columns) - set(df_before.columns))


def get_all_feature_columns(df: pd.DataFrame, target_col: str = "target") -> List[str]:
    """Return all feature columns (excluding target)."""
    return [c for c in df.columns if c != target_col]
