"""
tests/test_data.py
Pytest assertions for data validity:
  - clean_data() output shape and integrity
  - impute_missing_values() produces zero NaNs
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import pytest

from src.data_prep import (
    clean_data,
    impute_missing_values,
    generate_synthetic_data,
    UCI_COLUMNS,
)
from src.features import engineer_features


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def raw_df():
    """Generate a small synthetic dataset for testing."""
    return generate_synthetic_data(n=50, random_state=0)


@pytest.fixture
def clean_df(raw_df):
    return clean_data(raw_df)


@pytest.fixture
def imputed_df(clean_df):
    return impute_missing_values(clean_df, strategy="mice")


# ── Tests: clean_data ─────────────────────────────────────────────────────────
class TestCleanData:
    def test_returns_dataframe(self, raw_df):
        result = clean_data(raw_df)
        assert isinstance(result, pd.DataFrame)

    def test_no_duplicates(self, raw_df):
        # Inject duplicate rows
        df_with_dups = pd.concat([raw_df, raw_df.iloc[:5]], ignore_index=True)
        result = clean_data(df_with_dups)
        assert result.duplicated().sum() == 0

    def test_target_is_binary(self, clean_df):
        assert set(clean_df["target"].dropna().unique()).issubset({0, 1})

    def test_column_count(self, clean_df):
        assert clean_df.shape[1] == len(UCI_COLUMNS)

    def test_numeric_columns_are_numeric(self, clean_df):
        numeric = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        for col in numeric:
            assert pd.api.types.is_numeric_dtype(clean_df[col]), f"{col} is not numeric"

    def test_age_bounds(self, clean_df):
        assert clean_df["age"].dropna().between(18, 110).all()

    def test_trestbps_bounds(self, clean_df):
        assert clean_df["trestbps"].dropna().between(60, 250).all()

    def test_chol_bounds(self, clean_df):
        assert clean_df["chol"].dropna().between(100, 600).all()


# ── Tests: impute_missing_values ──────────────────────────────────────────────
class TestImputation:
    def test_no_nans_after_mice(self, clean_df):
        result = impute_missing_values(clean_df, strategy="mice")
        assert result.isnull().sum().sum() == 0

    def test_no_nans_after_knn(self, clean_df):
        result = impute_missing_values(clean_df, strategy="knn")
        assert result.isnull().sum().sum() == 0

    def test_shape_preserved(self, clean_df, imputed_df):
        assert imputed_df.shape == clean_df.shape

    def test_target_unchanged(self, clean_df, imputed_df):
        original_target = clean_df["target"].dropna()
        imputed_target  = imputed_df.loc[original_target.index, "target"]
        pd.testing.assert_series_equal(original_target, imputed_target, check_names=False)

    def test_synthetic_nan_injection(self):
        """Pass a dataframe WITH NaN values and assert zero NaN output."""
        df = generate_synthetic_data(n=60, random_state=1)
        df = clean_data(df)
        # Force additional NaNs
        df.loc[df.index[:10], "chol"] = np.nan
        df.loc[df.index[5:15], "thalach"] = np.nan
        result = impute_missing_values(df, strategy="mice")
        assert result.isnull().sum().sum() == 0

    def test_invalid_strategy_raises(self, clean_df):
        with pytest.raises(ValueError, match="Unknown strategy"):
            impute_missing_values(clean_df, strategy="invalid")


# ── Tests: engineer_features ──────────────────────────────────────────────────
class TestFeatureEngineering:
    def test_minimum_five_new_features(self, imputed_df):
        df_before = imputed_df.copy()
        df_after  = engineer_features(imputed_df)
        new_cols  = set(df_after.columns) - set(df_before.columns)
        assert len(new_cols) >= 5, f"Only {len(new_cols)} new features; need ≥5"

    def test_no_nulls_in_engineered(self, imputed_df):
        df_eng = engineer_features(imputed_df)
        from src.features import get_feature_names
        new_cols = get_feature_names(imputed_df, df_eng)
        for col in new_cols:
            assert df_eng[col].isnull().sum() == 0, f"NaN in engineered feature: {col}"

    def test_specific_features_present(self, imputed_df):
        df_eng = engineer_features(imputed_df)
        expected = [
            "age_thalach_ratio",
            "bp_chol_product",
            "high_risk_flag",
            "st_depression_flag",
            "vessel_thal_score",
        ]
        for feat in expected:
            assert feat in df_eng.columns, f"Missing expected feature: {feat}"
