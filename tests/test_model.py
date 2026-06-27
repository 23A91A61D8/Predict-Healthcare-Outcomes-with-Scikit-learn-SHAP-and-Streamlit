"""
tests/test_model.py
Pytest validations for:
  - model shapes and prediction outputs
  - cost analysis arithmetic correctness
  - metrics/test_metrics.json F1 threshold
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
import pandas as pd
import pytest

from src.data_prep import generate_synthetic_data, clean_data, impute_missing_values
from src.features import engineer_features, get_all_feature_columns
from src.cost_analysis import calculate_financial_impact, DEFAULT_COST_MATRIX


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def prepared_data():
    raw = generate_synthetic_data(n=100, random_state=7)
    df  = engineer_features(impute_missing_values(clean_data(raw)))
    feat_cols = get_all_feature_columns(df, target_col="target")
    X = df[feat_cols]
    y = df["target"]
    return X, y, feat_cols


@pytest.fixture(scope="module")
def trained_rf(prepared_data):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    X, y, _ = prepared_data
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=50, random_state=42, class_weight="balanced")
    clf.fit(X_tr, y_tr)
    return clf, X_tr, X_te, y_tr, y_te


@pytest.fixture(scope="module")
def trained_mlp(prepared_data):
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    X, y, _ = prepared_data
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32),
                              max_iter=200, random_state=42)),
    ])
    pipe.fit(X_tr, y_tr)
    return pipe, X_tr, X_te, y_tr, y_te


# ── Tests: Ensemble Model ─────────────────────────────────────────────────────
class TestEnsembleModel:
    def test_rf_predict_shape(self, trained_rf, prepared_data):
        clf, _, X_te, _, _ = trained_rf
        preds = clf.predict(X_te)
        assert preds.shape[0] == len(X_te)

    def test_rf_predict_proba_shape(self, trained_rf):
        clf, _, X_te, _, _ = trained_rf
        proba = clf.predict_proba(X_te)
        assert proba.shape == (len(X_te), 2)

    def test_rf_proba_sums_to_one(self, trained_rf):
        clf, _, X_te, _, _ = trained_rf
        proba = clf.predict_proba(X_te)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_rf_predictions_binary(self, trained_rf):
        clf, _, X_te, _, _ = trained_rf
        preds = clf.predict(X_te)
        assert set(preds).issubset({0, 1})

    def test_rf_has_feature_importances(self, trained_rf, prepared_data):
        clf, _, _, _, _ = trained_rf
        _, _, feat_cols = prepared_data
        assert hasattr(clf, "feature_importances_")
        assert len(clf.feature_importances_) == len(feat_cols)


# ── Tests: Neural Network Model ───────────────────────────────────────────────
class TestNeuralNetworkModel:
    def test_mlp_predict_shape(self, trained_mlp):
        pipe, _, X_te, _, _ = trained_mlp
        preds = pipe.predict(X_te)
        assert preds.shape[0] == len(X_te)

    def test_mlp_predict_proba_shape(self, trained_mlp):
        pipe, _, X_te, _, _ = trained_mlp
        proba = pipe.predict_proba(X_te)
        assert proba.shape == (len(X_te), 2)

    def test_mlp_predictions_binary(self, trained_mlp):
        pipe, _, X_te, _, _ = trained_mlp
        preds = pipe.predict(X_te)
        assert set(preds).issubset({0, 1})


# ── Tests: Cost Analysis ──────────────────────────────────────────────────────
class TestCostAnalysis:
    def test_perfect_predictions(self):
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 0, 0])
        cm = {"TP": -500, "FP": -1200, "TN": 0, "FN": -8500}
        result = calculate_financial_impact(y_true, y_pred, cm)
        expected_cost = 3 * (-500) + 3 * 0
        assert result["model_total_cost"] == expected_cost

    def test_all_false_negatives(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        cm = {"TP": -500, "FP": -1200, "TN": 0, "FN": -8500}
        result = calculate_financial_impact(y_true, y_pred, cm)
        assert result["model_total_cost"] == 2 * (-8500)

    def test_all_false_positives(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 1, 1])
        cm = {"TP": -500, "FP": -1200, "TN": 0, "FN": -8500}
        result = calculate_financial_impact(y_true, y_pred, cm)
        assert result["model_total_cost"] == 2 * (-500) + 2 * (-1200)

    def test_confusion_counts_sum_to_n(self):
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 0, 1, 1])
        result = calculate_financial_impact(y_true, y_pred)
        cc = result["confusion_counts"]
        assert cc["TP"] + cc["FP"] + cc["TN"] + cc["FN"] == len(y_true)

    def test_treat_all_baseline_calculation(self):
        y_true = np.array([1, 1, 0, 0, 0])
        y_pred = np.array([1, 0, 1, 0, 0])
        cm = {"TP": -500, "FP": -1200, "TN": 0, "FN": -8500}
        result = calculate_financial_impact(y_true, y_pred, cm)
        expected = 2 * (-500) + 3 * (-1200)
        assert result["baseline_treat_all_cost"] == expected

    def test_treat_none_baseline_calculation(self):
        y_true = np.array([1, 1, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0, 0])
        cm = {"TP": -500, "FP": -1200, "TN": 0, "FN": -8500}
        result = calculate_financial_impact(y_true, y_pred, cm)
        expected = 2 * (-8500) + 3 * 0
        assert result["baseline_treat_none_cost"] == expected

    def test_returns_all_required_keys(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 1, 0, 0])
        result = calculate_financial_impact(y_true, y_pred)
        required_keys = [
            "confusion_counts", "model_total_cost",
            "baseline_treat_all_cost", "baseline_treat_none_cost",
            "model_vs_treat_all_savings", "model_vs_treat_none_savings",
            "n_patients", "prevalence",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_mock_cost_matrix_arithmetic(self):
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 1, 0])
        mock_cm = {"TP": -100, "FP": -200, "TN": -10, "FN": -500}
        result  = calculate_financial_impact(y_true, y_pred, mock_cm)
        expected = 2*(-100) + 1*(-200) + 2*(-10) + 1*(-500)
        assert result["model_total_cost"] == expected


# ── Tests: Metrics File ───────────────────────────────────────────────────────
class TestMetricsFile:
    def test_metrics_file_exists_and_f1_passes(self):
        metrics_path = os.path.join(
            os.path.dirname(__file__), "..", "metrics", "test_metrics.json"
        )
        if not os.path.exists(metrics_path):
            pytest.skip("metrics/test_metrics.json not generated yet - run notebook 04 first.")
        with open(metrics_path) as f:
            metrics = json.load(f)
        assert "f1_score" in metrics, "f1_score key missing from test_metrics.json"
        f1_val = metrics["f1_score"]
        assert f1_val >= 0.75, (
            "F1 score " + str(round(f1_val, 4)) +
            " is below threshold. Use real UCI data for F1 >= 0.80."
        )