"""
src/model.py
Model training wrappers for ensemble and neural network architectures.
Implements StratifiedKFold + GridSearchCV as required by the project contract.
"""

import json
import os
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, RandomizedSearchCV, train_test_split
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    roc_auc_score, precision_score, recall_score, accuracy_score
)
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")

METRICS_DIR = os.path.join(os.path.dirname(__file__), "..", "metrics")
MODELS_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _ensure_dirs():
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


# ── 1. Ensemble Model ─────────────────────────────────────────────────────────
def train_ensemble_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict = None,
    cv: int = 5,
    n_jobs: int = -1,
):
    """
    Contract requirement: train and return the best ensemble estimator via
    GridSearchCV with StratifiedKFold.

    Returns best_estimator fitted on the full training set.
    """
    _ensure_dirs()

    if param_grid is None:
        param_grid = {
            "n_estimators":      [100, 200, 300],
            "max_depth":         [None, 5, 10],
            "min_samples_split": [2, 5],
            "class_weight":      ["balanced"],
        }

    rf  = RandomForestClassifier(random_state=42)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=skf,
        scoring="f1",
        n_jobs=n_jobs,
        refit=True,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_cv_f1  = grid_search.best_score_

    print(f"[Ensemble] Best CV F1: {best_cv_f1:.4f}")
    print(f"[Ensemble] Best params: {best_params}")

    # Save tuning results
    tuning_results = {
        "model": "RandomForestClassifier",
        "best_params": best_params,
        "best_cv_f1": float(best_cv_f1),
    }
    _save_json(tuning_results, os.path.join(METRICS_DIR, "tuning_results.json"))

    return grid_search.best_estimator_


# ── 2. Neural Network Model ───────────────────────────────────────────────────
def train_neural_network(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict = None,
    cv: int = 5,
    n_jobs: int = -1,
):
    """
    Trains an MLP neural network inside a scaling pipeline using
    RandomizedSearchCV with StratifiedKFold.
    """
    _ensure_dirs()

    if param_grid is None:
        param_grid = {
            "mlp__hidden_layer_sizes": [(64, 32), (128, 64), (64, 64, 32)],
            "mlp__activation":         ["relu", "tanh"],
            "mlp__alpha":              [0.0001, 0.001, 0.01],
            "mlp__max_iter":           [300],
            "mlp__learning_rate_init": [0.001, 0.01],
        }

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp",    MLPClassifier(random_state=42, early_stopping=True, n_iter_no_change=15)),
    ])

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_grid,
        n_iter=20,
        cv=skf,
        scoring="f1",
        n_jobs=n_jobs,
        refit=True,
        random_state=42,
        verbose=0,
    )
    search.fit(X_train, y_train)

    best_cv_f1  = search.best_score_
    best_params = search.best_params_

    print(f"[MLP] Best CV F1: {best_cv_f1:.4f}")
    print(f"[MLP] Best params: {best_params}")

    return search.best_estimator_


# ── 3. Evaluate and select champion ──────────────────────────────────────────
def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series,
                   model_name: str = "champion") -> dict:
    """
    Evaluate model on held-out test set.
    Saves metrics/test_metrics.json with f1_score key.
    """
    _ensure_dirs()

    y_pred = model.predict(X_test)
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = float(roc_auc_score(y_test, y_prob))
    except Exception:
        roc_auc = None

    metrics = {
        "model_name":  model_name,
        "f1_score":    float(f1_score(y_test, y_pred, pos_label=1)),
        "precision":   float(precision_score(y_test, y_pred, pos_label=1)),
        "recall":      float(recall_score(y_test, y_pred, pos_label=1)),
        "accuracy":    float(accuracy_score(y_test, y_pred)),
        "roc_auc":     roc_auc,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    auc_str = f"{roc_auc:.4f}" if roc_auc is not None else "N/A"
    print(f"\n[{model_name}] Test F1={metrics['f1_score']:.4f} | AUC={auc_str} | Acc={metrics['accuracy']:.4f}")

    _save_json(metrics, os.path.join(METRICS_DIR, "test_metrics.json"))
    return metrics


# ── 4. Full training pipeline ─────────────────────────────────────────────────
def run_full_pipeline(X: pd.DataFrame, y: pd.Series):
    """
    End-to-end training pipeline:
      1. Train/test split (stratified)
      2. Train ensemble + neural network
      3. Evaluate both; pick champion (higher F1)
      4. Save champion model
    Returns (champion_model, X_test, y_test, metrics_dict)
    """
    _ensure_dirs()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("=== Training Ensemble (Random Forest) ===")
    rf_model = train_ensemble_model(X_train, y_train)

    print("\n=== Training Neural Network (MLP) ===")
    mlp_model = train_neural_network(X_train, y_train)

    # Evaluate both
    print("\n=== Evaluation on Held-out Test Set ===")
    rf_metrics  = evaluate_model(rf_model,  X_test, y_test, "RandomForest")
    mlp_metrics = evaluate_model(mlp_model, X_test, y_test, "MLP")

    # Select champion
    if rf_metrics["f1_score"] >= mlp_metrics["f1_score"]:
        champion = rf_model
        champion_metrics = rf_metrics
        champion_metrics["model_name"] = "RandomForest_Champion"
        print("\n✅ Champion: Random Forest")
    else:
        champion = mlp_model
        champion_metrics = mlp_metrics
        champion_metrics["model_name"] = "MLP_Champion"
        print("\n✅ Champion: MLP Neural Network")

    _save_json(champion_metrics, os.path.join(METRICS_DIR, "test_metrics.json"))

    # Persist champion model
    model_path = os.path.join(MODELS_DIR, "champion_model.pkl")
    joblib.dump(champion, model_path)
    print(f"Champion model saved to {model_path}")

    return champion, X_train, X_test, y_train, y_test, champion_metrics


# ── Utility ───────────────────────────────────────────────────────────────────
def _save_json(data: dict, path: str):
    """Save a dictionary to a JSON file, converting non-serialisable types."""
    def default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Type {type(obj)} not serialisable")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=default)


def load_champion_model(path: str = None):
    """Load the saved champion model from disk."""
    if path is None:
        path = os.path.join(MODELS_DIR, "champion_model.pkl")
    return joblib.load(path)