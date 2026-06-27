"""
src/cost_analysis.py
Business logic for cost-effectiveness analysis of the predictive model.
Compares model policy vs "Treat Everyone" and "Treat No One" baselines.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


# ── Default cost matrix ───────────────────────────────────────────────────────
DEFAULT_COST_MATRIX = {
    "TP": -500,    # True Positive  → preventive treatment: moderate cost, avoided crisis
    "FP": -1200,   # False Positive → unnecessary treatment: wasted resource
    "TN":    0,    # True Negative  → correctly discharged: no extra cost
    "FN": -8500,   # False Negative → missed high-risk patient, emergency readmission
}

# Average cost of treating every patient preventively (regardless of risk)
PREVENTIVE_TREATMENT_COST = -1200  # per patient
# Average cost of emergency readmission
READMISSION_COST = -8500           # per patient


def calculate_financial_impact(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_matrix: Dict[str, float] = None,
) -> Dict[str, float]:
    """
    Contract requirement: calculate_financial_impact(y_true, y_pred, cost_matrix)
    -> dict with aggregate simulated cost and comparison metrics.

    Parameters
    ----------
    y_true      : Ground-truth binary labels (1 = disease present)
    y_pred      : Model predictions
    cost_matrix : Dict with keys TP, FP, TN, FN representing costs
                  (negative = expenditure, positive = saving)

    Returns
    -------
    dict with keys:
        model_total_cost, baseline_treat_all_cost, baseline_treat_none_cost,
        model_vs_treat_all_savings, model_vs_treat_none_savings,
        confusion_counts (TP, FP, TN, FN)
    """
    if cost_matrix is None:
        cost_matrix = DEFAULT_COST_MATRIX

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    # Confusion matrix components
    TP = int(np.sum((y_pred == 1) & (y_true == 1)))
    FP = int(np.sum((y_pred == 1) & (y_true == 0)))
    TN = int(np.sum((y_pred == 0) & (y_true == 0)))
    FN = int(np.sum((y_pred == 0) & (y_true == 1)))

    # Model cost
    model_cost = (
        TP * cost_matrix["TP"] +
        FP * cost_matrix["FP"] +
        TN * cost_matrix["TN"] +
        FN * cost_matrix["FN"]
    )

    # Baseline: treat everyone preventively
    # Every actual positive gets TP benefit; every actual negative gets FP cost
    n_pos = int(y_true.sum())
    n_neg = n - n_pos
    treat_all_cost = (
        n_pos * cost_matrix["TP"] +
        n_neg * cost_matrix["FP"]
    )

    # Baseline: treat no one → all FN and TN
    treat_none_cost = (
        n_pos * cost_matrix["FN"] +
        n_neg * cost_matrix["TN"]
    )

    return {
        "confusion_counts":              {"TP": TP, "FP": FP, "TN": TN, "FN": FN},
        "model_total_cost":              float(model_cost),
        "baseline_treat_all_cost":       float(treat_all_cost),
        "baseline_treat_none_cost":      float(treat_none_cost),
        "model_vs_treat_all_savings":    float(model_cost - treat_all_cost),
        "model_vs_treat_none_savings":   float(model_cost - treat_none_cost),
        "n_patients":                    n,
        "prevalence":                    float(n_pos / n),
        "cost_matrix_used":              cost_matrix,
    }


def format_financial_report(result: Dict) -> str:
    """
    Produce a human-readable financial impact report string.
    """
    cc = result["confusion_counts"]
    lines = [
        "=" * 55,
        "       FINANCIAL IMPACT ANALYSIS REPORT",
        "=" * 55,
        f"  Patients analysed      : {result['n_patients']}",
        f"  Disease prevalence     : {result['prevalence']:.1%}",
        "",
        "  Confusion Matrix Breakdown:",
        f"    True Positives  (TP) : {cc['TP']:>5}",
        f"    False Positives (FP) : {cc['FP']:>5}",
        f"    True Negatives  (TN) : {cc['TN']:>5}",
        f"    False Negatives (FN) : {cc['FN']:>5}",
        "",
        "  Simulated Financial Outcomes:",
        f"    Model policy cost    : ${result['model_total_cost']:>10,.0f}",
        f"    Treat-ALL baseline   : ${result['baseline_treat_all_cost']:>10,.0f}",
        f"    Treat-NONE baseline  : ${result['baseline_treat_none_cost']:>10,.0f}",
        "",
        "  Savings vs Baselines:",
        f"    vs Treat-ALL         : ${result['model_vs_treat_all_savings']:>10,.0f}",
        f"    vs Treat-NONE        : ${result['model_vs_treat_none_savings']:>10,.0f}",
        "=" * 55,
    ]
    return "\n".join(lines)


def sensitivity_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fn_cost_range: Tuple[float, float] = (-5000, -15000),
    steps: int = 10,
) -> pd.DataFrame:
    """
    Vary the False-Negative cost and compute resulting model savings.
    Useful for the dashboard's interactive cost slider.
    """
    fn_costs = np.linspace(fn_cost_range[0], fn_cost_range[1], steps)
    rows = []
    for fn_c in fn_costs:
        cm = dict(DEFAULT_COST_MATRIX)
        cm["FN"] = fn_c
        r = calculate_financial_impact(y_true, y_pred, cost_matrix=cm)
        rows.append({
            "fn_cost":                  fn_c,
            "model_total_cost":         r["model_total_cost"],
            "treat_all_cost":           r["baseline_treat_all_cost"],
            "savings_vs_treat_all":     r["model_vs_treat_all_savings"],
            "savings_vs_treat_none":    r["model_vs_treat_none_savings"],
        })
    return pd.DataFrame(rows)
