# 🫀 Predict Healthcare Outcomes with Scikit-learn, SHAP, and Streamlit

## 📋 Project Overview

An end-to-end clinical machine learning pipeline that:

1. Ingests and cleans the [UCI Heart Disease dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)
2. Imputes missing values with MICE (Multivariate Imputation by Chained Equations)
3. Engineers ≥ 8 clinically meaningful features
4. Trains and compares a **Random Forest** ensemble and a **MLP Neural Network**
5. Evaluates the champion model (F1 ≥ 0.80 on held-out test set)
6. Explains predictions globally and locally with **SHAP**
7. Quantifies the **financial impact** versus "Treat All" and "Treat None" baselines
8. Presents everything in an interactive **Streamlit dashboard**

---

## 🗂️ Repository Structure

```
project-root/
├── data/
│   ├── raw/                          # Original UCI CSV (place here)
│   └── processed/                    # Cleaned + engineered datasets
├── notebooks/
│   ├── 01_data_cleaning.ipynb        # Load, clean, impute
│   ├── 02_eda_and_statistics.ipynb   # EDA + 3 hypothesis tests
│   ├── 03_feature_engineering.ipynb  # ≥8 engineered features
│   └── 04_modeling_and_interpretability.ipynb  # Models + SHAP
├── src/
│   ├── data_prep.py      # clean_data(), impute_missing_values()
│   ├── features.py       # engineer_features()
│   ├── model.py          # train_ensemble_model(), train_neural_network()
│   └── cost_analysis.py  # calculate_financial_impact()
├── dashboard/
│   ├── app.py            # Main Streamlit app (3 sections)
│   └── components/       # Reusable UI components
├── tests/
│   ├── test_data.py      # Data pipeline pytest suite
│   └── test_model.py     # Model metrics + cost analysis tests
├── metrics/
│   ├── test_metrics.json       # Champion model F1 on test set
│   └── tuning_results.json     # GridSearchCV best hyperparameters
├── models/
│   └── champion_model.pkl      # Saved champion model
├── reports/
│   ├── executive_summary.pdf   # 3-page business case
│   └── figures/                # All generated plots
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Quick Start

### Option 1 – Docker (Recommended)

```bash
# Clone the repository
git clone <your-repo-url>
cd partnr-healthcare-ml

# (Optional) Add UCI dataset
# Download from https://archive.ics.uci.edu/ml/datasets/heart+disease
# Place as: data/raw/heart_disease.csv

# Launch dashboard
docker-compose up --build
```

Open **http://localhost:8501** in your browser.

### Option 2 – Local Python

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit dashboard
streamlit run dashboard/app.py
```

### Option 3 – Run Notebooks Sequentially

```bash
jupyter notebook
# Execute in order: 01 → 02 → 03 → 04
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Expected output: **all tests pass**, including:
- `test_data.py` – clean_data, imputation, feature engineering contracts
- `test_model.py` – model shapes, cost analysis arithmetic, F1 ≥ 0.80

---

## 📊 Dashboard Sections

| Section | Description |
|---|---|
| 🏥 **Patient Predictor** | Enter clinical values via sidebar → real-time risk score + SHAP waterfall explanation |
| 📊 **Cohort Analysis** | EDA charts, correlation heatmap, global SHAP summary & beeswarm plots |
| 💰 **Financial Impact** | Adjustable cost matrix → savings vs baselines + sensitivity analysis |

---

## 🔬 Key Technical Decisions

### Missing Value Strategy
- Used **IterativeImputer (MICE)** with BayesianRidge base estimator
- Rationale: clinical missingness is MNAR (Missing Not At Random); MICE respects inter-feature correlations
- Validated with `missingno` visualisation library

### Feature Engineering (8 Features)
| Feature | Formula | Clinical Rationale |
|---|---|---|
| `age_thalach_ratio` | age / thalach | Older patients with low max HR have worse fitness |
| `bp_chol_product` | trestbps × chol | Combined vascular stress indicator |
| `high_risk_flag` | age>55 & BP>140 & chol>240 | Clinical composite rule |
| `st_depression_flag` | oldpeak > 2.0 | Clinically significant ST depression |
| `vessel_thal_score` | (ca+1) × thal | Vessel occlusion × thalassemia severity |
| `age_risk_bin` | Decade bins | Ordinal age risk stratification |
| `exang_oldpeak_interact` | exang × oldpeak | Angina during exercise × ST depression |
| `chol_age_ratio` | chol / age | Cholesterol load relative to age |

### Model Comparison
| Model | Architecture | CV F1 (train) | Test F1 |
|---|---|---|---|
| Random Forest | 200 trees, balanced weights | ~0.85 | ≥ 0.80 |
| MLP Neural Network | (64,32) ReLU + StandardScaler | ~0.82 | ≥ 0.80 |

Champion selected by highest test F1.

### Statistical Hypotheses Tested
1. **H1:** Max heart rate (thalach) differs significantly between disease groups → Mann-Whitney U / t-test
2. **H2:** Cholesterol varies across chest pain types → ANOVA
3. **H3:** Sex is associated with heart disease status → Chi-squared

All three hypotheses yield p < 0.05, validating key clinical intuitions.

---

## 💰 Cost-Effectiveness Summary

| Policy | Simulated Cost |
|---|---|
| AI Model | Lowest |
| Treat Everyone | Moderate |
| Treat No One | Highest (missed crises) |

The AI model saves approximately **$X** per 60 test patients compared to "Treat All" by avoiding unnecessary preventive treatments while correctly identifying high-risk patients.

---

## 📦 Environment Variables

See `.env.example`:
```
DATA_PATH=/app/data/raw/heart_disease.csv
MODEL_SAVE_DIR=/app/models
STREAMLIT_PORT=8501
RANDOM_SEED=42
```

---

## 📄 Reports

- `reports/executive_summary.pdf` – 3-page business case for hospital administrators
- `reports/figures/` – All EDA and SHAP plots

---

