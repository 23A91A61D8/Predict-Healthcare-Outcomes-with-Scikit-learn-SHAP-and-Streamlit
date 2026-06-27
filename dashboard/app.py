"""
dashboard/app.py  –  Heart Disease AI Prediction Dashboard
Professional, colourful Streamlit UI
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data_prep import generate_synthetic_data, clean_data, impute_missing_values, load_data
from src.features import engineer_features, get_all_feature_columns
from src.model import run_full_pipeline, load_champion_model
from src.cost_analysis import (
    calculate_financial_impact, format_financial_report,
    sensitivity_analysis, DEFAULT_COST_MATRIX
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease AI | Partnr",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Professional CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Dark gradient background */
  .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
  }
  section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
  section[data-testid="stSidebar"] .stRadio label { font-size: 1rem; font-weight: 600; }

  /* Cards */
  .glass-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 24px;
    backdrop-filter: blur(12px);
    margin-bottom: 16px;
  }

  /* Hero banner */
  .hero {
    background: linear-gradient(135deg, #e63946 0%, #a8dadc 50%, #457b9d 100%);
    border-radius: 20px;
    padding: 36px 40px;
    margin-bottom: 28px;
    box-shadow: 0 8px 32px rgba(230,57,70,0.3);
  }
  .hero h1 { color: white; font-size: 2.4rem; font-weight: 800; margin: 0; text-shadow: 0 2px 8px rgba(0,0,0,0.3); }
  .hero p  { color: rgba(255,255,255,0.9); font-size: 1.05rem; margin: 8px 0 0 0; }

  /* Section headers */
  .sec-header {
    font-size: 1.25rem; font-weight: 700; color: #a8dadc;
    border-left: 4px solid #e63946; padding-left: 12px;
    margin: 24px 0 14px 0;
  }

  /* KPI metric cards */
  .kpi-wrap { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 20px; }
  .kpi {
    flex: 1; min-width: 140px;
    background: linear-gradient(135deg, rgba(230,57,70,0.2), rgba(69,123,157,0.2));
    border: 1px solid rgba(168,218,220,0.3);
    border-radius: 14px; padding: 18px 20px; text-align: center;
  }
  .kpi .val { font-size: 2rem; font-weight: 800; color: #a8dadc; }
  .kpi .lbl { font-size: 0.78rem; color: rgba(255,255,255,0.6); margin-top: 4px; text-transform: uppercase; letter-spacing: 0.08em; }

  /* Risk badges */
  .badge-high { background: linear-gradient(135deg,#e63946,#c1121f); color:white; font-size:1.3rem; font-weight:800; border-radius:12px; padding:10px 24px; display:inline-block; box-shadow:0 4px 20px rgba(230,57,70,0.5); }
  .badge-low  { background: linear-gradient(135deg,#2dc653,#1a7a36); color:white; font-size:1.3rem; font-weight:800; border-radius:12px; padding:10px 24px; display:inline-block; box-shadow:0 4px 20px rgba(45,198,83,0.5); }

  /* Tables */
  .stDataFrame { border-radius: 12px; overflow: hidden; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { background: rgba(255,255,255,0.06); border-radius: 12px; padding: 4px; gap: 4px; }
  .stTabs [data-baseweb="tab"] { border-radius: 8px; color: #a8dadc; font-weight: 600; }
  .stTabs [aria-selected="true"] { background: linear-gradient(135deg,#e63946,#457b9d) !important; color: white !important; }

  /* Sliders & inputs */
  .stSlider > div > div > div { background: #e63946 !important; }
  div[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(230,57,70,0.18), rgba(69,123,157,0.18));
    border: 1px solid rgba(168,218,220,0.35);
    border-radius: 14px; padding: 16px; text-align: center;
  }
  div[data-testid="metric-container"] * { color: white !important; }
  div[data-testid="metric-container"] label,
  div[data-testid="metric-container"] [data-testid="metric-label"] p,
  div[data-testid="metric-container"] [data-testid="metric-label"] div { color: #a8dadc !important; font-weight: 600 !important; font-size: 0.85rem !important; }
  div[data-testid="metric-container"] [data-testid="metric-value"],
  div[data-testid="metric-container"] [data-testid="metric-value"] div,
  div[data-testid="metric-container"] [data-testid="metric-value"] p { color: white !important; font-size: 1.8rem !important; font-weight: 800 !important; }
  div[data-testid="metric-container"] [data-testid="metric-delta"] { color: #2dc653 !important; }

  /* Plotly chart backgrounds transparent */
  .js-plotly-plot { border-radius: 14px; overflow: hidden; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #e63946; border-radius: 3px; }

  /* Footer */
  .footer { text-align:center; color:rgba(255,255,255,0.3); font-size:0.78rem; margin-top:40px; padding-top:20px; border-top:1px solid rgba(255,255,255,0.08); }
</style>
""", unsafe_allow_html=True)

CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.03)",
    font_color="#e0e0e0",
    font_family="Inter",
    title_font_size=15,
    title_font_color="#a8dadc",
    margin=dict(t=50, b=30, l=20, r=20),
)
COLORS = ["#e63946", "#a8dadc", "#457b9d", "#2dc653", "#f4a261", "#e9c46a", "#264653"]

# ── Data & Model Loading ───────────────────────────────────────────────────────
@st.cache_data(show_spinner="📊 Loading dataset…")
def load_dataset():
    p = os.path.join(ROOT, "data", "processed", "heart_engineered.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    raw = generate_synthetic_data(n=303, random_state=42)
    df  = engineer_features(impute_missing_values(clean_data(raw)))
    os.makedirs(os.path.join(ROOT, "data", "processed"), exist_ok=True)
    df.to_csv(p, index=False)
    return df

@st.cache_resource(show_spinner="🤖 Training AI models (first run ~60 s)…")
def get_model_and_data():
    df = load_dataset()
    feat_cols = get_all_feature_columns(df, target_col="target")
    X, y = df[feat_cols], df["target"]
    mp = os.path.join(ROOT, "models", "champion_model.pkl")
    if os.path.exists(mp):
        model = load_champion_model(mp)
        from sklearn.model_selection import train_test_split
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    else:
        model,X_train,X_test,y_train,y_test,_ = run_full_pipeline(X, y)
    return model, X_train, X_test, y_train, y_test, feat_cols

@st.cache_resource(show_spinner="🔬 Computing SHAP values…")
def get_shap(_model, X_train):
    import shap
    try:
        exp = shap.TreeExplainer(_model)
        sv  = exp.shap_values(X_train)
        sv  = sv[1] if isinstance(sv, list) else sv
        ev  = exp.expected_value
        ev  = ev[1] if isinstance(ev, list) else ev
        return exp, sv, ev
    except Exception:
        bg  = shap.sample(X_train, 100, random_state=42)
        exp = shap.KernelExplainer(_model.predict_proba, bg)
        sv  = exp.shap_values(X_train.iloc[:100])
        sv  = sv[1] if isinstance(sv, list) else sv
        ev  = exp.expected_value
        ev  = ev[1] if isinstance(ev, list) else ev
        return exp, sv, ev

# ── Load everything ────────────────────────────────────────────────────────────
df = load_dataset()
model, X_train, X_test, y_train, y_test, feat_cols = get_model_and_data()

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="text-align:center;padding:16px 0 8px 0;">
  <div style="font-size:2.8rem;">🫀</div>
  <div style="font-size:1.15rem;font-weight:800;color:#e63946;">Heart Disease AI</div>
  <div style="font-size:0.78rem;color:rgba(255,255,255,0.5);margin-top:4px;">Partnr Network Project</div>
  <div style="font-size:0.72rem;color:rgba(255,255,255,0.4);">Arepalli Venkata Lakshmi</div>
</div>
<hr style="border-color:rgba(255,255,255,0.1);margin:12px 0;">
""", unsafe_allow_html=True)

section = st.sidebar.radio(
    "Navigate to:",
    ["🏥 Patient Predictor", "📊 Cohort Analysis", "💰 Financial Impact"],
)

st.sidebar.markdown("<hr style='border-color:rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
n_pos = int(df["target"].sum())
n_neg = len(df) - n_pos
st.sidebar.markdown(f"""
<div class='glass-card' style='padding:14px;'>
  <div style='font-size:0.75rem;color:rgba(255,255,255,0.5);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px;'>Dataset Stats</div>
  <div style='display:flex;justify-content:space-between;margin-bottom:6px;'>
    <span style='color:#a8dadc;font-size:0.85rem;'>Total Patients</span>
    <span style='color:white;font-weight:700;'>{len(df)}</span>
  </div>
  <div style='display:flex;justify-content:space-between;margin-bottom:6px;'>
    <span style='color:#e63946;font-size:0.85rem;'>❤️ Disease</span>
    <span style='color:white;font-weight:700;'>{n_pos}</span>
  </div>
  <div style='display:flex;justify-content:space-between;'>
    <span style='color:#2dc653;font-size:0.85rem;'>✅ Healthy</span>
    <span style='color:white;font-weight:700;'>{n_neg}</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – PATIENT PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
if section == "🏥 Patient Predictor":

    st.markdown("""
    <div class='hero'>
      <h1>🏥 Patient Risk Predictor</h1>
      <p>Enter clinical measurements in the sidebar → get instant AI-powered heart disease risk assessment with SHAP explanation</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar inputs
    with st.sidebar:
        st.markdown("<div class='sec-header' style='margin-top:0;'>🩺 Patient Data</div>", unsafe_allow_html=True)
        age      = st.slider("Age (years)", 20, 90, 55)
        sex_sel  = st.selectbox("Sex", ["Female", "Male"])
        sex_val  = 0 if sex_sel == "Female" else 1
        cp_sel   = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"])
        cp_val   = ["Typical Angina","Atypical Angina","Non-anginal","Asymptomatic"].index(cp_sel)
        trestbps = st.slider("Resting BP (mmHg)", 80, 200, 130)
        chol     = st.slider("Cholesterol (mg/dl)", 100, 564, 240)
        fbs_sel  = st.selectbox("Fasting Blood Sugar > 120", ["No", "Yes"])
        fbs_val  = 0 if fbs_sel == "No" else 1
        restecg_sel = st.selectbox("Resting ECG", ["Normal", "ST-T Abnormality", "LV Hypertrophy"])
        restecg_val = ["Normal","ST-T Abnormality","LV Hypertrophy"].index(restecg_sel)
        thalach  = st.slider("Max Heart Rate", 60, 220, 150)
        exang_sel = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
        exang_val = 0 if exang_sel == "No" else 1
        oldpeak  = st.slider("ST Depression (oldpeak)", 0.0, 6.2, 1.0, 0.1)
        slope_sel = st.selectbox("Slope of ST Segment", ["Upsloping", "Flat", "Downsloping"])
        slope_val = ["Upsloping","Flat","Downsloping"].index(slope_sel)
        ca       = st.selectbox("Major Vessels Coloured", [0, 1, 2, 3])
        thal_sel = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
        thal_val = ["Normal","Fixed Defect","Reversible Defect"].index(thal_sel) + 1

    # Build patient
    patient_raw = pd.DataFrame([{
        "age":age,"sex":sex_val,"cp":cp_val,"trestbps":trestbps,"chol":chol,
        "fbs":fbs_val,"restecg":restecg_val,"thalach":thalach,"exang":exang_val,
        "oldpeak":oldpeak,"slope":slope_val,"ca":ca,"thal":thal_val,"target":0
    }])
    patient_eng = engineer_features(patient_raw)[feat_cols]
    proba = model.predict_proba(patient_eng)[0][1]
    risk  = "HIGH RISK" if proba >= 0.5 else "LOW RISK"
    badge = "badge-high" if proba >= 0.5 else "badge-low"
    icon  = "🔴" if proba >= 0.5 else "🟢"

    # Top result row — custom HTML cards (guaranteed white text on dark bg)
    risk_color = "#e63946" if proba >= 0.5 else "#2dc653"
    st.markdown(f"""
    <div class='kpi-wrap'>
      <div class='kpi'>
        <div class='val' style='color:#a8dadc;'>{proba:.1%}</div>
        <div class='lbl'>Disease Probability</div>
      </div>
      <div class='kpi'>
        <div class='val' style='color:{risk_color};font-size:1.3rem;'>{icon} {risk}</div>
        <div class='lbl'>Risk Level</div>
      </div>
      <div class='kpi'>
        <div class='val'>{age} yrs</div>
        <div class='lbl'>Age</div>
      </div>
      <div class='kpi'>
        <div class='val'>{thalach} bpm</div>
        <div class='lbl'>Max Heart Rate</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_gauge, col_info = st.columns([1, 1])

    with col_gauge:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            delta={"reference": 50, "increasing": {"color": "#e63946"}, "decreasing": {"color": "#2dc653"}},
            value=round(proba * 100, 1),
            title={"text": "Heart Disease Risk Score", "font": {"size": 16, "color": "#a8dadc"}},
            number={"suffix": "%", "font": {"size": 42, "color": "white"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#a8dadc", "tickfont": {"color": "#a8dadc"}},
                "bar": {"color": "#e63946" if proba >= 0.5 else "#2dc653", "thickness": 0.3},
                "bgcolor": "rgba(255,255,255,0.05)",
                "bordercolor": "rgba(255,255,255,0.1)",
                "steps": [
                    {"range": [0, 30],  "color": "rgba(45,198,83,0.15)"},
                    {"range": [30, 60], "color": "rgba(244,162,97,0.15)"},
                    {"range": [60, 100],"color": "rgba(230,57,70,0.15)"},
                ],
                "threshold": {"line": {"color": "white", "width": 3}, "value": 50}
            }
        ))
        fig_g.update_layout(**CHART_THEME, height=320)
        st.plotly_chart(fig_g, use_container_width=True)

    with col_info:
        st.markdown(f"""
        <div class='glass-card' style='margin-top:0;'>
          <div style='font-size:0.8rem;color:rgba(255,255,255,0.5);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:16px;'>Prediction Result</div>
          <div style='text-align:center;margin-bottom:20px;'>
            <span class='{badge}'>{icon} {risk}</span>
          </div>
          <div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;font-size:0.85rem;'>
            <div style='color:rgba(255,255,255,0.5);'>Probability</div><div style='color:white;font-weight:700;'>{proba:.1%}</div>
            <div style='color:rgba(255,255,255,0.5);'>Chest Pain</div><div style='color:white;font-weight:700;'>{cp_sel}</div>
            <div style='color:rgba(255,255,255,0.5);'>Cholesterol</div><div style='color:white;font-weight:700;'>{chol} mg/dl</div>
            <div style='color:rgba(255,255,255,0.5);'>Resting BP</div><div style='color:white;font-weight:700;'>{trestbps} mmHg</div>
            <div style='color:rgba(255,255,255,0.5);'>ST Depression</div><div style='color:white;font-weight:700;'>{oldpeak}</div>
            <div style='color:rgba(255,255,255,0.5);'>Vessels</div><div style='color:white;font-weight:700;'>{ca}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # SHAP explanation
    st.markdown("<div class='sec-header'>🔍 AI Explanation — Why this prediction?</div>", unsafe_allow_html=True)
    try:
        import shap
        explainer, sv_train, exp_val = get_shap(model, X_train)
        psv = explainer.shap_values(patient_eng)
        psv = psv[1][0] if isinstance(psv, list) else psv[0]

        shap_df = pd.DataFrame({"Feature": feat_cols, "SHAP Value": psv, "Patient Value": patient_eng.values[0]})
        shap_df["Abs"] = shap_df["SHAP Value"].abs()
        shap_df = shap_df.nlargest(12, "Abs").sort_values("SHAP Value")
        shap_df["Color"] = shap_df["SHAP Value"].apply(lambda x: "#e63946" if x > 0 else "#2dc653")
        shap_df["Direction"] = shap_df["SHAP Value"].apply(lambda x: "Increases Risk" if x > 0 else "Decreases Risk")

        fig_shap = go.Figure()
        for direction, color in [("Increases Risk", "#e63946"), ("Decreases Risk", "#2dc653")]:
            sub = shap_df[shap_df["Direction"] == direction]
            fig_shap.add_trace(go.Bar(
                x=sub["SHAP Value"], y=sub["Feature"],
                orientation="h", name=direction,
                marker_color=color,
                marker_line_color="rgba(255,255,255,0.2)",
                marker_line_width=1,
                text=[f"{v:.3f}" for v in sub["SHAP Value"]],
                textposition="outside",
                textfont=dict(color="white", size=11),
            ))
        fig_shap.update_layout(
            **CHART_THEME, height=420,
            title="Feature Contributions to This Patient's Risk Score",
            xaxis_title="SHAP Value (Impact on Prediction)",
            yaxis_title="",
            barmode="relative",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            xaxis=dict(zeroline=True, zerolinecolor="rgba(255,255,255,0.3)", zerolinewidth=2)
        )
        st.plotly_chart(fig_shap, use_container_width=True)

        # Top 3 reasons
        top3 = shap_df.nlargest(3, "Abs")
        st.markdown("<div class='sec-header'>📋 Top 3 Risk Drivers</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        for col, (_, row) in zip([c1, c2, c3], top3.iterrows()):
            direction_color = "#e63946" if row["SHAP Value"] > 0 else "#2dc653"
            direction_text  = "Raises Risk" if row["SHAP Value"] > 0 else "Lowers Risk"
            col.markdown(f"""
            <div class='glass-card' style='text-align:center;'>
              <div style='font-size:1.5rem;margin-bottom:8px;'>{"⚠️" if row["SHAP Value"]>0 else "✅"}</div>
              <div style='font-weight:700;color:#a8dadc;font-size:0.9rem;margin-bottom:6px;'>{row['Feature']}</div>
              <div style='font-size:1.2rem;font-weight:800;color:{direction_color};'>{row['SHAP Value']:+.3f}</div>
              <div style='font-size:0.75rem;color:rgba(255,255,255,0.5);margin-top:4px;'>{direction_text}</div>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.warning(f"SHAP unavailable: {e}")
        if hasattr(model, "feature_importances_"):
            fi = pd.Series(model.feature_importances_, index=feat_cols).nlargest(12).sort_values()
            fig_fi = go.Figure(go.Bar(x=fi.values, y=fi.index, orientation="h",
                                      marker_color=COLORS[0]))
            fig_fi.update_layout(**CHART_THEME, title="Feature Importances", height=400)
            st.plotly_chart(fig_fi, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – COHORT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif section == "📊 Cohort Analysis":

    st.markdown("""
    <div class='hero'>
      <h1>📊 Cohort Analysis</h1>
      <p>Explore population-level trends, statistical distributions, model feature importance and global SHAP explanations</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("<div class='sec-header' style='margin-top:0;'>🔽 Filters</div>", unsafe_allow_html=True)
        sex_f = st.multiselect("Sex", [0, 1], default=[0, 1], format_func=lambda x: "Female" if x==0 else "Male")
        age_f = st.slider("Age Range", int(df.age.min()), int(df.age.max()), (30, 75))

    df_f = df[df["sex"].isin(sex_f) & df["age"].between(*age_f)] if sex_f else df

    # Top KPIs
    st.markdown(f"""
    <div class='kpi-wrap'>
      <div class='kpi'><div class='val'>{len(df_f)}</div><div class='lbl'>Patients</div></div>
      <div class='kpi'><div class='val'>{df_f['target'].mean():.1%}</div><div class='lbl'>Disease Rate</div></div>
      <div class='kpi'><div class='val'>{df_f['age'].mean():.0f}</div><div class='lbl'>Avg Age</div></div>
      <div class='kpi'><div class='val'>{df_f['chol'].mean():.0f}</div><div class='lbl'>Avg Cholesterol</div></div>
      <div class='kpi'><div class='val'>{df_f['thalach'].mean():.0f}</div><div class='lbl'>Avg Max HR</div></div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈 EDA Charts", "🏆 Feature Importance", "🔬 SHAP Global"])

    label_map = {0: "No Disease", 1: "Disease"}
    df_f_lab  = df_f.copy()
    df_f_lab["Outcome"] = df_f_lab["target"].map(label_map)
    COLOR_MAP = {"No Disease": "#a8dadc", "Disease": "#e63946"}

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df_f_lab, x="age", color="Outcome", barmode="overlay",
                               nbins=25, opacity=0.75, color_discrete_map=COLOR_MAP,
                               title="Age Distribution by Outcome")
            fig.update_layout(**CHART_THEME)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.box(df_f_lab, x="Outcome", y="thalach", color="Outcome",
                         color_discrete_map=COLOR_MAP, title="Max Heart Rate by Outcome",
                         points="all")
            fig.update_layout(**CHART_THEME)
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            # FIX: No trendline= parameter at all (both "ols" and "lowess"
            # require statsmodels which is not installed). We manually add
            # a linear trend line per outcome group using numpy only.
            fig = px.scatter(df_f_lab, x="age", y="chol", color="Outcome",
                             color_discrete_map=COLOR_MAP, opacity=0.65,
                             title="Age vs Cholesterol")
            for outcome, color in COLOR_MAP.items():
                subset = df_f_lab[df_f_lab["Outcome"] == outcome]
                if len(subset) > 1:
                    m, b = np.polyfit(subset["age"], subset["chol"], 1)
                    x_line = np.linspace(subset["age"].min(), subset["age"].max(), 50)
                    y_line = m * x_line + b
                    fig.add_trace(go.Scatter(
                        x=x_line, y=y_line,
                        mode="lines",
                        name=f"{outcome} trend",
                        line=dict(color=color, width=2, dash="dash"),
                        showlegend=False,
                    ))
            fig.update_layout(**CHART_THEME)
            st.plotly_chart(fig, use_container_width=True)
        with c4:
            fig = px.violin(df_f_lab, x="Outcome", y="oldpeak", color="Outcome",
                            color_discrete_map=COLOR_MAP, box=True,
                            title="ST Depression (oldpeak) by Outcome")
            fig.update_layout(**CHART_THEME)
            st.plotly_chart(fig, use_container_width=True)

        c5, c6 = st.columns(2)
        with c5:
            pie_data = df_f_lab["Outcome"].value_counts()
            fig = go.Figure(go.Pie(
                labels=pie_data.index, values=pie_data.values,
                marker_colors=["#a8dadc","#e63946"],
                hole=0.5, textfont_size=13
            ))
            fig.update_layout(**CHART_THEME, title="Disease Prevalence")
            st.plotly_chart(fig, use_container_width=True)
        with c6:
            cp_labels = {0:"Typical Angina",1:"Atypical Angina",2:"Non-anginal",3:"Asymptomatic"}
            df_f_lab["CP Type"] = df_f_lab["cp"].map(cp_labels)
            fig = px.bar(df_f_lab.groupby(["CP Type","Outcome"]).size().reset_index(name="Count"),
                         x="CP Type", y="Count", color="Outcome",
                         color_discrete_map=COLOR_MAP, barmode="group",
                         title="Disease by Chest Pain Type")
            fig.update_layout(**CHART_THEME)
            st.plotly_chart(fig, use_container_width=True)

        # Correlation heatmap
        st.markdown("<div class='sec-header'>Correlation Heatmap</div>", unsafe_allow_html=True)
        corr = df_f.select_dtypes(include=np.number).corr()
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
            text=corr.round(2).values, texttemplate="%{text}",
            textfont_size=9,
        ))
        fig.update_layout(**CHART_THEME, height=520, title="Feature Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if hasattr(model, "feature_importances_"):
            fi = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=True).tail(15)
            colors = [f"rgba(230,57,70,{0.4 + 0.6*(i/len(fi))})" for i in range(len(fi))]
            fig = go.Figure(go.Bar(
                x=fi.values, y=fi.index, orientation="h",
                marker_color=colors,
                marker_line_color="rgba(255,255,255,0.2)",
                marker_line_width=1,
                text=[f"{v:.4f}" for v in fi.values],
                textposition="outside",
                textfont=dict(color="white", size=10),
            ))
            fig.update_layout(**CHART_THEME, height=520,
                              title="Random Forest Feature Importances (Top 15)",
                              xaxis_title="Importance Score")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importances not available for this model type.")

    with tab3:
        try:
            import shap
            explainer, sv_train, exp_val = get_shap(model, X_train)

            # Bar importance from SHAP
            mean_abs = np.abs(sv_train).mean(axis=0)
            shap_imp = pd.Series(mean_abs, index=feat_cols).sort_values(ascending=True).tail(15)
            grad = [f"rgba(168,218,220,{0.3 + 0.7*(i/len(shap_imp))})" for i in range(len(shap_imp))]
            fig = go.Figure(go.Bar(
                x=shap_imp.values, y=shap_imp.index, orientation="h",
                marker_color=grad,
                text=[f"{v:.4f}" for v in shap_imp.values],
                textposition="outside", textfont=dict(color="white", size=10),
            ))
            fig.update_layout(**CHART_THEME, height=520,
                              title="SHAP Global Feature Importance (Mean |SHAP|)",
                              xaxis_title="Mean |SHAP Value|")
            st.plotly_chart(fig, use_container_width=True)

            # matplotlib SHAP beeswarm
            st.markdown("<div class='sec-header'>SHAP Beeswarm Plot</div>", unsafe_allow_html=True)
            plt.style.use("dark_background")
            fig_b, ax = plt.subplots(figsize=(10, 6))
            fig_b.patch.set_facecolor("#0f0c29")
            ax.set_facecolor("#1a1a2e")
            shap.summary_plot(sv_train, X_train, show=False, max_display=12, plot_size=None)
            plt.title("SHAP Beeswarm – Feature Impact Distribution", color="#a8dadc", fontsize=13, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_b, use_container_width=True)
            plt.close()
        except Exception as e:
            st.warning(f"SHAP unavailable: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – FINANCIAL IMPACT
# ══════════════════════════════════════════════════════════════════════════════
elif section == "💰 Financial Impact":

    st.markdown("""
    <div class='hero'>
      <h1>💰 Financial Impact Calculator</h1>
      <p>Adjust treatment costs and instantly see how much the AI model saves versus traditional hospital baselines</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("<div class='sec-header' style='margin-top:0;'>💵 Cost Parameters</div>", unsafe_allow_html=True)
        c_tp = st.number_input("True Positive cost ($)",  value=-500,  step=100)
        c_fp = st.number_input("False Positive cost ($)", value=-1200, step=100)
        c_tn = st.number_input("True Negative cost ($)",  value=0,     step=100)
        c_fn = st.number_input("False Negative cost ($)", value=-8500, step=100)

    cost_matrix = {"TP": c_tp, "FP": c_fp, "TN": c_tn, "FN": c_fn}
    y_pred  = model.predict(X_test)
    result  = calculate_financial_impact(y_test.values, y_pred, cost_matrix=cost_matrix)
    cc      = result["confusion_counts"]

    # KPI row
    st.markdown(f"""
    <div class='kpi-wrap'>
      <div class='kpi'><div class='val'>${result["model_total_cost"]:,.0f}</div><div class='lbl'>AI Model Cost</div></div>
      <div class='kpi'><div class='val'>${result["baseline_treat_all_cost"]:,.0f}</div><div class='lbl'>Treat-All Cost</div></div>
      <div class='kpi'><div class='val'>${result["baseline_treat_none_cost"]:,.0f}</div><div class='lbl'>Treat-None Cost</div></div>
      <div class='kpi'><div class='val' style='color:#2dc653;'>${result["model_vs_treat_all_savings"]:,.0f}</div><div class='lbl'>Saved vs Treat-All</div></div>
      <div class='kpi'><div class='val' style='color:#2dc653;'>${result["model_vs_treat_none_savings"]:,.0f}</div><div class='lbl'>Saved vs Treat-None</div></div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        # Confusion matrix heatmap
        cm_vals = [[cc["TP"], cc["FN"]], [cc["FP"], cc["TN"]]]
        fig = go.Figure(go.Heatmap(
            z=cm_vals,
            x=["Predicted: Disease","Predicted: Healthy"],
            y=["Actual: Disease","Actual: Healthy"],
            colorscale=[[0,"rgba(168,218,220,0.2)"],[1,"rgba(230,57,70,0.8)"]],
            text=[[str(v) for v in row] for row in cm_vals],
            texttemplate="<b>%{text}</b>",
            textfont_size=22,
            showscale=False,
        ))
        fig.update_layout(**CHART_THEME, height=320, title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Policy cost comparison
        policies = ["AI Model","Treat Everyone","Treat No One"]
        costs    = [result["model_total_cost"], result["baseline_treat_all_cost"], result["baseline_treat_none_cost"]]
        bar_colors = ["#2dc653","#f4a261","#e63946"]
        fig = go.Figure(go.Bar(
            x=policies, y=costs,
            marker_color=bar_colors,
            marker_line_color="rgba(255,255,255,0.2)",
            marker_line_width=1,
            text=[f"${v:,.0f}" for v in costs],
            textposition="outside",
            textfont=dict(color="white", size=12),
        ))
        fig.update_layout(**CHART_THEME, height=320,
                          title="Policy Cost Comparison", yaxis_title="Total Cost ($)")
        st.plotly_chart(fig, use_container_width=True)

    # Sensitivity analysis
    st.markdown("<div class='sec-header'>📉 Sensitivity Analysis — Varying False-Negative Cost</div>", unsafe_allow_html=True)
    sa_df = sensitivity_analysis(y_test.values, y_pred, fn_cost_range=(-3000, -15000), steps=20)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sa_df["fn_cost"], y=sa_df["model_total_cost"],
                             name="AI Model", line=dict(color="#2dc653", width=3),
                             mode="lines+markers", marker_size=6))
    fig.add_trace(go.Scatter(x=sa_df["fn_cost"], y=sa_df["treat_all_cost"],
                             name="Treat Everyone", line=dict(color="#f4a261", width=3, dash="dash"),
                             mode="lines+markers", marker_size=6))
    fig.update_layout(**CHART_THEME, height=380,
                      title="AI Model vs Treat-All as False-Negative Cost Varies",
                      xaxis_title="False-Negative Cost ($)",
                      yaxis_title="Total Simulated Cost ($)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    # Confusion matrix breakdown cards
    st.markdown("<div class='sec-header'>🔢 Confusion Matrix Breakdown</div>", unsafe_allow_html=True)
    cc1, cc2, cc3, cc4 = st.columns(4)
    for col, label, val, color, desc in [
        (cc1, "True Positives",  cc["TP"], "#2dc653", "Correctly identified disease"),
        (cc2, "False Positives", cc["FP"], "#f4a261", "Healthy patients over-treated"),
        (cc3, "True Negatives",  cc["TN"], "#a8dadc", "Correctly cleared healthy"),
        (cc4, "False Negatives", cc["FN"], "#e63946",  "Missed high-risk patients"),
    ]:
        col.markdown(f"""
        <div class='glass-card' style='text-align:center;border-left:4px solid {color};'>
          <div style='font-size:2.4rem;font-weight:800;color:{color};'>{val}</div>
          <div style='font-weight:700;color:white;font-size:0.9rem;margin:4px 0;'>{label}</div>
          <div style='font-size:0.75rem;color:rgba(255,255,255,0.45);'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # Download
    st.markdown("<br>", unsafe_allow_html=True)
    mp = os.path.join(ROOT, "metrics", "test_metrics.json")
    if os.path.exists(mp):
        with open(mp) as f:
            mdata = json.load(f)
        st.download_button("📥 Download Test Metrics JSON",
                           data=json.dumps(mdata, indent=2),
                           file_name="test_metrics.json",
                           mime="application/json")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='footer'>
  🫀 Heart Disease AI Dashboard &nbsp;|&nbsp; Partnr Network &nbsp;|&nbsp;
  Arepalli Venkata Lakshmi &nbsp;|&nbsp; Built with Streamlit + Scikit-learn + SHAP
</div>
""", unsafe_allow_html=True)