# =============================================================================
# CREDIT CARD FRAUD DETECTION DASHBOARD
# Industry-Level Streamlit Application
# =============================================================================
# Features:
#   - Home Page with project overview and metrics
#   - Single Transaction Prediction with gauge chart
#   - Batch Prediction with CSV upload and download
#   - Model Insights with feature importance and performance metrics
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
import time
import io
from pathlib import Path

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Path setup ────────────────────────────────────────────────────────────────
# ROOT_DIR = Path(__file__).parent
# sys.path.insert(0, str(ROOT_DIR / "source"))

# MODEL_PATH = ROOT_DIR / "models" / "fraud_detection_rf.pkl"
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT_DIR / "models" / "fraud_detection_rf.pkl"
import streamlit as st
# tempraray lines
# st.write("ROOT_DIR:", ROOT_DIR)
# st.write("MODEL_PATH:", MODEL_PATH)
# st.write("Exists:", MODEL_PATH.exists())
# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield — Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "FraudShield: AI-powered real-time fraud detection dashboard.",
    },
)

# =============================================================================
# CUSTOM CSS — Dark financial-tech aesthetic
# =============================================================================
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

/* ── Root variables ── */
:root {
    --bg-primary:    #080c14;
    --bg-card:       #0d1526;
    --bg-elevated:   #111d35;
    --accent-blue:   #3b82f6;
    --accent-cyan:   #06b6d4;
    --accent-green:  #10b981;
    --accent-red:    #ef4444;
    --accent-amber:  #f59e0b;
    --text-primary:  #f0f4ff;
    --text-muted:    #8899bb;
    --border:        rgba(59,130,246,0.18);
    --glow:          rgba(59,130,246,0.12);
}

/* ── Global ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-primary) !important;
    font-family: 'Inter', sans-serif;
    color: var(--text-primary);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1120 0%, #060d1a 100%) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { font-family: 'Inter', sans-serif; }

/* ── Headings ── */
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.4rem !important; padding-bottom: 2rem !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1rem 1.25rem;
    box-shadow: 0 0 24px var(--glow);
    transition: transform .2s, box-shadow .2s;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 36px rgba(59,130,246,0.2);
}
[data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 2rem !important;
    color: var(--accent-blue) !important;
}
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: .8rem !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: .95rem !important;
    padding: .6rem 1.8rem !important;
    letter-spacing: .03em;
    transition: opacity .2s, transform .15s !important;
    box-shadow: 0 4px 20px rgba(59,130,246,0.35) !important;
}
.stButton > button:hover { opacity: .88; transform: translateY(-1px) !important; }

/* ── Inputs ── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: .85rem !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 3px var(--glow) !important;
}

/* ── Select boxes ── */
.stSelectbox > div > div {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}

/* ── Dataframes ── */
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    transition: border-color .2s;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent-blue) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: var(--bg-card); border-radius: 10px; gap: 4px; padding: 4px; }
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 7px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    color: var(--text-muted) !important;
}
.stTabs [aria-selected="true"] {
    background: var(--accent-blue) !important;
    color: #fff !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    color: var(--text-primary) !important;
}

/* ── Alert / success / error boxes ── */
.stAlert { border-radius: 10px !important; font-family: 'Inter', sans-serif !important; }

/* ── Custom HTML components ── */
.fraud-badge {
    display: inline-block;
    background: linear-gradient(135deg, #ef4444, #b91c1c);
    color: #fff;
    padding: .35rem 1rem;
    border-radius: 999px;
    font-family: 'DM Mono', monospace;
    font-size: .82rem;
    font-weight: 500;
    letter-spacing: .06em;
    box-shadow: 0 0 16px rgba(239,68,68,.4);
}
.legit-badge {
    display: inline-block;
    background: linear-gradient(135deg, #10b981, #065f46);
    color: #fff;
    padding: .35rem 1rem;
    border-radius: 999px;
    font-family: 'DM Mono', monospace;
    font-size: .82rem;
    font-weight: 500;
    letter-spacing: .06em;
    box-shadow: 0 0 16px rgba(16,185,129,.4);
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.55rem;
    font-weight: 700;
    color: #f0f4ff;
    margin-bottom: .2rem;
}
.section-sub {
    color: #8899bb;
    font-size: .88rem;
    margin-bottom: 1.4rem;
}
.info-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 20px var(--glow);
}
.info-card h4 {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    color: var(--accent-cyan);
    margin-bottom: .5rem;
}
.info-card p { color: var(--text-muted); font-size: .88rem; line-height: 1.6; margin: 0; }
.divider { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60a5fa, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
    margin: 0;
}
.hero-sub {
    font-size: 1.05rem;
    color: #8899bb;
    margin-top: .5rem;
    margin-bottom: 0;
}
.sidebar-logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 800;
    color: #60a5fa;
    letter-spacing: -.01em;
}
.sidebar-tagline { font-size: .78rem; color: #8899bb; margin-top: -.2rem; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# UTILITIES & CACHING
# =============================================================================

import joblib

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        return model, None

    except FileNotFoundError:
        return None, f"Model file not found at {MODEL_PATH}"

    except Exception as e:
        return None, f"Error loading model: {e}"
    
# temprary added    
# model, error = load_model()
# st.write("Model:", model)
# st.write("Error:", error)

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

def try_import_predict():
    """Attempt to import the custom predict function from source/predict.py."""
    try:
        from source.predict import predict_transaction
        return predict_transaction, None
    except ImportError:
        return None, "source/predict.py not found — using built-in prediction fallback."
    except Exception as e:
        return None, f"Error importing predict.py: {e}"


def builtin_predict(model, features: np.ndarray):
    """
    Fallback prediction using the loaded model directly.
    Returns (class_label, fraud_probability).
    """
    proba = model.predict_proba(features)[0]
    label = int(model.predict(features)[0])
    return label, float(proba[1])


def run_prediction(model, predict_fn, features: np.ndarray):
    """
    Unified prediction wrapper.
    Tries the imported predict_fn first, falls back to builtin_predict.
    Returns (label: int, fraud_prob: float).
    """
    if predict_fn is not None:
        try:
            result = predict_fn(model, features)
            # Accept (label, prob) tuple or just label
            if isinstance(result, (tuple, list)) and len(result) == 2:
                return int(result[0]), float(result[1])
            else:
                label = int(result)
                proba = model.predict_proba(features)[0]
                return label, float(proba[1])
        except Exception:
            pass
    return builtin_predict(model, features)


FEATURE_COLS = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]

def validate_df(df: pd.DataFrame):
    """Check that a DataFrame has all required feature columns."""
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    return missing


def plotly_gauge(prob: float) -> go.Figure:
    """Render a sleek fraud-probability gauge chart."""
    color = "#10b981" if prob < 0.4 else ("#f59e0b" if prob < 0.7 else "#ef4444")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 42, "color": color, "family": "DM Mono"}},
        delta={"reference": 50, "increasing": {"color": "#ef4444"}, "decreasing": {"color": "#10b981"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#8899bb",
                    "tickfont": {"color": "#8899bb", "size": 11}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "#0d1526",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  40], "color": "rgba(16,185,129,0.12)"},
                {"range": [40, 70], "color": "rgba(245,158,11,0.12)"},
                {"range": [70,100], "color": "rgba(239,68,68,0.12)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.75,
                "value": prob * 100,
            },
        },
        title={"text": "FRAUD PROBABILITY", "font": {"size": 13, "color": "#8899bb", "family": "Syne"}},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=30, b=10, l=30, r=30),
        height=280,
    )
    return fig


def plotly_feature_importance(model, top_n: int = 20) -> go.Figure:
    """Horizontal bar chart of top feature importances."""
    importances = model.feature_importances_
    feat_names = FEATURE_COLS
    df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
    df = df.sort_values("Importance", ascending=True).tail(top_n)

    fig = go.Figure(go.Bar(
        x=df["Importance"],
        y=df["Feature"],
        orientation="h",
        marker=dict(
            color=df["Importance"],
            colorscale=[[0, "#1e3a5f"], [0.5, "#3b82f6"], [1, "#06b6d4"]],
            showscale=False,
        ),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8899bb", family="DM Mono", size=11),
        xaxis=dict(gridcolor="rgba(59,130,246,0.1)", zeroline=False),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=20, t=20, b=20),
        height=520,
    )
    return fig


def plotly_fraud_dist(pred_df: pd.DataFrame) -> go.Figure:
    """Donut chart of fraud vs legitimate counts."""
    counts = pred_df["Prediction"].value_counts()
    labels = ["Legitimate" if k == 0 else "Fraudulent" for k in counts.index]
    colors = ["#10b981" if k == 0 else "#ef4444" for k in counts.index]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=counts.values,
        hole=.58,
        marker=dict(colors=colors, line=dict(color="#080c14", width=3)),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
        textfont=dict(size=13, family="DM Mono"),
    ))
    fig.add_annotation(
        text=f"<b>{len(pred_df)}</b><br><span style='font-size:11px'>records</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=18, color="#f0f4ff", family="DM Mono"),
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(font=dict(color="#8899bb"), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=10, b=10, l=10, r=10),
        height=300,
    )
    return fig


def plotly_prob_histogram(pred_df: pd.DataFrame) -> go.Figure:
    """Histogram of fraud probabilities for batch results."""
    fig = go.Figure()
    if "Fraud_Probability" not in pred_df.columns:
        return fig
    fig.add_trace(go.Histogram(
        x=pred_df["Fraud_Probability"],
        nbinsx=30,
        marker_color="#3b82f6",
        opacity=0.82,
        hovertemplate="Prob: %{x:.2f}<br>Count: %{y}<extra></extra>",
        name="All transactions",
    ))
    fraud_probs = pred_df[pred_df["Prediction"] == 1]["Fraud_Probability"]
    if len(fraud_probs):
        fig.add_trace(go.Histogram(
            x=fraud_probs,
            nbinsx=30,
            marker_color="#ef4444",
            opacity=0.72,
            name="Flagged as fraud",
        ))
    fig.update_layout(
        barmode="overlay",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8899bb", family="DM Mono", size=11),
        xaxis=dict(title="Fraud Probability", gridcolor="rgba(59,130,246,0.1)"),
        yaxis=dict(title="Count", gridcolor="rgba(59,130,246,0.1)"),
        legend=dict(font=dict(color="#8899bb"), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=10, b=10, l=10, r=10),
        height=300,
    )
    return fig


# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='padding: 1.2rem 0 1rem 0;'>
            <div class='sidebar-logo'>🛡️ FraudShield</div>
            <div class='sidebar-tagline'>AI Credit Card Fraud Detection</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        pages = {
            "🏠  Home":                    "Home",
            "🔍  Single Transaction":      "Single",
            "📂  Batch Prediction":        "Batch",
            "📊  Model Insights":          "Insights",
        }
        if "page" not in st.session_state:
            st.session_state.page = "Home"

        for label, key in pages.items():
            selected = st.session_state.page == key
            if st.sidebar.button(
                label,
                key=f"nav_{key}",
                use_container_width=True,
                type="primary" if selected else "secondary",
            ):
                st.session_state.page = key

        st.markdown("---")
        st.markdown("""
        <div style='font-size:.75rem; color:#8899bb; padding: .5rem 0;'>
            <b style='color:#60a5fa'>Model</b><br>Random Forest Classifier<br><br>
            <b style='color:#60a5fa'>Features</b><br>Time · Amount · V1–V28<br><br>
            <b style='color:#60a5fa'>Target</b><br>Class (0 = Legit, 1 = Fraud)
        </div>
        """, unsafe_allow_html=True)

        return st.session_state.page


# =============================================================================
# PAGE — HOME
# =============================================================================

def page_home(model):
    st.markdown("""
    <p class='hero-title'>Credit Card Fraud<br>Detection System</p>
    <p class='hero-sub'>Real-time AI-powered transaction screening · Random Forest · PCA-engineered features</p>
    """, unsafe_allow_html=True)
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── KPI cards ──────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Algorithm", "Random Forest", help="Ensemble decision-tree classifier")
    with k2:
        n_trees = getattr(model, "n_estimators", "N/A") if model else "N/A"
        st.metric("Estimators", n_trees, help="Number of trees in the forest")
    with k3:
        st.metric("Input Features", "30", help="Time, Amount, V1–V28")
    with k4:
        model_status = "✅ Loaded" if model else "❌ Not Found"
        st.metric("Model Status", model_status)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Info cards grid ────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class='info-card'>
            <h4>📋 Dataset Overview</h4>
            <p>European cardholder transactions over two days in September 2013.
            Contains 284,807 transactions — only 492 are fraudulent (0.172%).
            Features V1–V28 are PCA-transformed for privacy. <code>Time</code> and
            <code>Amount</code> are the only original features.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='info-card'>
            <h4>🤖 Model Architecture</h4>
            <p>Random Forest Classifier — an ensemble of decision trees trained via
            bagging. Handles severe class imbalance, provides feature importances,
            and outputs calibrated fraud probabilities. Robust to outliers and
            non-linear feature interactions.</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class='info-card'>
            <h4>🛡️ Fraud Detection</h4>
            <p>Fraudulent transactions are rare but costly. Our model flags
            high-risk transactions in milliseconds — enabling card blocks,
            step-up authentication, or analyst review before losses occur.</p>
        </div>
        """, unsafe_allow_html=True)

    # ── How it works ───────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p class='section-title'>⚙️ How It Works</p>", unsafe_allow_html=True)
    w1, w2, w3, w4 = st.columns(4)
    steps = [
        ("1", "Input Transaction", "Provide transaction features — Time, Amount, and the 28 PCA components."),
        ("2", "Preprocessing",     "Features are validated and formatted as a model-ready numeric vector."),
        ("3", "RF Inference",      "All decision trees vote; the ensemble produces a fraud probability score."),
        ("4", "Decision",          "Transactions above the threshold are flagged as potentially fraudulent."),
    ]
    for col, (num, title, desc) in zip([w1, w2, w3, w4], steps):
        with col:
            st.markdown(f"""
            <div class='info-card' style='text-align:center;'>
                <div style='font-family:DM Mono;font-size:2rem;color:#3b82f6;font-weight:700;'>{num}</div>
                <h4 style='text-align:center;margin-top:.3rem;'>{title}</h4>
                <p style='text-align:center;'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    # ── Feature reference ──────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📐 Feature Reference — All 30 Input Features"):
        feat_data = {
            "Feature": ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)],
            "Type": ["Original", "Original"] + ["PCA Component"] * 28,
            "Description": [
                "Seconds elapsed since the first transaction in the dataset",
                "Transaction amount in EUR",
            ] + [f"PCA-transformed anonymised feature component {i}" for i in range(1, 29)],
        }
        st.dataframe(pd.DataFrame(feat_data), use_container_width=True, hide_index=True)


# =============================================================================
# PAGE — SINGLE TRANSACTION PREDICTION
# =============================================================================

def page_single(model, predict_fn):
    st.markdown("<p class='section-title'>🔍 Single Transaction Prediction</p>", unsafe_allow_html=True)
    st.markdown("<p class='section-sub'>Enter transaction details and get an instant fraud probability score.</p>",
                unsafe_allow_html=True)

    if model is None:
        st.error("⚠️ Model not loaded. Please check `models/fraud_detection_rf.pkl`.")
        return

    with st.form("single_pred_form", clear_on_submit=False):
        st.markdown("##### 💳 Transaction Details")
        col_t, col_a = st.columns(2)
        with col_t:
            time_val = st.number_input("⏱ Time (seconds)", min_value=0.0, value=0.0,
                                    help="Seconds since first transaction in dataset")
        with col_a:
            amount_val = st.number_input("💰 Amount (EUR)", min_value=0.0, value=100.0,
                                        format="%.2f", help="Transaction amount")

        st.markdown("##### 🔢 PCA Components (V1 – V28)")
        st.caption("These are the 28 PCA-transformed features from the original dataset.")

        v_vals = {}
        cols_per_row = 4
        v_keys = [f"V{i}" for i in range(1, 29)]
        for row_start in range(0, 28, cols_per_row):
            row_keys = v_keys[row_start: row_start + cols_per_row]
            row_cols = st.columns(cols_per_row)
            for col, key in zip(row_cols, row_keys):
                with col:
                    v_vals[key] = st.number_input(key, value=0.0, format="%.4f",
                                                label_visibility="visible")

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🚀 Run Fraud Analysis", use_container_width=True)

    # ── Result ─────────────────────────────────────────────────────────────────
    if submitted:
        feature_vector = np.array([[
            time_val, *[v_vals[f"V{i}"] for i in range(1, 29)], amount_val
        ]])

        with st.spinner("Analysing transaction…"):
            time.sleep(0.4)   # small UX delay for realism
            try:
                label, fraud_prob = run_prediction(model, predict_fn, feature_vector)
            except Exception as e:
                st.error(f"Prediction error: {e}")
                return

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        r1, r2 = st.columns([1, 1])

        with r1:
            st.plotly_chart(plotly_gauge(fraud_prob), use_container_width=True, config={"displayModeBar": False})

        with r2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            if label == 1:
                st.markdown("### 🚨 Transaction Flagged")
                st.markdown("<span class='fraud-badge'>⚠ FRAUDULENT</span>", unsafe_allow_html=True)
                st.error(f"**Fraud Probability: {fraud_prob*100:.2f}%** — This transaction has been identified as high-risk. "
                        "Recommend blocking the card and initiating a cardholder verification.")
            else:
                st.markdown("### ✅ Transaction Approved")
                st.markdown("<span class='legit-badge'>✔ LEGITIMATE</span>", unsafe_allow_html=True)
                st.success(f"**Fraud Probability: {fraud_prob*100:.2f}%** — No suspicious patterns detected. "
                        "Transaction appears consistent with normal cardholder behaviour.")

            # Risk tier
            if fraud_prob < 0.3:
                risk, risk_color = "🟢 LOW RISK", "#10b981"
            elif fraud_prob < 0.6:
                risk, risk_color = "🟡 MEDIUM RISK", "#f59e0b"
            elif fraud_prob < 0.85:
                risk, risk_color = "🟠 HIGH RISK", "#f97316"
            else:
                risk, risk_color = "🔴 CRITICAL RISK", "#ef4444"

            st.markdown(f"""
            <div class='info-card' style='margin-top:1rem;'>
                <h4 style='color:{risk_color};'>{risk}</h4>
                <p>Fraud probability threshold: <b>50%</b><br>
                Computed probability: <b style='color:{risk_color};'>{fraud_prob*100:.2f}%</b><br>
                Model: Random Forest Classifier</p>
            </div>
            """, unsafe_allow_html=True)

        # ── Feature summary ────────────────────────────────────────────────────
        with st.expander("📋 Submitted Feature Values"):
            fdf = pd.DataFrame({
                "Feature": FEATURE_COLS,
                "Value": [time_val] + [v_vals[f"V{i}"] for i in range(1, 29)] + [amount_val],
            })
            st.dataframe(fdf, use_container_width=True, hide_index=True)


# =============================================================================
# PAGE — BATCH PREDICTION
# =============================================================================

def page_batch(model, predict_fn):
    st.markdown("<p class='section-title'>📂 Batch Prediction</p>", unsafe_allow_html=True)
    st.markdown("<p class='section-sub'>Upload a CSV file containing multiple transactions and download enriched predictions.</p>",
                unsafe_allow_html=True)

    if model is None:
        st.error("⚠️ Model not loaded. Please check `models/fraud_detection_rf.pkl`.")
        return

    # ── Template download ──────────────────────────────────────────────────────
    with st.expander("📥 Download CSV Template"):
        template_df = pd.DataFrame(columns=FEATURE_COLS)
        sample = {col: [np.random.randn() * 0.5] for col in FEATURE_COLS}
        sample["Time"]   = [12345.0]
        sample["Amount"] = [149.99]
        template_df = pd.DataFrame(sample)
        csv_template = template_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Template CSV", csv_template,
                        file_name="transaction_template.csv", mime="text/csv")
        st.dataframe(template_df, use_container_width=True, hide_index=True)

    # ── File upload ────────────────────────────────────────────────────────────
    uploaded = st.file_uploader("📁 Upload Transactions CSV", type=["csv"],
                                help="Must contain columns: Time, Amount, V1–V28")

    if uploaded is None:
        st.info("Upload a CSV file to begin batch prediction.")
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return

    st.markdown(f"**{len(df):,} rows × {df.shape[1]} columns detected**")
    with st.expander("🔎 Data Preview (first 20 rows)"):
        st.dataframe(df.head(20), use_container_width=True, hide_index=True)

    missing_cols = validate_df(df)
    if missing_cols:
        st.error(f"Missing required columns: `{', '.join(missing_cols)}`\n\nPlease ensure the CSV contains all 30 feature columns.")
        return

    st.success(f"✅ All {len(FEATURE_COLS)} required feature columns found.")

    # ── Run predictions ────────────────────────────────────────────────────────
    if st.button("⚡ Run Batch Prediction", use_container_width=True):
        with st.spinner(f"Processing {len(df):,} transactions…"):
            X = df[FEATURE_COLS].values
            labels, probs = [], []
            progress = st.progress(0, text="Predicting…")
            chunk = max(1, len(X) // 100)

            for i in range(0, len(X), chunk):
                batch_x = X[i: i + chunk]
                # Batch-mode: use model directly for speed
                batch_proba = model.predict_proba(batch_x)[:, 1]
                batch_labels = (batch_proba >= 0.5).astype(int)
                labels.extend(batch_labels.tolist())
                probs.extend(batch_proba.tolist())
                progress.progress(min(i + chunk, len(X)) / len(X),
                                text=f"Processed {min(i+chunk, len(X)):,} / {len(X):,}")

            progress.empty()

        result_df = df.copy()
        result_df["Prediction"]       = labels
        result_df["Fraud_Probability"] = [round(p, 4) for p in probs]
        result_df["Risk_Label"] = result_df["Fraud_Probability"].apply(
            lambda p: "LOW" if p < 0.3 else ("MEDIUM" if p < 0.6 else ("HIGH" if p < 0.85 else "CRITICAL"))
        )
        result_df["Status"] = result_df["Prediction"].map({0: "Legitimate", 1: "Fraudulent"})

        st.session_state["batch_result"] = result_df
        st.success(f"✅ Predictions complete for {len(result_df):,} transactions!")

    if "batch_result" not in st.session_state:
        return

    result_df = st.session_state["batch_result"]
    n_fraud   = int(result_df["Prediction"].sum())
    n_legit   = len(result_df) - n_fraud
    fraud_pct = n_fraud / len(result_df) * 100

    # ── Summary metrics ────────────────────────────────────────────────────────
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("<p class='section-title' style='font-size:1.2rem;'>📊 Batch Summary</p>",
                unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Total Transactions", f"{len(result_df):,}")
    with m2: st.metric("Legitimate", f"{n_legit:,}", delta=None)
    with m3: st.metric("Flagged Fraud", f"{n_fraud:,}",
                    delta=f"{fraud_pct:.2f}% of total",
                    delta_color="inverse")
    with m4:
        avg_risk = result_df["Fraud_Probability"].mean()
        st.metric("Avg Fraud Prob", f"{avg_risk*100:.2f}%")

    # ── Visualisations ─────────────────────────────────────────────────────────
    v1, v2 = st.columns(2)
    with v1:
        st.markdown("**Transaction Distribution**")
        st.plotly_chart(plotly_fraud_dist(result_df), use_container_width=True,
                        config={"displayModeBar": False})
    with v2:
        st.markdown("**Fraud Probability Distribution**")
        st.plotly_chart(plotly_prob_histogram(result_df), use_container_width=True,
                        config={"displayModeBar": False})

    # ── Results table ──────────────────────────────────────────────────────────
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("**Prediction Results Table**")

    filter_opt = st.radio("Filter results:", ["All", "Fraudulent Only", "Legitimate Only"],
                        horizontal=True, label_visibility="collapsed")
    if filter_opt == "Fraudulent Only":
        display_df = result_df[result_df["Prediction"] == 1]
    elif filter_opt == "Legitimate Only":
        display_df = result_df[result_df["Prediction"] == 0]
    else:
        display_df = result_df

    highlight_cols = ["Prediction", "Fraud_Probability", "Risk_Label", "Status"]
    st.dataframe(
        display_df[FEATURE_COLS[:5] + highlight_cols].head(500),
        use_container_width=True, hide_index=True,
    )
    if len(display_df) > 500:
        st.caption(f"Showing first 500 of {len(display_df):,} rows.")

    # ── Download ───────────────────────────────────────────────────────────────
    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Full Predictions CSV",
        csv_bytes,
        file_name="fraud_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )


# =============================================================================
# PAGE — MODEL INSIGHTS
# =============================================================================

def page_insights(model):
    st.markdown("<p class='section-title'>📊 Model Insights</p>", unsafe_allow_html=True)
    st.markdown("<p class='section-sub'>Feature importances, model diagnostics, and Random Forest internals.</p>",
                unsafe_allow_html=True)

    if model is None:
        st.error("⚠️ Model not loaded. Please check `models/fraud_detection_rf.pkl`.")
        return

    tab1, tab2, tab3 = st.tabs(["🌲 Feature Importance", "📐 Model Parameters", "🎯 Performance Guide"])

    # ── Tab 1: Feature importance ──────────────────────────────────────────────
    with tab1:
        top_n = st.slider("Show top N features", min_value=5, max_value=30, value=20, step=1)
        st.plotly_chart(plotly_feature_importance(model, top_n=top_n),
                        use_container_width=True, config={"displayModeBar": False})

        # Table
        importances = model.feature_importances_
        imp_df = pd.DataFrame({
            "Rank":       range(1, len(FEATURE_COLS) + 1),
            "Feature":    FEATURE_COLS,
            "Importance": importances,
            "% Explained": (importances / importances.sum() * 100).round(2),
        }).sort_values("Importance", ascending=False).reset_index(drop=True)
        imp_df["Rank"] = range(1, len(imp_df) + 1)

        with st.expander("📋 Full Feature Importance Table"):
            st.dataframe(imp_df, use_container_width=True, hide_index=True)

    # ── Tab 2: Model parameters ────────────────────────────────────────────────
    with tab2:
        params = model.get_params() if hasattr(model, "get_params") else {}
        if params:
            param_df = pd.DataFrame({
                "Parameter": list(params.keys()),
                "Value":     [str(v) for v in params.values()],
            })
            st.dataframe(param_df, use_container_width=True, hide_index=True)
        else:
            st.info("Model parameters not available.")

        # High-level model info
        info_items = {
            "Model Type":       type(model).__name__,
            "Number of Trees":  getattr(model, "n_estimators", "N/A"),
            "Max Depth":        str(getattr(model, "max_depth", "None (unlimited)")),
            "Min Samples Split":getattr(model, "min_samples_split", "N/A"),
            "Min Samples Leaf": getattr(model, "min_samples_leaf", "N/A"),
            "Max Features":     str(getattr(model, "max_features", "N/A")),
            "Bootstrap":        str(getattr(model, "bootstrap", "N/A")),
            "Feature Count":    getattr(model, "n_features_in_", len(FEATURE_COLS)),
        }
        c1, c2 = st.columns(2)
        for idx, (k, v) in enumerate(info_items.items()):
            with (c1 if idx % 2 == 0 else c2):
                st.metric(k, v)

    # ── Tab 3: Performance guide ───────────────────────────────────────────────
    with tab3:
        st.markdown("""
        <div class='info-card'>
            <h4>📈 Key Metrics for Fraud Detection</h4>
            <p>Standard accuracy is misleading on imbalanced datasets. Focus on:</p>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        metrics = [
            ("Precision",    "Of all flagged fraud, what % is truly fraud? High precision = few false alarms."),
            ("Recall",       "Of all actual fraud, what % did we catch? High recall = fewer missed fraud cases."),
            ("F1 Score",     "Harmonic mean of Precision and Recall. Balances both for imbalanced datasets."),
            ("AUC-ROC",      "Area under the ROC curve. Measures model's ability to discriminate fraud vs legit."),
        ]
        for col, (name, desc) in zip([m1, m2, m3, m4], metrics):
            with col:
                st.markdown(f"""
                <div class='info-card' style='text-align:center;'>
                    <h4 style='text-align:center;'>{name}</h4>
                    <p style='text-align:center;'>{desc}</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("""
        <div class='info-card' style='margin-top:1.2rem;'>
            <h4>🌲 Why Random Forest for Fraud Detection?</h4>
            <p>
            • <b>Handles imbalance</b>: Works well with <code>class_weight='balanced'</code> or SMOTE oversampling.<br>
            • <b>Robust features</b>: Naturally handles the 28 PCA components without scaling.<br>
            • <b>Interpretable</b>: Feature importances help explain which PCA components drive fraud signals.<br>
            • <b>Probabilistic output</b>: Probability scores allow flexible threshold tuning.<br>
            • <b>No overfitting</b>: Ensemble bagging prevents single tree memorisation.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Threshold tuning guide
        st.markdown("""
        <div class='info-card'>
            <h4>⚙️ Threshold Tuning Guide</h4>
            <p>
            The default decision threshold is 0.5. In fraud detection:<br><br>
            <b>Lower threshold (e.g. 0.3)</b> → Higher recall, more false positives → suits high-value cards.<br>
            <b>Higher threshold (e.g. 0.7)</b> → Higher precision, fewer false positives → suits low-risk accounts.<br><br>
            Choose based on the <b>cost of a missed fraud</b> vs <b>cost of a false alarm</b> for your use case.
            </p>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# MAIN APP ENTRY POINT
# =============================================================================

def main():
    # Load resources
    model, model_err = load_model()
    predict_fn, pred_warn = try_import_predict()

    # Sidebar navigation
    page = render_sidebar()

    # Surface non-blocking warnings
    if model_err:
        st.warning(f"⚠️ **Model Warning:** {model_err}")
    if pred_warn and page != "Home":
        st.info(f"ℹ️ {pred_warn}")

    # Route to selected page
    if page == "Home":
        page_home(model)
    elif page == "Single":
        page_single(model, predict_fn)
    elif page == "Batch":
        page_batch(model, predict_fn)
    elif page == "Insights":
        page_insights(model)


if __name__ == "__main__":
    main()
