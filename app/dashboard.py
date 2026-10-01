"""Decision-intelligence Streamlit dashboard with main-page centered controls and multi-model support for all targets."""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.anomaly import detect_anomalies
from src.config import DATA_PATH, EVAL_DIR, MODEL_DIR, TARGETS
from src.data_validation import validate_dataset
from src.decision_engine import DecisionEngine
from src.model_registry import MODEL_REGISTRY

st.set_page_config(page_title="Demand Intelligence", page_icon="📈", layout="wide")

TARGET_DISPLAY_NAMES = {
    "product_demand": "Product Demand",
    "inventory_requirement": "Inventory Requirement",
    "sales_revenue": "Sales Revenue",
    "staffing_requirement": "Staffing Requirement",
    "resource_utilization": "Resource Utilization",
}

AVAILABLE_MODELS = [
    "XGBoost",
    "SARIMA",
    "Temporal Attention",
    "LSTM",
    "Seasonal Naive",
    "Drift",
    "Auto (Production Candidate)",
    "Prophet",
]
OTHER_TARGET_MODELS = ["XGBoost"]

@st.cache_data
def load_data():
    path = DATA_PATH
    if not path.exists():
        from data.generator import SyntheticDemandGenerator
        SyntheticDemandGenerator().generate(save_path=path)
    df = pd.read_csv(path, parse_dates=["date"])
    report = validate_dataset(df)
    if not report.passed:
        raise RuntimeError(f"Dataset validation failed: {report}")
    return df, report


@st.cache_resource
def load_model(target: str, selected_model_name: str = "XGBoost", df_history: pd.DataFrame | None = None):
    model_to_load = selected_model_name

    # If Auto is chosen, read the benchmark winner if available
    if model_to_load == "Auto (Production Candidate)":
        if target != "product_demand":
                    model_to_load = "XGBoost"

        else :
            metrics_path = EVAL_DIR / "model_metrics.csv"
            if metrics_path.exists():
                m = pd.read_csv(metrics_path)
                ok = m[m["Production Candidate"] == True]
                model_to_load = str(ok.iloc[0]["Model"]) if not ok.empty else "XGBoost"
            else :
                model_to_load = "XGBoost"



    model_cls = MODEL_REGISTRY.get(model_to_load)
    if model_cls is None:
        model_cls = MODEL_REGISTRY["XGBoost"]
        model_to_load = "XGBoost"

    # 1. Check for pre-saved target-specific artifact
    ext = ".pt" if model_to_load in {"LSTM", "Temporal Attention"} else ".joblib"
    artifact_path = MODEL_DIR / f"{model_to_load.lower().replace(' ', '_')}_{target}{ext}"

    if artifact_path.exists():
        return model_cls.load(artifact_path)

    # 2. Check for general target artifact (XGBoost fallback)
    target_xgb_path = MODEL_DIR / f"xgboost_{target}.joblib"
    if target_xgb_path.exists() and model_to_load == "XGBoost":
        return MODEL_REGISTRY["XGBoost"].load(target_xgb_path)

    # 3. Dynamic fitting fallback if candidate is selected for non-demand target on the fly
    if df_history is not None:
        model = model_cls()
        model.fit(df_history, target, tune=False)
        return model

    return None


df, report = load_data()

st.title("📈 Intelligent Demand Forecasting & Resource Planning")
st.caption("2,000 daily observations • offline model training • probabilistic planning • what-if simulation")

# Header KPI Metrics
c = st.columns(4)
c[0].metric("Rows", f"{report.rows:,}")
c[1].metric("Date range", f"{report.date_start}→{report.date_end}")
c[2].metric("Missing values", report.missing_values)
c[3].metric("Validation", "PASSED" if report.passed else "FAILED")

st.markdown("---")

# Main Page Centered Controls
st.subheader("🎯 Forecast Setup & Controls")
ctrl_cols = st.columns([1.5, 1.5, 2.0])

with ctrl_cols[0]:
    target = st.selectbox(
        "Forecast Metric (Target)",
        list(TARGET_DISPLAY_NAMES.keys()),
        format_func=lambda x: TARGET_DISPLAY_NAMES[x],
        index=0,
    )

with ctrl_cols[1]:
    model_options = AVAILABLE_MODELS if target == "product_demand" else OTHER_TARGET_MODELS
    selected_model_option = st.selectbox(
        "Forecasting Model Architecture",
        AVAILABLE_MODELS,
        index=0,
    )

with ctrl_cols[2]:
    horizon = st.slider("Forecast Horizon (Days)", min_value=7, max_value=90, value=30)

st.markdown("---")

# Model Loading & Forecast Execution
model = load_model(target, selected_model_option, df_history=df)
if model is None:
    st.error("No trained artifact found. Run `python scripts/train_models.py` from the project root, then refresh.")
    st.stop()

result = model.forecast(df, horizon, confidence=0.95)

# Visual Dashboard Tabs
tabs = st.tabs(["Forecast", "Benchmark", "Business Impact", "What-if", "Anomalies", "Data Quality"])

with tabs[0]:
    fig = go.Figure()
    recent = df.tail(180)
    fig.add_trace(go.Scatter(x=recent["date"], y=recent[target], name="Historical", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=result.dates, y=result.upper, name="Upper", line=dict(width=0), showlegend=False))
    fig.add_trace(
        go.Scatter(
            x=result.dates,
            y=result.lower,
            name="95% Prediction Interval",
            fill="tonexty",
            fillcolor="rgba(31, 119, 180, 0.2)",
            line=dict(width=0),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result.dates,
            y=result.point,
            name="Forecast",
            mode="lines+markers",
            line=dict(color="#ff7f0e", width=2),
        )
    )
    fig.update_layout(
        height=520,
        title=f"{TARGET_DISPLAY_NAMES[target]} Forecast using {model.name}",
        xaxis_title="Date",
        yaxis_title=TARGET_DISPLAY_NAMES[target],
    )
    st.plotly_chart(fig, use_container_width=True)

    m_cols = st.columns(3)
    m_cols[0].metric("1-Day Forecast", f"{result.point[0]:,.1f}")
    m_cols[1].metric("Cumulative Forecast", f"{np.sum(result.point):,.0f}")
    m_cols[2].metric("Prediction Interval (Day 1)", f"{result.lower[0]:,.1f} – {result.upper[0]:,.1f}")

with tabs[1]:
    metrics_file = EVAL_DIR / "model_metrics.csv"
    if metrics_file.exists():
        m_df = pd.read_csv(metrics_file)
        st.dataframe(m_df, use_container_width=True, hide_index=True)
        if "Production Candidate" in m_df.columns and m_df["Production Candidate"].any():
            winner = m_df.loc[m_df["Production Candidate"].eq(True), "Model"].iloc[0]
            st.success(f"Benchmark Production Candidate: **{winner}**")

        if hasattr(model, "model") and hasattr(model.model, "feature_importances_"):
            imp = pd.DataFrame(
                {"Feature": model.prep.feature_columns, "Importance": model.model.feature_importances_}
            ).sort_values("Importance", ascending=False).head(10)
            st.subheader(f"{model.name} Top 10 Feature Importance")
            st.bar_chart(imp.set_index("Feature"))
    else:
        st.info("Run `python scripts/evaluate_models.py` to create the rolling-origin benchmark.")

with tabs[2]:
    st.subheader("Operational Recommendation")
    s = st.columns(4)
    with s[0]:
        service = st.slider("Service level (%)", 80, 99, 95)
    with s[1]:
        lead = st.slider("Lead time (days)", 1, 30, 7)
    with s[2]:
        buffer = st.slider("Labor buffer (%)", 0, 40, 10)
    with s[3]:
        capacity = st.number_input("Available capacity", 100, 5000, 1000)

    eng = DecisionEngine()
    rec = eng.recommend(
        point=result.point,
        lower=result.lower,
        upper=result.upper,
        lead_time_days=lead,
        service_level=service / 100.0,
        labor_buffer=buffer / 100.0,
        capacity=float(capacity),
    )

    k = st.columns(5)
    k[0].metric("Safety Stock", f"{rec.safety_stock:,.0f}")
    k[1].metric("Recommended Inventory", f"{rec.recommended_inventory:,.0f}")
    k[2].metric("Required Staff (Daily)", rec.required_staff)
    k[3].metric("Additional Staff Needed", rec.additional_staff)
    k[4].metric("Capacity Risk Tier", rec.risk)
    st.info(rec.action)

with tabs[3]:
    st.subheader("What-If Scenario Simulation (Model-Driven)")
    a = st.columns(4)
    with a[0]:
        growth = st.slider("Demand growth (%)", -30, 30, 0)
    with a[1]:
        price_change = st.slider("Price change (%)", -20, 20, 0)
    with a[2]:
        promo = st.checkbox("Promotion Active", False)
    with a[3]:
        current_staff = st.number_input("Current staff", 1, 500, 35)

    sim = DecisionEngine().what_if(
        base={"demand": float(result.point[0]), "price": float(df["price"].iloc[-1])},
        demand_growth=growth / 100.0,
        price_change=price_change / 100.0,
        promotion=promo,
        lead_time_days=lead,
        service_level=service / 100.0,
        labor_buffer=buffer / 100.0,
        capacity=float(capacity),
        current_staff=int(current_staff),
        model=model,
        history_df=df,
    )
    st.dataframe(pd.DataFrame([sim]), use_container_width=True, hide_index=True)

with tabs[4]:
    st.subheader(f"Historical Anomalies ({TARGET_DISPLAY_NAMES[target]})")
    an = detect_anomalies(df, target=target)
    an_count = int(an["anomaly"].sum())
    st.metric("Detected Historical Anomalies", an_count)
    if an_count > 0:
        st.dataframe(
            an[an["anomaly"]][["date", target, "anomaly_score"]].tail(30),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.write("No anomalies detected for the selected metric.")

with tabs[5]:
    st.subheader("Data Contract & Quality Validation")
    st.json(report.__dict__)
    st.dataframe(df.tail(50), use_container_width=True, hide_index=True)
