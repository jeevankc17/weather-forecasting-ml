# ─────────────────────────────────────────────────────────────
# app.py  — EnvirologApp Streamlit Dashboard for node-1
# Run:  streamlit run app.py
# Requires: node1_model.pkl and node1_features.pkl in same folder
# ─────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="EnvirologApp — node-1 Forecast",
    page_icon="🌿",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}

.main-title {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #39d353;
    letter-spacing: -1px;
    margin-bottom: 0;
}

.subtitle {
    font-size: 0.9rem;
    color: #8b949e;
    margin-top: 4px;
    margin-bottom: 2rem;
}

.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    transition: border-color 0.2s;
}

.metric-card:hover { border-color: #39d353; }

.metric-label {
    font-size: 0.72rem;
    font-family: 'Space Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #8b949e;
    margin-bottom: 6px;
}

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.7rem;
    font-weight: 700;
    color: #39d353;
}

.metric-unit {
    font-size: 0.8rem;
    color: #8b949e;
    margin-top: 2px;
}

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #8b949e;
    border-bottom: 1px solid #21262d;
    padding-bottom: 8px;
    margin: 1.5rem 0 1rem;
}

.status-pill {
    display: inline-block;
    background: #1a2f1a;
    color: #39d353;
    border: 1px solid #39d353;
    border-radius: 20px;
    padding: 2px 12px;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
}

.warn-pill {
    display: inline-block;
    background: #2d1f0a;
    color: #e3b341;
    border: 1px solid #e3b341;
    border-radius: 20px;
    padding: 2px 12px;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
}

.stDataFrame { background: #161b22; border-radius: 10px; }
div[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #21262d; }
.stButton > button {
    background: #39d353;
    color: #0d1117;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 0.8rem;
    letter-spacing: 1px;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.4rem;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model        = joblib.load("node1_model.pkl")
        feature_cols = joblib.load("node1_features.pkl")
        return model, feature_cols, True
    except FileNotFoundError:
        return None, None, False


# ── Forecast function ─────────────────────────────────────────
def forecast(df, model, steps):
    targets   = ["Temperature (°C)", "Humidity (%)", "Air Quality"]
    avg_gap   = df["Timestamp"].diff().dropna().dt.total_seconds().mean()
    last_time = df["Timestamp"].iloc[-1]
    history   = df[targets].values.tolist()
    results   = []

    for step in range(steps):
        row = []
        for col_idx in range(len(targets)):
            row += [history[-1][col_idx],
                    history[-2][col_idx],
                    history[-3][col_idx],
                    history[-4][col_idx],
                    history[-5][col_idx]]

        pred      = model.predict([row])[0]
        pred_time = last_time + pd.Timedelta(seconds=avg_gap * (step + 1))
        history.append(pred.tolist())

        results.append({
            "Step":             step + 1,
            "Predicted Time":   pred_time.strftime("%Y-%m-%d %H:%M:%S"),
            "Temperature (°C)": round(float(pred[0]), 2),
            "Humidity (%)":     round(float(pred[1]), 2),
            "Air Quality":      round(float(pred[2]), 1),
        })

    return pd.DataFrame(results)


# ── Header ────────────────────────────────────────────────────
col_title, col_status = st.columns([4, 1])
with col_title:
    st.markdown('<div class="main-title">🌿 EnvirologApp</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">node-1 · Environmental Sensor Forecast</div>', unsafe_allow_html=True)
with col_status:
    model, feature_cols, model_loaded = load_model()
    st.markdown("<br>", unsafe_allow_html=True)
    if model_loaded:
        st.markdown('<span class="status-pill">● MODEL READY</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="warn-pill">⚠ NO MODEL</span>', unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload CSV", type=["csv"],
        help="Upload your node-1 sensor CSV"
    )

    steps = st.slider(
        "Forecast steps", min_value=5, max_value=100,
        value=20, step=5,
        help="How many future readings to predict"
    )

    st.markdown("---")
    run_btn = st.button("▶  RUN FORECAST")

    st.markdown("---")
    st.markdown("**Model files needed:**")
    for fname in ["node1_model.pkl", "node1_features.pkl"]:
        exists = os.path.exists(fname)
        icon   = "✅" if exists else "❌"
        st.markdown(f"{icon} `{fname}`")

    st.markdown("---")
    st.caption("Train model first using `train.py`")


# ── Main panel ────────────────────────────────────────────────
if not model_loaded:
    st.warning("⚠️ Model not found. Run `python train.py` first to generate `node1_model.pkl`.")
    st.stop()

if uploaded_file is None:
    st.info("👈 Upload your CSV file from the sidebar to get started.")
    st.stop()

# Load and validate CSV
try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Failed to read uploaded CSV: {e}")
    st.stop()

if "Timestamp" not in df.columns:
    st.error("Uploaded CSV must include a 'Timestamp' column.")
    st.stop()

if not all(c in df.columns for c in ["Temperature (°C)", "Humidity (%)", "Air Quality"]):
    st.error("Uploaded CSV must include Temperature (°C), Humidity (%), and Air Quality columns.")
    st.stop()

df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

# ── Last known readings summary ───────────────────────────────
st.markdown('<div class="section-header">Last Known Readings</div>', unsafe_allow_html=True)

last = df.iloc[-1]
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🕒 Last Timestamp</div>
        <div class="metric-value" style="font-size:1rem">{pd.to_datetime(last['Timestamp']).strftime('%H:%M:%S')}</div>
        <div class="metric-unit">{pd.to_datetime(last['Timestamp']).strftime('%Y-%m-%d')}</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🌡️ Temperature</div>
        <div class="metric-value">{round(last['Temperature (°C)'], 1)}</div>
        <div class="metric-unit">°C</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">💧 Humidity</div>
        <div class="metric-value">{round(last['Humidity (%)'], 1)}</div>
        <div class="metric-unit">%</div>
    </div>""", unsafe_allow_html=True)

with c4:
    aq    = int(last['Air Quality'])
    color = "#39d353" if aq < 100 else "#e3b341" if aq < 150 else "#f85149"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🌬️ Air Quality</div>
        <div class="metric-value" style="color:{color}">{aq}</div>
        <div class="metric-unit">AQI index</div>
    </div>""", unsafe_allow_html=True)


# ── Run forecast ──────────────────────────────────────────────
if run_btn:
    with st.spinner("Forecasting..."):
        future_df = forecast(df, model, steps)

    # ── Forecast metrics (first prediction) ───────────────────
    st.markdown('<div class="section-header">Next Predicted Reading</div>', unsafe_allow_html=True)

    first = future_df.iloc[0]
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🕒 Predicted At</div>
            <div class="metric-value" style="font-size:1rem">{first['Predicted Time'].split(' ')[1]}</div>
            <div class="metric-unit">{first['Predicted Time'].split(' ')[0]}</div>
        </div>""", unsafe_allow_html=True)

    with m2:
        delta = round(first['Temperature (°C)'] - last['Temperature (°C)'], 2)
        arrow = "▲" if delta >= 0 else "▼"
        dcolor = "#f85149" if delta >= 0 else "#39d353"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🌡️ Temperature</div>
            <div class="metric-value">{first['Temperature (°C)']}</div>
            <div class="metric-unit" style="color:{dcolor}">{arrow} {abs(delta)} from last</div>
        </div>""", unsafe_allow_html=True)

    with m3:
        delta = round(first['Humidity (%)'] - last['Humidity (%)'], 2)
        arrow = "▲" if delta >= 0 else "▼"
        dcolor = "#39d353" if delta >= 0 else "#e3b341"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">💧 Humidity</div>
            <div class="metric-value">{first['Humidity (%)']}</div>
            <div class="metric-unit" style="color:{dcolor}">{arrow} {abs(delta)} from last</div>
        </div>""", unsafe_allow_html=True)

    with m4:
        aq    = int(first['Air Quality'])
        color = "#39d353" if aq < 100 else "#e3b341" if aq < 150 else "#f85149"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🌬️ Air Quality</div>
            <div class="metric-value" style="color:{color}">{aq}</div>
            <div class="metric-unit">AQI index</div>
        </div>""", unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────
    st.markdown('<div class="section-header">Forecast Charts</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🌡️ Temperature", "💧 Humidity", "🌬️ Air Quality"])

    with tab1:
        chart_df = future_df[["Predicted Time", "Temperature (°C)"]].set_index("Predicted Time")
        st.line_chart(chart_df, color="#39d353")

    with tab2:
        chart_df = future_df[["Predicted Time", "Humidity (%)"]].set_index("Predicted Time")
        st.line_chart(chart_df, color="#58a6ff")

    with tab3:
        chart_df = future_df[["Predicted Time", "Air Quality"]].set_index("Predicted Time")
        st.line_chart(chart_df, color="#e3b341")

    # ── Full forecast table ───────────────────────────────────
    st.markdown('<div class="section-header">Full Forecast Table</div>', unsafe_allow_html=True)
    st.dataframe(future_df, use_container_width=True, hide_index=True)

    # ── Download button ───────────────────────────────────────
    csv = future_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Download Forecast CSV",
        data=csv,
        file_name=f"node1_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )