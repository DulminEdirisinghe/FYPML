import os
import tempfile
from collections import Counter
from datetime import datetime

import streamlit as st
import torch
import plotly.graph_objects as go

from fuse_detector import load_all_models, detect

st.set_page_config(
    page_title="Drone Detection Dashboard",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# Custom CSS
# =========================
st.markdown("""
<style>
html, body, [class*="css"] {
    background: #08111f;
    color: white;
}

.block-container {
    padding-top: 0.8rem;
    padding-bottom: 0.8rem;
    max-width: 100%;
}

.topbar {
    background: linear-gradient(90deg, #0c1a2b, #10263d);
    border: 1px solid #1c3951;
    border-radius: 16px;
    padding: 16px 22px;
    margin-bottom: 18px;
}

.topbar-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 20px;
}

.topbar-title {
    font-size: 30px;
    font-weight: 800;
    color: white;
}

.topbar-subtitle {
    font-size: 14px;
    color: #9fb5c9;
    margin-top: 4px;
}

.topbar-date {
    font-size: 14px;
    color: #c7d6e6;
    background: #0f2235;
    border: 1px solid #24425d;
    border-radius: 10px;
    padding: 10px 14px;
    min-width: 180px;
    text-align: center;
}

.panel {
    background: rgba(9, 20, 34, 0.96);
    border: 1px solid #17324a;
    border-radius: 16px;
    padding: 16px;
    margin-bottom: 14px;
    min-height: 100%;
}

.panel-title {
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 12px;
    color: #eef4fb;
}

.metric-side {
    background: linear-gradient(180deg, #0e1c2c, #10253a);
    border: 1px solid #1f415d;
    border-radius: 16px;
    padding: 14px;
    margin-bottom: 14px;
}

.metric-side-number {
    font-size: 30px;
    font-weight: 800;
    color: white;
    margin-bottom: 4px;
}

.metric-side-label {
    font-size: 14px;
    color: #a7bacd;
    line-height: 1.4;
}

.status-pill {
    display: inline-block;
    padding: 8px 16px;
    border-radius: 999px;
    font-weight: 700;
    color: white;
    margin-top: 8px;
}

.status-green {
    background: linear-gradient(90deg, #138a52, #20bf78);
}

.status-orange {
    background: linear-gradient(90deg, #c67b00, #ffb020);
}

.status-red {
    background: linear-gradient(90deg, #b42318, #ef5350);
}

.stButton > button {
    background: linear-gradient(90deg, #0d6efd, #0aa2ff);
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: 700;
    padding: 0.7rem 1.2rem;
    width: 100%;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #0b5ed7, #0891e6);
    color: white;
}

.small-note {
    font-size: 13px;
    color: #9db1c4;
}

hr {
    border: none;
    border-top: 1px solid #18324a;
    margin: 12px 0;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Model loading
# =========================
@st.cache_resource
def get_models():
    return load_all_models()

classifier, yolo_model, transform, device = get_models()

# =========================
# Session state
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# Helpers
# =========================
def make_donut_chart(title, labels, values, colors):
    fig = go.Figure(
        data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.68,
            textinfo='percent',
            textfont=dict(color='white', size=14),
            marker=dict(colors=colors, line=dict(color="#08111f", width=2))
        )]
    )
    fig.update_layout(
        title=dict(text=title, font=dict(color="white", size=18)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        margin=dict(t=50, b=20, l=20, r=20),
        legend=dict(font=dict(color='white'))
    )
    return fig


def get_status_class(decision):
    if decision == "DroneType":
        return "status-green"
    elif decision == "Detected":
        return "status-orange"
    return "status-red"


def run_detection(image_path_a, image_path_b):
    result = detect(
        image_path_a=image_path_a,
        image_path_b=image_path_b,
        classifier=classifier,
        yolo_model=yolo_model,
        transform=transform,
        device=device
    )
    st.session_state.history.append(result)
    return result

# =========================
# Header
# =========================
current_date = datetime.now().strftime("%Y-%m-%d")

st.markdown(f"""
<div class="topbar">
    <div class="topbar-row">
        <div>
            <div class="topbar-title">DRONE DETECTION DASHBOARD</div>
            <div class="topbar-subtitle">Fusion-based drone monitoring using EfficientNet and YOLO</div>
        </div>
        <div class="topbar-date">Date<br><b>{current_date}</b></div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# Upload + Run section
# =========================
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown('<div class="panel-title">Input Control</div>', unsafe_allow_html=True)

up1, up2, up3 = st.columns([1.2, 1.2, 0.8])

image_path_a = None
image_path_b = None

with up1:
    scd_file = st.file_uploader("Upload SCD Image", type=["png", "jpg", "jpeg"], key="dashboard_scd")

with up2:
    wavelet_file = st.file_uploader("Upload Wavelet Image", type=["png", "jpg", "jpeg"], key="dashboard_wavelet")

if scd_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_a:
        tmp_a.write(scd_file.read())
        image_path_a = tmp_a.name

if wavelet_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_b:
        tmp_b.write(wavelet_file.read())
        image_path_b = tmp_b.name

with up3:
    run_clicked = st.button("Run Detection")

st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Run detection
# =========================
if run_clicked:
    if not image_path_a or not image_path_b:
        st.error("Please upload both SCD and wavelet images.")
    else:
        with st.spinner("Running fusion detector..."):
            try:
                run_detection(image_path_a, image_path_b)
            except Exception as e:
                st.error(f"Error: {str(e)}")

history = st.session_state.history
latest = history[-1] if history else None

# =========================
# Dashboard main
# =========================
left_stats, center_stats, right_stats = st.columns([0.95, 2.8, 0.95])

# Left side boxes
with left_stats:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Latest Detection</div>', unsafe_allow_html=True)

    if latest:
        detected_text = "Yes" if latest.get("detected", False) else "No"
        drone_type = latest.get("drone_type", "N/A")
        num_detections = latest.get("num_detections", 0)

        st.markdown(f"""
        <div class="metric-side">
            <div class="metric-side-number">{detected_text}</div>
            <div class="metric-side-label">Detected or Not</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-side">
            <div class="metric-side-number">{drone_type}</div>
            <div class="metric-side-label">Drone Type</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-side">
            <div class="metric-side-number">{num_detections}</div>
            <div class="metric-side-label">Number of Detections</div>
        </div>
        """, unsafe_allow_html=True)

        decision = latest.get("final_decision", "Unknown")
        status_class = get_status_class(decision)
        status_message = latest.get("status_message", decision)

        st.markdown(
            f'<div class="status-pill {status_class}">{status_message}</div>',
            unsafe_allow_html=True
        )
    else:
        st.info("No detection run yet.")

    st.markdown('</div>', unsafe_allow_html=True)

# Center section
with center_stats:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Detection Statistics</div>', unsafe_allow_html=True)

    if history:
        decision_counts = Counter([x.get("final_decision", "Unknown") for x in history])

        fig = make_donut_chart(
            "Final Decision Distribution",
            list(decision_counts.keys()),
            list(decision_counts.values()),
            ["#1fbf75", "#ffb020", "#ef5350", "#64748b"]
        )
        st.plotly_chart(fig, use_container_width=True)

        total_runs = len(history)
        total_detections = sum(int(x.get("num_detections", 0)) for x in history)
        typed_runs = sum(1 for x in history if x.get("final_decision") == "DroneType")

        row1, row2, row3 = st.columns(3)
        row1.metric("Total Runs", total_runs)
        row2.metric("Total Detections", total_detections)
        row3.metric("Type Classified Runs", typed_runs)
    else:
        st.info("No statistics available yet.")

    st.markdown('</div>', unsafe_allow_html=True)

# Right side boxes
with right_stats:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Fusion Outputs</div>', unsafe_allow_html=True)

    if latest:
        p4 = float(latest.get("P4_efficientnet_drone_prob", 0.0))
        f_val = float(latest.get("F_yolo_max_prob", 0.0))
        g_val = float(latest.get("G_fused_prob", 0.0))

        st.markdown(f"""
        <div class="metric-side">
            <div class="metric-side-number">{p4:.4f}</div>
            <div class="metric-side-label">EfficientNet Probability (P4)</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-side">
            <div class="metric-side-number">{f_val:.4f}</div>
            <div class="metric-side-label">YOLO Max Confidence (F)</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-side">
            <div class="metric-side-number">{g_val:.4f}</div>
            <div class="metric-side-label">Fused Probability (G)</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No fusion outputs available yet.")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Optional JSON
# =========================
if latest:
    st.markdown("### Latest Detection JSON")
    st.json(latest)