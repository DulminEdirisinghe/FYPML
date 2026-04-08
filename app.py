import os
from collections import Counter
from datetime import datetime

import streamlit as st
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

from pipeline import load_all_models, detect, CONFIG, get_latest_file

st.set_page_config(
    page_title="Drone Detection Dashboard",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# Auto refresh
# =========================
refresh_seconds = int(CONFIG.get("poll_interval", 2))
st_autorefresh(
    interval=refresh_seconds * 1000,
    key="stream_refresh"
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
    word-break: break-word;
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

if "last_processed_file" not in st.session_state:
    st.session_state.last_processed_file = None

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

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


def process_latest_stream_pair():
    image_a = get_latest_file(CONFIG['stream_folder_a'])
    if not image_a:
        return st.session_state.latest_result

    filename = os.path.basename(image_a)
    image_b = os.path.join(CONFIG['stream_folder_b'], filename)

    if not os.path.exists(image_b):
        return st.session_state.latest_result

    if st.session_state.last_processed_file == filename:
        return st.session_state.latest_result

    result = detect(
        image_path_a=image_a,
        image_path_b=image_b,
        classifier=classifier,
        yolo_model=yolo_model,
        transform=transform,
        device=device
    )

    result["stream_filename"] = filename
    result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result["stream_image_a_path"] = image_a
    result["stream_image_b_path"] = image_b

    st.session_state.last_processed_file = filename
    st.session_state.latest_result = result
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
            <div class="topbar-subtitle">Real-time fusion monitoring using EfficientNet and YOLO</div>
        </div>
        <div class="topbar-date">Date<br><b>{current_date}</b></div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# Process latest pair
# =========================
latest = process_latest_stream_pair()
history = st.session_state.history

# =========================
# Dashboard main
# =========================
left_stats, center_stats, right_stats = st.columns([0.95, 2.8, 0.95])

# Left side boxes
with left_stats:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Latest Detection</div>', unsafe_allow_html=True)

    if latest:
        detected_text = "Yes" if latest.get("final_decision") != "NOdrone" else "No"
        drone_type = latest.get("drone_type") or "-"

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

        decision = latest.get("final_decision", "Unknown")
        status_class = get_status_class(decision)
        status_message = latest.get("status_message", decision)

        st.markdown(
            f'<div class="status-pill {status_class}">{status_message}</div>',
            unsafe_allow_html=True
        )
    else:
        st.info("Waiting for stream input...")

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
        typed_runs = sum(1 for x in history if x.get("final_decision") == "DroneType")
        nodrone_runs = sum(1 for x in history if x.get("final_decision") == "NOdrone")

        row1, row2, row3 = st.columns(3)
        row1.metric("Total Runs", total_runs)
        row2.metric("Type Classified Runs", typed_runs)
        row3.metric("No-Drone Runs", nodrone_runs)

        if latest:
            st.markdown("#### Latest Stream Frame")
            st.write(f"**File:** {latest.get('stream_filename', 'N/A')}")
            st.write(f"**Timestamp:** {latest.get('timestamp', 'N/A')}")
    else:
        st.info("No statistics available yet.")

    st.markdown('</div>', unsafe_allow_html=True)

# Right side boxes
with right_stats:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Fusion Outputs</div>', unsafe_allow_html=True)

    if latest:
        p4 = float(latest.get("P4", 0.0))
        f_val = float(latest.get("F", 0.0))
        g_val = float(latest.get("G", 0.0))

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
# Latest Detection Images
# =========================
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown('<div class="panel-title">Latest Detection Images</div>', unsafe_allow_html=True)

if latest:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**EfficientNet Input( SCD Image )**")
        image_a_path = latest.get("stream_image_a_path")

        if image_a_path and os.path.exists(image_a_path):
            st.image(image_a_path, use_container_width=True)
        else:
            st.warning("SCD image not found")

    with col2:
        st.markdown("**YOLO Detection Output (Wavelet Image)**")
        yolo_img_path = latest.get("saved_detection_image")

        if yolo_img_path and os.path.exists(yolo_img_path):
            st.image(yolo_img_path, use_container_width=True)
        else:
            image_b_path = latest.get("stream_image_b_path")
            if image_b_path and os.path.exists(image_b_path):
                st.image(image_b_path, use_container_width=True)
            else:
                st.warning("YOLO image not found")
else:
    st.info("Waiting for first detection...")

st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Optional JSON
# =========================
if latest:
    st.markdown("### Latest Detection JSON")
    st.json(latest)