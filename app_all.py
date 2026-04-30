
import os
from datetime import datetime

import streamlit as st
from streamlit_autorefresh import st_autorefresh

from pipeline_all import load_all_models, detect, CONFIG, get_latest_file

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
st_autorefresh(interval=refresh_seconds * 1000, key="stream_refresh")

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
    margin-top: 6px;
    line-height: 1.5;
}

# .pair-card {
#     background: linear-gradient(180deg, #0d1c2d, #11263a);
#     border: 1px solid #1d3f5c;
#     border-radius: 16px;
#     padding: 14px;
#     margin-bottom: 16px;
# }

.section-gap {
    margin-top: 10px;
}

hr {
    border: none;
    border-top: 1px solid #18324a;
    margin: 14px 0;
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

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

if "last_processed_time_a" not in st.session_state:
    st.session_state.last_processed_time_a = None

if "last_processed_time_b" not in st.session_state:
    st.session_state.last_processed_time_b = None

if "last_update_message" not in st.session_state:
    st.session_state.last_update_message = "Waiting for stream input..."

# =========================
# Helpers
# =========================
def get_status_class(decision):
    if decision == "DroneType":
        return "status-green"
    elif decision == "Detected":
        return "status-orange"
    return "status-red"


def process_latest_stream_pair():
    folder_a = CONFIG["stream_folder_a"]
    folder_b = CONFIG["stream_folder_b"]

    image_a = get_latest_file(folder_a)
    image_b = get_latest_file(folder_b)

    if not image_a or not image_b:
        if not image_a and not image_b:
            st.session_state.last_update_message = "Waiting for images in both folders..."
        elif not image_a:
            st.session_state.last_update_message = "Waiting for SCD image in folder A..."
        else:
            st.session_state.last_update_message = "Waiting for wavelet image in folder B..."
        return st.session_state.latest_result

    try:
        time_a = os.path.getctime(image_a)
        time_b = os.path.getctime(image_b)
    except OSError:
        st.session_state.last_update_message = "Could not read image timestamps."
        return st.session_state.latest_result

    has_new_a = (
        st.session_state.last_processed_time_a is None
        or time_a > st.session_state.last_processed_time_a
    )
    has_new_b = (
        st.session_state.last_processed_time_b is None
        or time_b > st.session_state.last_processed_time_b
    )

    if not (has_new_a and has_new_b):
        return st.session_state.latest_result

    try:
        result = detect(
            image_path_a=image_a,
            image_path_b=image_b,
            classifier=classifier,
            yolo_model=yolo_model,
            transform=transform,
            device=device
        )

        result["stream_filename_a"] = os.path.basename(image_a)
        result["stream_filename_b"] = os.path.basename(image_b)
        result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result["stream_image_a_path"] = image_a
        result["stream_image_b_path"] = image_b
        result["timestamp_a"] = time_a
        result["timestamp_b"] = time_b

        st.session_state.last_processed_time_a = time_a
        st.session_state.last_processed_time_b = time_b
        st.session_state.latest_result = result
        st.session_state.history.append(result)

        # Keep only last 1000 results
        st.session_state.history = st.session_state.history[-1000:]

        st.session_state.last_update_message = "New pair processed successfully."
        return result

    except Exception as e:
        st.session_state.last_update_message = f"Processing failed: {e}"
        return st.session_state.latest_result


def show_pair_images(item):
    st.markdown("""
    <div class="pair-card"></div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**SCD Image / EfficientNet Input**")
        img_a = item.get("stream_image_a_path")
        if img_a and os.path.exists(img_a):
            st.image(img_a, use_container_width=True)
        else:
            st.warning("Image A not found")

    with col2:
        st.markdown("**Wavelet / YOLO Output**")
        yolo_img = item.get("saved_detection_image")
        img_b = item.get("stream_image_b_path")

        if yolo_img and os.path.exists(yolo_img):
            st.image(yolo_img, use_container_width=True)
        elif img_b and os.path.exists(img_b):
            st.image(img_b, use_container_width=True)
        else:
            st.warning("Image B not found")

    decision = item.get("final_decision", "Unknown")
    status_class = get_status_class(decision)
    status_message = item.get("status_message", decision)

    st.markdown(
        f'<div class="status-pill {status_class}">{status_message}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="small-note">
            <b>Time:</b> {item.get('timestamp', 'N/A')}<br>
            <b>Folder A:</b> {item.get('stream_filename_a', 'N/A')}<br>
            <b>Folder B:</b> {item.get('stream_filename_b', 'N/A')}<br>
        </div>
        """,
        unsafe_allow_html=True
    )


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
left_stats, center_stats, right_stats = st.columns([1.0, 2.8, 1.0])

# Left side
with left_stats:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Latest Detection</div>', unsafe_allow_html=True)

    if latest:
        detected_text = "Detected" if latest.get("final_decision") != "NOdrone" else "Not Detected"
        drone_type = latest.get("drone_type", "-")

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
        st.info(st.session_state.last_update_message)

    st.markdown(
        f"<p class='small-note'>{st.session_state.last_update_message}</p>",
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Center section
with center_stats:
    if history:
        recent_items = history[-2:][::-1]

        for idx, item in enumerate(recent_items, start=1):
            show_pair_images(item)

            if idx < len(recent_items):
                st.markdown("<hr>", unsafe_allow_html=True)

        total_runs = len(history)
        typed_runs = sum(1 for x in history if x.get("final_decision") == "DroneType")
        uncertain_runs = sum(1 for x in history if x.get("final_decision") == "Detected")
        nodrone_runs = sum(1 for x in history if x.get("final_decision") == "NOdrone")

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Runs", total_runs)
        c2.metric("Typed Runs", typed_runs)
        c3.metric("Uncertain", uncertain_runs)
        c4.metric("No-Drone", nodrone_runs)
    else:
        st.info("No image pairs available yet.")

# Right side
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
# Latest JSON
# =========================
if latest:
    st.markdown("### Latest Detection JSON")
    st.json(latest)