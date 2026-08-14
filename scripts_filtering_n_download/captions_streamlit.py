import json
import math
import tempfile
import subprocess

from pathlib import Path

import streamlit as st


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="How2 Caption Viewer",
    layout="wide"
)

st.title("How2 Caption Segmentation Viewer")


# ============================================================
# DATA
# ============================================================

DATA_ROOT = Path("/mnt/private-share/speechDatabases/how2")

JSONL_FILE = "gabi_scripts/how2_dev5_segmented-captions100.jsonl"

@st.cache_data
def load_data(path):

    items = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            if line.strip():
                items.append(json.loads(line))

    return items


data = load_data(JSONL_FILE)


# ============================================================
# VIDEO CUTTER
# ============================================================

CACHE_DIR = Path("/tmp/how2_caption_cache")
CACHE_DIR.mkdir(exist_ok=True, parents=True)


def make_clip(video_path, start, end, idx):

    video_path = Path(video_path)

    stem = video_path.stem

    out_path = CACHE_DIR / (
        f"{stem}_{idx}_{start:.2f}_{end:.2f}.mp4"
    )

    if out_path.exists():
        return str(out_path)

    duration = max(end - start, 0.05)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start),
        "-to", str(end),
        "-i", str(video_path),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-strict", "experimental",
        str(out_path)
    ]

    subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    return str(out_path)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Controls")

page_size = st.sidebar.slider(
    "Items per page",
    1,
    20,
    4
)


# ============================================================
# PAGINATION
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = 0

total_pages = math.ceil(len(data) / page_size)

st.session_state.page = max(
    0,
    min(st.session_state.page, total_pages - 1)
)


# ============================================================
# NAVIGATION
# ============================================================

c1, c2, c3, c4, c5 = st.columns([1, 1, 2, 1, 1])

with c1:
    if st.button("⏮ First"):
        st.session_state.page = 0

with c2:
    if st.button("◀ Prev"):
        st.session_state.page = max(
            0,
            st.session_state.page - 1
        )

with c3:
    st.markdown(
        f"""
        <div style='text-align:center;font-size:18px'>
        Page {st.session_state.page + 1} / {total_pages}
        </div>
        """,
        unsafe_allow_html=True
    )

with c4:
    if st.button("Next ▶"):
        st.session_state.page = min(
            total_pages - 1,
            st.session_state.page + 1
        )

with c5:
    if st.button("Last ⏭"):
        st.session_state.page = total_pages - 1


# ============================================================
# PAGE ITEMS
# ============================================================

start_idx = st.session_state.page * page_size
end_idx = start_idx + page_size

page_items = data[start_idx:end_idx]


# ============================================================
# DISPLAY
# ============================================================

for item_idx, item in enumerate(page_items):

    st.markdown("---")

    video_path = item["video_segment"]
    transcript = item.get("text", "")
    captions = item.get("captions", [])

    left, middle, right = st.columns([1.2, 0.15, 1.8])

    # ========================================================
    # ORIGINAL VIDEO
    # ========================================================

    with left:

        st.markdown("### Original")

        if Path(video_path).exists():
            st.video(video_path)
        else:
            st.error("Missing video")

        st.markdown(
            f"""
            <div style="
                font-size:14px;
                line-height:1.5;
                margin-top:8px;
                padding:10px;
                border-radius:8px;
                background:#11111110;
            ">
            {transcript}
            </div>
            """,
            unsafe_allow_html=True
        )

    # ========================================================
    # ARROW
    # ========================================================

    with middle:

        st.markdown(
            """
            <div style="
                display:flex;
                justify-content:center;
                align-items:center;
                height:300px;
                font-size:42px;
            ">
            ➜
            </div>
            """,
            unsafe_allow_html=True
        )

    # ========================================================
    # SEGMENTED CAPTIONS
    # ========================================================

    with right:

        st.markdown("### Caption Segments")

        for cap_idx, cap in enumerate(captions):

            start, end, caption = cap

            clip_path = make_clip(
                video_path,
                start,
                end,
                cap_idx
            )

            st.video(clip_path)

            st.markdown(
                f"""
                <div style="
                    font-size:13px;
                    line-height:1.4;
                    margin-top:4px;
                    margin-bottom:18px;
                    padding:8px;
                    border-radius:8px;
                    background:#11111110;
                ">
                <b>{start:.2f}s → {end:.2f}s</b><br>
                {caption}
                </div>
                """,
                unsafe_allow_html=True
            )

    # ========================================================
    # PATHS
    # ========================================================

    with st.expander("paths"):

        st.write(video_path)
        st.write(item.get("audio_segment"))
