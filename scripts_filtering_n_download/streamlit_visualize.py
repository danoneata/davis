import json
from pathlib import Path
import math
import os 

import streamlit as st

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="How2 Compact Viewer",
    layout="wide"
)

st.title("How2 Dataset Viewer (Paginated)")

# ============================================================
# DATA
# ============================================================

DATA_ROOT = Path("/mnt/private-share/speechDatabases/how2/processed_how2")
JSONL_FILE = DATA_ROOT / "how2_val.json"


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
# SIDEBAR
# ============================================================

st.sidebar.header("Controls")

page_size = st.sidebar.slider("Items per page", 3, 60, 12)

# ============================================================
# PAGINATION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = 0

total_pages = math.ceil(len(data) / page_size)

# clamp page
st.session_state.page = max(0, min(st.session_state.page, total_pages - 1))

# ============================================================
# PAGE NAVIGATION
# ============================================================

col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])

with col1:
    if st.button("⏮ First"):
        st.session_state.page = 0

with col2:
    if st.button("◀ Prev"):
        st.session_state.page = max(0, st.session_state.page - 1)

with col3:
    st.markdown(
        f"<div style='text-align:center; font-size:16px;'>"
        f"Page {st.session_state.page + 1} / {total_pages}"
        f"</div>",
        unsafe_allow_html=True
    )

with col4:
    if st.button("Next ▶"):
        st.session_state.page = min(total_pages - 1, st.session_state.page + 1)

with col5:
    if st.button("Last ⏭"):
        st.session_state.page = total_pages - 1

# ============================================================
# SLICE DATA
# ============================================================

start_idx = st.session_state.page * page_size
end_idx = start_idx + page_size

page_items = data[start_idx:end_idx]

st.write(f"Showing {len(page_items)} items")

# ============================================================
# GRID DISPLAY
# ============================================================

cols_per_row = 3

rows = [
    page_items[i:i + cols_per_row]
    for i in range(0, len(page_items), cols_per_row)
]

for row in rows:

    cols = st.columns(cols_per_row)

    for col, item in zip(cols, row):

        with col:

            video_path = os.path.join(DATA_ROOT,item["video_path"])
            text = item.get("text", "")

            st.markdown("###")

            # ----------------------------
            # VIDEO
            # ----------------------------

            if Path(video_path).exists():
                st.video(video_path)
            else:
                st.error("Missing video")

            # ----------------------------
            # TEXT
            # ----------------------------

            st.markdown(
                f"""
                <div style="
                    font-size: 13px;
                    line-height: 1.3;
                    margin-top: 6px;
                ">
                {text}
                </div>
                """,
                unsafe_allow_html=True
            )
            # ----------------------------
            # AUDIO
            # ----------------------------

            audio_path = os.path.join(DATA_ROOT, item["audio_path"])

            if Path(audio_path).exists():
                st.audio(audio_path)
            else:
                st.warning("Missing audio")

            # ----------------------------
            # PATHS
            # ----------------------------

            with st.expander("paths"):
                st.write(video_path)
                st.write(item.get("audio_path"))