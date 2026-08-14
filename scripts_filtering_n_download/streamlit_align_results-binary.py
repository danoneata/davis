import os
import streamlit as st
import json
from pathlib import Path
import math

st.set_page_config(page_title="Alignment Viewer", layout="wide")

st.title("Video-Audio Alignment Viewer")

DATA_ROOT = "/mnt/private-share/speechDatabases/how2/processed_how2/"
JSONL_PATH = "how2_val-aligned.json"
PAGE_SIZE = 100


@st.cache_data
def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def matches_query(entry, q):
    if not q:
        return True
    q = q.lower()
    return (
        q in entry.get("text", "").lower()
        or q in entry.get("scene_caption", "").lower()
    )


def matches_alignment(entry, selected_labels):
    """
    Returns True if:
    - nothing selected
    - both selected
    - entry label matches selected filter
    """
    label = entry.get("alignment_label", "").lower().strip()

    if len(selected_labels) == 0:
        return True

    if len(selected_labels) == 2:
        return True

    return label in selected_labels


if not Path(JSONL_PATH).exists():
    st.error(f"File not found: {JSONL_PATH}")
    st.stop()

entries = load_jsonl(JSONL_PATH)

# --------------------------------
# GLOBAL STATS
# --------------------------------
aligned_count = sum(
    1
    for e in entries
    if e.get("alignment_label", "").lower().strip() == "aligned"
)

not_aligned_count = sum(
    1
    for e in entries
    if e.get("alignment_label", "").lower().strip() == "not_aligned"
)

st.subheader("Dataset Statistics")

col1, col2, col3 = st.columns(3)

col1.metric("Total Samples", len(entries))
col2.metric("Aligned", aligned_count)
col3.metric("Not Aligned", not_aligned_count)

st.divider()

# --------------------------------
# SEARCH + FILTERS
# --------------------------------
query = st.text_input("Search (text or caption):")

selected_labels = st.multiselect(
    "Filter by alignment",
    options=["aligned", "not_aligned"],
    default=[],
)

filtered = [
    e
    for e in entries
    if matches_query(e, query)
    and matches_alignment(e, selected_labels)
]

# --------------------------------
# PAGINATION
# --------------------------------
total_pages = max(1, math.ceil(len(filtered) / PAGE_SIZE))

page = st.number_input(
    "Page",
    min_value=1,
    max_value=total_pages,
    value=1,
)

start = (page - 1) * PAGE_SIZE
end = start + PAGE_SIZE
page_entries = filtered[start:end]

st.write(
    f"{len(page_entries)} / {len(filtered)} samples "
    f"(Page {page}/{total_pages})"
)

# --------------------------------
# RENDER
# --------------------------------
for i, entry in enumerate(page_entries, start=start + 1):
    st.divider()
    st.markdown(f"### Sample {i}")

    col_video, col_text = st.columns([1, 3])

    with col_video:
        video_path = os.path.join(DATA_ROOT,entry.get("video_path"))

        if video_path and Path(video_path).exists():
            st.video(video_path)
        else:
            st.warning("Missing video")

    with col_text:
        st.markdown("**Text**")
        st.write(entry.get("text", ""))

        st.markdown("**Scene Caption**")
        st.write(entry.get("video_caption", ""))

        label = entry.get("alignment_label", "").lower().strip()

        st.markdown("**Alignment**")

        if label == "aligned":
            st.success("✔ ALIGNED")
        else:
            st.error("✘ NOT ALIGNED")
