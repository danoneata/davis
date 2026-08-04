"""Streamlit viewer for the How2 train alignment scores.

Shows each segment's video together with its ASR transcript, video caption and the
two alignment scores, sorted by whichever score you pick.

    uv run streamlit run show_alignment_scores.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from toolz import partition_all

sns.set_theme(style="whitegrid", font_scale=0.7)

SCORES_PATH = (
    Path(__file__).parent / "data" / "filelists" / "how2-train-alignment-scores.jsonl"
)
# The JSON records point at `how2_{train,...}-{video,audio}-segments/`; the files
# actually live under these roots, so we keep only the basename and re-root it.
SEGMENTS_ROOT = Path("/data/how2/clean_code/how2/segments/train")
AUDIOS_ROOT = Path("/data/how2/clean_code/how2/audios/train")

SCORES = ["lm_score", "denseav_score"]
NUM_COLS = 3
PAGE_SIZE = 12


@st.cache_data(show_spinner="Loading alignment scores…")
def load_scores(path: str) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    df["segment_id"] = df["video_path"].map(lambda p: Path(p).stem)
    return df


def video_path(segment_id: str) -> Path:
    return SEGMENTS_ROOT / f"{segment_id}.mp4"


def audio_path(segment_id: str) -> Path:
    return AUDIOS_ROOT / f"{segment_id}.wav"


def plot_distribution(df: pd.DataFrame, selected: pd.DataFrame, score: str):
    """Histogram of `score` over all segments, with the current selection on top."""
    fig, ax = plt.subplots(figsize=(3.2, 1.6))
    # lm_score is an integer grade, so give it one bar per value. For the continuous
    # score we pass explicit edges computed over *all* segments: `bins=60` is a count,
    # so the selection would otherwise get its own 60 bins over its own narrower range
    # and come out misaligned with the grey bars behind it.
    if score == "lm_score":
        kwargs = {"discrete": True}
    else:
        kwargs = {"bins": np.histogram_bin_edges(df[score], bins=60)}
    sns.histplot(df[score], color="0.8", edgecolor=None, ax=ax, **kwargs)
    if len(selected) < len(df):
        sns.histplot(selected[score], color="tab:blue", edgecolor=None, ax=ax, **kwargs)
    ax.set_xlabel(score)
    ax.set_ylabel("")
    ax.set_yticks([])
    fig.tight_layout(pad=0.2)
    return fig


def render_card(container, row: pd.Series) -> None:
    path = video_path(row["segment_id"])
    container.markdown(
        "segment: `{}` · lm_score: **{}** · denseav_score: **{:.3f}**".format(
            row["segment_id"],
            row["lm_score"],
            row["denseav_score"],
        )
    )
    container.video(str(path))

    container.markdown(f"**ASR:** {row['text']}")
    container.markdown(f"**Caption:** {row['video_caption']}")


def main() -> None:
    st.set_page_config(layout="wide")
    st.title("How2 train · alignment scores")

    df = load_scores(str(SCORES_PATH))

    with st.sidebar:
        sort_by = st.selectbox("Sort by", SCORES)
        ascending = st.radio("Order", ["descending", "ascending"]) == "ascending"

        st.divider()
        st.caption("Filters")
        lm_min, lm_max = int(df["lm_score"].min()), int(df["lm_score"].max())
        lm_range = st.slider("lm_score", lm_min, lm_max, (lm_min, lm_max))
        dn_min, dn_max = float(df["denseav_score"].min()), float(
            df["denseav_score"].max()
        )
        dn_range = st.slider("denseav_score", dn_min, dn_max, (dn_min, dn_max))
        query = st.text_input("Text search (ASR or caption)")

    selected = df[
        df["lm_score"].between(*lm_range) & df["denseav_score"].between(*dn_range)
    ]
    if query:
        pattern = query.strip()
        selected = selected[
            selected["text"].str.contains(pattern, case=False, regex=False)
            | selected["video_caption"].str.contains(pattern, case=False, regex=False)
        ]
    # `kind="stable"` keeps ties in file order, so paging is reproducible.
    selected = selected.sort_values(sort_by, ascending=ascending, kind="stable")

    with st.sidebar:
        st.divider()
        st.caption("Distributions (grey: all, blue: selected)")
        for score in SCORES:
            fig = plot_distribution(df, selected, score)
            st.pyplot(fig)
            plt.close(fig)

        st.divider()
        page_size = st.number_input("Segments per page", 1, 48, PAGE_SIZE)

    num_pages = max(1, (len(selected) + page_size - 1) // page_size)
    with st.sidebar:
        page = st.number_input("Page", 1, num_pages, 1)

    st.markdown(
        f"{len(selected):,} of {len(df):,} segments · "
        f"sorted by `{sort_by}` ({'ascending' if ascending else 'descending'}) · "
        f"page {page} of {num_pages:,}"
    )

    start = (page - 1) * page_size
    rows = selected.iloc[start : start + page_size]
    for group in partition_all(NUM_COLS, rows.iterrows()):
        cols = st.columns(NUM_COLS)
        for col, (_, row) in zip(cols, group):
            render_card(col, row)
        st.divider()


if __name__ == "__main__":
    main()
