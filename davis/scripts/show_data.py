from pathlib import Path
from toolz import partition_all

import streamlit as st


st.set_page_config(layout="wide")


# TODO: Use a more descriptive name for the base directory.
BASE_DIR = Path("/data/how2")
NUM_COLS = 4


VIDEO_DIRS = {
    "msr-vtt": BASE_DIR / "msr-vtt" / "TestVideo",
    "youcook2": BASE_DIR / "youcook2" / "val",
    "cross-task": BASE_DIR / "cross-task" / "raw_videos",
}

SELECTED_FILES = {
    "msr-vtt": ["video{}.mp4".format(i) for i in [8790, 8819, 8836, 8867, 8903, 8915]],
    "youcook2": ["hs2h7nb5PHQ_3.mp4", "hs2h7nb5PHQ_4.mp4"],
}


def show_videos(dataset: str) -> None:
    video_dir = VIDEO_DIRS[dataset]
    num_to_show = 6 * NUM_COLS
    files = list(video_dir.glob("*.mp4"))[10 :num_to_show + 10]
    # files = SELECTED_FILES[dataset]
    # files = [video_dir / file for file in files]
    for row_files in partition_all(NUM_COLS, files):
        cols = st.columns(NUM_COLS)
        for col, video_file in zip(cols, row_files):
            col.video(str(video_file))


if __name__ == "__main__":
    with st.sidebar:
        dataset = st.selectbox("Dataset", list(VIDEO_DIRS.keys()))
    show_videos(dataset)
