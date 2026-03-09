from pathlib import Path
from toolz import partition_all

import streamlit as st


st.set_page_config(layout="wide")


BASE_DIR = Path("/data/how2")
NUM_COLS = 4


def show_videos_from_dir(video_dir: Path) -> None:
    num_to_show = 6 * NUM_COLS
    files = list(video_dir.glob("*.mp4"))[:num_to_show]
    for row_files in partition_all(NUM_COLS, files):
        cols = st.columns(NUM_COLS)
        for col, video_file in zip(cols, row_files):
            col.video(str(video_file))


def show_msr_vtt():
    msr_vtt_dir = BASE_DIR / "msr-vtt" / "TrainValVideo"
    show_videos_from_dir(msr_vtt_dir)


def show_youcook2():
    youcook2_dir = BASE_DIR / "youcook2" / "test"
    show_videos_from_dir(youcook2_dir)


SHOW_FUNCS = {
    "MSR-VTT": show_msr_vtt,
    "YouCook2": show_youcook2,
}


if __name__ == "__main__":
    with st.sidebar:
        dataset = st.selectbox("Dataset", list(SHOW_FUNCS.keys()))

    show_func = SHOW_FUNCS[dataset]
    show_func()
