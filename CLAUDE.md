# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

DAVIS: Data-efficient learning of foundation audio-visual models. Research project on visually-grounded speech models for video, funded by UEFISCDI (project PN-IV-P2-2.1-TE-2023-1632). `docs/index.html` is the public-facing project page.

## Setup

```bash
uv pip install -e .
```

## Running scripts

The Streamlit data viewer:

```bash
streamlit run davis/scripts/show_data.py
```

The audio-presence checker (Click CLI, dataset is `msr-vtt` or `youcook2`):

```bash
python -m davis.scripts.find_videos_with_audio <dataset>
```

## Architecture

- `davis/scripts/show_data.py` is the canonical source for dataset layout — `VIDEO_DIRS` and `SELECTED_FILES` are imported by other scripts. When adding a new dataset, register it here first.
- Video data lives under `/data/how2/<dataset>/...` (hard-coded `BASE_DIR` in `show_data.py`). Scripts assume this path exists on the host.
- `find_videos_with_audio.py` shells out to `ffprobe`, so `ffmpeg`/`ffprobe` must be on `PATH`.
- ID lists for `find_videos_with_audio.py` are read from `/data/how2/avlnet-data/msrvtt_val_ids.txt` and `/data/how2/avlnet-data/youcook_val_ids_common.txt` (hard-coded absolute paths, so they need the same host state).
