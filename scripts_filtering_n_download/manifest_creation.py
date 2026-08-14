# create_how2_jsonl.py

import json
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

DATA_ROOT = Path("/mnt/private-share/speechDatabases/how2")

VIDEO_SEGMENTS_DIR = DATA_ROOT / "300h-subset-video-segments"
AUDIO_MONO_DIR = DATA_ROOT / "300h-subset-audio-mono"

TEXT_FILE = (
    DATA_ROOT
    / "300h-subset-text"
    / "data"
    / "dev5"
    / "text.id.en"
)

SEGMENT_FILE = (
    DATA_ROOT
    / "300h-subset-text"
    / "data"
    / "dev5"
    / "segments"
)

OUTPUT_JSONL = DATA_ROOT / "how2_dev5.jsonl"

# ============================================================
# LOAD TEXT MAPPING
# ============================================================

text_mapping = {}

with open(TEXT_FILE, "r", encoding="utf-8") as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        # first token = segment_name
        # rest = transcript
        parts = line.split(maxsplit=1)

        segment_name = parts[0]

        text = parts[1] if len(parts) > 1 else ""

        text_mapping[segment_name] = text

# ============================================================
# LOAD SEGMENT INFO
# ============================================================

segment_mapping = {}

with open(SEGMENT_FILE, "r", encoding="utf-8") as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        # format:
        # segment_name audio_name start end
        parts = line.split()

        if len(parts) != 4:
            continue

        segment_name = parts[0]
        audio_name = parts[1]
        start_time = float(parts[2])
        end_time = float(parts[3])

        segment_mapping[segment_name] = {
            "audio_name": audio_name,
            "start": start_time,
            "end": end_time
        }

# ============================================================
# CREATE JSONL
# ============================================================

video_files = sorted(
    VIDEO_SEGMENTS_DIR.glob("*.mp4")
)

num_written = 0
num_missing = 0

with open(OUTPUT_JSONL, "w", encoding="utf-8") as out_f:

    for video_path in video_files:

        segment_name = video_path.stem

        # --------------------------------------------
        # segment metadata
        # --------------------------------------------

        if segment_name not in segment_mapping:

            print(f"Missing segment info: {segment_name}")
            num_missing += 1
            continue

        seg_info = segment_mapping[segment_name]

        audio_name = seg_info["audio_name"]

        start_time = seg_info["start"]
        end_time = seg_info["end"]

        # --------------------------------------------
        # audio path
        # --------------------------------------------

        audio_path = (
            AUDIO_MONO_DIR
            / f"{audio_name}.wav"
        )

        # --------------------------------------------
        # text
        # --------------------------------------------

        text = text_mapping.get(segment_name, "")

        # --------------------------------------------
        # json entry
        # --------------------------------------------

        entry = {
            "video_segment": str(video_path),
            "audio": str(audio_path),
            "audio_segment": {
                "start": start_time,
                "end": end_time
            },
            "start": start_time,
            "end": end_time,
            "text": text
        }

        out_f.write(
            json.dumps(entry, ensure_ascii=False)
            + "\n"
        )

        num_written += 1

# ============================================================
# DONE
# ============================================================

print(f"Written entries: {num_written}")
print(f"Missing entries: {num_missing}")
print(f"Saved to: {OUTPUT_JSONL}")