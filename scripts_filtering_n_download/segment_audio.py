import json
from pathlib import Path

import soundfile as sf

# ============================================================
# PATHS
# ============================================================

DATA_ROOT = Path("/mnt/private-share/speechDatabases/how2")

INPUT_JSONL = DATA_ROOT / "how2_dev5.jsonl"

OUTPUT_AUDIO_DIR = DATA_ROOT / "300h-subset-audio-segments_dev5"
OUTPUT_JSONL = DATA_ROOT / "how2_dev5_segmented.jsonl"

OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# PROCESS
# ============================================================

def extract_audio_segment(audio_path, start, end, out_path):

    audio, sr = sf.read(audio_path)

    start_sample = int(start * sr)
    end_sample = int(end * sr)

    segment = audio[start_sample:end_sample]

    sf.write(out_path, segment, sr)

    return str(out_path)

# ============================================================
# RUN
# ============================================================

count = 0
missing_audio = 0

with open(INPUT_JSONL, "r", encoding="utf-8") as fin, \
     open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:

    for line in fin:

        item = json.loads(line)

        audio_path = item["audio"]

        start = float(item["start"])
        end = float(item["end"])

        video_segment_path = Path(item["video_segment"])

        segment_name = video_segment_path.stem

        out_audio_path = OUTPUT_AUDIO_DIR / f"{segment_name}.wav"

        # ----------------------------------------------------
        # skip if audio missing
        # ----------------------------------------------------

        if not Path(audio_path).exists():
            print(f"[MISSING AUDIO] {audio_path}")
            missing_audio += 1
            continue

        # ----------------------------------------------------
        # extract segment
        # ----------------------------------------------------

        extract_audio_segment(
            audio_path,
            start,
            end,
            out_audio_path
        )

        # ----------------------------------------------------
        # update manifest
        # ----------------------------------------------------

        new_item = dict(item)

        new_item["audio_segment"] = str(out_audio_path)

        fout.write(json.dumps(new_item) + "\n")

        count += 1

# ============================================================
# DONE
# ============================================================

print(f"Done. Created {count} audio segments")
print(f"Missing audio files: {missing_audio}")
print(f"Saved audio dir: {OUTPUT_AUDIO_DIR}")
print(f"Saved manifest: {OUTPUT_JSONL}")
