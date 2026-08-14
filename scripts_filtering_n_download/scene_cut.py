import json
import subprocess
import tempfile
from pathlib import Path
import sys
from tqdm import tqdm

INPUT_JSONL = sys.argv[1]
OUTPUT_JSONL = sys.argv[2]

def detect_scenes(video_path):
    """
    Run PySceneDetect on a video and return scene boundaries.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        stats_file = Path(tmpdir) / "scenes.csv"

        # Run scenedetect
        cmd = [
            "scenedetect",
            "-i", str(video_path),
            "detect-content",
            "list-scenes",
            "-o", str(tmpdir)
        ]

        subprocess.run(cmd, check=True)

        # Find generated CSV
        csv_files = list(Path(tmpdir).glob("*.csv"))
        if not csv_files:
            return []

        csv_path = csv_files[0]

        scenes = []

        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Skip header lines
        for line in lines:
            if line.startswith("Scene Number"):
                continue
            if "," not in line:
                continue

            parts = [p.strip() for p in line.split(",")]

            # PySceneDetect CSV format:
            # Scene Number, Start Frame, Start Timecode, Start Time (seconds),
            # End Frame, End Timecode, End Time (seconds), Length (frames), Length (timecode), Length (seconds)

            try:
                start_sec = float(parts[3])
                end_sec = float(parts[6])

                scenes.append({
                    "start": start_sec,
                    "end": end_sec
                })
            except Exception:
                continue

        return scenes


with open(INPUT_JSONL, "r", encoding="utf-8") as infile, \
     open(OUTPUT_JSONL, "w", encoding="utf-8") as outfile:

    for line in tqdm(infile):
        line = line.strip()
        if not line:
            continue

        entry = json.loads(line)

        video_path = entry["video_segment"]

        try:
            scenes = detect_scenes(video_path)
        except subprocess.CalledProcessError as e:
            print(f"Scene detection failed for {video_path}: {e}")
            scenes = []

        entry["scenes"] = scenes

        outfile.write(json.dumps(entry) + "\n")

print(f"Done. Output written to {OUTPUT_JSONL}")
