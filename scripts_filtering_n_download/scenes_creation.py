import os
import json
import subprocess
from pathlib import Path
import sys
from tqdm import tqdm

# =========================
# CONFIG
# =========================

input_jsonl = sys.argv[1]
output_jsonl = sys.argv[2]

# Output folder for generated scene videos
output_scene_dir = "/mnt/private-share/speechDatabases/how2/300h-subset-video-segments-scenes"

os.makedirs(output_scene_dir, exist_ok=True)

# =========================
# PROCESS
# =========================

with open(input_jsonl, "r") as fin, open(output_jsonl, "w") as fout:

    for line_idx, line in enumerate(tqdm(fin)):

        line = line.strip()

        if not line:
            continue

        try:
            item = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[ERROR] Line {line_idx}: invalid JSON")
            print(e)
            continue

        video_segment = item["video_segment"]
        audio = item["audio"]
        audio_segment = item["audio_segment"]
        start_audio = item["start"]
        end_audio = item["end"]
        text = item["text"]

        scenes = item.get("scenes", [])

        video_name = Path(video_segment).stem

        for scene_idx, scene in enumerate(scenes):

            start_scene = scene["start"]
            end_scene = scene["end"]

            duration = end_scene - start_scene

            output_scene_path = os.path.join(
                output_scene_dir,
                f"{video_name}_scene{scene_idx}.mp4"
            )

            # =========================
            # Create scene video with ffmpeg
            # =========================

            cmd = [
                "ffmpeg",
                "-y",
                "-i", video_segment,
                "-ss", str(start_scene),
                "-t", str(duration),
                "-c:v", "libx264",
                "-c:a", "aac",
                output_scene_path
            ]

            print("Running:", " ".join(cmd))

            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )

            # =========================
            # Write output jsonl
            # =========================

            out_item = {
                "video_segment": video_segment,
                "video_scene": output_scene_path,
                "start_scene": start_scene,
                "end_scene": end_scene,
                "audio": audio,
                "audio_segment": audio_segment,
                "start_audio": start_audio,
                "end_audio": end_audio,
                "text": text
            }

            fout.write(json.dumps(out_item) + "\n")

print("Done!")
print("Scene videos saved to:", output_scene_dir)
print("Output JSONL:", output_jsonl)
