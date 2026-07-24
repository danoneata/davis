import json
import glob
import numpy as np
import subprocess
import os
from tqdm import tqdm
import sys


config = {
    "data_path": "/mnt/how2/how2_300h/300h-subset",
    "segments_path": "/root/how2/segments/dev5",
    "audios_path": "/root/how2/audios",
    "embeddings_path": "/root/how2/embeddings",
    "segment_format": "mp4",
    "n_splits": 10
}



def create_segment(video_path: str, segment_name: str, start: float, end: float, config) -> None:
    video_name = os.path.basename(video_path).split(".")[0]
    segment_path = os.path.join(config['segments_path'], f"{segment_name}.{config['segment_format']}")
    if os.path.exists(segment_path):
        return
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel", "quiet",
            "-ss", f"{start:.2f}",
            "-i", video_path,
            "-y",
            "-t", f"{end - start:.2f}",
            "-c:v", "libx264",
            "-c:a", "aac",
            segment_path
        ]
    )



def main() -> None:
    video_paths = [
        path for path in sorted(glob.glob(config['data_path'] + "/*"))
        if not path.endswith(".json") and not path.endswith(".vtt")
    ]

    with open("./dev5.txt", "r") as f:
        lines = f.readlines()

    for line in tqdm(lines):
        segment_name, video_name, start, end = line.strip().split(" ")
        start, end = float(start), float(end)
        video_path = next(video_path for video_path in video_paths if os.path.basename(video_path).startswith(video_name))
        create_segment(video_path, segment_name, start, end, config)
    

if __name__ == "__main__":
    main()