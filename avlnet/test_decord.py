import numpy as np
import torch
from decord import VideoReader, cpu

def test():
    # create a dummy video or just test the logic
    fps = 30.0
    num_frames = 300
    duration = num_frames / fps
    num_features = int(round(duration * 1.5))
    
    clip_len = 32
    clip_fps = 15
    frame_stride = max(1, int(round(fps / clip_fps)))
    
    for i in range(num_features):
        timestamp = i / 1.5
        center_frame_idx = int(round(timestamp * fps))
        half_clip_len = (clip_len * frame_stride) // 2
        start_frame_idx = center_frame_idx - half_clip_len
        
        clip_indices = [
            min(max(0, start_frame_idx + j * frame_stride), num_frames - 1)
            for j in range(clip_len)
        ]
        print(f"Feature {i}: timestamp {timestamp:.2f}s, center {center_frame_idx}, indices {clip_indices[0]} to {clip_indices[-1]}, len {len(clip_indices)}")

test()
