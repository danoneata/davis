import torch
from transformers import VideoMAEModel, VideoMAEImageProcessor
import numpy as np

processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")

# dummy video: 32 frames, H, W, C
video = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(32)]
inputs = processor(list(video), return_tensors="pt")
print(inputs["pixel_values"].shape)

try:
    with torch.no_grad():
        outputs = model(**inputs)
    print(outputs.last_hidden_state.shape)
except Exception as e:
    print(f"Error: {e}")
