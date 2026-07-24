import torch
from transformers import VideoMAEModel, VideoMAEImageProcessor
import numpy as np

processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-large-finetuned-kinetics")
model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-large-finetuned-kinetics")

# dummy video: 16 frames, H, W, C
video = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(16)]
inputs = processor([video], return_tensors="pt")
print("Input shape:", inputs["pixel_values"].shape)

with torch.no_grad():
    outputs = model(**inputs)
print("Output shape:", outputs.last_hidden_state.shape)
