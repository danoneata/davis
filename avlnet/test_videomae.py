import torch
from transformers import VideoMAEModel, VideoMAEImageProcessor

processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")

# dummy video: B, T, C, H, W
video = [torch.randn(3, 224, 224) for _ in range(16)]
inputs = processor(list(video), return_tensors="pt")
print(inputs["pixel_values"].shape)

with torch.no_grad():
    outputs = model(**inputs)

print(outputs.last_hidden_state.shape)
