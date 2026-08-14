import av
import numpy as np
import torch
import sys
import time

from transformers import (
    BitsAndBytesConfig,
    VideoLlavaForConditionalGeneration,
    VideoLlavaProcessor,
)

video_path = sys.argv[1]

# -------------------------
# CONFIG (keep it light)
# -------------------------
MAX_FRAMES = 4
MAX_NEW_TOKENS = 100

# -------------------------
# VIDEO LOADER
# -------------------------
def load_video(path, num_frames=4):
    container = av.open(path)
    total_frames = container.streams.video[0].frames

    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)

    frames = []
    container.seek(0)

    for i, frame in enumerate(container.decode(video=0)):
        if i > indices[-1]:
            break
        if i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))

    return np.stack(frames)

# -------------------------
# LOAD MODEL (4-bit)
# -------------------------
print("Loading model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = VideoLlavaForConditionalGeneration.from_pretrained(
    "LanguageBind/Video-LLaVA-7B-hf",
    quantization_config=bnb_config,
    device_map="cuda:0",#"auto",
)

processor = VideoLlavaProcessor.from_pretrained(
    "LanguageBind/Video-LLaVA-7B-hf"
)

print("Model loaded.")

# -------------------------
# LOAD VIDEO
# -------------------------
print("Loading video...")
video = load_video(video_path, MAX_FRAMES)
print("Video frames:", video.shape)

# -------------------------
# PROMPT
# -------------------------
prompt = "USER: <video>\nDescribe briefly what is EXACTLY happening in the video, the scene and actions too. BUT do not extrapolate to anything that doesn't happen in the video. Don't talk about how people feel or think! DO NOT SAY WHAT IT SEEMS TO HAPPEN, JUST THE EXACT CONTENT!\nASSISTANT:"

# -------------------------
# INPUTS
# -------------------------
inputs = processor(
    text=prompt,
    videos=video,
    return_tensors="pt"
)

# force correct device (IMPORTANT)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# -------------------------
# INFERENCE
# -------------------------
print("Running inference...")

start = time.time()

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )

print("Done in:", time.time() - start, "seconds")

# -------------------------
# DECODE
# -------------------------
result = processor.batch_decode(
    output,
    skip_special_tokens=True
)[0]

print("\n--- OUTPUT ---\n")
print(result)
