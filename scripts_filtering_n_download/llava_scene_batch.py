import os
import json
import time
import sys
import av
import numpy as np
import torch

from transformers import (
    BitsAndBytesConfig,
    VideoLlavaForConditionalGeneration,
    VideoLlavaProcessor,
)

# =========================================================
# USAGE
# =========================================================
#
# python caption_scenes.py input.jsonl output.jsonl 8
#
# argv[1] -> input jsonl
# argv[2] -> output jsonl
# argv[3] -> batch size
#
# =========================================================

INPUT_JSONL = sys.argv[1]
OUTPUT_JSONL = sys.argv[2]
BATCH_SIZE = 16
DATA_ROOT = "/mnt/private-share/speechDatabases/how2/processed_how2"
# =========================================================
# CONFIG
# =========================================================

MAX_FRAMES = 4
MAX_NEW_TOKENS = 64

PROMPT = (
    "USER: <video>\n"
    "Describe briefly what is EXACTLY happening in the video, "
    #"most importantly the SCENE and ACTIONS. "
    "Do not give unimportant details. "
    "Focus on actions."
    "Do not talk about how people feel, think, how they look or what they wear! "
    "DO NOT SAY WHAT IT SEEMS TO HAPPEN, JUST THE EXACT CONTENT!\n"
    "ASSISTANT:"
)

# =========================================================
# VIDEO LOADER
# =========================================================

def load_video(path, num_frames=4):
    """
    Efficient frame sampling.
    """

    try:
        container = av.open(path)

        stream = container.streams.video[0]

        total_frames = stream.frames

        if total_frames <= 0:
            # fallback decode
            frames_all = []
            for frame in container.decode(video=0):
                frames_all.append(frame.to_ndarray(format="rgb24"))

            container.close()

            if len(frames_all) == 0:
                return None

            indices = np.linspace(
                0,
                len(frames_all) - 1,
                min(num_frames, len(frames_all))
            ).astype(int)

            selected = [frames_all[i] for i in indices]

            while len(selected) < num_frames:
                selected.append(selected[-1])

            return np.stack(selected)

        indices = np.linspace(
            0,
            total_frames - 1,
            num_frames
        ).astype(int)

        frames = []

        container.seek(0)

        for i, frame in enumerate(container.decode(video=0)):

            if i > indices[-1]:
                break

            if i in indices:
                frames.append(frame.to_ndarray(format="rgb24"))

        container.close()

        if len(frames) == 0:
            return None

        while len(frames) < num_frames:
            frames.append(frames[-1])

        return np.stack(frames)

    except Exception as e:
        print(f"[ERROR] loading video {path}: {e}")
        return None

# =========================================================
# LOAD MODEL
# =========================================================

print("Loading model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = VideoLlavaForConditionalGeneration.from_pretrained(
    "LanguageBind/Video-LLaVA-7B-hf",
    quantization_config=bnb_config,
    device_map="cuda:0",
    torch_dtype=torch.float16,
)

processor = VideoLlavaProcessor.from_pretrained(
    "LanguageBind/Video-LLaVA-7B-hf"
)

model.eval()

print("Model loaded.")

# =========================================================
# READ INPUT
# =========================================================

data = []

with open(INPUT_JSONL, "r") as f:
    for line in f:
        line = line.strip()

        if not line:
            continue

        data.append(json.loads(line))

print(f"Loaded {len(data)} entries")

# =========================================================
# BATCH PROCESSING
# =========================================================

total_start = time.time()

with open(OUTPUT_JSONL, "w") as fout:

    for batch_start in range(0, len(data), BATCH_SIZE):

        batch = data[batch_start: batch_start + BATCH_SIZE]

        print(
            f"\nProcessing batch "
            f"{batch_start // BATCH_SIZE + 1} / "
            f"{(len(data) + BATCH_SIZE - 1) // BATCH_SIZE}"
        )

        # -------------------------------------------------
        # LOAD VIDEOS
        # -------------------------------------------------

        valid_items = []
        videos = []

        load_start = time.time()

        for item in batch:

            video_path = os.path.join(DATA_ROOT,item["video_path"])

            video = load_video(video_path, MAX_FRAMES)

            if video is None:
                print(f"[SKIP] {video_path}")
                continue

            videos.append(video)
            valid_items.append(item)

        if len(videos) == 0:
            continue

        print(
            f"Loaded {len(videos)} videos "
            f"in {time.time() - load_start:.2f}s"
        )

        # -------------------------------------------------
        # PREPROCESS
        # -------------------------------------------------

        prompts = [PROMPT] * len(videos)

        inputs = processor(
            text=prompts,
            videos=videos,
            padding=True,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(model.device)
            for k, v in inputs.items()
        }

        # -------------------------------------------------
        # GENERATE
        # -------------------------------------------------

        infer_start = time.time()

        with torch.no_grad():

            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
            )

        print(
            f"Inference done in "
            f"{time.time() - infer_start:.2f}s"
        )

        # -------------------------------------------------
        # DECODE
        # -------------------------------------------------

        decoded = processor.batch_decode(
            outputs,
            skip_special_tokens=True
        )

        # -------------------------------------------------
        # SAVE
        # -------------------------------------------------

        for item, caption in zip(valid_items, decoded):

            # cleanup
            if "ASSISTANT:" in caption:
                caption = caption.split("ASSISTANT:")[-1].strip()

            item["video_caption"] = caption

            fout.write(json.dumps(item) + "\n")

        fout.flush()

        # -------------------------------------------------
        # CLEANUP
        # -------------------------------------------------

        del inputs
        del outputs

        torch.cuda.empty_cache()

# =========================================================
# DONE
# =========================================================

print("\n===================================")
print("DONE")
print("===================================")
print(f"Saved to: {OUTPUT_JSONL}")
print(f"Total time: {time.time() - total_start:.2f}s")
