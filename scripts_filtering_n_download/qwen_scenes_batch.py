# -*- coding: utf-8 -*-
import os
import json
import sys
import torch

from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# =========================================================
# ARGS
# =========================================================
INPUT_JSONL = sys.argv[1]
OUTPUT_JSONL = sys.argv[2]
BATCH_SIZE = int(sys.argv[3])

# =========================================================
# MODEL
# =========================================================
checkpoint_path = "Qwen/Qwen3-VL-4B-Instruct-FP8"

processor = AutoProcessor.from_pretrained(checkpoint_path)

llm = LLM(
    model=checkpoint_path,
    trust_remote_code=True,
    gpu_memory_utilization=0.70,
    enforce_eager=False,
    tensor_parallel_size=torch.cuda.device_count(),
    seed=0
)

sampling_params = SamplingParams(
    temperature=0,
    max_tokens=256,
    top_k=-1,
    stop_token_ids=[],
)

# =========================================================
# PROMPT
# =========================================================
def build_messages(video_path):
    return [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {
                    "type": "text",
                    "text": (
                        "Describe briefly what is EXACTLY happening in the video. "
                        "Only describe visible actions and events. "
                        "Do NOT infer emotions, intentions, or anything not directly shown."
                    ),
                },
            ],
        }
    ]

# =========================================================
# INPUT PREP FOR VLLM
# =========================================================
def prepare_inputs_for_vllm(messages, processor):
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    return {
        "prompt": text,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs
    }

# =========================================================
# LOAD DATA
# =========================================================
data = []
with open(INPUT_JSONL, "r") as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

print(f"Loaded {len(data)} samples")

# =========================================================
# BATCH PROCESSING
# =========================================================
with open(OUTPUT_JSONL, "w") as fout:

    for i in range(0, len(data), BATCH_SIZE):

        batch = data[i:i + BATCH_SIZE]

        print(f"\nBatch {i // BATCH_SIZE + 1}")

        # -------------------------
        # BUILD VLLM INPUTS
        # -------------------------
        inputs = []
        valid_items = []

        for item in batch:
            video_path = item["video_scene"]

            messages = build_messages(video_path)

            try:
                inputs.append(
                    prepare_inputs_for_vllm(messages, processor)
                )
                valid_items.append(item)
            except Exception as e:
                print(f"[SKIP] {video_path} -> {e}")

        if not inputs:
            continue

        # -------------------------
        # RUN VLLM INFERENCE
        # -------------------------
        outputs = llm.generate(inputs, sampling_params=sampling_params)

        # -------------------------
        # PARSE OUTPUTS
        # -------------------------
        for item, out in zip(valid_items, outputs):

            caption = out.outputs[0].text.strip()

            item["scene_caption"] = caption

            fout.write(json.dumps(item) + "\n")

        fout.flush()

print("\nDONE ->", OUTPUT_JSONL)
