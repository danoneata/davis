import sys
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import sys

# Mobile-VideoGPT Imports
from mobilevideogpt.utils import preprocess_input

# Model and tokenizer paths
pretrained_path = "Amshaker/Mobile-VideoGPT-1.5B" 
video_path = sys.argv[1]
prompt = "Can you describe what is happening in the video in detail?"

# Load model and tokenizer
config = AutoConfig.from_pretrained(pretrained_path)
tokenizer = AutoTokenizer.from_pretrained(pretrained_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    pretrained_path,
    config=config,
    torch_dtype=torch.float16
).cuda()

# Preprocess input
input_ids, video_frames, context_frames, stop_str = preprocess_input(
    model, tokenizer, video_path, prompt
)

# Run inference
with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=torch.stack(video_frames, dim=0).half().cuda(),
        context_images=torch.stack(context_frames, dim=0).half().cuda(),
        do_sample=False,
        temperature=0,
        top_p=1,
        num_beams=1,
        max_new_tokens=1024,
        use_cache=True,
    )
# Decode output
outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
if outputs.endswith(stop_str):
    outputs = outputs[:-len(stop_str)].strip()

print("🤖 Mobile-VideoGPT Output:", outputs)
