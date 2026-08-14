import json
import torch
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

model_name = "Qwen/Qwen3-4B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)


# =========================
# LOAD PROMPT TEMPLATE
# =========================
def load_prompt_template(path):
    with open(path, "r") as f:
        return f.read()


def build_messages(text, scene_caption, prompt_template):
    prompt = prompt_template.format(
        text=text,
        scene_caption=scene_caption
    )

    return [
        {
            "role": "system",
            "content": "You are a strict binary classifier for video-text alignment."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]


def run_qwen(text, scene_caption, prompt_template):
    messages = build_messages(text, scene_caption, prompt_template)

    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    inputs = tokenizer([chat_text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False
        )

    output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist()

    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip().lower()

    # =========================
    # SAFE LABEL PARSING
    # =========================
    if content == "aligned":
        label = "aligned"
    elif content == "not_aligned":
        label = "not_aligned"
    else:
        label = "not_aligned"  # fallback safety

    return label, content


def process_jsonl(input_path, output_path, prompt_path):
    prompt_template = load_prompt_template(prompt_path)

    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for line in tqdm(fin):
            item = json.loads(line)

            label, raw_output = run_qwen(
                item["text"],
                item["video_caption"],
                prompt_template
            )

            item["alignment_label"] = label
            item["model_output"] = raw_output

            fout.write(json.dumps(item) + "\n")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    prompt_path = sys.argv[3]

    process_jsonl(input_path, output_path, prompt_path)
