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
            "content": "You are a strict video-text alignment scorer. Output ONLY an integer from 1 to 10."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]


def parse_score(content: str):
    """
    Robust parsing for model output -> integer score (1–10)
    """
    if content is None:
        return 1

    content = content.strip()

    # Try direct integer parse
    try:
        score = int(content)
    except ValueError:
        # Extract first integer if model outputs extra text
        digits = "".join([c for c in content if c.isdigit()])
        if digits:
            score = int(digits[0])  # safest single-digit fallback
        else:
            score = 1  # worst-case fallback

    # Clamp to valid range
    score = max(1, min(10, score))
    return score


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
            max_new_tokens=8,   # small since output is only 1 digit
            do_sample=False
        )

    output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist()

    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip().lower()

    score = parse_score(content)

    return score, content


def process_jsonl(input_path, output_path, prompt_path):
    prompt_template = load_prompt_template(prompt_path)

    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for line in tqdm(fin):
            item = json.loads(line)

            score, raw_output = run_qwen(
                item["text"],
                item["video_caption"],
                prompt_template
            )

            item["alignment_score"] = score
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