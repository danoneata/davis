import sys
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "mistralai/Mistral-7B-Instruct-v0.3"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

def build_prompt(text, scene_caption):
    return f"""You are a multimodal aligner consistency checker.

You will be given:
1. A spoken transcript from a video.
2. A visual description of the same video scene created automatically using a Visual Language Model.

Your task is to decide whether the speech and the visual scene are describing the same situation.

Rules:
- Return "aligned" if the speech somehow matches or describes the visual content.
- Return "not_aligned" if they refer to very different actions, objects, or contexts.
- Ignore minor wording differences.
- Note that the visual description may contain hallucinations or sometimes not make much sense. However, you should note that the Visual Language Model only got the vision video part, so it has to make sense strictly out of vision. So you should align contents with this in mind.
- Out of the transcription, think of it as what it is describing, not as what it is spoken.
- The visual description can be noisy, but think outside the box how a Visual Language Model might make errors and then decide if the noisy description actually makes sense with the spoken transcript.
- Focus on semantic meaning, not exact words.

FINALLY, MOST IMPORTANT, ASK YOURSELF: 'How would what is being spoken look like and be described by someone else?

Return ONLY one word: aligned or not_aligned.

Transcript:
{text}

Scene description:
{scene_caption}
"""

def classify(text, scene_caption):
    prompt = build_prompt(text, scene_caption)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.0
        )

    result = tokenizer.decode(output[0], skip_special_tokens=True)

    # extract last token decision
    if "not_aligned" in result:
        return "not_aligned"
    return "aligned"


def process_jsonl(input_file, output_file):
    with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            item = json.loads(line)

            label = classify(item["text"], item["scene_caption"])

            item["alignment_label"] = label

            f_out.write(json.dumps(item) + "\n")


# Example run
process_jsonl(sys.argv[1],sys.argv[2])
