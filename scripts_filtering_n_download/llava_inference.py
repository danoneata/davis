import av
import numpy as np
import torch

from transformers import BitsAndBytesConfig,VideoLlavaForConditionalGeneration, VideoLlavaProcessor
import sys

video_path = sys.argv[1]

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`list[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

# Load the model in half-precision

# specify how to quantize the model
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf", quantization_config=quantization_config, device_map="auto")
processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")

# Load the video as an np.arrau, sampling uniformly 8 frames
video_path = hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename=input_video, repo_type="dataset")
container = av.open(video_path)
total_frames = container.streams.video[0].frames
indices = np.arange(0, total_frames, total_frames / 8).astype(int)
video = read_video_pyav(container, indices)

# For better results, we recommend to prompt the model in the following format
prompt = "USER: <video>\Please explain in detail what happens in the video. ASSISTANT:"
inputs = processor(text=prompt, videos=video, return_tensors="pt").to(model.device)

out = model.generate(**inputs, max_new_tokens=60)
print(out)
decoded = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)

print(decoded)
