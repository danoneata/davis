import glob
import os
import pathlib
import pickle
from tqdm import tqdm

import hydra
import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from decord import VideoReader, cpu
from omegaconf import DictConfig
from PIL import Image
from transformers import AutoModel, AutoVideoProcessor, AutoImageProcessor, Wav2Vec2FeatureExtractor, WavLMModel



@torch.no_grad()
def get_vjepa2_embedding(
    video_path: str,
    model: AutoModel,
    processor: AutoVideoProcessor
) -> np.ndarray:
    num_frames = model.config.frames_per_clip

    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    frames = vr.get_batch(indices).asnumpy()  # T H W C
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2)

    inputs = processor(frames, return_tensors="pt")
    inputs = {k: v.to('cuda') for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, skip_predictor=True)

    # Encoder tokens: [batch, seq_len, hidden_size]
    token_features = outputs.last_hidden_state

    # Single vector for the whole video
    embedding = token_features.mean(dim=1)  # [batch, hidden_size]

    return embedding[0].float().cpu().numpy()


def get_dinov2_embeddings_1fps(
    video_path: str,
    model: AutoModel,
    processor: AutoImageProcessor,
    pool: str = "cls",   # "cls" or "mean"
) -> np.ndarray:
    vr = VideoReader(video_path, ctx=cpu(0))
    num_frames = len(vr)
    fps = float(vr.get_avg_fps())

    frame_indices = np.arange(0, num_frames, int(round(fps)))


    embeddings = []

    for idx in frame_indices:
        frame = vr[idx].asnumpy()  # H, W, C in RGB for decord
        image = Image.fromarray(frame)

        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        hidden = outputs.last_hidden_state

        if pool == "cls":
            emb = hidden[:, 0, :]          # CLS token
        elif pool == "mean":
            emb = hidden.mean(dim=1)       # mean over all tokens
        else:
            raise ValueError(f"Unsupported pool='{pool}', use 'cls' or 'mean'.")

        embeddings.append(emb[0].float().cpu().numpy())

    return np.stack(embeddings, axis=0).max(axis=0)


def get_wavlm_embedding(
    audio_path: str,
    model: WavLMModel,
    processor: Wav2Vec2FeatureExtractor
) -> np.ndarray:
    waveform, sr = torchaudio.load(audio_path)
    assert waveform.shape[0] == 1, "Audio must be mono"
    assert sr == 16000, "Audio must be 16 kHz"
    # TODO: IF THE AUDIO IS NOT ALREADY PREPROCESSED UNCOMMENT THIS
    # if waveform.shape[0] > 1:
    #     waveform = waveform.mean(dim=0, keepdim=True)
    # if sr != target_sr:
    #     waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    #     sr = target_sr
    waveform = waveform.squeeze(0)

    # Processor prepares input_values for the model
    inputs = processor(
        waveform.numpy(),
        sampling_rate=sr,
        return_tensors="pt",
        padding=False,
    )

    input_values = inputs.input_values.to('cuda')
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to('cuda')

    with torch.no_grad():
        outputs = model(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    # Final layer frame-level features: [batch, frames, hidden]
    last_hidden = outputs.last_hidden_state[0].cpu().numpy()

    # Mean pooled utterance-level embedding
    audio_embedding = last_hidden.mean(axis=0)

    return audio_embedding


def main():
    config = {
        "segments_path": "/root/how2/segments/dev5",
        "audios_path": "/root/how2/audios/dev5",
        "embeddings_path": "/root/how2/embeddings/dev5",
    }
    os.makedirs(config["embeddings_path"], exist_ok=True)
    
    video_model_name = "facebook/vjepa2-vitl-fpc64-256"
    video_processor = AutoVideoProcessor.from_pretrained(video_model_name)
    video_model = AutoModel.from_pretrained(
        video_model_name,
        torch_dtype=torch.float16,
    ).to('cuda')
    video_model.eval()

    image_model_name = "facebook/dinov2-base"
    image_processor = AutoImageProcessor.from_pretrained(image_model_name)
    image_model = AutoModel.from_pretrained(image_model_name).to("cuda")
    image_model.eval()

    audio_model_name = "microsoft/wavlm-base-plus"
    audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(audio_model_name)
    audio_model = WavLMModel.from_pretrained(audio_model_name).to('cuda')
    audio_model.eval()

    segment_names = sorted(os.listdir(config['segments_path']))
    # segment_names = segment_names[:int(len(segment_names) / 2)]
    for segment_name in tqdm(segment_names):
        segment_path = os.path.join(config["segments_path"], segment_name)
        audio_path = os.path.join(config["audios_path"], segment_name.replace(".mp4", ".wav"))
        embedding_path = os.path.join(config["embeddings_path"], segment_name.replace(".mp4", ".pkl"))

        if os.path.exists(embedding_path):
            continue

        try:
            video_embedding = get_vjepa2_embedding(segment_path, video_model, video_processor)
            image_embedding = get_dinov2_embeddings_1fps(segment_path, image_model, image_processor, pool="cls")
            audio_embedding = get_wavlm_embedding(audio_path, audio_model, audio_processor)
        except Exception as e:
            print(f"Error processing {segment_name}: {e}")
            continue
        
        with open(embedding_path, "wb") as f:
            pickle.dump({
                "video_embedding": video_embedding,
                "image_embedding": image_embedding,
                "audio_embedding": audio_embedding,
            }, f)



if __name__ == "__main__":
    main()