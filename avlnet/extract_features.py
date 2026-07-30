import os
import pickle
import subprocess
import tempfile
import warnings

from pathlib import Path

import fire
import numpy as np
import soundfile as sf
import torch
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoVideoProcessor,
    Wav2Vec2FeatureExtractor,
    WavLMModel,
)

from utils import read_file

warnings.filterwarnings("ignore", message="Support for mismatched key_padding_mask and attn_mask")


DATA_DIR = Path(__file__).parent / "data"

DATASETS = {
    "msr-vtt": {
        "videos_path": Path("/data/how2/msr-vtt/TestVideo"),
        "selected_files": Path("/data/how2/avlnet-data/msrvtt_val_ids.txt"),
        "embeddings_path": DATA_DIR / "embeddings" / "msr-vtt",
        "audios_path": DATA_DIR / "audios" / "msr-vtt",
    },
    "youcook2": {
        "videos_path": Path("/data/how2/youcook2/val"),
        "selected_files": Path("/data/how2/avlnet-data/youcook_val_ids_common.txt"),
        "embeddings_path": DATA_DIR / "embeddings" / "youcook2",
        "audios_path": DATA_DIR / "audios" / "youcook2",
    },
}

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".avi", ".mov"}
VIDEO_MODEL_NAME = "facebook/vjepa2-vitl-fpc64-256"
IMAGE_MODEL_NAME = "facebook/dinov2-base"
AUDIO_MODEL_NAME = "microsoft/wavlm-base-plus"


@torch.no_grad()
def get_vjepa2_embedding(video_path, model, processor):
    num_frames = model.config.frames_per_clip
    vr = VideoReader(str(video_path), ctx=cpu(0))
    total_frames = len(vr)
    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    frames = vr.get_batch(indices).asnumpy()
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2)

    inputs = processor(frames, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    outputs = model(**inputs, skip_predictor=True)
    token_features = outputs.last_hidden_state
    embedding = token_features.mean(dim=1)
    return embedding[0].float().cpu().numpy()


@torch.no_grad()
def get_dinov2_embeddings_1fps(video_path, model, processor, pool: str = "cls"):
    vr = VideoReader(str(video_path), ctx=cpu(0))
    num_frames = len(vr)
    fps = float(vr.get_avg_fps())
    frame_indices = np.arange(0, num_frames, max(1, int(round(fps))))

    embeddings = []
    for idx in frame_indices:
        frame = vr[idx].asnumpy()
        image = Image.fromarray(frame)
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        outputs = model(**inputs)
        hidden = outputs.last_hidden_state
        if pool == "cls":
            emb = hidden[:, 0, :]
        elif pool == "mean":
            emb = hidden.mean(dim=1)
        else:
            raise ValueError(f"Unsupported pool='{pool}', use 'cls' or 'mean'.")
        embeddings.append(emb[0].float().cpu().numpy())
    return np.stack(embeddings, axis=0).max(axis=0)


@torch.no_grad()
def get_wavlm_embedding(audio_path, model, processor):
    waveform, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    assert waveform.ndim == 1, f"Audio must be mono, got shape {waveform.shape}"
    assert sr == 16000, f"Audio must be 16 kHz, got {sr}"

    inputs = processor(
        waveform, sampling_rate=sr, return_tensors="pt", padding=False
    )
    input_values = inputs.input_values.to("cuda")
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to("cuda")

    outputs = model(input_values, attention_mask=attention_mask, output_hidden_states=True)
    last_hidden = outputs.last_hidden_state[0].cpu().numpy()
    return last_hidden.mean(axis=0)


def extract_audio(video_path: Path, wav_path: Path) -> None:
    """Extract mono 16 kHz WAV from `video_path`, matching the upstream How2
    pipeline: ffmpeg → sox (`gain -4`, mono, resample to 16 kHz). Skips if the
    output already exists. Requires both `ffmpeg` and `sox` on PATH.
    """
    if wav_path.exists():
        return
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        raw_wav = Path(td) / "raw.wav"
        subprocess.run(
            [
                "ffmpeg", "-loglevel", "quiet", "-y",
                "-i", str(video_path),
                "-vn",
                str(raw_wav),
            ],
            check=True,
        )
        subprocess.run(
            [
                "sox", str(raw_wav), str(wav_path),
                "gain", "-4",
                "channels", "1",
                "rate", "16000",
            ],
            check=True,
        )


def load_models():
    video_processor = AutoVideoProcessor.from_pretrained(VIDEO_MODEL_NAME)
    video_model = AutoModel.from_pretrained(
        VIDEO_MODEL_NAME, torch_dtype=torch.float16
    ).to("cuda")
    video_model.eval()

    image_processor = AutoImageProcessor.from_pretrained(IMAGE_MODEL_NAME)
    image_model = AutoModel.from_pretrained(IMAGE_MODEL_NAME).to("cuda")
    image_model.eval()

    audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(AUDIO_MODEL_NAME)
    audio_model = WavLMModel.from_pretrained(AUDIO_MODEL_NAME).to("cuda")
    audio_model.eval()

    return {
        "video": (video_model, video_processor),
        "image": (image_model, image_processor),
        "audio": (audio_model, audio_processor),
    }


def list_videos(videos_path: Path) -> list:
    return sorted(
        p for p in videos_path.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def main(dataset_name: str):
    """Extract V-JEPA2 + DINOv2 + WavLM features for every video in a registered dataset.

    Paths per dataset come from `DATASETS`. Writes one <segment_id>.pkl per video to
    `embeddings_path`, skipping any that already exist. Audio is extracted on the fly
    to `audios_path` (also cached, so re-runs skip ffmpeg).

    Example:
        uv run extract_features.py msr-vtt
        uv run extract_features.py youcook2
    """
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Unknown dataset `{dataset_name}`. Available: {sorted(DATASETS)}"
        )
    cfg = DATASETS[dataset_name]
    videos_path = cfg["videos_path"]
    embeddings_path = cfg["embeddings_path"]
    audios_path = cfg["audios_path"]

    if not videos_path.exists():
        raise FileNotFoundError(f"Videos path does not exist: {videos_path}")
    embeddings_path.mkdir(parents=True, exist_ok=True)

    videos = list_videos(videos_path)
    print(f"Dataset `{dataset_name}`: {len(videos)} video(s) in {videos_path}")
    print(f"Writing embeddings to {embeddings_path}")
    print(f"Caching audio to {audios_path}")

    models = load_models()
    video_model, video_processor = models["video"]
    image_model, image_processor = models["image"]
    audio_model, audio_processor = models["audio"]

    selected_files = read_file(cfg["selected_files"])

    n_done = n_skipped = n_failed = 0
    for video_file in tqdm(videos, desc=f"Extracting {dataset_name}"):
        segment_id = video_file.stem
        emb_path = embeddings_path / f"{segment_id}.pkl"
        wav_path = audios_path / f"{segment_id}.wav"

        if segment_id not in selected_files:
            continue

        if emb_path.exists():
            n_skipped += 1
            continue

        try:
            extract_audio(video_file, wav_path)
            video_embedding = get_vjepa2_embedding(video_file, video_model, video_processor)
            image_embedding = get_dinov2_embeddings_1fps(
                video_file, image_model, image_processor, pool="cls"
            )
            audio_embedding = get_wavlm_embedding(wav_path, audio_model, audio_processor)
        except Exception as e:
            print(f"[error] {segment_id}: {e}")
            n_failed += 1
            continue

        with open(emb_path, "wb") as f:
            pickle.dump(
                {
                    "video_embedding": video_embedding,
                    "image_embedding": image_embedding,
                    "audio_embedding": audio_embedding,
                },
                f,
            )
        n_done += 1

    print(
        f"Done: {n_done} new, {n_skipped} already-existing, {n_failed} failed."
    )


if __name__ == "__main__":
    fire.Fire(main)
