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

warnings.filterwarnings(
    "ignore", message="Support for mismatched key_padding_mask and attn_mask"
)


DATA_DIR = Path(__file__).parent / "data"

# `kind` selects the extraction pipeline: "video" datasets have one video file per
# item (V-JEPA2 + DINOv2 + WavLM); "image" datasets have a still image paired with
# one or more spoken captions, so there is no video embedding to compute.
DATASETS = {
    "msr-vtt": {
        "kind": "video",
        "videos_path": Path("/data/how2/msr-vtt/TestVideo"),
        "selected_files": Path("/data/how2/avlnet-data/msrvtt_val_ids.txt"),
        # Antonio's embeddings path
        # "embeddings_path": Path("/data/how2/evaluation_data/msr-vtt/embeddings"),
        "embeddings_path": DATA_DIR / "embeddings" / "msr-vtt",
        "audios_path": DATA_DIR / "audios" / "msr-vtt",
    },
    "youcook2": {
        "kind": "video",
        "videos_path": Path("/data/how2/youcook2/val"),
        "selected_files": Path("/data/how2/avlnet-data/youcook_val_ids_common.txt"),
        # Antonio's embeddings path
        "embeddings_path": Path("/data/how2/evaluation_data/youcook2/embeddings"),
        # "embeddings_path": DATA_DIR / "embeddings" / "youcook2",
        "audios_path": DATA_DIR / "audios" / "youcook2",
    },
    "flickr8k": {
        "kind": "image",
        "images_path": Path("/data/how2/flickr8k/flickr8k-images/Flicker8k_Dataset"),
        # Already mono 16 kHz, so we read them in place instead of re-encoding.
        "audios_path": Path("/data/how2/flickr8k/flickr8k-audio/wavs"),
        "wav2capt_path": Path("/data/how2/flickr8k/flickr8k-audio/wav2capt.txt"),
        # Transcripts of the spoken captions; used by the viewers, not by extraction.
        "captions_path": Path("/data/how2/flickr8k/flickr8k-text/Flickr8k.token.txt"),
        "selected_files": Path(
            "/data/how2/flickr8k/flickr8k-text/Flickr_8k.testImages.txt"
        ),
        "embeddings_path": DATA_DIR / "embeddings" / "flickr8k",
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
def _embed_image(image, model, processor, pool: str = "cls"):
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
    return emb[0].float().cpu().numpy()


def get_dinov2_embeddings_1fps(video_path, model, processor, pool: str = "cls"):
    vr = VideoReader(str(video_path), ctx=cpu(0))
    num_frames = len(vr)
    fps = float(vr.get_avg_fps())
    frame_indices = np.arange(0, num_frames, max(1, int(round(fps))))

    embeddings = [
        _embed_image(Image.fromarray(vr[idx].asnumpy()), model, processor, pool)
        for idx in frame_indices
    ]
    return np.stack(embeddings, axis=0).max(axis=0)


def get_dinov2_embedding(image_path, model, processor, pool: str = "cls"):
    """Same as `get_dinov2_embeddings_1fps` for a still image: with a single frame
    the max-over-frames pooling collapses to the frame embedding itself.
    """
    image = Image.open(image_path).convert("RGB")
    return _embed_image(image, model, processor, pool)


@torch.no_grad()
def get_wavlm_embedding(audio_path, model, processor, gain_db: float = 0.0):
    waveform, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    assert waveform.ndim == 1, f"Audio must be mono, got shape {waveform.shape}"
    assert sr == 16000, f"Audio must be 16 kHz, got {sr}"

    # WavLM's feature extractor has do_normalize=False, so the input level matters.
    # `extract_audio` bakes a -4 dB sox gain into the cached WAVs; datasets whose
    # audio we read as-is have to apply the same gain here to stay comparable.
    if gain_db:
        waveform = waveform * (10.0 ** (gain_db / 20.0))

    inputs = processor(waveform, sampling_rate=sr, return_tensors="pt", padding=False)
    input_values = inputs.input_values.to("cuda")
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to("cuda")

    outputs = model(
        input_values, attention_mask=attention_mask, output_hidden_states=True
    )
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
                "ffmpeg",
                "-loglevel",
                "quiet",
                "-y",
                "-i",
                str(video_path),
                "-vn",
                str(raw_wav),
            ],
            check=True,
        )
        subprocess.run(
            [
                "sox",
                str(raw_wav),
                str(wav_path),
                "gain",
                "-4",
                "channels",
                "1",
                "rate",
                "16000",
            ],
            check=True,
        )


def load_models(with_video: bool = True):
    models = {}

    if with_video:
        video_processor = AutoVideoProcessor.from_pretrained(VIDEO_MODEL_NAME)
        video_model = AutoModel.from_pretrained(
            VIDEO_MODEL_NAME, torch_dtype=torch.float16
        ).to("cuda")
        video_model.eval()
        models["video"] = (video_model, video_processor)

    image_processor = AutoImageProcessor.from_pretrained(IMAGE_MODEL_NAME)
    image_model = AutoModel.from_pretrained(IMAGE_MODEL_NAME).to("cuda")
    image_model.eval()

    audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(AUDIO_MODEL_NAME)
    audio_model = WavLMModel.from_pretrained(AUDIO_MODEL_NAME).to("cuda")
    audio_model.eval()

    models["image"] = (image_model, image_processor)
    models["audio"] = (audio_model, audio_processor)
    return models


def list_videos(videos_path: Path) -> list:
    return sorted(
        p
        for p in videos_path.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def read_wav2capt(path: Path) -> dict:
    """Map image id -> [wav filename per caption index], from Flickr8k's wav2capt.txt.

    Lines look like `<image_id>_<n>.wav <image_id>.jpg #<n>`.
    """
    captions = {}
    for line in read_file(path):
        wav_name, image_name, caption_index = line.split()
        image_id = Path(image_name).stem
        captions.setdefault(image_id, {})[int(caption_index.lstrip("#"))] = wav_name
    return {
        image_id: [wavs[i] for i in sorted(wavs)] for image_id, wavs in captions.items()
    }


def extract_image_dataset(dataset_name: str, cfg: dict):
    """Extract DINOv2 + WavLM features for an image dataset with spoken captions.

    Writes one <image_id>.pkl per image holding a single `image_embedding` and an
    `audio_embeddings` array of shape (n_captions, AUDIO_DIM) -- Flickr8k has five
    spoken captions per image, and keeping them all lets the evaluation average
    over captions instead of arbitrarily picking one.
    """
    images_path = cfg["images_path"]
    audios_path = cfg["audios_path"]
    embeddings_path = cfg["embeddings_path"]

    for path in (images_path, audios_path):
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
    embeddings_path.mkdir(parents=True, exist_ok=True)

    captions = read_wav2capt(cfg["wav2capt_path"])
    image_ids = [Path(name).stem for name in read_file(cfg["selected_files"]) if name]

    print(f"Dataset `{dataset_name}`: {len(image_ids)} image(s) in {images_path}")
    print(f"Reading audio from {audios_path}")
    print(f"Writing embeddings to {embeddings_path}")

    models = load_models(with_video=False)
    image_model, image_processor = models["image"]
    audio_model, audio_processor = models["audio"]

    n_done = n_skipped = n_failed = 0
    for image_id in tqdm(image_ids, desc=f"Extracting {dataset_name}"):
        emb_path = embeddings_path / f"{image_id}.pkl"
        if emb_path.exists():
            n_skipped += 1
            continue

        try:
            image_embedding = get_dinov2_embedding(
                images_path / f"{image_id}.jpg",
                image_model,
                image_processor,
                pool="cls",
            )
            audio_embeddings = np.stack(
                [
                    get_wavlm_embedding(
                        audios_path / wav_name,
                        audio_model,
                        audio_processor,
                        gain_db=-4.0,
                    )
                    for wav_name in captions[image_id]
                ]
            )
        except Exception as e:
            print(f"[error] {image_id}: {e}")
            n_failed += 1
            continue

        with open(emb_path, "wb") as f:
            pickle.dump(
                {
                    "image_embedding": image_embedding,
                    "audio_embeddings": audio_embeddings,
                },
                f,
            )
        n_done += 1

    print(f"Done: {n_done} new, {n_skipped} already-existing, {n_failed} failed.")


def main(dataset_name: str):
    """Extract V-JEPA2 + DINOv2 + WavLM features for every item in a registered dataset.

    Paths per dataset come from `DATASETS`. Writes one <segment_id>.pkl per video to
    `embeddings_path`, skipping any that already exist. Audio is extracted on the fly
    to `audios_path` (also cached, so re-runs skip ffmpeg).

    Image datasets (`kind: image`, e.g. Flickr8k) have no video track, so they skip
    V-JEPA2 and are handled by `extract_image_dataset`.

    Example:
        uv run extract_features.py msr-vtt
        uv run extract_features.py youcook2
        uv run extract_features.py flickr8k
    """
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Unknown dataset `{dataset_name}`. Available: {sorted(DATASETS)}"
        )
    cfg = DATASETS[dataset_name]
    if cfg.get("kind", "video") == "image":
        return extract_image_dataset(dataset_name, cfg)

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
            video_embedding = get_vjepa2_embedding(
                video_file,
                video_model,
                video_processor,
            )
            image_embedding = get_dinov2_embeddings_1fps(
                video_file,
                image_model,
                image_processor,
                pool="cls",
            )
            audio_embedding = get_wavlm_embedding(
                wav_path,
                audio_model,
                audio_processor,
            )
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

    print(f"Done: {n_done} new, {n_skipped} already-existing, {n_failed} failed.")


if __name__ == "__main__":
    fire.Fire(main)
