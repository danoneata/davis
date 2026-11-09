#!/usr/bin/env python3
"""
Extract video / image / audio embeddings (GPU, batched) for every clip listed in JSONL manifests.

Manifest line (video + audio, or image + audio; "duration" is optional):
    {"video_path": ".../ID_0.0_10.0.mp4", "audio_path": ".../ID_0.0_10.0.wav", "duration": 10.0}
    {"image_path": ".../ID.jpg",          "audio_path": ".../ID.wav"}
"video_path" is used if present, else "image_path". Images need --mode image_only.

Output: <output_dir>/<audio file name without extension>.pkl, all arrays float16:
    (keyed by the audio, so one image with several audios, e.g. Flickr8k <id>_0..4.wav, gives one .pkl per audio)
    {
        "video_embedding": V-JEPA 2, mean over all encoder tokens           (1024,)      full mode only, else None
        "image_embedding": DINOv2 per frame, then averaged over the frames    (D,)
                           frames = one per second (full) or the middle frame (image_only)
                           D = 768 (cls / mean) or 1536 (cls+mean)
        "audio_embedding": WavLM / SPEAR last layer, ~50 frames/s average-pooled in time order
                           to --audio_fps (default 5 -> one vector per 200 ms)   (ceil(5 * seconds), 768 | 1280)
        "audio_embedding_pooled": mean of audio_embedding over time                (768 | 1280,)
    }

Modes:
    full        video + image + audio
    image_only  image + audio (no V-JEPA). Videos: the middle frame (or --image_frames 1fps);
                images: the image itself, preprocessed exactly like a single video frame.

Examples:
    python extract_embeddings.py --manifest train_*.jsonl --output_dir /data/emb
    python extract_embeddings.py --manifest train_*.jsonl --output_dir /data/emb_img --mode image_only --audio_model spear

    # one process per GPU (8 GPUs)
    for i in 0 1 2 3 4 5 6 7; do
        CUDA_VISIBLE_DEVICES=$i python extract_embeddings.py --manifest train_*.jsonl \
            --output_dir /data/emb --num_shards 8 --shard_id $i &
    done

    # disk usage estimate from the manifest durations (no GPU needed)
    python extract_embeddings.py --manifest train_*.jsonl --estimate_storage

Re-running the same command resumes: clips that already have a .pkl are skipped.
"""

import argparse
import json
import logging
import math
import os
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from decord import VideoReader, cpu
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, Wav2Vec2FeatureExtractor, WavLMModel

# --------------------------------------------------------------------------- #
# Models (change here)
# --------------------------------------------------------------------------- #
VJEPA_MODEL = "facebook/vjepa2-vitl-fpc64-256"
DINO_MODEL = "facebook/dinov2-base"
AUDIO_MODELS = {
    "wavlm": "microsoft/wavlm-base-plus",
    "spear": "marcoyang/spear-xlarge-speech-audio-v2",
}

# Frame preprocessing, same recipe as both models' HF processors, but split CPU / GPU for speed:
#   worker (CPU): decord decodes frames directly at shortest side 292, then center-crops 256x256 (uint8)
#   GPU:          V-JEPA 2 uses the 256 crop as is (its processor: resize 292 + crop 256)
#                 DINOv2 resizes the crop to 224 (~ its processor: resize 256 + crop 224, same field of view)
DECODE_SHORT_SIDE = 292
CROP_SIZE = 256
DINO_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff")

SAMPLE_RATE = 16000
MIN_AUDIO_SAMPLES = SAMPLE_RATE // 5  # shorter audio is zero-padded to 0.2 s (SPEAR crashes on ~empty input)
AUDIO_MODEL_FPS = 50                  # WavLM and SPEAR both output 50 frames per second

# Only used by --estimate_storage
EMB_DIMS = {"video": 1024, "image": 768, "wavlm": 768, "spear": 1280}
PICKLE_OVERHEAD_BYTES = 500

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}

log = logging.getLogger("extract")


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
def is_image(path: str) -> bool:
    return path.lower().endswith(IMAGE_EXTS)


def clip_key(audio_path: str) -> str:
    """'/a/b/---4IxR2xOg_0.0_10.0.wav' -> '---4IxR2xOg_0.0_10.0' (the output file name).

    Keyed by the audio file: for scene clips it has the same name as the video, and for
    datasets with several audios per image (Flickr8k: <id>.jpg -> <id>_0..4.wav) every audio
    gets its own output file instead of overwriting the others.
    """
    return os.path.splitext(os.path.basename(audio_path))[0]


def read_manifests(paths):
    video_paths, audio_paths, durations = [], [], []
    for path in paths:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                video_paths.append(row.get("video_path") or row["image_path"])
                audio_paths.append(row["audio_path"])
                durations.append(float(row.get("duration", 0.0)))
    return video_paths, audio_paths, np.asarray(durations, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Data loading (runs in DataLoader worker processes, CPU only: decode + crop, nothing else)
# --------------------------------------------------------------------------- #
class ClipDataset(Dataset):
    def __init__(self, video_paths, audio_paths, args, audio_processor):
        # Paths are kept as numpy byte arrays, not Python lists: with millions of entries,
        # lists get slowly copied into every worker (copy-on-write + refcounts) and RAM explodes.
        self.video_paths = np.array([p.encode() for p in video_paths])
        self.audio_paths = np.array([p.encode() for p in audio_paths])
        self.use_video = args.mode == "full"
        self.image_frames = args.image_frames
        self.vjepa_num_frames = args.vjepa_num_frames
        self.audio_processor = audio_processor  # WavLM feature extractor, None for SPEAR (takes raw waveform)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, i):
        video_path = self.video_paths[i].decode()
        sample = {"key": clip_key(self.audio_paths[i].decode())}
        try:
            sample.update(self.load_image(video_path) if is_image(video_path) else self.load_video(video_path))
            sample["audio"] = self.load_audio(self.audio_paths[i].decode())
        except Exception as e:  # a broken file must not kill the whole run
            return {"key": sample["key"], "error": f"{type(e).__name__}: {e}"}
        return sample

    def image_frame_indices(self, num_frames, fps):
        if self.image_frames == "middle":
            return np.array([num_frames // 2])
        return np.arange(0, num_frames, max(1, int(round(fps))))  # "1fps"

    def load_video(self, path):
        # Probe size / length, then decode already resized (scaling inside the decoder is ~free)
        probe = VideoReader(path, ctx=cpu(0), num_threads=1)
        n, fps = len(probe), probe.get_avg_fps()
        h, w = probe[0].shape[:2]
        del probe
        scale = DECODE_SHORT_SIDE / min(h, w)
        vr = VideoReader(path, ctx=cpu(0), num_threads=1,
                         width=max(CROP_SIZE, round(w * scale)), height=max(CROP_SIZE, round(h * scale)))

        image_idx = self.image_frame_indices(n, fps)
        video_idx = np.linspace(0, n - 1, self.vjepa_num_frames).astype(int) if self.use_video else image_idx[:0]

        # Decode every needed frame exactly once, then center crop
        wanted = np.unique(np.concatenate([image_idx, video_idx]))
        frames = vr.get_batch(wanted).asnumpy()  # (N, H, W, 3) uint8 RGB
        top = (frames.shape[1] - CROP_SIZE) // 2
        left = (frames.shape[2] - CROP_SIZE) // 2
        frames = frames[:, top:top + CROP_SIZE, left:left + CROP_SIZE]

        out = {"image": torch.from_numpy(np.ascontiguousarray(frames[np.searchsorted(wanted, image_idx)]))}
        if self.use_video:
            out["video"] = torch.from_numpy(np.ascontiguousarray(frames[np.searchsorted(wanted, video_idx)]))
        return out

    def load_image(self, path):
        """Single still image -> same preprocessing as one decoded video frame: short side 292, center crop 256."""
        img = Image.open(path)
        img.draft("RGB", (DECODE_SHORT_SIDE, DECODE_SHORT_SIDE))  # JPEG: decode at reduced scale (>= target), much faster
        img = img.convert("RGB")
        w, h = img.size
        scale = DECODE_SHORT_SIDE / min(h, w)
        nw, nh = max(CROP_SIZE, round(w * scale)), max(CROP_SIZE, round(h * scale))
        img = img.resize((nw, nh), Image.BILINEAR)
        top, left = (nh - CROP_SIZE) // 2, (nw - CROP_SIZE) // 2
        frame = np.array(img.crop((left, top, left + CROP_SIZE, top + CROP_SIZE)))  # writable copy
        return {"image": torch.from_numpy(frame)[None]}  # (1, 256, 256, 3) uint8

    def load_audio(self, path):
        wav, sr = sf.read(path, dtype="float32", always_2d=True)  # (samples, channels)
        wav = torch.from_numpy(wav.mean(axis=1))  # mono
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        if len(wav) < MIN_AUDIO_SAMPLES:
            wav = F.pad(wav, (0, MIN_AUDIO_SAMPLES - len(wav)))
        if self.audio_processor is not None:  # WavLM feature extractor (normalises if the checkpoint wants it)
            wav = self.audio_processor(wav.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values[0]
        return wav


def collate(samples):
    good = [s for s in samples if "error" not in s]
    batch = {
        "keys": [s["key"] for s in good],
        "failed": [(s["key"], s["error"]) for s in samples if "error" in s],
    }
    if not good:
        return batch

    # All DINOv2 frames of the batch go through the model together; counts split them back per clip
    batch["image"] = torch.cat([s["image"] for s in good])  # (sum F, 256, 256, 3) uint8
    batch["image_counts"] = [len(s["image"]) for s in good]
    if "video" in good[0]:
        batch["video"] = torch.stack([s["video"] for s in good])  # (B, T, 256, 256, 3) uint8
    batch["audio_lens"] = torch.tensor([len(s["audio"]) for s in good])
    batch["audio"] = pad_sequence([s["audio"] for s in good], batch_first=True)  # zero-padded
    return batch


def make_batches(durations, batch_size):
    """Group clips of similar length together -> almost no audio padding, fewer wasted FLOPs.
    Longest first, so an out-of-memory error shows up immediately rather than hours in."""
    order = np.argsort(-durations, kind="stable")
    return [order[i:i + batch_size].tolist() for i in range(0, len(order), batch_size)]


def worker_init(_):
    torch.set_num_threads(1)  # many workers x many threads each = slower, not faster


# --------------------------------------------------------------------------- #
# GPU: preprocessing + encoders
# --------------------------------------------------------------------------- #
def normalize_frames(frames, dtype, size=None):
    """uint8 (N, H, W, 3) on GPU -> normalized (N, 3, size, size) in `dtype`."""
    x = frames.permute(0, 3, 1, 2).float().div_(255)
    if size is not None and size != x.shape[-1]:
        x = F.interpolate(x, size=(size, size), mode="bicubic", antialias=True, align_corners=False).clamp_(0, 1)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return ((x - mean) / std).to(dtype)


def encode_video(model, frames, dtype):
    """V-JEPA 2 encoder on (B, T, 256, 256, 3) uint8, mean over all tokens -> list of (1024,)."""
    b, t = frames.shape[:2]
    pixels = normalize_frames(frames.flatten(0, 1), dtype).view(b, t, 3, CROP_SIZE, CROP_SIZE)
    tokens = model(pixel_values_videos=pixels, skip_predictor=True).last_hidden_state
    return list(tokens.mean(dim=1))


def encode_image(model, frames, counts, pool, dtype):
    """DINOv2 on every frame of the batch, pooled per frame, then averaged over each clip's frames -> list of (D,)."""
    pixels = normalize_frames(frames, dtype, size=DINO_SIZE)
    hidden = model(pixel_values=pixels).last_hidden_state  # (frames, 1 + patches, 768)
    cls = hidden[:, 0]
    mean = hidden[:, 1:].mean(dim=1)  # patch tokens only (excludes CLS)
    if pool == "cls":
        feats = cls
    elif pool == "mean":
        feats = mean
    else:  # "cls+mean"
        feats = torch.cat([cls, mean], dim=-1)
    return [f.float().mean(dim=0) for f in feats.split(counts)]  # average over the clip's frames


def pool_time(feats, factor):
    """(T, D) -> (ceil(T / factor), D): mean of each consecutive group of `factor` frames, time order kept.
    The last group may be shorter and is averaged over the frames it actually has."""
    if factor == 1:
        return feats
    return F.avg_pool1d(feats.T[None].float(), factor, factor, ceil_mode=True)[0].T


def encode_wavlm(model, wav, lens, use_attention_mask, pool_factor):
    """WavLM last layer -> list of (T_i, 768) at AUDIO_MODEL_FPS / pool_factor frames per second."""
    mask = None
    if use_attention_mask:  # HF: only pass a mask if the checkpoint's feature extractor returns one
        mask = (torch.arange(wav.shape[1], device=wav.device)[None] < lens[:, None]).long()
    hidden = model(wav, attention_mask=mask).last_hidden_state
    out_lens = model._get_feat_extract_output_lengths(lens).tolist()
    return [pool_time(h[:n], pool_factor) for h, n in zip(hidden, out_lens)]


def encode_spear(model, wav, lens, dtype, pool_factor):
    """SPEAR last layer -> list of (T_i, 1280) at AUDIO_MODEL_FPS / pool_factor frames per second."""
    spear = model.model  # the HF wrapper keeps the actual SPEAR model in `.model`
    # fbank stays in fp32: under fp16 autocast the mel-filterbank matmul can overflow on loud audio
    feats, feat_lens = spear.compute_fbank(wav.float(), lens)
    with torch.autocast(wav.device.type, dtype=dtype, enabled=dtype != torch.float32):
        out = spear.model.forward_encoder(feats, feat_lens, return_dict=True)
    return [pool_time(h[:n], pool_factor) for h, n in zip(out["encoder_out"], out["encoder_out_lens"].tolist())]


def load_models(args, device, dtype):
    """Returns (models, audio_processor). HF models run directly in `dtype`; SPEAR keeps fp32 weights + autocast."""
    models = {}
    models["image"] = AutoModel.from_pretrained(DINO_MODEL, attn_implementation="sdpa").to(device, dtype).eval()
    if args.mode == "full":
        models["video"] = AutoModel.from_pretrained(VJEPA_MODEL, attn_implementation="sdpa").to(device, dtype).eval()

    audio_name = AUDIO_MODELS[args.audio_model]
    if args.audio_model == "wavlm":
        audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(audio_name)
        models["audio"] = WavLMModel.from_pretrained(audio_name).to(device, dtype).eval()
    else:
        audio_processor = None
        models["audio"] = AutoModel.from_pretrained(audio_name, trust_remote_code=True).to(device).eval()
    return models, audio_processor


def to_numpy_fp16(tensors):
    """One GPU->CPU copy for the whole batch (instead of one per clip), then split back per clip."""
    shapes = [t.shape for t in tensors]
    flat = torch.cat([t.reshape(-1).to(torch.float16) for t in tensors]).cpu().numpy()
    out, start = [], 0
    for shape in shapes:
        size = math.prod(shape)
        out.append(flat[start:start + size].reshape(shape))
        start += size
    return out


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def save_pickle(path, obj):
    tmp = path + ".tmp"  # write-then-rename: a crash never leaves a half-written .pkl behind
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


class AsyncWriter:
    """Writes files from a thread pool so the GPU never waits on the disk."""

    def __init__(self, num_threads=8, max_pending=1024):
        self.pool = ThreadPoolExecutor(num_threads)
        self.slots = threading.Semaphore(max_pending)  # bounds RAM if the disk is slower than the GPU

    def submit(self, path, obj):
        self.slots.acquire()
        future = self.pool.submit(save_pickle, path, obj)
        future.add_done_callback(self._done)

    def _done(self, future):
        self.slots.release()
        if future.exception() is not None:
            log.error("write failed: %s", future.exception())

    def close(self):
        self.pool.shutdown(wait=True)


# --------------------------------------------------------------------------- #
# Storage estimate
# --------------------------------------------------------------------------- #
def estimate_storage(durations, args):
    bytes_per_value = 2  # float16
    image_dim = EMB_DIMS["image"] * (2 if args.image_pool == "cls+mean" else 1)
    audio_frames = np.ceil(durations * args.audio_fps)

    values = image_dim + (audio_frames + 1) * EMB_DIMS[args.audio_model]  # +1: audio_embedding_pooled
    if args.mode == "full":
        values = values + EMB_DIMS["video"]
    file_bytes = values * bytes_per_value + PICKLE_OVERHEAD_BYTES
    on_disk = np.ceil(file_bytes / 4096) * 4096  # file systems allocate at least one 4 KB block per file

    print(f"clips:              {len(durations):,}")
    print(f"total duration:     {durations.sum() / 3600:,.1f} h (mean {durations.mean():.2f} s/clip)")
    print(f"setup:              mode={args.mode} audio={args.audio_model} ({args.audio_fps} fps) "
          f"image_pool={args.image_pool} image_frames={args.image_frames}")
    print(f"average file size:  {file_bytes.mean() / 1e3:,.1f} KB")
    print(f"total:              {file_bytes.sum() / 1e9:,.1f} GB  (~{on_disk.sum() / 1e9:,.1f} GB on disk)")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", nargs="+", required=True, help="one or more JSONL manifests")
    p.add_argument("--output_dir", help="where the .pkl files go (required unless --estimate_storage)")
    p.add_argument("--mode", choices=["full", "image_only"], default="full",
                   help="full = V-JEPA 2 + DINOv2 + audio; image_only = DINOv2 + audio")
    p.add_argument("--audio_model", choices=list(AUDIO_MODELS), default="spear")
    p.add_argument("--audio_fps", type=int, default=5, choices=[1, 2, 5, 10, 25, 50],
                   help="audio frames kept per second (average pooling of the 50 frames/s model output)")
    p.add_argument("--image_pool", choices=["cls", "mean", "cls+mean"], default="cls+mean")
    p.add_argument("--image_frames", choices=["1fps", "middle"], default=None,
                   help="frames fed to DINOv2 (default: 1fps in full mode, middle frame in image_only)")
    p.add_argument("--vjepa_num_frames", type=int, default=16,
                   help="frames sampled uniformly over the clip for V-JEPA 2 (default: the trained value, 64)")
    p.add_argument("--batch_size", type=int, default=16, help="clips per batch")
    p.add_argument("--num_workers", type=int, default=min(16, os.cpu_count() or 1),
                   help="CPU decoding processes; raise it if the GPU is waiting on data")
    p.add_argument("--dtype", choices=list(DTYPES), default="fp16", help="inference precision on the GPU")
    p.add_argument("--num_shards", type=int, default=1, help="split the work across N processes / GPUs")
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--estimate_storage", action="store_true", help="print the expected disk usage and exit")
    args = p.parse_args()

    if not args.output_dir and not args.estimate_storage:
        p.error("--output_dir is required")
    if args.image_frames is None:
        args.image_frames = "1fps" if args.mode == "full" else "middle"
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    video_paths, audio_paths, durations = read_manifests(args.manifest)
    log.info("read %d clips from %d manifest(s)", len(video_paths), len(args.manifest))
    n_images = sum(map(is_image, video_paths))
    if n_images and args.mode == "full":
        raise SystemExit(f"{n_images} manifest entries are images: use --mode image_only (V-JEPA needs video)")
    if args.estimate_storage:
        estimate_storage(durations, args)
        return

    # This process's shard, minus clips that are already done
    os.makedirs(args.output_dir, exist_ok=True)
    done = {f[:-4] for f in os.listdir(args.output_dir) if f.endswith(".pkl")}
    todo = [i for i in range(args.shard_id, len(video_paths), args.num_shards)
            if clip_key(audio_paths[i]) not in done]
    log.info("shard %d/%d: %d clips to do (%d already done overall)",
             args.shard_id, args.num_shards, len(todo), len(done))
    if not todo:
        return
    video_paths = [video_paths[i] for i in todo]
    audio_paths = [audio_paths[i] for i in todo]
    durations = durations[todo]
    if not durations.any():
        # No durations in the manifest: sort batches by audio file size instead (~ length for wav/flac),
        # so clips of similar length still end up together and padding stays low
        log.info("no durations in manifest: using audio file sizes to group batches")
        durations = np.array([os.path.getsize(p) for p in audio_paths], dtype=np.float64)

    # Models
    device, dtype = torch.device(args.device), DTYPES[args.dtype]
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    models, audio_processor = load_models(args, device, dtype)
    if "video" in models:
        config = models["video"].config
        args.vjepa_num_frames = args.vjepa_num_frames or config.frames_per_clip
        if args.vjepa_num_frames % config.tubelet_size:
            raise SystemExit(f"--vjepa_num_frames must be a multiple of {config.tubelet_size}")
    pool_factor = AUDIO_MODEL_FPS // args.audio_fps
    use_attention_mask = audio_processor is not None and audio_processor.return_attention_mask

    # Data
    dataset = ClipDataset(video_paths, audio_paths, args, audio_processor)
    batches = make_batches(durations, args.batch_size)
    loader = DataLoader(
        dataset,
        batch_sampler=batches,
        collate_fn=collate,
        num_workers=args.num_workers,
        worker_init_fn=worker_init,
        pin_memory=device.type == "cuda",
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    writer = AsyncWriter()
    failed_log = open(os.path.join(args.output_dir, f"failed_shard{args.shard_id}.txt"), "a")
    n_ok = n_failed = 0
    start = time.time()

    for batch in tqdm(loader, total=len(batches), unit="batch", smoothing=0.05):
        for key, error in batch["failed"]:
            failed_log.write(f"{key}\t{error}\n")
            n_failed += 1
        if not batch["keys"]:
            continue

        n = len(batch["keys"])
        out = {"video_embedding": [None] * n}

        with torch.inference_mode():
            image = batch["image"].to(device, non_blocking=True)
            out["image_embedding"] = to_numpy_fp16(
                encode_image(models["image"], image, batch["image_counts"], args.image_pool, dtype))

            if "video" in models:
                video = batch["video"].to(device, non_blocking=True)
                out["video_embedding"] = to_numpy_fp16(encode_video(models["video"], video, dtype))

            lens = batch["audio_lens"].to(device)
            if args.audio_model == "wavlm":
                wav = batch["audio"].to(device, dtype, non_blocking=True)
                feats = encode_wavlm(models["audio"], wav, lens, use_attention_mask, pool_factor)
            else:
                wav = batch["audio"].to(device, non_blocking=True)
                feats = encode_spear(models["audio"], wav, lens, dtype, pool_factor)
            out["audio_embedding"] = to_numpy_fp16(feats)
            # mean over the stored frames (same definition as add_pooled_audio.py), fp32 on the GPU
            out["audio_embedding_pooled"] = to_numpy_fp16([f.float().mean(0) for f in feats])

        for i, key in enumerate(batch["keys"]):
            record = {name: values[i] for name, values in out.items()}
            writer.submit(os.path.join(args.output_dir, key + ".pkl"), record)
        n_ok += n

    writer.close()
    failed_log.close()
    elapsed = time.time() - start
    log.info("done: %d clips in %.1f min (%.1f clips/s), %d failed (see failed_shard%d.txt)",
             n_ok, elapsed / 60, n_ok / max(elapsed, 1e-6), n_failed, args.shard_id)


if __name__ == "__main__":
    main()
