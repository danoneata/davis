"""Small shared utilities."""
from __future__ import absolute_import, division, unicode_literals, print_function

from pathlib import Path
import os
from src.model import Net
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


def load_config(configs_dir: Path, config_name: str) -> dict:
    path = configs_dir / f"{config_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Config `{config_name}` not found at {path}. "
            f"Available: {sorted(p.stem for p in configs_dir.glob('*.yaml'))}"
        )
    cfg = OmegaConf.load(path)
    return cfg

import json
from pathlib import PurePosixPath


def read_manifest(ids_path):
    """
    Read clip records from either:
      - a plain text file with one id per line:   1s1TPdCUvhQ_0
      - a JSONL file with one record per line:    {"video_path": ".../-07Hlp9ZerU_0.mp4", ...}

    Returns a list of dicts with keys "id", "original_video_id" and
    "speaker_id". For JSON records the id is the filename stem of audio_path
    (falling back to video_path), e.g. "-07Hlp9ZerU_0", so an image with
    several audios (Flickr8k <id>_0..4.wav) gives one record per audio. When
    "original_video_id" is missing it is derived from the id by stripping the
    segment suffix (001evy9D-mw_10 -> 001evy9D-mw); an explicit null
    "original_video_id" and a missing/null "speaker_id" are None.
    """
    records = {}
    with open(ids_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            if line.startswith("{"):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"{ids_path}:{line_no}: invalid JSON line ({e})"
                    )

                path = record.get("audio_path") or record.get("video_path")
                if not path:
                    raise ValueError(
                        f"{ids_path}:{line_no}: record has no "
                        f"'video_path' or 'audio_path'"
                    )
                seg_id = PurePosixPath(path).stem   # strips dir + extension
            else:
                seg_id = line
                record = {}

            # Drop duplicates but keep the order, so a clip can't appear twice
            # in the dataset (and the sampler can't put it in a batch twice).
            if seg_id in records:
                continue
            records[seg_id] = {
                "id": seg_id,
                # Absent key -> derived from the id; explicit null -> None
                # (all null ids are grouped together by the dataset).
                "original_video_id": (
                    record["original_video_id"] if "original_video_id" in record
                    else seg_id.rsplit("_", 1)[0]
                ),
                "speaker_id": record.get("speaker_id"),
            }

    return list(records.values())


def read_file(ids_path):
    """Read only the clip ids of a manifest (see read_manifest)."""
    return [r["id"] for r in read_manifest(ids_path)]

def build_model(image_dim, 
                video_feat_dim, 
                audio_dim, 
                embd_dim,
                image_only,
                audio_pool=True,
                normalize=True,
                ):
    
    if image_only: # for datasets like Flickr8k
        video_dim = image_dim
    else:
        video_dim = image_dim + video_feat_dim
    
    return Net(embd_dim=embd_dim, video_dim=video_dim, audio_dim=audio_dim,
               audio_pool=audio_pool, normalize=normalize)


def build_dataloader(
    ids_path,
    embeddings_path,
    embeddings_path_split,
    normalize,
    image_only,
    batch_size,
    num_workers,
    shuffle,
    drop_last,
    split,
    batch_mode="random",
    seed=0,
    audio_pool=True,
    io_threads=8,
    prefetch_factor=4,
    mining_size=-1,
):
    """
    batch_mode selects how batches are formed:
      - "random":  plain DataLoader batches (shuffled if shuffle=True).
      - "video":   clips from the same original_video_id are packed together.
      - "speaker": clips from the same speaker_id are packed together.
    mining_size caps how many clips of one group go into a single batch
    (-1 = no cap); the rest of the group lands in other batches, nothing is dropped.
    Evaluation loaders should use "random" with shuffle=False.
    """
    if not ids_path:
        return None

    # Avoid circular import.
    from src.dataset import BATCH_MODES, Feature_DataLoader, GroupedBatchSampler

    if batch_mode not in BATCH_MODES:
        raise ValueError(f"batch_mode must be one of {BATCH_MODES}, got '{batch_mode}'.")

    dataset = Feature_DataLoader(
        ids_path=ids_path,
        embeddings_path=embeddings_path,
        embeddings_path_split=embeddings_path_split,
        split=split,
        normalize=normalize,
        image_only=image_only,
        audio_pool=audio_pool,
        io_threads=io_threads,
    )

    # Keep workers alive across epochs (no re-spawn after every epoch / val)
    # and let each one queue more batches ahead.
    worker_kwargs = (
        dict(persistent_workers=True, prefetch_factor=prefetch_factor)
        if num_workers > 0 else {}
    )

    print(f"{ids_path}: {len(dataset)} clips")

    if len(dataset) == 0:
        raise ValueError(
            f"No embeddings found for any id in {ids_path}."
        )

    if batch_mode == "random":
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=True,
            collate_fn=dataset.custom_collate,
            **worker_kwargs,
        )

    batch_sampler = GroupedBatchSampler(
        groups=dataset.group_indices(batch_mode),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        seed=seed,
        mining_size=mining_size,
    )

    print(
        f"[{split}] {len(batch_sampler)} {batch_mode}-grouped batches, "
        f"target batch size={batch_size}, mining_size={mining_size}"
    )

    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=dataset.custom_collate,
        **worker_kwargs,
    )


def is_better_metric(value, best, mode="max", min_delta=0.0):
    if best is None:
        return True
    if mode == "max":
        return value > best + min_delta
    return value < best - min_delta


def sanitize_metric_name(metric_name):
    return "".join(c if c.isalnum() else "_" for c in metric_name)


def checkpoint_filename(epoch, metric_name, metric_value):
    metric_label = sanitize_metric_name(metric_name)
    metric_pct = metric_value * 100.0
    return f"e{epoch}_{metric_label}={metric_pct:.2f}pct.pth"


def update_top_checkpoints(net, run_checkpoint_dir, epoch, metric_value, metric_name, top_checkpoints, top_n, mode):
    if top_n <= 0:
        return top_checkpoints

    previous_top = list(top_checkpoints)
    top_checkpoints.append({"epoch": epoch, "metric": metric_value, "path": None})
    reverse = mode == "max"
    top_checkpoints.sort(key=lambda e: e["metric"], reverse=reverse)

    if len(top_checkpoints) > top_n:
        top_checkpoints = top_checkpoints[:top_n]

    new_top = []
    retained_paths = set()
    for entry in top_checkpoints:
        desired_path = os.path.join(run_checkpoint_dir, checkpoint_filename(entry["epoch"], metric_name, entry["metric"]))
        if entry.get("path") and entry["path"] != desired_path:
            if os.path.isfile(entry["path"]):
                os.replace(entry["path"], desired_path)
        if entry.get("path") is None or not os.path.isfile(desired_path):
            net.module.save_checkpoint(desired_path)
        entry["path"] = desired_path
        retained_paths.add(desired_path)
        new_top.append(entry)

    for old in previous_top:
        old_path = old.get("path")
        if old_path and old_path not in retained_paths and os.path.isfile(old_path):
            os.remove(old_path)

    return new_top
