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


def read_file(ids_path):
    """
    Read clip ids from either:
      - a plain text file with one id per line:   1s1TPdCUvhQ_0
      - a JSONL file with one record per line:    {"video_path": ".../-07Hlp9ZerU_0.mp4", ...}

    For JSON records the id is the filename stem of video_path
    (falling back to audio_path), e.g. "-07Hlp9ZerU_0".
    """
    ids = []
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

                path = record.get("video_path") or record.get("audio_path")
                if not path:
                    raise ValueError(
                        f"{ids_path}:{line_no}: record has no "
                        f"'video_path' or 'audio_path'"
                    )
                seg_id = PurePosixPath(path).stem   # strips dir + extension
            else:
                seg_id = line

            ids.append(seg_id)

    # Drop duplicates but keep the order, so a clip can't appear twice in the
    # dataset (and the sampler can't put it in a batch twice).
    return list(dict.fromkeys(ids))

def build_model(image_dim, 
                video_feat_dim, 
                audio_dim, 
                embd_dim,
                image_only
                ):
    
    if image_only: # for datasets like Flickr8k
        video_dim = image_dim
    else:
        video_dim = image_dim + video_feat_dim
    
    return Net(embd_dim=embd_dim, video_dim=video_dim, audio_dim=audio_dim)


def build_dataloader(ids_path,
                     embeddings_path,
                     normalize,
                     image_only,
                     batch_size,
                     num_workers,
                     shuffle,
                     drop_last,
                     split):
    if not ids_path:
        return None
    # Avoid circular import: import Feature_DataLoader locally
    from src.dataset import Feature_DataLoader

    dataset = Feature_DataLoader(
        ids_path=ids_path,
        embeddings_path=embeddings_path,
        split=split,
        normalize=normalize,
        image_only=image_only,
    )
    print(f"{ids_path}: {len(dataset)} clips")
    if len(dataset) == 0:
        raise ValueError(f"No embeddings found for any id in {ids_path}.")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=drop_last,
    )

def build_dataloader_negmine(
    ids_path,
    embeddings_path,
    normalize,
    image_only,
    batch_size,
    num_workers,
    shuffle,
    drop_last,
    split,
):
    if not ids_path:
        return None

    # Avoid circular import.
    from src.dataset_negmine import (
        Feature_DataLoader,
        VideoGroupedBatchSampler,
    )

    dataset = Feature_DataLoader(
        ids_path=ids_path,
        embeddings_path=embeddings_path,
        split=split,
        normalize=normalize,
        image_only=image_only,
    )

    print(f"{ids_path}: {len(dataset)} clips")

    if len(dataset) == 0:
        raise ValueError(
            f"No embeddings found for any id in {ids_path}."
        )

    # ================================================================
    # TRAINING
    #
    # We group complete videos into batches.
    #
    # Example with batch_size=32:
    #
    #   video A -> 10 clips
    #   video B -> 8 clips
    #   video C -> 7 clips
    #   video D -> 12 clips
    #
    # batches:
    #
    #   A + B + C = 25
    #   D         = 12
    #
    # D is NOT split between batches.
    # ================================================================
    if "train" in split:
        batch_sampler = VideoGroupedBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )

        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )

        print(
            f"[train] {len(batch_sampler)} grouped batches, "
            f"target batch size={batch_size}"
        )

        return loader

    # ================================================================
    # VALIDATION / TEST
    #
    # Keep normal DataLoader behavior. Evaluation does not need the
    # special grouped-negative sampling strategy.
    # ================================================================
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=True,
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
