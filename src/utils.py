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

def read_file(path):
    """Read a text file of ids, one per line.

    This is the same helper referenced by ``extract_features.py`` (used there to
    select which videos in a raw video folder to process) and by
    ``dataset.py`` here (used to pick which pre-extracted embeddings to load for
    train/val/test).

    Blank lines and lines starting with '#' are ignored, surrounding whitespace is
    stripped, and duplicate ids are dropped while preserving first-seen order.

    Returns:
        A list of id strings. Membership checks (``seg_id in read_file(...)``)
        work fine on the returned list.
    """
    path = Path(path)
    seen = set()
    ids = []
    with open(path, "r") as f:
        for line in f:
            seg_id = line.strip()
            if not seg_id or seg_id.startswith("#"):
                continue
            if seg_id not in seen:
                seen.add(seg_id)
                ids.append(seg_id)
    return ids

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
