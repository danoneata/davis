import os
import pickle
import sys

import fire
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

from extract_features import DATASETS
from how2_splits import load_split, parse_dataset_name
from retrieval import (
    DIRECTIONS,
    V2A,
    build_problem,
    metrics_from_ranks,
    score_problem,
)
from train import DualEncoder, load_config


HOW2_SPLITS = {
    "how2-train": "train",
    "how2-val": "val",
    "how2-dev5": "dev5",
}
DEFAULT_DATASETS = ("how2-dev5", "msr-vtt", "youcook2")
# Rows pushed through the adapters per forward pass.
ENCODE_CHUNK = 4096


def default_datasets(cfg: dict) -> list[str]:
    """`DEFAULT_DATASETS`, with How2 test being the test split the run was trained for."""
    tag = cfg.get("split_tag")
    if tag is None:
        return list(DEFAULT_DATASETS)
    return [f"{name}-{tag}" if parse_dataset_name(name) else name for name in DEFAULT_DATASETS]


def warn_on_split_mismatch(datasets: list[str], cfg: dict):
    """Original and tagged splits overlap, so crossing them scores on training data."""
    tag = cfg.get("split_tag")
    for name in datasets:
        how2 = parse_dataset_name(name)
        if how2 is not None and how2[1] != tag:
            print(
                f"WARNING: `{name}` is not from the split this model was trained on "
                f"(split_tag={tag}); it may contain training segments.",
                file=sys.stderr,
            )


def _load_external_dataset(name: str):
    """Load the per-video embeddings written by `extract_features.py <name>`.

    One .pkl per video, so we stack them into (n, dim) arrays to match the layout
    of the grouped How2 pickles.
    """
    emb_dir = DATASETS[name]["embeddings_path"]
    if not emb_dir.exists():
        raise FileNotFoundError(
            f"Embeddings directory does not exist: {emb_dir}. "
            f"Run: uv run extract_features.py {name}"
        )

    paths = sorted(emb_dir.glob("*.pkl"))
    if not paths:
        raise FileNotFoundError(f"No .pkl files in {emb_dir}")

    video, image, audio = [], [], []
    for path in tqdm(paths, desc=f"Loading {name}"):
        with open(path, "rb") as f:
            d = pickle.load(f)
        if d["image_embedding"].ndim == 2:
            # Some feature sets store the raw per-frame DINOv2 embeddings. Pool them
            # with `max`, not `mean`: both extraction pipelines the model was trained
            # on max over the 1 fps frames (`extract_features.get_dinov2_embeddings_1fps`
            # and How2's own cache_embeddings.py), so mean pooling would feed it a
            # statistic it has never seen.
            d["image_embedding"] = d["image_embedding"].max(axis=0)
        video.append(d["video_embedding"])
        image.append(d["image_embedding"])
        audio.append(d["audio_embedding"])

    return (
        np.stack(video).astype(np.float32),
        np.stack(image).astype(np.float32),
        np.stack(audio).astype(np.float32),
    )


def _load_image_dataset(name: str):
    """Load the per-image embeddings written by `extract_features.py <name>`.

    These datasets pair a still image with several spoken captions, so the audio side
    is longer than the visual one: the clips are flattened into a single gallery and
    `audio_to_visual` records which image each belongs to. Every caption is then a
    correct answer for its image, rather than one caption being picked arbitrarily.

    There is no video track, so the video array is zero-width: `use_video=False`
    models ignore it, and `use_video=True` models are rejected up front.
    """
    emb_dir = DATASETS[name]["embeddings_path"]
    if not emb_dir.exists():
        raise FileNotFoundError(
            f"Embeddings directory does not exist: {emb_dir}. "
            f"Run: uv run extract_features.py {name}"
        )

    paths = sorted(emb_dir.glob("*.pkl"))
    if not paths:
        raise FileNotFoundError(f"No .pkl files in {emb_dir}")

    image, audio = [], []
    for path in tqdm(paths, desc=f"Loading {name}"):
        with open(path, "rb") as f:
            d = pickle.load(f)
        image.append(d["image_embedding"])
        audio.append(d["audio_embeddings"])

    image = np.stack(image).astype(np.float32)
    audio = np.stack(audio).astype(np.float32)  # (n, n_captions, dim)
    video = np.zeros((len(image), 0), dtype=np.float32)
    n, n_captions = audio.shape[:2]
    return {
        "video": video,
        "image": image,
        "audio": audio.reshape(n * n_captions, audio.shape[-1]),
        "audio_to_visual": np.repeat(np.arange(n), n_captions),
    }


def _load_dataset(name: str, cfg: dict) -> dict:
    """Return the retrieval pool for `name`: {video, image, audio, audio_to_visual}.

    Video datasets pair one clip with one visual item, so `audio_to_visual` is the
    identity; see `_load_image_dataset` for the multi-caption case.
    """
    how2 = parse_dataset_name(name)
    if how2 is not None:
        d = load_split(cfg["root"], *how2)
        arrays = d["video_embeddings"], d["image_embeddings"], d["audio_embeddings"]
    elif name in DATASETS:
        if DATASETS[name].get("kind", "video") == "image":
            return _load_image_dataset(name)
        arrays = _load_external_dataset(name)
    else:
        raise ValueError(
            f"Unknown dataset `{name}`. Available: "
            f"how2 splits {list(HOW2_SPLITS)} (optionally `-<tag>`, e.g. "
            f"how2-dev5-final) or external {sorted(DATASETS)}."
        )
    video, image, audio = arrays
    return {
        "video": video,
        "image": image,
        "audio": audio,
        "audio_to_visual": np.arange(len(image)),
    }


@torch.no_grad()
def _encode_pool(model, pool: dict, device) -> dict:
    """Project a pool into the joint space, L2-normalised and chunked.

    The two sides are encoded separately because they no longer have the same length
    on multi-caption datasets. This inlines what `DualEncoder.forward` does to each
    side (train.py) rather than calling it, since forward takes the two together.
    """
    normalize = torch.nn.functional.normalize
    video = torch.from_numpy(pool["video"])
    image = torch.from_numpy(pool["image"])
    audio = torch.from_numpy(pool["audio"])

    zvisual = []
    for start in range(0, len(image), ENCODE_CHUNK):
        sl = slice(start, start + ENCODE_CHUNK)
        visual = model.compose_video(video[sl].to(device), image[sl].to(device))
        zvisual.append(normalize(model.video_adapter(visual), p=2, dim=1).cpu())

    zaudio = []
    for start in range(0, len(audio), ENCODE_CHUNK):
        chunk = audio[start : start + ENCODE_CHUNK].to(device)
        zaudio.append(normalize(model.audio_adapter(chunk), p=2, dim=1).cpu())

    return {
        "zvisual": torch.cat(zvisual),
        "zaudio": torch.cat(zaudio),
        "audio_to_visual": pool["audio_to_visual"],
    }


def main(config_name: str, datasets=None, direction: str = V2A):
    """Evaluate the best checkpoint for `config_name` on one or more datasets.

    By default, reports R@1/5/10, MRR and median rank on: how2-dev5 (How2 test, or
    how2-dev5-<tag> for a config with a `split_tag`), msr-vtt, youcook2.

    Example:
        uv run evaluate.py default
        uv run evaluate.py default --datasets msr-vtt,youcook2
        uv run evaluate.py default --datasets how2-val,how2-dev5
        uv run evaluate.py run-01-novideo --datasets flickr8k --direction both

    `datasets` may be a comma-separated string or a list. Each name is either
    a How2 split (how2-train/how2-val/how2-dev5, optionally suffixed with a filelist
    tag as in how2-dev5-final) or a key of extract_features.DATASETS.

    `direction` is `v2a` (visual queries, audio gallery -- the default and what
    training reports), `a2v`, or `both`. The two are not transposes of each other on
    image datasets: `v2a` queries 1,000 Flickr8k images against all 5,000 clips with
    all five captions of the query image counting as correct, while `a2v` queries each
    of the 5,000 clips against the 1,000 images with a single correct answer.
    """
    cfg = load_config(config_name)

    if datasets is None:
        datasets = default_datasets(cfg)
    elif isinstance(datasets, str):
        datasets = [s.strip() for s in datasets.split(",") if s.strip()]
    else:
        datasets = list(datasets)

    if direction == "both":
        directions = list(DIRECTIONS)
    elif direction in DIRECTIONS:
        directions = [direction]
    else:
        raise ValueError(
            f"Unknown direction `{direction}`. Available: {[*DIRECTIONS, 'both']}."
        )

    warn_on_split_mismatch(datasets, cfg)

    checkpoint_dir = cfg.get("checkpoint_dir", "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, config_name, "best.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = DualEncoder(**ckpt.get("config", {})).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    def get_result(name):
        if (
            name in DATASETS
            and DATASETS[name].get("kind", "video") == "image"
            and model.use_video
        ):
            raise ValueError(
                f"Dataset `{name}` has no video track, but `{config_name}` was "
                f"trained with use_video=true. Evaluate a use_video=false model "
                f"on it instead (e.g. run-01-novideo)."
            )

        # Encoded once and scored per direction: the projections do not depend on
        # which side asks the questions.
        encoded = _encode_pool(model, _load_dataset(name, cfg), device)
        for direction in directions:
            problem = build_problem(encoded, direction)
            preds = score_problem(
                problem, device=device, desc=f"Scoring {name} {direction}"
            )
            for metric_name, value in metrics_from_ranks(preds["gt_rank"]).items():
                if metric_name in {"r1", "r5", "r10"}:
                    value = 100 * value
                yield {
                    "name": name,
                    "direction": direction,
                    "metric": metric_name,
                    "value": value,
                }

    results = {
        (r["metric"], r["name"], r["direction"]): r["value"]
        for name in datasets
        for r in get_result(name)
    }

    selected_metrics = ["r10", "mrr"]
    keys = [
        (metric, name, d)
        for metric in selected_metrics
        for name in datasets
        for d in directions
    ]
    # With a single direction the header stays the two-level metric/dataset one that
    # these tables have always had, so old and new output line up column for column.
    columns = pd.MultiIndex.from_tuples(
        keys if len(directions) > 1 else [k[:2] for k in keys]
    )
    df = pd.DataFrame([[results[k] for k in keys]], columns=columns)
    print(config_name)
    print(df.to_csv(index=False, float_format="%.4f"))


if __name__ == "__main__":
    fire.Fire(main)
