import os
import pickle

import fire
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

from extract_features import DATASETS
from siglip import SigLIP, evaluate, load_config


HOW2_SPLITS = {
    "how2-train": "train",
    "how2-val": "val",
    "how2-dev5": "dev5",
}
DEFAULT_DATASETS = ("how2-dev5", "msr-vtt", "youcook2")


class _ArrayDataset(torch.utils.data.Dataset):
    def __init__(self, video, image, audio):
        self.video, self.image, self.audio = video, image, audio

    def __len__(self):
        return len(self.video)

    def __getitem__(self, idx):
        return self.video[idx], self.image[idx], self.audio[idx]


def _load_how2_split(root: str, split: str):
    path = os.path.join(root, f"{split}_grouped_embeddings.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"How2 embeddings not found: {path}")
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["video_embeddings"], d["image_embeddings"], d["audio_embeddings"]


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
            d["image_embedding"] = d["image_embedding"].mean(axis=0)
        video.append(d["video_embedding"])
        image.append(d["image_embedding"])
        audio.append(d["audio_embedding"])

    return (
        np.stack(video).astype(np.float32),
        np.stack(image).astype(np.float32),
        np.stack(audio).astype(np.float32),
    )


def _load_dataset(name: str, cfg: dict):
    if name in HOW2_SPLITS:
        return _load_how2_split(cfg["root"], HOW2_SPLITS[name])
    if name in DATASETS:
        return _load_external_dataset(name)
    raise ValueError(
        f"Unknown dataset `{name}`. Available: "
        f"how2 splits {list(HOW2_SPLITS)} or external {sorted(DATASETS)}."
    )


def main(config_name: str, datasets=None):
    """Evaluate the best checkpoint for `config_name` on one or more datasets.

    By default, reports R@1/5/10, MRR and median rank on: how2-dev5 (How2 test),
    msr-vtt, youcook2.

    Example:
        uv run evaluate.py default
        uv run evaluate.py default --datasets msr-vtt,youcook2
        uv run evaluate.py default --datasets how2-val,how2-dev5

    `datasets` may be a comma-separated string or a list. Each name is either
    a How2 split (how2-train/how2-val/how2-dev5) or a key of extract_features.DATASETS.
    """
    if datasets is None:
        datasets = list(DEFAULT_DATASETS)
    elif isinstance(datasets, str):
        datasets = [s.strip() for s in datasets.split(",") if s.strip()]
    else:
        datasets = list(datasets)

    cfg = load_config(config_name)
    checkpoint_dir = cfg.get("checkpoint_dir", "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, config_name, "best.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = SigLIP(**ckpt.get("config", {})).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    def get_result(name):
        video, image, audio = _load_dataset(name, cfg)
        dataset = _ArrayDataset(
            torch.from_numpy(video),
            torch.from_numpy(image),
            torch.from_numpy(audio),
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=5000,
            shuffle=False,
            num_workers=0,
        )
        for metric_name, metric_value in evaluate(model, dataloader).items():
            if metric_name in {"r1", "r5", "r10"}:
                metric_value = 100 * metric_value
            yield {
                "name": name,
                "metric": metric_name,
                "value": metric_value,
            }

    results = [r for name in datasets for r in get_result(name)]
    df = pd.DataFrame(results)

    selected_metrics = ["r10", "mrr"]
    df = df[df["metric"].isin(selected_metrics)]
    df = df.pivot(index="name", columns="metric", values="value")
    df = pd.DataFrame(df[selected_metrics].unstack()).T
    print(config_name)
    print(df.to_csv(index=False, float_format="%.4f"))


if __name__ == "__main__":
    fire.Fire(main)
