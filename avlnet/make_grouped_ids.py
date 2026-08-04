"""Recover the segment ids behind a `<split>_grouped_embeddings.pkl`.

The grouping script (`/data/how2/clean_code/segments/group_embeddings.py`) appended
per-segment embeddings in `os.listdir` order and never saved their ids, so the
grouped arrays cannot be joined against anything keyed by segment id -- which is
what filtering the training set needs.

`os.listdir` still reproduces that order today, so this script replays it and freezes
the result into `<split>_grouped_ids.json`. The ordering is only assumed, never
trusted: before writing, a random sample of per-segment embeddings is compared
against the corresponding rows of the grouped array, and any mismatch aborts. Run it
once per split; re-run it only if the grouped pickle is regenerated.

    uv run make_grouped_ids.py train
"""

import json
import os
import pickle
from pathlib import Path

import fire
import numpy as np

EMBEDDINGS_ROOT = Path("/data/how2/clean_code/how2/embeddings")
N_SPOT_CHECKS = 200


def grouped_ids_path(root: Path, split: str) -> Path:
    return Path(root) / f"{split}_grouped_ids.json"


def load_grouped_ids(root: str, split: str) -> list[str]:
    """Segment ids, one per row of `<split>_grouped_embeddings.pkl`."""
    path = grouped_ids_path(Path(root), split)
    if not path.exists():
        raise FileNotFoundError(
            f"Segment ids not found at {path}. They are needed to map a filter onto "
            f"the grouped embeddings. Run: uv run make_grouped_ids.py {split}"
        )
    with open(path) as f:
        return json.load(f)


def main(split: str = "train", root: str = str(EMBEDDINGS_ROOT), seed: int = 0):
    """Write `<root>/<split>_grouped_ids.json` for `<root>/<split>_grouped_embeddings.pkl`."""
    root = Path(root)
    segments_dir = root / split
    grouped_path = root / f"{split}_grouped_embeddings.pkl"
    if not segments_dir.exists():
        raise FileNotFoundError(f"Per-segment embeddings not found: {segments_dir}")
    if not grouped_path.exists():
        raise FileNotFoundError(f"Grouped embeddings not found: {grouped_path}")

    names = os.listdir(segments_dir)
    with open(grouped_path, "rb") as f:
        grouped = pickle.load(f)
    video, audio = grouped["video_embeddings"], grouped["audio_embeddings"]

    if len(names) != len(video):
        raise RuntimeError(
            f"{len(names)} per-segment files but {len(video)} rows in {grouped_path.name}. "
            f"The listdir order cannot be replayed -- the grouped embeddings must be "
            f"regenerated with the ids saved alongside them."
        )

    rng = np.random.default_rng(seed)
    n_checks = min(N_SPOT_CHECKS, len(names))
    for i in rng.choice(len(names), size=n_checks, replace=False):
        with open(segments_dir / names[i], "rb") as f:
            d = pickle.load(f)
        video_ok = np.allclose(d["video_embedding"], video[i], atol=1e-5)
        audio_ok = np.allclose(d["audio_embedding"], audio[i], atol=1e-5)
        if not (video_ok and audio_ok):
            raise RuntimeError(
                f"Row {i} of {grouped_path.name} does not match {names[i]} "
                f"(video_ok={video_ok}, audio_ok={audio_ok}). The listdir order no "
                f"longer reproduces the grouping; regenerate the grouped embeddings "
                f"with the ids saved alongside them."
            )

    ids = [Path(name).stem for name in names]
    out_path = grouped_ids_path(root, split)
    with open(out_path, "w") as f:
        json.dump(ids, f)
    print(f"Wrote {len(ids)} ids to {out_path} ({n_checks} spot-checks passed).")


if __name__ == "__main__":
    fire.Fire(main)
