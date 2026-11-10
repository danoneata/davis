"""Load How2 embeddings, either as originally split or through a tagged filelist.

The original splits are the grouped pickles themselves. A tagged split is defined by
`data/filelists/how2_<split>-<tag>.json` (JSONL despite the extension; `tag: final` is
the speaker-separated split) and cuts across them: `how2_train-final` holds segments
from all three of the original train, val and dev5. So nothing is regenerated. Each
record's `video_path` says which grouped pickle holds it, and the row comes from that
pickle's `<split>_grouped_ids.json` sidecar (see `make_grouped_ids.py`).
"""

import json
import os
import pickle
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from make_grouped_ids import load_grouped_ids

FILELISTS_DIR = Path(__file__).parent / "data" / "filelists"
SOURCE_SPLITS = ("train", "val", "dev5")
EMBEDDING_KEYS = ("video_embeddings", "image_embeddings", "audio_embeddings")

_DATASET_NAME = re.compile(r"how2-(train|val|dev5)(?:-(.+))?")
_VIDEO_PATH = re.compile(r"how2_(train|val|dev5)-video-segments/(.+)\.mp4")


def parse_dataset_name(name: str) -> tuple[str, str | None] | None:
    """`how2-dev5-final` -> `("dev5", "final")`, `how2-dev5` -> `("dev5", None)`.

    `None` when `name` is not a How2 split.
    """
    match = _DATASET_NAME.fullmatch(name)
    return match.groups() if match else None


def filelist_path(split: str, tag: str) -> Path:
    return FILELISTS_DIR / f"how2_{split}-{tag}.json"


@lru_cache(maxsize=None)
def read_records(split: str, tag: str) -> dict[str, dict]:
    """Segment id -> its filelist record (scores, text, speaker...), in filelist order."""
    path = filelist_path(split, tag)
    if not path.exists():
        raise FileNotFoundError(f"Filelist not found: {path}")

    records = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            match = _VIDEO_PATH.fullmatch(record["video_path"])
            if match is None:
                raise ValueError(f"Unexpected video_path `{record['video_path']}` in {path}.")
            records[match.group(2)] = record
    return records


@lru_cache(maxsize=None)
def read_filelist(split: str, tag: str) -> dict[str, str]:
    """Segment id -> the original split holding it, in filelist order."""
    return {
        segment_id: _VIDEO_PATH.fullmatch(record["video_path"]).group(1)
        for segment_id, record in read_records(split, tag).items()
    }


def load_grouped(root: str, split: str) -> dict:
    path = os.path.join(root, f"{split}_grouped_embeddings.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"How2 embeddings not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=None)
def _load_source(root: str, source: str) -> tuple[dict, dict[str, int]]:
    """A grouped pickle and its segment id -> row map; shared by the tagged splits."""
    grouped = load_grouped(root, source)
    ids = load_grouped_ids(root, source)
    n_rows = len(grouped["video_embeddings"])
    if len(ids) != n_rows:
        raise RuntimeError(
            f"{len(ids)} ids but {n_rows} rows for `{source}`: the sidecar is stale. "
            f"Run: uv run make_grouped_ids.py {source}"
        )
    return grouped, {segment_id: i for i, segment_id in enumerate(ids)}


def load_tagged_split(root: str, split: str, tag: str) -> tuple[dict, list[str]]:
    """Embedding arrays for a tagged split, and the segment id of each row.

    Segments without embeddings are dropped and counted.
    """
    segments = read_filelist(split, tag)
    chunks = {key: [] for key in EMBEDDING_KEYS}
    ids = []
    for source in SOURCE_SPLITS:
        members = [s for s, segment_source in segments.items() if segment_source == source]
        if not members:
            continue
        grouped, position = _load_source(root, source)
        found = [s for s in members if s in position]
        rows = np.array([position[s] for s in found], dtype=np.int64)
        for key in EMBEDDING_KEYS:
            chunks[key].append(grouped[key][rows])
        ids.extend(found)

    print(
        f"Split how2_{split}-{tag}: {len(ids)} segments"
        f" ({len(segments) - len(ids)} of {len(segments)} in the filelist have no embeddings)."
    )
    return {key: np.concatenate(chunks[key]) for key in EMBEDDING_KEYS}, ids


def load_split(root: str, split: str, tag: str | None = None) -> dict:
    """Embedding arrays for a split: tagged if `tag` is set, else the original one."""
    if tag is None:
        return load_grouped(root, split)
    return load_tagged_split(root, split, tag)[0]


def load_split_ids(root: str, split: str, tag: str | None = None) -> list[str]:
    """The segment id of each row of `load_split(root, split, tag)`."""
    if tag is None:
        return load_grouped_ids(root, split)
    return load_tagged_split(root, split, tag)[1]


@lru_cache(maxsize=None)
def _read_audio_features(path: str, key: str) -> tuple[np.ndarray, dict[str, int]]:
    with open(path, "rb") as f:
        features = pickle.load(f)
    if key not in features:
        raise KeyError(f"`{key}` not in {path}. Available: {sorted(features)}")
    return features[key], {segment_id: i for i, segment_id in enumerate(features["ids"])}


def with_audio_features(embeddings: dict, ids: list[str], spec: dict) -> dict:
    """`embeddings` with `audio_embeddings` replaced by external per-segment features.

    `spec` is the `audio_features` config key: `{path, key, l2_normalize}`. `path` is a
    pickle holding `ids` and one or more `(n, dim)` arrays (e.g. the CLIP text
    features written by `clip_baseline.py extract_text`); `key` picks the array. Rows
    are matched to `ids` -- the segment id of each row of `embeddings` -- by id, never
    by position, so the external file's order does not matter. A segment with no
    feature is an error rather than a silent drop.
    """
    features, row_of = _read_audio_features(str(spec["path"]), spec["key"])
    missing = [s for s in ids if s not in row_of]
    if missing:
        raise KeyError(
            f"{len(missing)} of {len(ids)} segments have no `{spec['key']}` in "
            f"{spec['path']} (e.g. {missing[:3]})."
        )
    audio = features[[row_of[s] for s in ids]].astype(np.float32)
    if spec.get("l2_normalize", False):
        audio = audio / np.linalg.norm(audio, axis=1, keepdims=True)
    return {**embeddings, "audio_embeddings": audio}


def load_split_with_features(root: str, split: str, tag: str | None, audio_features: dict | None) -> dict:
    """`load_split`, with the audio side swapped for `audio_features` when it is set."""
    if audio_features is None:
        return load_split(root, split, tag)
    if tag is None:
        embeddings, ids = load_grouped(root, split), load_grouped_ids(root, split)
    else:
        embeddings, ids = load_tagged_split(root, split, tag)  # one pass for both
    return with_audio_features(embeddings, ids, audio_features)
