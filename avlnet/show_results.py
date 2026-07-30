import hashlib
import os
import pickle
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from tqdm import tqdm

from evaluate import HOW2_SPLITS
from extract_features import DATASETS, list_videos
from siglip import SigLIP

SEGMENTS_ROOT = Path("/data/how2/clean_code/how2/segments")
EMBEDDINGS_ROOT = Path("/data/how2/clean_code/how2/embeddings")
CACHE_ROOT = Path(__file__).parent / "cache"
DEFAULT_CHECKPOINT = Path(__file__).parent / "checkpoints" / "run-00" / "best.pt"
# How2 splits plus the external evaluation datasets registered in extract_features.
DATASETS_ALL = ["how2-val", "how2-dev5", "how2-train", *sorted(DATASETS)]
MAX_TOPK = 20
BATCH_SIZE = 256


def embeddings_dir(dataset: str) -> Path:
    """Directory of per-segment embedding .pkl files, as written by feature extraction."""
    if dataset in HOW2_SPLITS:
        return EMBEDDINGS_ROOT / HOW2_SPLITS[dataset]
    if dataset in DATASETS:
        return DATASETS[dataset]["embeddings_path"]
    raise ValueError(f"Unknown dataset `{dataset}`. Available: {DATASETS_ALL}")


def videos_dir(dataset: str) -> Path:
    if dataset in HOW2_SPLITS:
        return SEGMENTS_ROOT / HOW2_SPLITS[dataset]
    return DATASETS[dataset]["videos_path"]


def cache_tag(dataset: str) -> str:
    """Filename component for on-disk caches.

    For How2 this is the bare split name, so caches written before multi-dataset
    support stay valid. External datasets get a hash of their embeddings dir, since
    `DATASETS` can be pointed at different feature sets for the same dataset.
    """
    if dataset in HOW2_SPLITS:
        return HOW2_SPLITS[dataset]
    digest = hashlib.md5(str(embeddings_dir(dataset)).encode()).hexdigest()[:6]
    return f"{dataset}-{digest}"


def build_pool_cache(dataset: str) -> Path:
    """Load all per-segment .pkl files for `dataset` in sorted order and cache to disk."""
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_ROOT / f"pool_{cache_tag(dataset)}.pkl"
    if cache_path.exists():
        return cache_path

    emb_dir = embeddings_dir(dataset)
    if not emb_dir.exists():
        raise FileNotFoundError(
            f"Embeddings directory does not exist: {emb_dir}. "
            f"Run: uv run extract_features.py {dataset}"
        )
    files = sorted(f for f in os.listdir(emb_dir) if f.endswith(".pkl"))

    ids, video, image, audio = [], [], [], []
    for name in tqdm(files, desc=f"Loading {dataset} embeddings"):
        try:
            with open(emb_dir / name, "rb") as f:
                d = pickle.load(f)
        except Exception:
            continue
        image_embedding = d["image_embedding"]
        if image_embedding.ndim == 2:
            # Some datasets store per-frame image features; pool as in evaluate.py.
            image_embedding = image_embedding.mean(axis=0)
        ids.append(name[:-4])
        video.append(d["video_embedding"])
        image.append(image_embedding)
        audio.append(d["audio_embedding"])

    payload = {
        "ids": ids,
        "video_embeddings": np.stack(video).astype(np.float32),
        "image_embeddings": np.stack(image).astype(np.float32),
        "audio_embeddings": np.stack(audio).astype(np.float32),
    }
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f)
    return cache_path


@st.cache_resource(show_spinner="Loading pool embeddings…")
def load_pool(dataset: str) -> dict:
    cache_path = build_pool_cache(dataset)
    with open(cache_path, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner="Loading model…")
def load_model(checkpoint_path: str) -> tuple[SigLIP, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    model = SigLIP(**config).to(device)
    state_dict = (
        ckpt["model_state_dict"]
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt
        else ckpt
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


@st.cache_resource(show_spinner="Encoding pool…")
def encode_pool(
    dataset: str,
    checkpoint_path: str,
    pool_size: int,
    seed: int,
) -> dict:
    """Return {'ids', 'zvideo', 'zaudio'} for the selected pool subset."""
    pool = load_pool(dataset)
    model, device = load_model(checkpoint_path)

    n = len(pool["ids"])
    if pool_size >= n:
        indices = np.arange(n)
    else:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(n, size=pool_size, replace=False))

    video = torch.from_numpy(pool["video_embeddings"][indices]).to(device)
    image = torch.from_numpy(pool["image_embeddings"][indices]).to(device)
    audio = torch.from_numpy(pool["audio_embeddings"][indices]).to(device)
    video = model.compose_video(video, image)

    with torch.no_grad():
        zvideo, zaudio = model(video, audio, normalize=True)

    ids = [pool["ids"][i] for i in indices]
    return {"ids": ids, "zvideo": zvideo.cpu(), "zaudio": zaudio.cpu()}


@st.cache_resource(show_spinner=False)
def video_index(dataset: str) -> dict:
    """Map segment id → video file for datasets whose extensions vary."""
    path = videos_dir(dataset)
    if not path.exists():
        return {}
    return {p.stem: p for p in list_videos(path)}


def video_path(dataset: str, segment_id: str) -> Path:
    # How2 segment dirs are large, so build the path directly instead of listing them.
    if dataset in HOW2_SPLITS:
        return videos_dir(dataset) / f"{segment_id}.mp4"
    default = videos_dir(dataset) / f"{segment_id}.mp4"
    return video_index(dataset).get(segment_id, default)


def _predictions_cache_key(
    dataset: str, checkpoint_path: str, pool_size: int, seed: int, direction: str
) -> str:
    ckpt_mtime = (
        os.path.getmtime(checkpoint_path) if os.path.exists(checkpoint_path) else 0
    )
    raw = f"{cache_tag(dataset)}|{checkpoint_path}|{ckpt_mtime}|{pool_size}|{seed}|{direction}|k{MAX_TOPK}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


@st.cache_resource(show_spinner="Precomputing retrievals…")
def precompute_predictions(
    dataset: str,
    checkpoint_path: str,
    pool_size: int,
    seed: int,
    direction: str,
) -> dict:
    """Compute top-MAX_TOPK results and ground-truth rank for every query.

    Returns {'topk_idx': (N, MAX_TOPK), 'topk_score': (N, MAX_TOPK), 'gt_rank': (N,)}.
    Persisted to disk under cache/preds_<dataset>_<hash>.pkl.
    """
    key = _predictions_cache_key(dataset, checkpoint_path, pool_size, seed, direction)
    disk_path = CACHE_ROOT / f"preds_{cache_tag(dataset)}_{key}.pkl"
    if disk_path.exists():
        with open(disk_path, "rb") as f:
            return pickle.load(f)

    encoded = encode_pool(dataset, checkpoint_path, pool_size, seed)
    if direction == "audio→video":
        query_mat, gallery = encoded["zaudio"], encoded["zvideo"]
    else:
        query_mat, gallery = encoded["zvideo"], encoded["zaudio"]

    n = query_mat.shape[0]
    k = min(MAX_TOPK, n)
    topk_idx = np.zeros((n, k), dtype=np.int64)
    topk_score = np.zeros((n, k), dtype=np.float32)
    gt_rank = np.zeros(n, dtype=np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gallery = gallery.to(device)
    query_mat = query_mat.to(device)

    with torch.no_grad():
        for start in tqdm(range(0, n, BATCH_SIZE), desc=f"Scoring {direction}"):
            end = min(start + BATCH_SIZE, n)
            scores = query_mat[start:end] @ gallery.T  # (b, N)
            vals, idx = torch.topk(scores, k=k, dim=1)
            topk_idx[start:end] = idx.cpu().numpy()
            topk_score[start:end] = vals.cpu().numpy()

            b = end - start
            gt_scores = scores[
                torch.arange(b, device=device), torch.arange(start, end, device=device)
            ]
            gt_rank[start:end] = (
                ((scores > gt_scores.unsqueeze(1)).sum(dim=1) + 1).cpu().numpy()
            )

    preds = {"topk_idx": topk_idx, "topk_score": topk_score, "gt_rank": gt_rank}
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    with open(disk_path, "wb") as f:
        pickle.dump(preds, f)
    return preds


def retrieve(preds: dict, query_idx: int, k: int) -> list:
    """Slice the precomputed top-K matrices for a single query."""
    idx = preds["topk_idx"][query_idx, :k]
    score = preds["topk_score"][query_idx, :k]
    return [(rank, int(i), float(s)) for rank, (i, s) in enumerate(zip(idx, score))]


def render_card(
    container,
    dataset: str,
    segment_id: str,
    rank: int,
    score: float,
    is_gt: bool,
):
    label = f"#{rank + 1} · score `{score:.3f}` · `{segment_id}`"
    if is_gt:
        label = "✅ " + label
    container.markdown(label)
    path = video_path(dataset, segment_id)
    if path.exists():
        container.video(str(path))
    else:
        container.warning(f"Missing video file: {path}")


def render_query_card(container, dataset: str, segment_id: str, badge: str):
    container.markdown(f"`{segment_id}` · {badge}")
    path = video_path(dataset, segment_id)
    if path.exists():
        container.video(str(path))
    else:
        container.warning(f"Missing video file: {path}")


def render_query_row(
    dataset: str,
    query_id: str,
    query_idx: int,
    ids: list,
    gt_rank: int,
    results: list,
    top_k: int,
    n_cols: int = 5,
):
    if gt_rank == 1:
        badge = f"✅ GT@1"
    elif gt_rank <= top_k:
        badge = f"⚠️  GT rank {gt_rank} (in top-{top_k})"
    else:
        badge = f"❌ GT rank {gt_rank} (outside top-{top_k})"

    col, *_ = st.columns(n_cols)
    col.markdown(f"### Query")
    render_query_card(col, dataset, query_id, badge)

    results = [
        (ids[pool_idx], rank, score, pool_idx == query_idx)
        for rank, pool_idx, score in results
    ]

    col, *_ = st.columns(n_cols)
    col.markdown(f"### Retrieved results")
    for row_start in range(0, len(results), n_cols):
        cols = st.columns(n_cols)
        for col, cell in zip(cols, results[row_start : row_start + n_cols]):
            sid, rank, score, is_gt = cell
            render_card(col, dataset, sid, rank, score, is_gt)


def select_query_indices(
    candidate_indices: list,
    gt_rank_all: np.ndarray,
    ids: list,
    n: int,
    ordering: str,
    seed: int,
) -> list:
    arr = np.array(candidate_indices, dtype=np.int64)
    if ordering == "random":
        rng = np.random.default_rng(seed)
        rng.shuffle(arr)
    elif ordering == "rank ↑":
        arr = arr[np.argsort(gt_rank_all[arr], kind="stable")]
    elif ordering == "rank ↓":
        arr = arr[np.argsort(-gt_rank_all[arr], kind="stable")]
    elif ordering == "id ↑":
        arr = arr[np.argsort([ids[i] for i in arr], kind="stable")]
    elif ordering == "id ↓":
        order = np.argsort([ids[i] for i in arr], kind="stable")[::-1]
        arr = arr[order]
    return arr[:n].tolist()


def get_models():
    return os.listdir("checkpoints")


def main():
    st.set_page_config(page_title="SigLIP retrieval viewer", layout="wide")
    st.title("SigLIP qualitative retrieval")

    with st.sidebar:
        st.header("Configuration")
        model_name = st.selectbox("Model", get_models(), index=0)

        dataset = st.selectbox("Dataset", DATASETS_ALL, index=0)
        direction = st.radio("Direction", ["audio→video", "video→audio"], index=0)

        try:
            pool = load_pool(dataset)
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()
        n_total = len(pool["ids"])
        st.caption(f"Dataset `{dataset}` has **{n_total}** segments.")

        preset = st.selectbox(
            "Pool size",
            ["full", "500", "2000", "10000", "custom"],
            index=0,
        )
        if preset == "full":
            pool_size = n_total
        elif preset == "custom":
            pool_size = st.number_input(
                "Custom pool size",
                min_value=10,
                max_value=n_total,
                value=min(2000, n_total),
                step=100,
            )
        else:
            pool_size = min(int(preset), n_total)

        top_k = st.slider("Top-K", min_value=1, max_value=MAX_TOPK, value=10)

        st.markdown("---")
        n_queries = st.slider(
            "Number of queries to show", min_value=1, max_value=100, value=20
        )
        ordering = st.selectbox(
            "Order queries by",
            ["random", "rank ↑", "rank ↓", "id ↑", "id ↓"],
            index=0,
        )
        only_errors = st.checkbox("Only show errors (GT not in top-K)", value=False)

    seed = 42
    checkpoint_path = f"checkpoints/{model_name}/best.pt"
    encoded = encode_pool(dataset, checkpoint_path, int(pool_size), int(seed))
    preds = precompute_predictions(
        dataset, checkpoint_path, int(pool_size), int(seed), direction
    )
    ids = encoded["ids"]
    gt_rank_all = preds["gt_rank"]
    n_errors = int((gt_rank_all > top_k).sum())
    acc = 1.0 - n_errors / len(ids)
    st.caption(
        f"Retrieval pool: **{len(ids)}** segments (dataset `{dataset}`, direction `{direction}`). "
        f"R@{top_k}: **{acc:.1%}** — {n_errors} errors (GT outside top-{top_k})."
    )

    if only_errors:
        error_mask = gt_rank_all > top_k
        candidate_indices = np.nonzero(error_mask)[0].tolist()
    else:
        candidate_indices = list(range(len(ids)))

    query_caption = (
        "Query modality: **audio track** of each segment. Retrieving videos."
        if direction == "audio→video"
        else "Query modality: **video** of each segment. Retrieving audios."
    )
    st.caption(query_caption)

    selected = select_query_indices(
        candidate_indices,
        gt_rank_all,
        ids,
        int(n_queries),
        ordering,
        seed,
    )
    st.caption(
        f"Showing **{len(selected)}** query(ies) ordered by *{ordering}* "
        f"from **{len(candidate_indices)}** candidate(s)."
    )

    for i, query_idx in enumerate(selected):
        query_id = ids[query_idx]
        results = retrieve(preds, query_idx, k=top_k)
        render_query_row(
            dataset,
            query_id,
            query_idx,
            ids,
            int(gt_rank_all[query_idx]),
            results,
            top_k,
        )
        if i < len(selected) - 1:
            st.divider()


if __name__ == "__main__":
    main()
