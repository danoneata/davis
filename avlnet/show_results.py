import hashlib
import os
import pickle
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from tqdm import tqdm

from evaluate import HOW2_SPLITS
from extract_features import DATASETS, list_videos, read_wav2capt
from siglip import SigLIP
from utils import read_file

SEGMENTS_ROOT = Path("/data/how2/clean_code/how2/segments")
EMBEDDINGS_ROOT = Path("/data/how2/clean_code/how2/embeddings")
CACHE_ROOT = Path(__file__).parent / "cache"
DEFAULT_CHECKPOINT = Path(__file__).parent / "checkpoints" / "run-00" / "best.pt"
# How2 splits plus the external evaluation datasets registered in extract_features.
DATASETS_ALL = ["how2-val", "how2-dev5", "how2-train", *sorted(DATASETS)]
MAX_TOPK = 20
BATCH_SIZE = 256
# The two sides of a retrieval pair. "visual" is a video for video datasets and a
# still image for image datasets; it is the side `SigLIP.video_adapter` encodes.
VISUAL, AUDIO = "visual", "audio"


def dataset_kind(dataset: str) -> str:
    """"video" (one video per item) or "image" (a still image with spoken captions)."""
    if dataset in HOW2_SPLITS:
        return "video"
    if dataset in DATASETS:
        return DATASETS[dataset].get("kind", "video")
    raise ValueError(f"Unknown dataset `{dataset}`. Available: {DATASETS_ALL}")


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


def _embedding_files(dataset: str) -> tuple:
    emb_dir = embeddings_dir(dataset)
    if not emb_dir.exists():
        raise FileNotFoundError(
            f"Embeddings directory does not exist: {emb_dir}. "
            f"Run: uv run extract_features.py {dataset}"
        )
    return emb_dir, sorted(f for f in os.listdir(emb_dir) if f.endswith(".pkl"))


def build_video_pool_cache(dataset: str, cache_path: Path) -> Path:
    emb_dir, files = _embedding_files(dataset)

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
        "kind": "video",
        "ids": ids,
        "video_embeddings": np.stack(video).astype(np.float32),
        "image_embeddings": np.stack(image).astype(np.float32),
        "audio_embeddings": np.stack(audio).astype(np.float32),
    }
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f)
    return cache_path


def build_image_pool_cache(dataset: str, cache_path: Path) -> Path:
    """Cache an image dataset: one image embedding and n_captions audio embeddings each.

    Files are read in `sorted` order, matching `evaluate._load_image_dataset`, so row
    indices here line up with the ones the reported metrics are computed over.
    """
    emb_dir, files = _embedding_files(dataset)

    ids, image, audio = [], [], []
    for name in tqdm(files, desc=f"Loading {dataset} embeddings"):
        with open(emb_dir / name, "rb") as f:
            d = pickle.load(f)
        ids.append(name[:-4])
        image.append(d["image_embedding"])
        audio.append(d["audio_embeddings"])

    payload = {
        "kind": "image",
        "ids": ids,
        "image_embeddings": np.stack(image).astype(np.float32),
        # (n, n_captions, dim)
        "audio_embeddings": np.stack(audio).astype(np.float32),
    }
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f)
    return cache_path


def build_pool_cache(dataset: str) -> Path:
    """Load all per-item .pkl files for `dataset` in sorted order and cache to disk."""
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    if dataset_kind(dataset) == "image":
        # Distinct filename: image payloads have a different shape from video ones,
        # so a stale cache of the other kind must never be picked up.
        cache_path = CACHE_ROOT / f"pool_{cache_tag(dataset)}_img.pkl"
        builder = build_image_pool_cache
    else:
        cache_path = CACHE_ROOT / f"pool_{cache_tag(dataset)}.pkl"
        builder = build_video_pool_cache
    if cache_path.exists():
        return cache_path
    return builder(dataset, cache_path)


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


def subsample(n: int, pool_size: int, seed: int) -> np.ndarray:
    if pool_size >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=pool_size, replace=False))


@st.cache_resource(show_spinner="Encoding pool…")
def encode_pool(
    dataset: str,
    checkpoint_path: str,
    pool_size: int,
    seed: int,
    caption: str,
) -> dict:
    """Encode the selected pool subset into the joint space.

    Returns the two sides of the retrieval problem plus `audio_to_visual`, which maps
    each audio row to the index of the visual item it belongs to. That mapping is the
    only place the many-to-one structure of image datasets lives: video datasets get
    the identity, Flickr8k gets each image repeated once per selected caption.

    `caption` is "all" or a caption index as a string; it is ignored for video datasets.
    """
    pool = load_pool(dataset)
    model, device = load_model(checkpoint_path)

    indices = subsample(len(pool["ids"]), pool_size, seed)
    visual_ids = [pool["ids"][i] for i in indices]
    image = torch.from_numpy(pool["image_embeddings"][indices]).to(device)

    if pool.get("kind") == "image":
        # No video track: a zero-width tensor, as in `evaluate._load_image_dataset`.
        video = torch.zeros(len(indices), 0, device=device)
        audio_np = pool["audio_embeddings"][indices]  # (m, n_captions, dim)
        caption_indices = (
            list(range(audio_np.shape[1])) if caption == "all" else [int(caption)]
        )
        audio_np = audio_np[:, caption_indices].reshape(-1, audio_np.shape[-1])
        audio_ids = [f"{sid}_{c}" for sid in visual_ids for c in caption_indices]
        audio_to_visual = np.repeat(np.arange(len(visual_ids)), len(caption_indices))
    else:
        video = torch.from_numpy(pool["video_embeddings"][indices]).to(device)
        audio_np = pool["audio_embeddings"][indices]
        audio_ids = list(visual_ids)
        audio_to_visual = np.arange(len(visual_ids))

    visual = model.compose_video(video, image)
    audio = torch.from_numpy(audio_np).to(device)

    with torch.no_grad():
        zvisual, zaudio = model(visual, audio, normalize=True)

    return {
        "visual_ids": visual_ids,
        "audio_ids": audio_ids,
        "zvisual": zvisual.cpu(),
        "zaudio": zaudio.cpu(),
        "audio_to_visual": audio_to_visual,
    }


def group_by_visual(audio_to_visual: np.ndarray, n_visual: int) -> np.ndarray:
    """Invert `audio_to_visual` into an (n_visual, n_captions) array of audio indices.

    Every visual item is assumed to have the same number of audio clips, which holds
    for both the identity mapping and Flickr8k's fixed five captions per image.
    """
    counts = np.bincount(audio_to_visual, minlength=n_visual)
    if not (counts == counts[0]).all():
        raise ValueError("Uneven number of audio clips per visual item.")
    order = np.argsort(audio_to_visual, kind="stable")
    return order.reshape(n_visual, counts[0])


def build_problem(encoded: dict, direction: str) -> dict:
    """Turn an encoded pool into a directed retrieval problem.

    `positives` holds, for every query, the gallery indices that count as correct: one
    per query except for image→audio on a multi-caption dataset, where every caption of
    the query image is a correct answer.
    """
    audio_to_visual = encoded["audio_to_visual"]
    n_visual = len(encoded["visual_ids"])
    if direction == "audio→video":
        query, gallery, query_kind = AUDIO, VISUAL, AUDIO
        positives = audio_to_visual[:, None]
    else:
        query, gallery, query_kind = VISUAL, AUDIO, VISUAL
        positives = group_by_visual(audio_to_visual, n_visual)
    return {
        "query_ids": encoded[f"{query}_ids"],
        "gallery_ids": encoded[f"{gallery}_ids"],
        "zquery": encoded[f"z{query}"],
        "zgallery": encoded[f"z{gallery}"],
        "positives": positives.astype(np.int64),
        "query_kind": query_kind,
        "gallery_kind": AUDIO if query_kind == VISUAL else VISUAL,
    }


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


def image_path(dataset: str, image_id: str) -> Path:
    return DATASETS[dataset]["images_path"] / f"{image_id}.jpg"


@st.cache_resource(show_spinner=False)
def wav_names(dataset: str) -> dict:
    """Image id → wav filename per caption index, from the dataset's wav2capt.txt."""
    return read_wav2capt(DATASETS[dataset]["wav2capt_path"])


def audio_path(dataset: str, image_id: str, caption_index: int) -> Path:
    """Path of the wav for one spoken caption.

    Goes through wav2capt.txt rather than assuming `<image_id>_<n>.wav`, since that
    file is what defines which wav is caption `n` -- the same mapping extraction used.
    """
    names = wav_names(dataset).get(image_id)
    audios_path = DATASETS[dataset]["audios_path"]
    if names is None or caption_index >= len(names):
        return audios_path / f"{image_id}_{caption_index}.wav"
    return audios_path / names[caption_index]


@st.cache_resource(show_spinner=False)
def load_captions(dataset: str) -> dict:
    """(image id, caption index) → transcript, from Flickr8k's token file."""
    path = DATASETS[dataset].get("captions_path")
    if path is None or not Path(path).exists():
        return {}
    captions = {}
    for line in read_file(path):
        key, _, text = line.partition("\t")
        image_name, _, caption_index = key.partition("#")
        if not caption_index.isdigit():
            continue
        captions[Path(image_name).stem, int(caption_index)] = text.strip()
    return captions


def split_audio_id(item_id: str) -> tuple[str, int]:
    """Split `<image_id>_<caption index>` -- image ids themselves contain underscores."""
    image_id, _, caption_index = item_id.rpartition("_")
    return image_id, int(caption_index)


def render_media(container, dataset: str, item_id: str, kind: str):
    """Render one item, as a video, an image or an audio player with its transcript."""
    if dataset_kind(dataset) == "image":
        if kind == AUDIO:
            image_id, caption_index = split_audio_id(item_id)
            path = audio_path(dataset, image_id, caption_index)
            if not path.exists():
                container.warning(f"Missing audio file: {path}")
                return
            container.audio(str(path))
            text = load_captions(dataset).get((image_id, caption_index))
            if text:
                container.caption(text)
            return
        path = image_path(dataset, item_id)
        if path.exists():
            container.image(str(path))
        else:
            container.warning(f"Missing image file: {path}")
        return

    # Video datasets: the audio track lives in the video file, so both sides render it.
    path = video_path(dataset, item_id)
    if path.exists():
        container.video(str(path))
    else:
        container.warning(f"Missing video file: {path}")


def render_card(
    container,
    dataset: str,
    item_id: str,
    kind: str,
    rank: int,
    score: float,
    is_gt: bool,
):
    label = f"#{rank + 1} · score `{score:.3f}` · `{item_id}`"
    if is_gt:
        label = "✅ " + label
    container.markdown(label)
    render_media(container, dataset, item_id, kind)


def render_query_card(container, dataset: str, item_id: str, kind: str, badge: str):
    container.markdown(f"`{item_id}` · {badge}")
    render_media(container, dataset, item_id, kind)


def render_query_row(
    dataset: str,
    problem: dict,
    query_idx: int,
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
    render_query_card(
        col, dataset, problem["query_ids"][query_idx], problem["query_kind"], badge
    )

    gallery_ids = problem["gallery_ids"]
    positives = set(problem["positives"][query_idx].tolist())
    results = [
        (gallery_ids[gallery_idx], rank, score, gallery_idx in positives)
        for rank, gallery_idx, score in results
    ]

    col, *_ = st.columns(n_cols)
    col.markdown(f"### Retrieved results")
    for row_start in range(0, len(results), n_cols):
        cols = st.columns(n_cols)
        for col, cell in zip(cols, results[row_start : row_start + n_cols]):
            item_id, rank, score, is_gt = cell
            render_card(
                col, dataset, item_id, problem["gallery_kind"], rank, score, is_gt
            )


def _predictions_cache_key(
    dataset: str,
    checkpoint_path: str,
    pool_size: int,
    seed: int,
    direction: str,
    caption: str,
) -> str:
    ckpt_mtime = (
        os.path.getmtime(checkpoint_path) if os.path.exists(checkpoint_path) else 0
    )
    # `v2` distinguishes these from caches written before ranking against a set of
    # positives, whose ground-truth ranks were diagonal-only.
    raw = (
        f"v2|{cache_tag(dataset)}|{checkpoint_path}|{ckpt_mtime}|{pool_size}|{seed}|"
        f"{direction}|{caption}|k{MAX_TOPK}"
    )
    return hashlib.md5(raw.encode()).hexdigest()[:12]


@st.cache_resource(show_spinner="Precomputing retrievals…")
def precompute_predictions(
    dataset: str,
    checkpoint_path: str,
    pool_size: int,
    seed: int,
    direction: str,
    caption: str,
) -> dict:
    """Compute top-MAX_TOPK results and ground-truth rank for every query.

    Returns {'topk_idx': (N, MAX_TOPK), 'topk_score': (N, MAX_TOPK), 'gt_rank': (N,)},
    where the indices are into the gallery. The ground-truth rank is the rank of the
    best-scoring correct answer, which for a single positive is just its own rank.
    Persisted to disk under cache/preds_<dataset>_<hash>.pkl.
    """
    key = _predictions_cache_key(
        dataset, checkpoint_path, pool_size, seed, direction, caption
    )
    disk_path = CACHE_ROOT / f"preds_{cache_tag(dataset)}_{key}.pkl"
    if disk_path.exists():
        with open(disk_path, "rb") as f:
            return pickle.load(f)

    problem = build_problem(
        encode_pool(dataset, checkpoint_path, pool_size, seed, caption), direction
    )
    query_mat, gallery = problem["zquery"], problem["zgallery"]

    n = query_mat.shape[0]
    k = min(MAX_TOPK, gallery.shape[0])
    topk_idx = np.zeros((n, k), dtype=np.int64)
    topk_score = np.zeros((n, k), dtype=np.float32)
    gt_rank = np.zeros(n, dtype=np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gallery = gallery.to(device)
    query_mat = query_mat.to(device)
    positives = torch.from_numpy(problem["positives"]).to(device)

    with torch.no_grad():
        for start in tqdm(range(0, n, BATCH_SIZE), desc=f"Scoring {direction}"):
            end = min(start + BATCH_SIZE, n)
            scores = query_mat[start:end] @ gallery.T  # (b, n_gallery)
            vals, idx = torch.topk(scores, k=k, dim=1)
            topk_idx[start:end] = idx.cpu().numpy()
            topk_score[start:end] = vals.cpu().numpy()

            # Rank of the best correct answer: the number of gallery items scoring
            # strictly higher than it, plus one.
            best = scores.gather(1, positives[start:end]).max(dim=1).values
            gt_rank[start:end] = (
                ((scores > best.unsqueeze(1)).sum(dim=1) + 1).cpu().numpy()
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


def direction_label(dataset: str, direction: str) -> str:
    if dataset_kind(dataset) == "image":
        return direction.replace("video", "image")
    return direction


def main():
    st.set_page_config(page_title="SigLIP retrieval viewer", layout="wide")
    st.title("SigLIP qualitative retrieval")

    with st.sidebar:
        st.header("Configuration")
        model_name = st.selectbox("Model", get_models(), index=0)

        dataset = st.selectbox("Dataset", DATASETS_ALL, index=0)
        is_image = dataset_kind(dataset) == "image"
        item_name = "images" if is_image else "segments"
        direction = st.radio(
            "Direction",
            ["audio→video", "video→audio"],
            index=0,
            format_func=lambda d: direction_label(dataset, d),
        )

        try:
            pool = load_pool(dataset)
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()
        n_total = len(pool["ids"])
        st.caption(f"Dataset `{dataset}` has **{n_total}** {item_name}.")

        caption = "all"
        if is_image:
            n_captions = pool["audio_embeddings"].shape[1]
            caption = st.selectbox(
                "Caption",
                ["all", *(str(c) for c in range(n_captions))],
                index=0,
                format_func=lambda c: (
                    f"all {n_captions} captions" if c == "all" else f"caption #{c}"
                ),
                help=(
                    "A single caption index reproduces one of the trials that "
                    "`evaluate.py` averages over; `all` uses every caption at once."
                ),
            )

        preset = st.selectbox(
            "Pool size",
            ["full", "500", "2000", "10000", "custom"],
            index=0,
        )
        if preset == "full":
            pool_size = n_total
        elif preset == "custom":
            pool_size = st.number_input(
                f"Custom pool size ({item_name})",
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
    model, _ = load_model(checkpoint_path)
    if is_image and model.use_video:
        st.error(
            f"Dataset `{dataset}` has no video track, but `{model_name}` was trained "
            f"with use_video=true. Pick a use_video=false checkpoint instead "
            f"(e.g. run-01-novideo)."
        )
        st.stop()

    problem = build_problem(
        encode_pool(dataset, checkpoint_path, int(pool_size), int(seed), caption),
        direction,
    )
    preds = precompute_predictions(
        dataset, checkpoint_path, int(pool_size), int(seed), direction, caption
    )
    query_ids = problem["query_ids"]
    gt_rank_all = preds["gt_rank"]
    n_errors = int((gt_rank_all > top_k).sum())
    acc = 1.0 - n_errors / len(query_ids)
    st.caption(
        f"**{len(query_ids)}** queries against a gallery of "
        f"**{len(problem['gallery_ids'])}** (dataset `{dataset}`, direction "
        f"`{direction_label(dataset, direction)}`). "
        f"R@{top_k}: **{acc:.1%}** — {n_errors} errors (GT outside top-{top_k})."
    )

    if only_errors:
        error_mask = gt_rank_all > top_k
        candidate_indices = np.nonzero(error_mask)[0].tolist()
    else:
        candidate_indices = list(range(len(query_ids)))

    visual_name = "image" if is_image else "video"
    modality = {VISUAL: visual_name, AUDIO: "audio"}
    st.caption(
        f"Query modality: **{modality[problem['query_kind']]}**. "
        f"Retrieving {modality[problem['gallery_kind']]}s."
    )

    selected = select_query_indices(
        candidate_indices,
        gt_rank_all,
        query_ids,
        int(n_queries),
        ordering,
        seed,
    )
    st.caption(
        f"Showing **{len(selected)}** query(ies) ordered by *{ordering}* "
        f"from **{len(candidate_indices)}** candidate(s)."
    )

    for i, query_idx in enumerate(selected):
        results = retrieve(preds, query_idx, k=top_k)
        render_query_row(
            dataset,
            problem,
            query_idx,
            int(gt_rank_all[query_idx]),
            results,
            top_k,
        )
        if i < len(selected) - 1:
            st.divider()


if __name__ == "__main__":
    main()
