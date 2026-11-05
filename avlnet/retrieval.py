"""Scoring a retrieval problem: which gallery items count as correct, and where they rank.

Shared by `evaluate.py`, which turns the ranks into corpus-level metrics, and
`show_results.py`, which shows the top-K list behind a single query -- so the two can
never disagree about what "the rank of the correct answer" means.
"""

import numpy as np
import torch
from tqdm import tqdm


# The two sides of a retrieval pair. "visual" is a video for video datasets and a
# still image for image datasets; it is the side `DualEncoder.video_adapter` encodes.
VISUAL, AUDIO = "visual", "audio"

# The retrieval directions, named query-to-gallery. `v2a` reads as "video->audio" on
# video datasets and "image->audio" on image ones -- the visual side is the same
# either way, which is why the codes do not spell out which it is.
V2A, A2V = "v2a", "a2v"
DIRECTIONS = (V2A, A2V)

# Queries scored per pass. The gallery stays resident on the device, so this bounds
# only the size of the score matrix.
SCORE_CHUNK = 256


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

    `encoded` carries `zvisual`, `zaudio` and `audio_to_visual`, the last mapping every
    audio row to the index of the visual item it belongs to. That mapping is the only
    place the many-to-one structure of image datasets lives: video datasets pass the
    identity, Flickr8k repeats each image once per caption. `visual_ids`/`audio_ids`
    are optional -- the viewer needs them to label results, the metrics do not.

    `positives` holds, for every query, the gallery indices that count as correct: one
    per query except for `v2a` on a multi-caption dataset, where every caption of the
    query image is a correct answer.
    """
    if direction not in DIRECTIONS:
        raise ValueError(f"Unknown direction `{direction}`. Available: {list(DIRECTIONS)}")

    audio_to_visual = np.asarray(encoded["audio_to_visual"])
    n_visual = len(encoded["zvisual"])
    if direction == A2V:
        query, gallery, query_kind = AUDIO, VISUAL, AUDIO
        positives = audio_to_visual[:, None]
    else:
        query, gallery, query_kind = VISUAL, AUDIO, VISUAL
        positives = group_by_visual(audio_to_visual, n_visual)
    return {
        "query_ids": encoded.get(f"{query}_ids"),
        "gallery_ids": encoded.get(f"{gallery}_ids"),
        "zquery": encoded[f"z{query}"],
        "zgallery": encoded[f"z{gallery}"],
        "positives": positives.astype(np.int64),
        "query_kind": query_kind,
        "gallery_kind": AUDIO if query_kind == VISUAL else VISUAL,
    }


@torch.no_grad()
def score_problem(
    problem: dict,
    topk: int = None,
    device=None,
    chunk: int = SCORE_CHUNK,
    desc: str = "Scoring",
) -> dict:
    """Rank each query's positives against the whole gallery.

    Returns `{'gt_rank': (n_query,)}`: the rank of the *best-scoring* correct answer,
    which for a single positive is just its own rank. With `topk` set it also returns
    `topk_idx`/`topk_score` of shape (n_query, min(topk, n_gallery)), computed in the
    same pass -- sharing the pass over the score matrix is the only reason the top-K
    lists and the ranks come out of one function.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    zquery = torch.as_tensor(problem["zquery"]).to(device)
    zgallery = torch.as_tensor(problem["zgallery"]).to(device)
    positives = torch.as_tensor(problem["positives"]).to(device)

    n = zquery.shape[0]
    gt_rank = np.zeros(n, dtype=np.int64)
    k = None if topk is None else min(topk, zgallery.shape[0])
    topk_idx = None if k is None else np.zeros((n, k), dtype=np.int64)
    topk_score = None if k is None else np.zeros((n, k), dtype=np.float32)

    for start in tqdm(range(0, n, chunk), desc=desc, leave=False):
        end = min(start + chunk, n)
        scores = zquery[start:end] @ zgallery.T  # (b, n_gallery)

        # Rank of the best correct answer: the number of gallery items scoring
        # strictly higher than it, plus one.
        best = scores.gather(1, positives[start:end]).max(dim=1).values
        gt_rank[start:end] = ((scores > best.unsqueeze(1)).sum(dim=1) + 1).cpu().numpy()

        if k is not None:
            vals, idx = torch.topk(scores, k=k, dim=1)
            topk_idx[start:end] = idx.cpu().numpy()
            topk_score[start:end] = vals.cpu().numpy()

    preds = {"gt_rank": gt_rank}
    if k is not None:
        preds["topk_idx"] = topk_idx
        preds["topk_score"] = topk_score
    return preds


def metrics_from_ranks(ranks: np.ndarray) -> dict:
    """R@1/5/10 (fractions in [0, 1]), MRR and median rank, from ground-truth ranks.

    Note `medr` averages the two middle values on an even count, where `train.evaluate`
    takes the lower one (`torch.median`); the two can differ by half a rank.
    """
    ranks = np.asarray(ranks, dtype=np.float64)
    return {
        "r1": float((ranks <= 1).mean()),
        "r5": float((ranks <= 5).mean()),
        "r10": float((ranks <= 10).mean()),
        "mrr": float((1.0 / ranks).mean()),
        "medr": float(np.median(ranks)),
    }
