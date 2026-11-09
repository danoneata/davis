"""Retrieval metrics (R@1/5/10, median rank).

Adapted from the original AVLnet metrics.py (Rouditchenko et al., Interspeech 2021).
The MSR-VTT-specific "count missing test clips as mistakes" branch was dropped since
it was specific to that benchmark's fixed 1000-clip test protocol; test_set_size is
now always just the number of rows actually present.
"""
from __future__ import absolute_import, division, unicode_literals, print_function

import numpy as np
import torch as th


def compute_metrics(x, eval_lang_retrieval=False):
    """
    Args:
        x: [n_audio, n_video] similarity matrix (row i, col i = matching pair).
        eval_lang_retrieval: if True, evaluate video->audio retrieval instead of
            the default audio->video retrieval (transposes x first).
    """
    if eval_lang_retrieval:
        print("Retrieving audio given input video clips")
        x = x.T
    else:
        print("Retrieving video clips given input audio")

    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]

    metrics = {}
    test_set_size = x.shape[0]
    metrics['acc'] = float(np.sum(ind <= 0)) / test_set_size
    metrics['R@5'] = float(np.sum(ind < 5)) / test_set_size
    metrics['R@10'] = float(np.sum(ind < 10)) / test_set_size
    metrics['MR'] = np.median(ind) + 1
    return metrics


def print_computed_metrics(metrics):
    r1 = metrics['acc']
    r5 = metrics['R@5']
    r10 = metrics['R@10']
    mr = metrics['MR']
    print('R@1: {:.4f} - R@5: {:.4f} - R@10: {:.4f} - Median R: {}'.format(r1, r5, r10, mr))


def rank_metrics(ranks, prefix=""):
    """R@1/5/10, MRR and median rank from a 1D tensor of 1-based ranks."""
    ranks = ranks.float()
    return {
        f"{prefix}acc": (ranks <= 1).float().mean().item(),
        f"{prefix}R@5": (ranks <= 5).float().mean().item(),
        f"{prefix}R@10": (ranks <= 10).float().mean().item(),
        f"{prefix}MRR": (1.0 / ranks).mean().item(),
        f"{prefix}MR": ranks.median().item(),
    }


def within_group_ranks(logits, groups):
    """
    Rank of the correct candidate (diagonal) when each query is ranked only
    against candidates of its own group (e.g. same speaker).

    Args:
        logits: [n_query, n_cand] similarities, row i matches column i.
        groups: list of n group labels (str). Queries with an empty label, or
            whose group has a single clip (rank would trivially be 1), are
            dropped.

    Returns:
        ranks: 1-based ranks of the kept queries.
        sizes: number of candidates (group size) for each kept query.
    """
    label_to_int = {g: i for i, g in enumerate(dict.fromkeys(groups))}
    g = th.tensor([label_to_int[x] for x in groups], device=logits.device)
    known = th.tensor([x != "" for x in groups], device=logits.device)

    same = g.unsqueeze(1) == g.unsqueeze(0)
    target_scores = logits.diagonal().unsqueeze(1)
    ranks = ((logits > target_scores) & same).sum(dim=1) + 1
    sizes = same.sum(dim=1)

    keep = known & (sizes > 1)
    return ranks[keep], sizes[keep]


def chance_rank_metrics(sizes, prefix=""):
    """Expected R@1/5/10 and MRR of a random ranking among `sizes` candidates."""
    sizes = sizes.float()
    mrr = th.tensor([sum(1.0 / r for r in range(1, int(n) + 1)) / n for n in sizes.tolist()])
    return {
        f"{prefix}acc": (1.0 / sizes).mean().item(),
        f"{prefix}R@5": (sizes.clamp(max=5) / sizes).mean().item(),
        f"{prefix}R@10": (sizes.clamp(max=10) / sizes).mean().item(),
        f"{prefix}MRR": mrr.mean().item(),
    }


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def retrieval_ranks(audio, video, visual_keys=None, chunk=4096):
    """Full-corpus retrieval ranks in both directions, allowing several audios
    per visual item (e.g. Flickr8k: 5 spoken captions per image).

    audio, video: (N, D) embeddings, row i = i-th (audio, visual) pair.
    visual_keys:  length-N list; rows with the same key share one visual item
                  (the same image). None -> every row is its own item (1-to-1).

    Returns (a2v, v2a), 1D tensors of 1-based ranks:
      a2v (N,): each audio vs. the M unique visual items; rank of its own item.
      v2a (M,): each unique visual item vs. all N audios; rank of its best-scoring
                own audio (= first correct hit, the usual Flickr8k protocol).
    With 1-to-1 data both reduce to the plain diagonal ranking.
    """
    n = audio.shape[0]
    if visual_keys is None:
        item = th.arange(n, device=audio.device)
        first = item
    else:
        first_row = {}  # visual key -> first row with that key
        for i, k in enumerate(visual_keys):
            first_row.setdefault(k, i)
        index = {k: j for j, k in enumerate(first_row)}
        item = th.tensor([index[k] for k in visual_keys], device=audio.device)
        first = th.tensor(list(first_row.values()), device=audio.device)
    items = video[first]  # (M, D), one embedding per unique visual item
    m = items.shape[0]

    a2v = []
    for s in range(0, n, chunk):
        logits = audio[s:s + chunk] @ items.t()                       # (b, M)
        target = logits.gather(1, item[s:s + chunk, None])            # (b, 1)
        a2v.append((logits > target).sum(1) + 1)

    v2a = []
    for s in range(0, m, chunk):
        logits = items[s:s + chunk] @ audio.t()                       # (b, N)
        own = item[None, :] == th.arange(s, s + logits.shape[0], device=audio.device)[:, None]
        best_pos = logits.masked_fill(~own, float("-inf")).max(1, keepdim=True).values
        v2a.append((logits > best_pos).sum(1) + 1)

    return th.cat(a2v), th.cat(v2a)
