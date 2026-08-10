"""Retrieval metrics (R@1/5/10, median rank).

Adapted from the original AVLnet metrics.py (Rouditchenko et al., Interspeech 2021).
The MSR-VTT-specific "count missing test clips as mistakes" branch was dropped since
it was specific to that benchmark's fixed 1000-clip test protocol; test_set_size is
now always just the number of rows actually present.
"""
from __future__ import absolute_import, division, unicode_literals, print_function

import numpy as np


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
