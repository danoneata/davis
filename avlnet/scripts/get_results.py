"""Regenerate the result tables of the reports, one experiment at a time.

    uv run scripts/get_results.py                              # list the experiments
    uv run scripts/get_results.py impact_of_negative_pool
    uv run scripts/get_results.py impact_of_negative_pool --fmt csv > table.csv

Each experiment is a function returning a `Table` (a header and rows of cells),
registered in `EXPERIMENTS`. The function fixes the systems and settings, so a
table regenerates the same way every time; to change one, edit its function. The
output format is shared: `markdown` (default) or `csv`.

To add an experiment, write `def my_experiment() -> Table` and add it to
`EXPERIMENTS`; its docstring is what the listing shows.
"""

import contextlib
import hashlib
import os
import pickle
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import fire
import numpy as np
import torch

# The avlnet modules live one level up, and configs hold paths relative to it
# (`checkpoint_dir`, `audio_features.path`), so run from there wherever we are called.
AVLNET_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AVLNET_DIR))
os.chdir(AVLNET_DIR)

from clip_baseline import HOW2_ROOT, OUT_PATH, _normalize, _score, _speech_encoded  # noqa: E402
from how2_splits import load_split_ids  # noqa: E402
from retrieval import A2V, V2A, metrics_from_ranks  # noqa: E402


@dataclass
class Table:
    header: list[str]
    rows: list[list[str]]


# Where `cache_function` keeps the ranks; delete it (or a file in it) to recompute.
CACHE_DIR = Path("/tmp/avlnet-get-results")
# `metrics_from_ranks` metrics that are fractions in [0, 1], printed as percentages.
FRACTION_METRICS = {"r1", "r5", "r10", "below"}


# --- shared helpers -----------------------------------------------------------------

def cache_function(path, func):
    """`func`, but its result is pickled to `path` on the first call and read back after.

    Once `path` exists the wrapped function returns its contents without calling `func`
    at all, whatever the arguments, so the path must identify everything the result
    depends on (see `_ranks_cache_path`). The file is written to a temporary name first
    and renamed, so an interrupted run leaves no half-written cache behind.
    """
    path = Path(path)

    def cached(*args, **kwargs):
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)
        result = func(*args, **kwargs)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(result, f)
        tmp.replace(path)
        return result

    return cached


def _ranks_cache_path(systems: dict[str, str], fold: str) -> Path:
    """A cache file for `_ranks(systems, fold)`, keyed on what the ranks depend on.

    The key covers the fold, each system and the modification time of the file its
    scores come from: the config's `best.pt`, or the CLIP features for `clip:` rows. So
    editing an experiment's systems, retraining a config or re-extracting CLIP features
    each give a new file instead of silently reusing stale ranks. Row labels are not in
    the key: they only name the rows.
    """
    from train import load_config

    parts = [fold]
    for system in sorted(set(systems.values())):
        if system.startswith("clip:"):
            source = Path(OUT_PATH)
        else:
            cfg = load_config(system)
            source = Path(cfg.get("checkpoint_dir", "checkpoints")) / system / "best.pt"
        parts.append(f"{system}@{source.stat().st_mtime_ns}")
    digest = hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]
    return CACHE_DIR / f"ranks-{fold}-{digest}.pkl"


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_clip_features() -> tuple[dict, dict[str, int]]:
    with open(OUT_PATH, "rb") as f:
        feats = pickle.load(f)
    return feats, {s: i for i, s in enumerate(feats["ids"])}


def _ranks(systems: dict[str, str], fold: str) -> dict[str, dict[str, np.ndarray]]:
    """gt_rank per direction for each system on a dev5 fold, all on the same pool.

    `systems` maps a row label to either a config name or `clip:<field>` for zero-shot
    CLIP (central frame vs. the CLIP text of `<field>`, from `clip_baseline.py extract`).
    The pool is the fold's segments with speech embeddings (`load_split_ids`).
    """
    device = _device()
    ids = load_split_ids(HOW2_ROOT, "dev5", fold)
    ranks = {}
    feats = row_of = None
    for label, system in systems.items():
        if system.startswith("clip:"):
            if feats is None:
                feats, row_of = _load_clip_features()
                missing = [s for s in ids if s not in row_of]
                if missing:
                    raise RuntimeError(
                        f"{len(missing)} segments of {fold} have no CLIP features in "
                        f"{OUT_PATH}. Run: uv run clip_baseline.py extract"
                    )
                idx = np.array([row_of[s] for s in ids])
                zimage = _normalize(feats["image_embeddings"][idx])
            field = system.removeprefix("clip:")
            ranks[label] = _score(zimage, _normalize(feats[f"{field}_embeddings"][idx]), device)
        else:
            encoded = _speech_encoded(system, fold, device)
            ranks[label] = _score(encoded["zvisual"], encoded["zaudio"], device)
    return ranks


# --- experiments ----------------------------------------------------------------------

def impact_of_negative_pool(metric_name: str) -> Table:
    """Results by batch composition (random, 16x8 ... 1x128 speakers x segments), with
    CLIP-ASR and the text model; A -> V and V -> A on `final` and `no_filter2k`,
    difference = final - no_filter2k.

    `metric_name` is any key of `retrieval.metrics_from_ranks`; fractions (R@K, below)
    are printed as percentages. `final` is filtered to `lm_score >= 8`, `no_filter2k`
    is not.
    """
    def get_metric(ranks: np.ndarray) -> float:
        """`metric_name` for a one-to-one pool (every query has n - 1 negatives)."""
        value = metrics_from_ranks(ranks, n_negatives=len(ranks) - 1)[metric_name]
        return 100 * value if metric_name in FRACTION_METRICS else value

    systems = {
        "CLIP-ASR": "clip:text",
        "Speech, random batches (run-06-all)": "run-06-all",
        "Speech, 16 x 8 (run-07-spk-all)": "run-07-spk-all",
        "Speech, 8 x 16 (run-09-spk-all-8x16)": "run-09-spk-all-8x16",
        "Speech, 4 x 32 (run-09-spk-all-4x32)": "run-09-spk-all-4x32",
        "Speech, 2 x 64 (run-09-spk-all-2x64)": "run-09-spk-all-2x64",
        "Speech, 1 x 128 (run-09-spk-all-1x128)": "run-09-spk-all-1x128",
        "Text (run-10-text-asr-8x16)": "run-10-text-asr-8x16",
    }
    folds = {"final": "final", "no_filter2k": "final-no_filter2k"}
    directions = {"A → V": A2V, "V → A": V2A}  # query -> gallery, in column order

    values = {}  # (label, direction label, fold label) -> metric
    for fold_label, fold in folds.items():
        ranks = cache_function(_ranks_cache_path(systems, fold), _ranks)(systems, fold)
        for label, by_direction in ranks.items():
            for dir_label, direction in directions.items():
                values[label, dir_label, fold_label] = get_metric(by_direction[direction])

    header = ["System"]
    for dir_label in directions:
        header += [f"{dir_label} {f}" for f in folds] + [f"{dir_label} final − no_filter2k"]
    rows = []
    for label in systems:
        cells = [label]
        for dir_label in directions:
            cells += [f"{values[label, dir_label, f]:.1f}" for f in folds]
            diff = values[label, dir_label, "final"] - values[label, dir_label, "no_filter2k"]
            cells.append(f"{diff:+.1f}")
        rows.append(cells)
    return Table(header, rows)


EXPERIMENTS = {
    "impact_of_negative_pool": partial(impact_of_negative_pool, metric_name="r10"),
}


# --- output -----------------------------------------------------------------------------

def format_table(table: Table, fmt: str = "markdown") -> str:
    if fmt == "csv":
        quote = lambda c: f'"{c}"' if "," in c else c  # noqa: E731
        return "\n".join(",".join(map(quote, r)) for r in [table.header, *table.rows])
    if fmt == "markdown":
        lines = ["| " + " | ".join(table.header) + " |", "|" + " --- |" * len(table.header)]
        lines += ["| " + " | ".join(r) + " |" for r in table.rows]
        return "\n".join(lines)
    raise ValueError(f"Unknown fmt `{fmt}`. Available: markdown, csv")


def _listing() -> str:
    lines = ["Experiments:"]
    for name, fn in EXPERIMENTS.items():
        # A `partial` carries its own generic docstring; describe the wrapped function.
        doc = (fn.func if isinstance(fn, partial) else fn).__doc__ or ""
        summary = " ".join(doc.split("\n\n")[0].split())
        if isinstance(fn, partial) and fn.keywords:
            summary += f" [{', '.join(f'{k}={v}' for k, v in fn.keywords.items())}]"
        lines.append(f"  {name}: {summary}")
    return "\n".join(lines)


def main(experiment: str = None, fmt: str = "markdown"):
    """Print the table of `experiment` (see `EXPERIMENTS`); with no name, list them."""
    if experiment is None:
        print(_listing())
        return
    if experiment not in EXPERIMENTS:
        raise SystemExit(f"Unknown experiment `{experiment}`.\n{_listing()}")
    # Loading prints progress ("Split how2_dev5-...") to stdout; keep stdout the table.
    with contextlib.redirect_stdout(sys.stderr):
        table = EXPERIMENTS[experiment]()
    print(format_table(table, fmt))


if __name__ == "__main__":
    fire.Fire(main)
