"""Compare two directories of per-item embedding pickles.

Useful whenever the same dataset is extracted twice -- a different pipeline, a
different machine, a bug fix -- and the question is whether the two agree.

Feature sets do not always store the same shape for the same modality: some keep
the raw per-frame DINOv2 embeddings, others the pooled vector. Anything 2-D is
collapsed with `--pool` (default `max`, matching the extraction pipelines) so the
two sides are comparable.

Example:
    uv run compare_embeddings.py data/embeddings/youcook2 \\
        /data/how2/evaluation_data/youcook2/embeddings
"""

import pickle
from pathlib import Path

import fire
import numpy as np
from tqdm import tqdm

POOLS = {
    "max": lambda x: x.max(axis=0),
    "mean": lambda x: x.mean(axis=0),
    # `none` leaves 2-D arrays alone, for comparing two per-frame feature sets
    # element-wise; shapes then have to match exactly.
    "none": lambda x: x,
}
# Reported per key: the share of items whose cosine similarity clears these.
COSINE_LEVELS = (0.9999, 0.99, 0.9)


def load(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def pool_array(array, how: str) -> np.ndarray:
    """Collapse a per-frame (n_frames, dim) array to (dim,); pass 1-D through."""
    array = np.asarray(array, dtype=np.float64)
    return POOLS[how](array) if array.ndim == 2 else array


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator else float("nan")


def compare_arrays(a, b, how: str, rtol: float, atol: float) -> dict:
    a, b = pool_array(a, how), pool_array(b, how)
    if a.shape != b.shape:
        return {"shape_mismatch": f"{a.shape} vs {b.shape}"}
    return {
        "cosine": cosine(a, b),
        "max_abs": float(np.abs(a - b).max()),
        # Relative to B's scale, so "0.05" reads as 5% of a typical component
        # rather than an unanchored absolute number.
        "rel": float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12)),
        "allclose": bool(np.allclose(a, b, rtol=rtol, atol=atol)),
    }


def summarize(key: str, records: list, rtol: float, atol: float) -> str:
    mismatched = [r for r in records if "shape_mismatch" in r]
    usable = [r for r in records if "shape_mismatch" not in r]
    if not usable:
        return f"  {key:18s} all {len(mismatched)} items shape-mismatched"

    cosines = np.array([r["cosine"] for r in usable])
    close = sum(r["allclose"] for r in usable)
    levels = "  ".join(
        f"cos>{level}: {100 * (cosines > level).mean():5.1f}%" for level in COSINE_LEVELS
    )
    lines = [
        f"  {key:18s} n={len(usable):<5d} "
        f"allclose(rtol={rtol:g}, atol={atol:g}): {close}/{len(usable)}"
        f" ({100 * close / len(usable):.1f}%)",
        f"  {'':18s} cosine  min={cosines.min():.6f}  median={np.median(cosines):.6f}",
        f"  {'':18s} {levels}",
        f"  {'':18s} worst max_abs={max(r['max_abs'] for r in usable):.4g}"
        f"  worst rel={max(r['rel'] for r in usable):.4g}",
    ]
    if mismatched:
        lines.append(f"  {'':18s} plus {len(mismatched)} shape-mismatched")
    return "\n".join(lines)


def main(
    path_a: str,
    path_b: str,
    pool: str = "max",
    rtol: float = 1e-3,
    atol: float = 1e-4,
    n_worst: int = 10,
    limit: int = None,
):
    """Report how closely the embeddings in `path_a` and `path_b` agree.

    Items are matched by filename stem, and only keys present on both sides are
    compared -- so feature sets with different schemas still line up on what they
    share. `pool` is how a per-frame 2-D array is collapsed (`max` or `mean`).

    Cosine similarity is the number to read: these embeddings are used for
    retrieval, where direction is what matters and scale does not.
    """
    if pool not in POOLS:
        raise ValueError(f"Unknown pool `{pool}`. Available: {sorted(POOLS)}")

    dir_a, dir_b = Path(path_a), Path(path_b)
    for d in (dir_a, dir_b):
        if not d.exists():
            raise FileNotFoundError(f"Directory does not exist: {d}")

    files_a = {p.stem: p for p in dir_a.glob("*.pkl")}
    files_b = {p.stem: p for p in dir_b.glob("*.pkl")}
    common = sorted(set(files_a) & set(files_b))
    print(f"A: {dir_a}  ({len(files_a)} items)")
    print(f"B: {dir_b}  ({len(files_b)} items)")
    print(
        f"common: {len(common)}   only in A: {len(set(files_a) - set(files_b))}"
        f"   only in B: {len(set(files_b) - set(files_a))}"
    )
    if not common:
        print("Nothing to compare.")
        return
    if limit:
        common = common[:limit]
        print(f"limited to the first {len(common)}")
    print(f"pooling 2-D arrays with `{pool}`\n")

    by_key, skipped, keys_a, keys_b = {}, [], set(), set()
    for stem in tqdm(common, desc="Comparing", leave=False):
        try:
            a, b = load(files_a[stem]), load(files_b[stem])
        except Exception as e:
            skipped.append((stem, f"{type(e).__name__}: {e}"))
            continue
        keys_a |= set(a)
        keys_b |= set(b)
        for key in sorted(set(a) & set(b)):
            record = compare_arrays(a[key], b[key], pool, rtol, atol)
            record["stem"] = stem
            by_key.setdefault(key, []).append(record)

    only_a, only_b = sorted(keys_a - keys_b), sorted(keys_b - keys_a)
    if only_a or only_b:
        print(f"keys only in A: {only_a or '-'}   only in B: {only_b or '-'}\n")

    for key, records in by_key.items():
        print(summarize(key, records, rtol, atol))
        usable = sorted(
            (r for r in records if "cosine" in r), key=lambda r: r["cosine"]
        )[:n_worst]
        if usable and usable[0]["cosine"] < COSINE_LEVELS[0]:
            print(f"  {'':18s} worst items:")
            for r in usable:
                print(
                    f"  {'':20s} {r['stem']:26s} cos={r['cosine']:.6f}"
                    f"  max_abs={r['max_abs']:.4g}  rel={r['rel']:.4g}"
                )
        print()

    if skipped:
        print(f"unreadable: {len(skipped)} (first 5: {skipped[:5]})")


if __name__ == "__main__":
    fire.Fire(main)
