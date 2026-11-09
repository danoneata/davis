#!/usr/bin/env python3
"""
Add "audio_embedding_pooled" (temporal mean of "audio_embedding" [T, D] -> [D], float16)
to every .pkl in an embeddings dir produced by extract_feats.py. All existing keys are kept;
each file is rewritten in place (use --safe for tmp+atomic-rename, ~30x slower).

Resumable: files that already contain the key are skipped without being rewritten.

Usage:
    python add_pooled_audio.py --input_dir /data/emb
    python add_pooled_audio.py --input_dir /data/emb --workers 64 --pool max
"""

import argparse
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np

KEY_IN = "audio_embedding"
KEY_OUT = "audio_embedding_pooled"

_DIR = None
_POOL = None
_FORCE = False
_SAFE = False


def _init(input_dir, pool, force, safe):
    global _DIR, _POOL, _FORCE, _SAFE
    _DIR, _POOL, _FORCE, _SAFE = input_dir, pool, force, safe


def _pool(a):
    if a.ndim == 1:  # already a single vector
        return a.astype(np.float16, copy=False)
    if _POOL == "max":
        return a.max(axis=0).astype(np.float16)
    # accumulate in fp32: fp16 sums over many frames lose precision / can overflow
    return a.mean(axis=0, dtype=np.float32).astype(np.float16)


def _process_chunk(names):
    """Returns (n_written, n_skipped, n_failed, failed_lines)."""
    w = s = f = 0
    failed = []
    dump = pickle.dumps
    for name in names:
        path = os.path.join(_DIR, name)
        try:
            with open(path, "r+b" if not _SAFE else "rb") as fh:
                rec = pickle.loads(fh.read())
                if not _FORCE and KEY_OUT in rec:
                    s += 1
                    continue
                a = rec.get(KEY_IN)
                if a is None:
                    f += 1
                    failed.append(f"{name}\tno {KEY_IN}")
                    continue
                rec[KEY_OUT] = _pool(np.asarray(a))
                data = dump(rec, protocol=pickle.HIGHEST_PROTOCOL)
                if not _SAFE:
                    # in place: no new inode / rename, so no directory-lock contention
                    fh.seek(0)
                    fh.write(data)
                    fh.truncate()
            if _SAFE:
                tmp = path + ".tmp"
                with open(tmp, "wb") as fh:
                    fh.write(data)
                os.replace(tmp, path)
            w += 1
        except Exception as e:  # keep going; report at the end
            f += 1
            failed.append(f"{name}\t{e!r}")
    return w, s, f, failed


def list_pkls(d):
    with os.scandir(d) as it:
        return [e.name for e in it if e.name.endswith(".pkl")]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_dir", required=True)
    p.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 8),
                   help="processes (32 was the sweet spot on a 64-core box; more just contends on the dir)")
    p.add_argument("--chunk", type=int, default=512, help="files per task")
    p.add_argument("--pool", choices=["mean", "max"], default="mean")
    p.add_argument("--force", action="store_true", help="recompute even if the key already exists")
    p.add_argument("--safe", action="store_true",
                   help="write tmp file + atomic rename (crash-proof, but much slower on big dirs). "
                        "Default rewrites in place: a hard crash can corrupt only the files being "
                        "written at that moment (<= --workers), and they get listed in the failed log on rerun")
    p.add_argument("--failed_log", default=None, help="default: <input_dir>/../add_pooled_audio_failed.tsv")
    args = p.parse_args()

    t0 = time.time()
    names = list_pkls(args.input_dir)
    n = len(names)
    print(f"found {n:,} .pkl files in {time.time() - t0:.1f}s; {args.workers} workers", flush=True)
    if n == 0:
        return

    chunks = [names[i:i + args.chunk] for i in range(0, n, args.chunk)]
    del names
    W = S = F = done = 0
    errors = []
    failed_log = args.failed_log or os.path.join(os.path.dirname(os.path.abspath(args.input_dir)), "add_pooled_audio_failed.tsv")
    t0 = time.time()
    with Pool(args.workers, initializer=_init, initargs=(args.input_dir, args.pool, args.force, args.safe)) as pool:
        for w, s, f, fl in pool.imap_unordered(_process_chunk, chunks):
            W += w; S += s; F += f
            done += w + s + f
            errors.extend(fl)
            el = time.time() - t0
            rate = done / el if el else 0
            eta = (n - done) / rate if rate else 0
            sys.stderr.write(f"\r{done:,}/{n:,}  {rate:,.0f} files/s  ETA {eta / 60:.1f} min  "
                             f"(written {W:,}, skipped {S:,}, failed {F:,})   ")
            sys.stderr.flush()
    sys.stderr.write("\n")
    el = time.time() - t0
    print(f"done in {el / 60:.1f} min ({done / max(el, 1e-9):,.0f} files/s): written {W:,}, skipped {S:,}, failed {F:,}")
    if errors:
        with open(failed_log, "w") as fh:
            fh.write("\n".join(errors) + "\n")
        print(f"failed files listed in {failed_log}; first few:")
        for e in errors[:10]:
            print("  ", e)


if __name__ == "__main__":
    main()
