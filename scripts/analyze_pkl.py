#!/usr/bin/env python3
"""Usage: python inspect_pkl.py file1.pkl [file2.pkl ...]   (or a folder: python inspect_pkl.py embs_test/)"""
import glob, os, pickle, sys
import numpy as np

paths = []
for arg in sys.argv[1:]:
    paths += sorted(glob.glob(os.path.join(arg, "*.pkl")))[:5] if os.path.isdir(arg) else [arg]

for path in paths:
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"\n{os.path.basename(path)}  ({os.path.getsize(path) / 1024:.1f} KB)")
    for key, value in data.items():
        if value is None:
            print(f"  {key:16s} None")
        elif isinstance(value, np.ndarray):
            v = value.astype(np.float32)
            print(f"  {key:16s} shape={value.shape}  dtype={value.dtype}  "
                  f"min={v.min():.3f} max={v.max():.3f} mean={v.mean():.3f}  nan/inf={(~np.isfinite(v)).sum()}")
        else:
            print(f"  {key:16s} {type(value).__name__}: {value}")
