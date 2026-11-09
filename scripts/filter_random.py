#!/usr/bin/env python3
"""
Randomly subsample lines (or JSON array elements) from an input file,
producing one output file per seed.

Examples:
    # keep 10% of lines, 5 different seeds -> 5 files
    python subsample.py data.jsonl --percentage 10 --seeds 5
    #   data_10perc_5seeds_seed0.json ... data_10perc_5seeds_seed4.json

    # keep exactly 1000 lines, 3 seeds
    python subsample.py data.jsonl --num-samples 1000 --seeds 3
    #   data_1000samp_3seeds_seed0.json ... data_1000samp_3seeds_seed2.json

Input handling:
    * If the file is a JSON array ([...]), its elements are sampled and the
      output is written as a JSON array.
    * Otherwise it is treated line by line (e.g. JSONL); blank lines are
      ignored and the selected lines are written out unchanged.
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path


def load_items(path: Path):
    """Return (items, mode) where mode is 'array' or 'lines'."""
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data, "array"
        except json.JSONDecodeError:
            pass  # fall back to line mode
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return lines, "lines"


def write_items(path: Path, items, mode: str):
    with path.open("w", encoding="utf-8") as f:
        if mode == "array":
            json.dump(items, f, ensure_ascii=False, indent=2)
            f.write("\n")
        else:
            for ln in items:
                f.write(ln + "\n")


def format_number(x: float) -> str:
    """10.0 -> '10', 12.5 -> '12.5'"""
    return str(int(x)) if float(x).is_integer() else f"{x:g}"


def main():
    p = argparse.ArgumentParser(
        description="Subsample a file by percentage or exact count, once per seed."
    )
    p.add_argument("input", type=Path, help="Input file (JSONL / text lines or JSON array)")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("-p", "--percentage", type=float,
                       help="Percentage of lines to keep (0 < p <= 100)")
    group.add_argument("-n", "--num-samples", type=int,
                       help="Exact number of lines to keep")
    p.add_argument("-s", "--seeds", type=int, default=1,
                   help="Number of seeds = number of output files (default: 1)")
    p.add_argument("--seed-start", type=int, default=0,
                   help="First seed value; seeds used are seed_start .. seed_start+seeds-1 (default: 0)")
    p.add_argument("-o", "--output-dir", type=Path, default=None,
                   help="Output directory (default: same directory as input)")
    p.add_argument("--shuffle", action="store_true",
                   help="Shuffle the selected lines (default: keep original order)")
    args = p.parse_args()

    if not args.input.is_file():
        sys.exit(f"Error: input file not found: {args.input}")
    if args.seeds < 1:
        sys.exit("Error: --seeds must be >= 1")

    items, mode = load_items(args.input)
    total = len(items)
    if total == 0:
        sys.exit("Error: input file contains no lines/items")

    # Work out how many to keep and the filename tag
    if args.percentage is not None:
        if not (0 < args.percentage <= 100):
            sys.exit("Error: --percentage must be in (0, 100]")
        k = max(1, round(total * args.percentage / 100))
        tag = f"{format_number(args.percentage)}perc"
    else:
        if args.num_samples < 1:
            sys.exit("Error: --num-samples must be >= 1")
        if args.num_samples > total:
            sys.exit(f"Error: --num-samples ({args.num_samples}) exceeds "
                     f"number of lines in input ({total})")
        k = args.num_samples
        tag = f"{args.num_samples}samp"

    out_dir = args.output_dir or args.input.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{args.input.stem}_{tag}_{args.seeds}seeds"

    print(f"Input: {args.input} ({total} {'items' if mode == 'array' else 'lines'})")
    print(f"Keeping {k} per file across {args.seeds} seed(s)")

    for seed in range(args.seed_start, args.seed_start + args.seeds):
        rng = random.Random(seed)
        idx = rng.sample(range(total), k)
        if not args.shuffle:
            idx.sort()
        selected = [items[i] for i in idx]

        out_path = out_dir / f"{base}_seed{seed}.json"
        write_items(out_path, selected, mode)
        print(f"  seed {seed}: {out_path}")


if __name__ == "__main__":
    main()
