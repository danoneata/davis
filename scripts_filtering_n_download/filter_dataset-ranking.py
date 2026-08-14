#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter a JSONL manifest by LM/DenseAV scores."
    )

    parser.add_argument(
        "--input_manifest",
        type=str,
        required=True,
        help="Input JSONL manifest.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output txt file containing one segment ID per line.",
    )

    parser.add_argument(
        "--filtering",
        choices=["lm", "denseav", "both"],
        required=True,
        help="Filtering criterion.",
    )

    # Optional percentile filtering
    parser.add_argument(
        "--top",
        type=float,
        default=None,
        help="Keep the top N%%. If omitted, threshold filtering is used.",
    )

    # Optional threshold filtering
    parser.add_argument(
        "--lm_score",
        type=float,
        default=None,
        help="Minimum LM score.",
    )

    parser.add_argument(
        "--denseav_score",
        type=float,
        default=None,
        help="Minimum DenseAV score.",
    )

    args = parser.parse_args()

    if args.top is None and args.lm_score is None and args.denseav_score is None:
        parser.error(
            "Specify either --top or one/both score thresholds."
        )

    return args


def get_seg_id(video_path):
    # Example:
    # how2_val-video-segments/-d5dN54tH2E_10.mp4
    return Path(video_path).stem.lstrip("-")


def top_indices(scores, top_percent):
    n = len(scores)
    k = max(1, int(round(n * top_percent / 100.0)))

    ranked = sorted(
        range(n),
        key=lambda i: scores[i],
        reverse=True,
    )

    return set(ranked[:k])


def main():
    args = parse_args()

    entries = []
    lm_scores = []
    denseav_scores = []

    with open(args.input_manifest, "r") as f:
        for line in f:
            item = json.loads(line)
            entries.append(item)
            lm_scores.append(float(item["lm_score"]))
            denseav_scores.append(float(item["denseav_score"]))

    if args.top is not None:
        # Percentile filtering

        if args.filtering == "lm":
            keep = top_indices(lm_scores, args.top)

        elif args.filtering == "denseav":
            keep = top_indices(denseav_scores, args.top)

        else:
            lm_keep = top_indices(lm_scores, args.top)
            dense_keep = top_indices(denseav_scores, args.top)
            keep = lm_keep & dense_keep

    else:
        # Threshold filtering

        if args.filtering == "lm":
            if args.lm_score is None:
                raise ValueError("--lm_score is required.")

            keep = {
                i for i, score in enumerate(lm_scores)
                if score >= args.lm_score
            }

        elif args.filtering == "denseav":
            if args.denseav_score is None:
                raise ValueError("--denseav_score is required.")

            keep = {
                i for i, score in enumerate(denseav_scores)
                if score >= args.denseav_score
            }

        else:
            if args.lm_score is None or args.denseav_score is None:
                raise ValueError(
                    "--lm_score and --denseav_score are required for filtering=both."
                )

            keep = {
                i
                for i, (lm, dense)
                in enumerate(zip(lm_scores, denseav_scores))
                if lm >= args.lm_score and dense >= args.denseav_score
            }

    keep = sorted(keep)

    with open(args.output_path, "w") as f:
        for idx in keep:
            f.write(get_seg_id(entries[idx]["video_path"]) + "\n")

    print(f"Loaded {len(entries)} entries.")
    print(f"Selected {len(keep)} entries.")
    print(f"Saved segment IDs to {args.output_path}")


if __name__ == "__main__":
    main()
