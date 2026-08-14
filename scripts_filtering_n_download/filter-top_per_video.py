#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Select the best segment for each root video file based on "
            "lm or denseav score. Ties are broken using the other metric."
        )
    )

    parser.add_argument(
        "--input_path",
        required=True,
        help="Path to the input JSONL file.",
    )

    parser.add_argument(
        "--output_path",
        required=True,
        help="Path to the output JSONL file.",
    )

    parser.add_argument(
        "--filter",
        required=True,
        choices=["lm", "denseav"],
        help="Metric to maximize. Ties are broken using the other metric.",
    )

    return parser.parse_args()


def get_seg_id(video_path):
    """
    Extract the segment ID from a video_path.

    Example:
        how2_train-video-segments/--8pSDeC-fg_4.mp4
        -> --8pSDeC-fg_4
    """
    return Path(video_path).stem


def get_root_id(seg_id):
    """
    Extract the root video ID from a segment ID.

    Example:
        --8pSDeC-fg_4 -> --8pSDeC-fg
        --7E2sU6zP4_13 -> --7E2sU6zP4
    """
    match = re.match(r"^(.*)_(\d+)$", seg_id)

    if not match:
        raise ValueError(f"Could not extract root ID from seg_id: {seg_id}")

    return match.group(1)


def select_best_segment(records, metric):
    """
    Select one record per root file.

    Primary sort:
        highest requested metric

    Tie-break:
        highest score on the other metric

    Final deterministic tie-break:
        lexicographically smallest seg_id
    """
    other_metric = "denseav_score" if metric == "lm" else "lm_score"

    # Group records by root video ID.
    groups = {}

    for record in records:
        video_path = record.get("video_path")
        if not video_path:
            continue

        seg_id = get_seg_id(video_path)
        root_id = get_root_id(seg_id)

        groups.setdefault(root_id, []).append(record)

    selected = []

    for root_id, group in groups.items():
        def sort_key(record):
            seg_id = get_seg_id(record["video_path"])

            # Higher scores are better, hence negative values.
            primary = record.get(
                f"{metric}_score",
                float("-inf"),
            )
            secondary = record.get(
                other_metric,
                float("-inf"),
            )

            # seg_id is only a final deterministic tie-breaker.
            return (-primary, -secondary, seg_id)

        best = sorted(group, key=sort_key)[0]

        selected.append(best)

    # Keep output deterministic by root ID.
    selected.sort(
        key=lambda record: get_root_id(get_seg_id(record["video_path"]))
    )

    return selected


def main():
    args = parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    records = []

    with input_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {e}"
                ) from e

            records.append(record)

    selected = select_best_segment(records, args.filter)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in selected:
            seg_id = get_seg_id(record["video_path"])
            f.write(seg_id + "\n")

    print(f"Input records:    {len(records)}")
    print(f"Root files:       {len(selected)}")
    print(f"Filter metric:    {args.filter}")
    print(f"Output:           {output_path}")


if __name__ == "__main__":
    main()