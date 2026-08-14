#!/usr/bin/env python3

import argparse
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

from tqdm import tqdm


# ----------------------------
# Resolve video file (any format)
# ----------------------------

VIDEO_EXTS = [".mp4", ".mkv", ".webm"]


def find_video_file(video_dir, root_name):
    base = Path(video_dir)

    for ext in VIDEO_EXTS:
        candidate = base / f"{root_name}{ext}"
        if candidate.exists():
            return candidate

    return None


# ----------------------------
# Loading
# ----------------------------

def load_segments(path):
    segments = []
    root_ids = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                seg_id, root_name, start, end = line.split()
                segments.append((seg_id, root_name, float(start), float(end)))
                root_ids.add(root_name)

    return segments, root_ids


def load_transcripts(path):
    transcripts = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                seg_id, text = line.rstrip("\n").split(maxsplit=1)
                transcripts[seg_id] = text

    return transcripts


def scan_available(video_dir, audio_dir):
    video_roots = set()
    audio_roots = set()

    for ext in VIDEO_EXTS:
        video_roots.update({p.stem for p in Path(video_dir).glob(f"*{ext}")})

    audio_roots = {p.stem for p in Path(audio_dir).glob("*.wav")}

    return video_roots, audio_roots


# ----------------------------
# FFmpeg
# ----------------------------

def trim_video(src, dst, start, duration):
    subprocess.run([
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-ss", str(start),
        "-i", str(src),
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-c:a", "aac",
        str(dst),
    ], check=True)


def trim_audio(src, dst, start, duration):
    subprocess.run([
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-ss", str(start),
        "-i", str(src),
        "-t", str(duration),
        "-c:a", "pcm_s16le",
        str(dst),
    ], check=True)


# ----------------------------
# Worker
# ----------------------------

def process_segment(job, video_dir, audio_dir, video_out_dir, audio_out_dir, transcripts):
    seg_id, root_name, start, end = job
    duration = end - start

    video_src = find_video_file(video_dir, root_name)
    audio_src = Path(audio_dir) / f"{root_name}.wav"

    if video_src is None or not audio_src.exists():
        return None

    video_dst = Path(video_out_dir) / f"{seg_id}.mp4"
    audio_dst = Path(audio_out_dir) / f"{seg_id}.wav"

    try:
        trim_video(video_src, video_dst, start, duration)
        trim_audio(audio_src, audio_dst, start, duration)
    except Exception:
        return None

    return {
        "video_path": str(video_dst),
        "audio_path": str(audio_dst),
        "original_video_id": root_name,
        "start": start,
        "end": end,
        "text": transcripts.get(seg_id, ""),
    }


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--segments", required=True)
    ap.add_argument("--transcripts", required=True)
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--output-prefix", required=True)
    ap.add_argument("--workers", type=int, default=32)

    args = ap.parse_args()

    print("\nLoading metadata...")

    segments, segment_roots = load_segments(args.segments)
    transcripts = load_transcripts(args.transcripts)

    video_roots, audio_roots = scan_available(args.video_dir, args.audio_dir)

    print("\n================ DATASET STATS ================\n")

    print(f"Segments total: {len(segments):,}")
    print(f"Unique root_ids in segments: {len(segment_roots):,}")

    print(f"Video files found (mp4/mkv/webm): {len(video_roots):,}")
    print(f"Audio files found (wav): {len(audio_roots):,}")

    video_overlap = segment_roots & video_roots
    audio_overlap = segment_roots & audio_roots

    missing_video = segment_roots - video_roots
    missing_audio = segment_roots - audio_roots

    print(f"\nRoot overlap (segments ∩ video): {len(video_overlap):,}")
    print(f"Root overlap (segments ∩ audio): {len(audio_overlap):,}")

    print(f"\nMissing video roots: {len(missing_video):,}")
    print(f"Missing audio roots: {len(missing_audio):,}")

    if missing_video:
        print("\nExample missing video roots:", list(missing_video)[:10])

    if missing_audio:
        print("\nExample missing audio roots:", list(missing_audio)[:10])

    print("\n==============================================\n")

    video_out_dir = Path(f"{args.output_prefix}-video-segments")
    audio_out_dir = Path(f"{args.output_prefix}-audio-segments")

    video_out_dir.mkdir(parents=True, exist_ok=True)
    audio_out_dir.mkdir(parents=True, exist_ok=True)

    worker = partial(
        process_segment,
        video_dir=args.video_dir,
        audio_dir=args.audio_dir,
        video_out_dir=str(video_out_dir),
        audio_out_dir=str(audio_out_dir),
        transcripts=transcripts,
    )

    manifest_path = Path(f"{args.output_prefix}.json")

    kept = 0
    skipped = 0

    print(f"Starting processing with {args.workers} workers...\n")

    with open(manifest_path, "w", encoding="utf-8") as manifest:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:

            for record in tqdm(
                ex.map(worker, segments, chunksize=16),
                total=len(segments),
                desc="Processing segments",
                unit="seg",
            ):
                if record is None:
                    skipped += 1
                    continue

                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1

    print("\nDONE")
    print(f"Manifest: {manifest_path}")
    print(f"Kept: {kept:,}")
    print(f"Skipped: {skipped:,}")


if __name__ == "__main__":
    main()