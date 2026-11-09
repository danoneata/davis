#!/usr/bin/env python3

import argparse
import json
import os
import wave
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

VIDEO_EXTS = (".mp4",)
IMAGE_EXTS = (".jpg", ".jpeg")


def wav_duration(audio_path):
    """Duration in seconds read from the wav header."""
    try:
        with wave.open(audio_path, "rb") as w:
            return round(w.getnframes() / float(w.getframerate()), 2)
    except Exception:
        import soundfile as sf  # fallback for non-PCM wavs
        return round(sf.info(audio_path).duration, 2)


def process_video(args):
    """Scene mode: one <stem>.wav per <stem>.mp4/.jpg."""
    video_path, audio_dir = args

    stem = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join(audio_dir, stem + ".wav")

    if not os.path.isfile(audio_path):
        return []

    try:
        file_id, start, end = stem.rsplit("_", 2)
        duration = round(float(end) - float(start), 1)
    except (ValueError, IndexError):
        # Filenames without _<start>_<end> (e.g. plain images): use wav length.
        if not video_path.lower().endswith(IMAGE_EXTS):
            return []
        try:
            file_id, duration = stem, wav_duration(audio_path)
        except Exception:
            return []

    return [{
        "video_path": video_path,
        "audio_path": audio_path,
        "duration": duration,
        "original_video_id": file_id,
    }]


def process_flickr8k(args):
    """Flickr8k mode: one image <file_id>.jpg -> many audios <file_id>_<k>.wav."""
    video_path, audio_paths = args

    file_id = os.path.splitext(os.path.basename(video_path))[0]
    entries = []
    for audio_path in audio_paths:
        try:
            duration = wav_duration(audio_path)
        except Exception:
            continue
        entries.append({
            "video_path": video_path,
            "audio_path": audio_path,
            "duration": duration,
            "original_video_id": file_id,
        })
    return entries


def list_files(directory, exts):
    with os.scandir(directory) as entries:
        return [
            entry.path
            for entry in entries
            if entry.name.lower().endswith(exts) and entry.is_file()
        ]


def load_ids(ids_file):
    """One id per line; extensions (e.g. .jpg) are stripped."""
    ids = set()
    with open(ids_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(os.path.splitext(os.path.basename(line))[0])
    return ids


def video_id_candidates(video_path):
    """Possible ids for a file: full stem, and <file_id> from <file_id>_<start>_<end>."""
    stem = os.path.splitext(os.path.basename(video_path))[0]
    parts = stem.rsplit("_", 2)
    return (stem, parts[0]) if len(parts) == 3 else (stem,)


def main():
    parser = argparse.ArgumentParser(
        description="Create a JSONL manifest pairing video (.mp4) or image "
        "(.jpg) files with .wav audio files."
    )
    parser.add_argument("video_dir")
    parser.add_argument("audio_dir")
    parser.add_argument("output_file", type=Path)
    parser.add_argument("workers", type=int)
    parser.add_argument(
        "--flickr8k",
        action="store_true",
        help="Each image <file_id>.jpg may have several audios "
        "<file_id>_<k>.wav; each audio becomes a separate entry. "
        "Auto-enabled if such audios are detected.",
    )
    parser.add_argument(
        "--ids_file",
        default=None,
        help="Optional file with one id per line (e.g. Flickr_8k.testImages.txt); "
        "extensions are ignored. Only videos/images whose id is listed are kept.",
    )
    args = parser.parse_args()

    video_dir = os.path.abspath(args.video_dir)
    audio_dir = os.path.abspath(args.audio_dir)
    output_file = args.output_file
    workers = args.workers

    if not os.path.isdir(video_dir):
        raise RuntimeError(f"Video directory does not exist: {video_dir}")
    if not os.path.isdir(audio_dir):
        raise RuntimeError(f"Audio directory does not exist: {audio_dir}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Prefer .mp4; fall back to images if no videos are present.
    video_paths = list_files(video_dir, VIDEO_EXTS)
    if not video_paths:
        video_paths = list_files(video_dir, IMAGE_EXTS)
        print("No .mp4 files found; using .jpg images as video_path.")

    if args.ids_file:
        ids = load_ids(args.ids_file)
        matched = set()
        kept = []
        for path in video_paths:
            for cand in video_id_candidates(path):
                if cand in ids:
                    kept.append(path)
                    matched.add(cand)
                    break
        print(f"Ids in {args.ids_file}: {len(ids)} "
              f"(matched {len(matched)}, missing {len(ids) - len(matched)})")
        video_paths = kept

    print(f"Video/image files: {len(video_paths)}")
    print(f"Workers: {workers}")

    # Group audios by file_id: <file_id>_<k>.wav -> <file_id> (flickr8k style)
    audio_by_id = defaultdict(list)
    if not args.flickr8k:
        # Auto-detect: any image that has <stem>_<k>.wav audios.
        video_stems = {os.path.splitext(os.path.basename(p))[0] for p in video_paths}
    for audio_path in list_files(audio_dir, (".wav",)):
        stem = os.path.splitext(os.path.basename(audio_path))[0]
        file_id, sep, idx = stem.rpartition("_")
        if sep and idx.isdigit():
            audio_by_id[file_id].append(audio_path)
    if not args.flickr8k and any(s in audio_by_id for s in video_stems):
        print("Detected multiple audios per file (<file_id>_<k>.wav); "
              "enabling --flickr8k mode.")
        args.flickr8k = True

    if args.flickr8k:
        for paths in audio_by_id.values():
            paths.sort()

        worker_fn = process_flickr8k
        tasks = (
            (path, audio_by_id.get(os.path.splitext(os.path.basename(path))[0], []))
            for path in video_paths
        )
    else:
        worker_fn = process_video
        tasks = ((path, audio_dir) for path in video_paths)

    written = 0
    skipped = 0

    with output_file.open("w", encoding="utf-8") as f:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = pool.map(worker_fn, tasks, chunksize=256)

            for entries in tqdm(
                results,
                total=len(video_paths),
                desc="Creating manifest",
                unit="file",
            ):
                if not entries:
                    # Missing audio and malformed filenames are both skipped.
                    skipped += 1
                    continue

                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    written += 1

    print(f"Manifest entries:        {written}")
    print(f"Skipped videos/images:   {skipped}")
    print(f"Output:                  {output_file}")


if __name__ == "__main__":
    main()
