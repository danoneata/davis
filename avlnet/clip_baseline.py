"""CLIP text-to-video baseline on the How2 dev5 `final` folds.

Replaces the audio side of the retrieval problem with text, embedded by CLIP: either
the transcript (`text` in the filelist) or the generated video description
(`video_caption`). The visual side is CLIP's embedding of the segment's central frame.

    uv run clip_baseline.py extract            # once, ~20k segments
    uv run clip_baseline.py evaluate           # CLIP + speech models, same pools

Features are written to `data/embeddings/how2-clip-vitl14/dev5-final-no_filter.pkl`,
which covers all three dev5 filelists (`final` and `final-no_filter2k` are subsets of
`final-no_filter`). Unlike the grouped How2 pickles, it stores the segment IDs.
"""

import pickle
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from how2_splits import load_split_ids, read_records
from retrieval import DIRECTIONS, build_problem, metrics_from_ranks, score_problem

# decord, transformers and extract_features (which loads transformers too) are imported
# inside the extraction code: they take seconds to import and scoring never needs them.

MODEL_NAME = "openai/clip-vit-large-patch14"
SEGMENTS_DIR = Path("/data/how2/clean_code/how2/segments")
HOW2_ROOT = "/data/how2/clean_code/how2/embeddings"
OUT_PATH = Path(__file__).parent / "data/embeddings/how2-clip-vitl14/dev5-final-no_filter.pkl"
EXTRACT_TAG = "final-no_filter"
FOLDS = ("final", "final-no_filter2k", "final-no_filter")
SPEECH_MODELS = ("run-07-spk-all", "run-09-spk-all-8x16")
TEXT_FIELDS = {"text": "clip-asr", "video_caption": "clip-caption"}
FRAME_TIMEOUT_SECONDS = 60


def segment_video_path(record: dict) -> Path:
    source, name = record["video_path"].split("-video-segments/")
    return SEGMENTS_DIR / source.removeprefix("how2_") / name


class CentralFrames(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from decord import VideoReader, cpu
        from extract_features import read_frames, time_limit

        path = self.paths[i]
        try:
            with time_limit(FRAME_TIMEOUT_SECONDS, f"Timed out decoding {path}"):
                vr = VideoReader(str(path), ctx=cpu(0))
                return read_frames(vr, [len(vr) // 2])[0]
        except Exception as e:
            print(f"[error] {path}: {e}")
            return None


def _collate(batch):
    return batch


@torch.no_grad()
def _embed_texts(texts, model, processor, batch_size=256):
    embeddings, truncated = [], []
    for start in tqdm(range(0, len(texts), batch_size), desc="Text", leave=False):
        chunk = texts[start : start + batch_size]
        full = processor.tokenizer(chunk, truncation=False)["input_ids"]
        truncated.extend(len(ids) > processor.tokenizer.model_max_length for ids in full)
        inputs = processor(text=chunk, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        embeddings.append(model.get_text_features(**inputs).float().cpu().numpy())
    return np.concatenate(embeddings), np.array(truncated)


def extract(limit: int = None, out_path: str = None, batch_size: int = 64, num_workers: int = 8):
    """Embed central frames, transcripts and captions of every dev5 `final-no_filter` segment."""
    out_path = Path(out_path) if out_path else OUT_PATH
    if out_path.exists():
        print(f"{out_path} exists; delete it to re-extract.")
        return

    records = read_records("dev5", EXTRACT_TAG)
    ids = list(records)[:limit]
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(MODEL_NAME).to("cuda").eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    loader = torch.utils.data.DataLoader(
        CentralFrames([segment_video_path(records[s]) for s in ids]),
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_collate,
    )
    image_embeddings, ok = [], []
    with torch.no_grad():
        for frames in tqdm(loader, desc="Frames"):
            good = [f for f in frames if f is not None]
            ok.extend(f is not None for f in frames)
            if not good:
                continue
            inputs = processor(images=good, return_tensors="pt")
            image_embeddings.append(
                model.get_image_features(pixel_values=inputs["pixel_values"].to("cuda"))
                .float()
                .cpu()
                .numpy()
            )
    ok = np.array(ok)
    ids = [s for s, keep in zip(ids, ok) if keep]
    print(f"{len(ids)} segments embedded, {(~ok).sum()} failed.")

    out = {"model": MODEL_NAME, "ids": ids, "image_embeddings": np.concatenate(image_embeddings)}
    for field in TEXT_FIELDS:
        out[f"{field}_embeddings"], out[f"{field}_truncated"] = _embed_texts(
            [records[s][field] for s in ids], model, processor
        )
        print(f"{field}: {out[f'{field}_truncated'].mean():.1%} truncated at 77 tokens.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print(f"Wrote {out_path}")


def _markdown(df: pd.DataFrame, fmt: str = "{:.1f}") -> str:
    """A pipe table, without pulling in `tabulate` for `DataFrame.to_markdown`."""
    cell = lambda v: fmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + " --- |" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


TEXT_OUT_PATH = OUT_PATH.parent / "text-final.pkl"
# Every segment a `split_tag: final` run trains, validates or is tested on.
# `dev5-final-no_filter` covers the other two dev5 filelists.
TEXT_FILELISTS = (("train", "final"), ("val", "final"), ("dev5", "final-no_filter"))


def extract_text(out_path: str = None):
    """CLIP text features of `text` and `video_caption` for the train, val and dev5 filelists.

    These are the `audio_features` of the text-vision runs (see `train.run_training`).
    """
    out_path = Path(out_path) if out_path else TEXT_OUT_PATH
    if out_path.exists():
        print(f"{out_path} exists; delete it to re-extract.")
        return

    records = {}
    for split, tag in TEXT_FILELISTS:
        split_records = read_records(split, tag)
        overlap = records.keys() & split_records.keys()
        if overlap:
            raise RuntimeError(f"{len(overlap)} segments of {split}-{tag} are in another filelist.")
        records.update(split_records)
    ids = list(records)
    print(f"{len(ids)} segments from {[f'{s}-{t}' for s, t in TEXT_FILELISTS]}.")

    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(MODEL_NAME).to("cuda").eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    out = {"model": MODEL_NAME, "ids": ids, "filelists": TEXT_FILELISTS}
    for field in TEXT_FIELDS:
        out[f"{field}_embeddings"], out[f"{field}_truncated"] = _embed_texts(
            [records[s][field] for s in ids], model, processor
        )
        print(f"{field}: {out[f'{field}_truncated'].mean():.1%} truncated at 77 tokens.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print(f"Wrote {out_path}")


def _normalize(x):
    return torch.nn.functional.normalize(torch.as_tensor(x, dtype=torch.float32), dim=1)


def _score(zvisual, zaudio, device):
    """gt_rank per direction for a one-to-one pool."""
    encoded = {"zvisual": zvisual, "zaudio": zaudio, "audio_to_visual": np.arange(len(zvisual))}
    return {
        d: score_problem(build_problem(encoded, d), device=device)["gt_rank"] for d in DIRECTIONS
    }


def _speech_encoded(config_name, fold, device):
    from evaluate import _encode_pool, _load_dataset
    from train import DualEncoder, load_config

    cfg = load_config(config_name)
    ckpt = torch.load(
        Path(cfg.get("checkpoint_dir", "checkpoints")) / config_name / "best.pt",
        map_location=device,
        weights_only=False,
    )
    model = DualEncoder(**ckpt.get("config", {})).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return _encode_pool(model, _load_dataset(f"how2-dev5-{fold}", cfg), device)


def evaluate(features: str = None, speech_models=SPEECH_MODELS, breakdown: bool = True):
    """R@1/5/10, MRR, MedR, % below of CLIP (transcript, caption) and speech models per dev5 fold.

    Every system is scored on the same pool: the segments of the fold that have speech
    embeddings (`how2_splits.load_split_ids`), in that order.
    """
    if isinstance(speech_models, str):  # `--speech_models a,b` arrives unsplit
        speech_models = [m.strip() for m in speech_models.split(",") if m.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(features or OUT_PATH, "rb") as f:
        feats = pickle.load(f)
    row_of = {s: i for i, s in enumerate(feats["ids"])}

    rows, ranks_by = [], {}
    for fold in FOLDS:
        ids = load_split_ids(HOW2_ROOT, "dev5", fold)
        missing = [s for s in ids if s not in row_of]
        if missing:
            raise RuntimeError(f"{len(missing)} segments of {fold} have no CLIP features.")
        idx = np.array([row_of[s] for s in ids])

        systems = {}
        zimage = _normalize(feats["image_embeddings"][idx])
        for field, name in TEXT_FIELDS.items():
            systems[name] = _score(zimage, _normalize(feats[f"{field}_embeddings"][idx]), device)
        for config_name in speech_models:
            enc = _speech_encoded(config_name, fold, device)
            systems[config_name] = _score(enc["zvisual"], enc["zaudio"], device)

        for name, by_direction in systems.items():
            for direction, ranks in by_direction.items():
                ranks_by[fold, name, direction] = ranks
                m = metrics_from_ranks(ranks, n_negatives=len(ids) - 1)  # one positive each
                rows.append({"fold": fold, "n": len(ids), "system": name, "direction": direction,
                             **{k: 100 * m[k] for k in ("r1", "r5", "r10", "below")},
                             "mrr": m["mrr"], "medr": m["medr"]})

        if breakdown:
            records = read_records("dev5", fold)
            truncated = feats["text_truncated"][idx]
            n_words = np.array([len(records[s]["text"].split()) for s in ids])
            terciles = pd.qcut(n_words, 3, labels=["short", "mid", "long"])
            groups = {"not truncated": ~truncated, "truncated": truncated}
            groups.update({f"{t} ({n_words[terciles == t].min()}-{n_words[terciles == t].max()} words)":
                           np.asarray(terciles == t) for t in ["short", "mid", "long"]})
            print(f"\n## {fold}: R@10 by transcript length (ranks against the full pool)")
            out = []
            for gname, mask in groups.items():
                row = {"group": gname, "n": int(mask.sum())}
                for name in systems:
                    for d in DIRECTIONS:
                        row[f"{name} {d}"] = 100 * (ranks_by[fold, name, d][mask] <= 10).mean()
                out.append(row)
            print(_markdown(pd.DataFrame(out)))

    df = pd.DataFrame(rows)
    print("\n## All metrics")
    print(_markdown(df, "{:.2f}"))
    for metric in ("r10", "mrr", "below"):
        print(f"\n## {metric}")
        table = df.pivot_table(index=["direction", "system"], columns="fold", values=metric,
                               sort=False)[list(FOLDS)].reset_index()
        print(_markdown(table, "{:.3f}" if metric == "mrr" else "{:.1f}"))


if __name__ == "__main__":
    fire.Fire({"extract": extract, "extract_text": extract_text, "evaluate": evaluate})
