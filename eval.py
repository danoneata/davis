"""Standalone evaluation script for the AVLnet-style audio-video retrieval model.

Loads settings from a YAML config in `configs/` (same mechanism as train.py),
but any field can also be overridden directly on the command line. CLI flags
always win over the config file.

Examples:
    # Just use everything from configs/default.yaml
    python eval.py default

    # Use the config, but point at a different checkpoint and test set
    python eval.py default --pretrain_path=checkpoints/exp1/e10.pth \\
        --test_ids_path=data/other_test_ids.txt

    # Evaluate video->audio retrieval instead of audio->video
    python eval.py default --eval_lang_retrieval=1

    # No config at all, everything from the command line
    python eval.py --embeddings_path=data/feats --test_ids_path=data/test_ids.txt \\
        --pretrain_path=checkpoints/exp1/e10.pth
"""
from __future__ import absolute_import, division, unicode_literals, print_function

import inspect
import json

import numpy as np
import torch as th
import wandb
from pathlib import Path

from torch.utils.data import DataLoader

from src.utils import (
    load_config,
    build_dataloader,
    build_model,

)
from src.metrics import rank_metrics, retrieval_ranks



th.backends.cudnn.benchmark = True

CONFIGS_DIR = Path(__file__).parent / "configs"

def visual_key_map(manifest_path):
    """clip id -> visual item key, read from a JSONL manifest.

    The clip id is the audio file stem (as in read_manifest); the visual key is
    the video/image file stem. Several clips can share one visual item (Flickr8k:
    <id>.jpg with audios <id>_0..4.wav). Plain-text manifests -> {} (1-to-1).
    """
    keys = {}
    if not manifest_path:
        return keys
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            r = json.loads(line)
            visual = r.get("video_path") or r.get("image_path")
            clip = r.get("audio_path") or visual
            if clip and visual:
                keys[Path(clip).stem] = Path(visual).stem
    return keys


@th.no_grad()
def embed_split(model, eval_dataloader):
    """Run the model over a whole split -> (audio (N, D), video (N, D), clip ids)."""
    model.eval()
    all_audio, all_video, ids = [], [], []
    for data in eval_dataloader:
        video = data["video"].cuda()
        audio = data["audio"].cuda()
        audio_mask = data["audio_mask"].cuda() if "audio_mask" in data else None
        audio_out, video_out = model(video, audio, audio_mask)
        all_audio.append(audio_out)
        all_video.append(video_out)
        ids.extend(data["video_id"])
    return th.cat(all_audio), th.cat(all_video), ids


@th.no_grad()
def eval_retrieval(model, eval_dataloader, dataset_name,
                    eval_lang_retrieval=False,
                    wandb_log=False, step=None, visual_keys=None):
    """Full-corpus retrieval eval: embeds the whole split, then ranks every
    query against every candidate in the split (not just within-batch).

    visual_keys (clip id -> visual item, see visual_key_map) handles several
    audios per image: images are deduplicated, audio->video ranks the unique
    images, video->audio counts a hit when any of the image's audios is ranked
    high enough (first correct hit)."""
    mode_str = "audio given video" if eval_lang_retrieval else "video given audio"
    print(f"Evaluating retrieval on {dataset_name} data: retrieving {mode_str}")

    all_audio, all_video, ids = embed_split(model, eval_dataloader)
    keys = [visual_keys.get(i, i) for i in ids] if visual_keys else None
    a2v, v2a = retrieval_ranks(all_audio, all_video, keys)
    metrics = rank_metrics(v2a if eval_lang_retrieval else a2v)

    print(f"{dataset_name} metrics: " +
          ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    if wandb_log:
        wandb.log(
            {f"{dataset_name}_{k.lower()}": v for k, v in metrics.items()},
            step=step,
        )

    return metrics


def run_eval(
    dataset_name: str = None,
    embeddings_path: str = None,
    val_ids_path: str = None,
    test_ids_path: str = None,
    val_embs_path: str = None,
    test_embs_path: str = None,
    image_dim: int = 768,
    video_feat_dim: int = 1024,
    audio_dim: int = 768,
    embd_dim: int = 4096,
    normalize: bool = True,
    image_only: bool = False,
    audio_pool: bool = True,
    batch_size_val: int = 10000,
    num_thread_reader: int = 8,
    pretrain_path: str = "",
    eval_lang_retrieval: int = 0,
    wandb_log: bool = False,
    wandb_project: str = "avlnet",
    experiment_name: str = None,
    **_ignored,
):
    """Run evaluation only. `**_ignored` swallows training-only config fields
    (lr, epochs, seed, etc.) so the same YAML used for train.py can be reused
    here without erroring out."""
    if _ignored:
        print(f"Ignoring config fields not used by eval: {sorted(_ignored)}")

    if val_ids_path is None and test_ids_path is None:
        raise ValueError("Provide at least one of val_ids_path / test_ids_path.")

    if not pretrain_path:
        print("WARNING: no pretrain_path given, evaluating a randomly initialized model.")

    net = build_model(image_dim, video_feat_dim, audio_dim, embd_dim, image_only, audio_pool, normalize)
    if pretrain_path:
        net.load_checkpoint(pretrain_path)
    net.cuda()
    net.eval()

    if dataset_name == "flickr8k":
        dataloader_val = build_dataloader(None, embeddings_path, None, normalize, image_only, batch_size_val, num_thread_reader, shuffle=False, drop_last=False, split="val", audio_pool=audio_pool)
        dataloader_test = build_dataloader(
            test_ids_path, embeddings_path, test_embs_path, normalize, image_only, batch_size_val, num_thread_reader, shuffle=False, drop_last=False, split="flickr8k", audio_pool=audio_pool
        )
    elif dataset_name == "msr-vtt":
        dataloader_val = build_dataloader(None, embeddings_path, None, normalize, image_only, batch_size_val, num_thread_reader,shuffle=False, drop_last=False, split="val", audio_pool=audio_pool)
        dataloader_test = build_dataloader(
            test_ids_path, embeddings_path, test_embs_path, normalize, image_only, batch_size_val, num_thread_reader, shuffle=False, drop_last=False, split="msr-vtt", audio_pool=audio_pool
        )
    elif dataset_name == "youcook2":
        dataloader_val = build_dataloader(None, embeddings_path, None, normalize, image_only, batch_size_val, num_thread_reader, shuffle=False, drop_last=False, split="val", audio_pool=audio_pool)
        dataloader_test = build_dataloader(
            test_ids_path, embeddings_path, test_embs_path, normalize, image_only,  batch_size_val, num_thread_reader, shuffle=False, drop_last=False, split="youcook2", audio_pool=audio_pool
        )
    else:
        dataloader_val = build_dataloader(
            val_ids_path, embeddings_path, val_embs_path, normalize, image_only, batch_size_val, num_thread_reader, shuffle=False, drop_last=False, split="val_how2", audio_pool=audio_pool
        )
        dataloader_test = build_dataloader(
            test_ids_path, embeddings_path, test_embs_path, normalize, image_only, batch_size_val, num_thread_reader, shuffle=False, drop_last=False, split="dev5_how2", audio_pool=audio_pool
        )

    if wandb_log:
        wandb.init(
            project=wandb_project,
            name=experiment_name,
            config={
                "embeddings_path": embeddings_path,
                "val_ids_path": val_ids_path,
                "test_ids_path": test_ids_path,
                "pretrain_path": pretrain_path,
                "eval_lang_retrieval": eval_lang_retrieval,
            },
        )

    val_metrics = None
    test_metrics = None

    if dataloader_val is not None:
        val_metrics = eval_retrieval(net, dataloader_val, dataset_name, eval_lang_retrieval, wandb_log=wandb_log,
                                     visual_keys=visual_key_map(val_ids_path))
    if dataloader_test is not None:
        test_metrics = eval_retrieval(net, dataloader_test, dataset_name, eval_lang_retrieval, wandb_log=wandb_log,
                                      visual_keys=visual_key_map(test_ids_path))

    print("\n" + "=" * 60)
    print("FINAL RETRIEVAL RESULTS")
    print("=" * 60)

    if val_metrics is not None:
        print(
            f"VAL : acc={val_metrics['acc']:.4f} | R@5={val_metrics['R@5']:.4f} | "
            f"R@10={val_metrics['R@10']:.4f} | MRR={val_metrics['MRR']:.4f} | MR={val_metrics['MR']:.2f}"
        )
    if test_metrics is not None:
        print(
            f"TEST: acc={test_metrics['acc']:.4f} | R@5={test_metrics['R@5']:.4f} | "
            f"R@10={test_metrics['R@10']:.4f} | MRR={test_metrics['MRR']:.4f} | MR={test_metrics['MR']:.2f}"
        )
    print("=" * 60)

    if wandb_log:
        wandb.finish()

    return {"val": val_metrics, "test": test_metrics}


def main(config_name: str = None, **overrides):
    """
    Usage:
        python eval.py <config_name> [--field=value ...]
        python eval.py --field=value ...   (no config file at all)

    Any `--key=value` flag overrides the matching field loaded from
    `configs/<config_name>.yaml`. Flags for fields not present in the config
    simply add new ones.
    """
    cfg = dict(load_config(CONFIGS_DIR,config_name)) if config_name else {}
    cfg.update(overrides)

    print(f"Running eval" + (f" with config `{config_name}`" if config_name else "") + ":")
    for k, v in sorted(cfg.items()):
        print(f"  {k}: {v}")

    return run_eval(**cfg)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
