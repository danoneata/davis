"""Run the model once on val + test and dump top-k retrievals for the Streamlit viewer.

Takes exactly the same arguments as eval.py (config name + --field=value overrides),
plus --out and --topk.

    python3 dump_retrievals.py default \
        --pretrain_path=checkpoints/<exp>/e65_R_5=14.48pct.pth \
        --out=retrievals_e65.pkl

Both directions are stored, so the app can switch between them:
    a2v : query = audio, candidates = videos   (eval_lang_retrieval=0)
    v2a : query = video, candidates = audios   (eval_lang_retrieval=1)
"""
import pickle
from pathlib import Path

import numpy as np
import torch as th

from src.utils import load_config, build_dataloader_negmine, build_model

th.backends.cudnn.benchmark = True
CONFIGS_DIR = Path(__file__).parent / "configs"


def build_loaders(cfg):
    """Same split logic as run_eval() in eval.py. Returns {split: (loader, manifest_path)}."""
    name = cfg.get("dataset_name")
    args = (
        cfg["embeddings_path"],
        cfg.get("normalize", True),
        cfg.get("image_only", False),
        cfg.get("batch_size_val", 10000),
        cfg.get("num_thread_reader", 8),
    )
    kw = dict(shuffle=False, drop_last=False)  # order must be deterministic!

    if name in ("flickr8k", "msr-vtt", "youcook2"):
        val = build_dataloader_negmine(None, *args, split="val", **kw)
        test = build_dataloader_negmine(cfg.get("test_ids_path"), *args, split=name, **kw)
        return {"val": (val, None), "test": (test, cfg.get("test_ids_path"))}

    val = build_dataloader_negmine(cfg.get("val_ids_path"), *args, split="val_how2", **kw)
    test = build_dataloader_negmine(cfg.get("test_ids_path"), *args, split="dev5_how2", **kw)
    return {"val": (val, cfg.get("val_ids_path")), "test": (test, cfg.get("test_ids_path"))}


@th.no_grad()
def embed(model, loader):
    all_a, all_v, keys = [], [], {}
    for data in loader:
        audio_out, video_out = model(data["video"].cuda(), data["audio"].cuda())
        all_a.append(audio_out.float())
        all_v.append(video_out.float())
        # If the dataset returns string ids/paths, keep them: lets the app align
        # embeddings to manifest rows even if the dataset filtered/reordered anything.
        for k, v in data.items():
            if isinstance(v, (list, tuple)) and len(v) and isinstance(v[0], str):
                keys.setdefault(k, []).extend(v)
    return th.cat(all_a), th.cat(all_v), keys


@th.no_grad()
def rank_and_topk(queries, cands, k, chunk=4096):
    """Same rank definition as eval.py: rank = 1 + #candidates scoring strictly above the GT."""
    n = queries.shape[0]
    top_idx, top_sc, gt_rank = [], [], []
    for s in range(0, n, chunk):
        logits = queries[s:s + chunk] @ cands.t()
        b = logits.shape[0]
        tgt = logits[th.arange(b), th.arange(s, s + b)].unsqueeze(1)
        gt_rank.append(((logits > tgt).sum(1) + 1).cpu())
        sc, ix = logits.topk(min(k, cands.shape[0]), dim=1)
        top_idx.append(ix.cpu())
        top_sc.append(sc.cpu())
    return (
        th.cat(top_idx).numpy().astype(np.int64),
        th.cat(top_sc).numpy().astype(np.float32),
        th.cat(gt_rank).numpy().astype(np.int64),
    )


def summarize(ranks):
    r = ranks.astype(np.float64)
    return {
        "acc": float((r <= 1).mean()),
        "R@5": float((r <= 5).mean()),
        "R@10": float((r <= 10).mean()),
        "MRR": float((1.0 / r).mean()),
        "MR": float(np.median(r)),
    }


def main(config_name: str = None, out: str = "retrievals.pkl", topk: int = 10, **overrides):
    cfg = dict(load_config(CONFIGS_DIR, config_name)) if config_name else {}
    cfg.update(overrides)

    pretrain_path = cfg.get("pretrain_path")
    if not pretrain_path:
        print("WARNING: no pretrain_path given, using a randomly initialized model.")

    net = build_model(
        cfg.get("image_dim", 768), cfg.get("video_feat_dim", 1024), cfg.get("audio_dim", 768),
        cfg.get("embd_dim", 4096), cfg.get("image_only", False),
    )
    if pretrain_path:
        net.load_checkpoint(pretrain_path)
    net.cuda().eval()

    results = {"pretrain_path": pretrain_path, "config_name": config_name, "topk": topk, "splits": {}}

    for split, (loader, manifest_path) in build_loaders(cfg).items():
        if loader is None:
            print(f"[{split}] no dataloader, skipping")
            continue
        print(f"[{split}] embedding ...")
        A, V, keys = embed(net, loader)
        n = A.shape[0]
        entry = {"manifest_path": manifest_path, "n": n, "keys": keys}
        for direction, (q, c) in {"a2v": (A, V), "v2a": (V, A)}.items():
            idx, sc, gt = rank_and_topk(q, c, topk)
            entry[direction] = {"top_idx": idx, "top_scores": sc, "gt_rank": gt}
            m = summarize(gt)
            print(f"[{split}] {direction}: N={n} " + ", ".join(f"{k}={v:.4f}" for k, v in m.items()))
        results["splits"][split] = entry

    with open(out, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved retrievals to {out}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
