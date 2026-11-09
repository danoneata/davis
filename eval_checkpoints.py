"""Evaluate the best checkpoint of every run in a checkpoint directory on a fixed
set of retrieval benchmarks, in both directions, and write CSV tables.

Each subdir of --checkpoint_dir is one training run, named
    <model>_<batch_mode>_<av|img>      e.g. how2_train-final-lm_score8_video_img
holding the top-N checkpoints saved by train.py (e14_R_10=21.45pct.pth, ...).
The best one (highest validation score in the file name) is evaluated.

Eval sets (EVAL_SETS below):
    dev5, dev5_unfiltered  -> every run (av and img)
    flickr8k               -> img runs only (it has images, no video for V-JEPA)
Flickr8k has ~5 spoken captions per image. Images are deduplicated:
    audio->video: each audio ranks the unique images
    video->audio: each image ranks all audios; hit = best of its own audios in top K
(see src.metrics.retrieval_ranks; on 1-to-1 sets like dev5 this is the usual diagonal).

Usage:
    python eval_checkpoints.py [--config default] [--checkpoint_dir checkpoints]
        [--out_dir eval_results] [--gpu 0] [--pattern '*'] [--rerun]

Outputs in --out_dir:
    results.jsonl          one line per run, all metrics (cache: finished runs are skipped)
    results_R@10.csv       one row per run: model, batch_mode, modality, checkpoint,
                           R@10 (%) video->audio and audio->video per eval set
    results_long.csv       one row per (run, eval set, direction) with all metrics
"""
import argparse
import csv
import fnmatch
import json
import os
import re
import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parent

HOW2_EMBS = "/mnt/private-share/speechDatabases/how2/processed_how2/embeddings-spear/"
GABI = "/mnt/private-share/speechDatabases/davis-gabi"
# name -> (manifest, embeddings dir, usable by av models?)
EVAL_SETS = {
    "dev5": (f"{GABI}/how2-spkr-sep/how2_dev5-final.json", HOW2_EMBS, True),
    "dev5_unfiltered": (f"{GABI}/how2-spkr-sep/how2_dev5-final-no_filter2k.json", HOW2_EMBS, True),
    "flickr8k": (str(REPO / "flickr8k.json"), f"{GABI}/embeddings-spear/flickr8k", False),
}
DIRECTIONS = {"v2a": "video->audio", "a2v": "audio->video"}
METRICS = ["R@1", "R@5", "R@10", "MRR", "MR"]

RUN_RE = re.compile(r"^(?P<model>.+)_(?P<mode>random|speaker|video)_(?P<mod>av|img)$")
CKPT_RE = re.compile(r"^e(\d+)_.*=([-\d.]+)pct\.pth$")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="default", help="config in configs/ (model dims, audio_pool, ...)")
    p.add_argument("--checkpoint_dir", default=None, help="default: checkpoint_dir from the config")
    p.add_argument("--out_dir", default="eval_results")
    p.add_argument("--pattern", default="*", help="glob on run dir names")
    p.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES")
    p.add_argument("--workers", type=int, default=0, help="DataLoader workers for loading eval features")
    p.add_argument("--rerun", action="store_true", help="ignore results.jsonl and evaluate everything again")
    return p.parse_args()


def best_checkpoint(run_dir):
    """Highest-scoring checkpoint by the validation value in its file name."""
    best = None
    for f in run_dir.glob("e*.pth"):
        m = CKPT_RE.match(f.name)
        if m and (best is None or float(m.group(2)) > best[2]):
            best = (f, int(m.group(1)), float(m.group(2)))
    return best


class Evaluator:
    """Loads each eval set's features once per modality (cached on the GPU),
    then only the small model forward + ranking runs per checkpoint."""

    def __init__(self, cfg, workers):
        import torch as th
        from eval import visual_key_map
        from src.metrics import rank_metrics, retrieval_ranks
        from src.utils import build_dataloader, build_model
        self.th = th
        self.visual_key_map, self.rank_metrics, self.retrieval_ranks = visual_key_map, rank_metrics, retrieval_ranks
        self.build_dataloader, self.build_model = build_dataloader, build_model
        self.cfg, self.workers = cfg, workers
        self.cache = {}

    def features(self, name, image_only):
        key = (name, image_only)
        if key not in self.cache:
            manifest, embs, _ = EVAL_SETS[name]
            c = self.cfg
            loader = self.build_dataloader(
                manifest, None, embs, c["normalize"], image_only, 4096, self.workers,
                shuffle=False, drop_last=False, split=name,
                audio_pool=c["audio_pool"], io_threads=c.get("io_threads", 8),
            )
            batches, ids = [], []
            for data in loader:
                mask = data["audio_mask"].cuda() if "audio_mask" in data else None
                batches.append((data["video"].cuda(), data["audio"].cuda(), mask))
                ids.extend(data["video_id"])
            vk = self.visual_key_map(manifest)
            keys = [vk.get(i, i) for i in ids]
            print(f"  [{name}, {'img' if image_only else 'av'}] {len(ids)} audios, "
                  f"{len(set(keys))} visual items", flush=True)
            self.cache[key] = (batches, keys)
        return self.cache[key]

    def model(self, ckpt, image_only):
        c = self.cfg
        net = self.build_model(c["image_dim"], c["video_feat_dim"], c["audio_dim"], c["embd_dim"],
                               image_only, c["audio_pool"], c["normalize"])
        net.load_state_dict(self.th.load(ckpt, map_location="cpu"))  # strict on purpose
        return net.cuda().eval()

    def evaluate(self, ckpt, image_only):
        th = self.th
        net = self.model(ckpt, image_only)
        results = {}
        for name, (_, _, av_ok) in EVAL_SETS.items():
            if not image_only and not av_ok:
                continue
            try:
                batches, keys = self.features(name, image_only)
                with th.no_grad():
                    outs = [net(v, a, m) for v, a, m in batches]
                audio = th.cat([o[0] for o in outs])
                video = th.cat([o[1] for o in outs])
                a2v, v2a = self.retrieval_ranks(audio, video, keys)
                res = {}
                for d, ranks in (("v2a", v2a), ("a2v", a2v)):
                    m = self.rank_metrics(ranks)
                    m["R@1"] = m.pop("acc")
                    m["n_queries"] = len(ranks)
                    res[d] = m
                results[name] = res
                print(f"    {name:16s} V->A R@10={100 * res['v2a']['R@10']:6.2f}  "
                      f"A->V R@10={100 * res['a2v']['R@10']:6.2f}", flush=True)
            except Exception as e:
                msg = f"{type(e).__name__}: {e}".splitlines()[0][:300]
                results[name] = {"error": msg}
                print(f"    {name}: ERROR {msg}", flush=True)
        del net
        th.cuda.empty_cache()
        return results


def write_tables(out_dir, rows):
    rows = sorted(rows, key=lambda r: (r["modality"], r["model"], r["batch_mode"]))
    pct = lambda x: f"{100 * x:.2f}"

    wide_cols = ["model", "batch_mode", "modality", "best_epoch", "val_R@10", "checkpoint"] + [
        f"{s} {DIRECTIONS[d]} R@10" for s in EVAL_SETS for d in DIRECTIONS]
    with open(out_dir / "results_R@10.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(wide_cols)
        for r in rows:
            cells = []
            for s in EVAL_SETS:
                e = r["eval"].get(s)
                for d in DIRECTIONS:
                    if e is None:
                        cells.append("-")          # not applicable (flickr8k for av)
                    elif "error" in e:
                        cells.append("error")
                    else:
                        cells.append(pct(e[d]["R@10"]))
            w.writerow([r["model"], r["batch_mode"], r["modality"], r["best_epoch"],
                        f"{r['best_val']:.2f}", Path(r["checkpoint"]).name] + cells)

    long_cols = ["model", "batch_mode", "modality", "best_epoch", "checkpoint", "eval_set",
                 "direction"] + [f"{m} (%)" if m != "MR" else m for m in METRICS] + ["n_queries", "error"]
    with open(out_dir / "results_long.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(long_cols)
        for r in rows:
            base = [r["model"], r["batch_mode"], r["modality"], r["best_epoch"], r["checkpoint"]]
            for s, e in r["eval"].items():
                for d, label in DIRECTIONS.items():
                    if "error" in e:
                        w.writerow(base + [s, label] + [""] * (len(METRICS) + 1) + [e["error"]])
                        continue
                    m = e[d]
                    vals = [f"{m['MR']:.1f}" if k == "MR" else pct(m[k]) for k in METRICS]
                    w.writerow(base + [s, label] + vals + [m["n_queries"], ""])


def main():
    args = parse_args()
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cfg = OmegaConf.to_container(OmegaConf.load(REPO / "configs" / f"{args.config}.yaml"), resolve=True)
    ckpt_root = Path(args.checkpoint_dir or cfg.get("checkpoint_dir", "checkpoints"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for d in sorted(p for p in ckpt_root.iterdir() if p.is_dir() and fnmatch.fnmatch(p.name, args.pattern)):
        m = RUN_RE.match(d.name)
        if not m:
            print(f"skipping {d.name}: name is not <model>_<random|speaker|video>_<av|img>")
            continue
        runs.append((d, m.group("model"), m.group("mode"), m.group("mod")))
    n_img = sum(r[3] == "img" for r in runs)
    print(f"{len(runs)} runs in {ckpt_root} ({len(runs) - n_img} av, {n_img} img)")

    results_path = out_dir / "results.jsonl"
    rows = [json.loads(l) for l in open(results_path) if l.strip()] if results_path.exists() else []
    if args.rerun:  # drop only the runs selected by --pattern, keep the others
        selected = {r[0].name for r in runs}
        rows = [r for r in rows if r["run"] not in selected]
        with open(results_path, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    done = {r["run"] for r in rows}

    evaluator = None
    failed = []
    for k, (d, model, mode, mod) in enumerate(runs, 1):
        tag = f"[{k}/{len(runs)}] {d.name}"
        if d.name in done:
            print(f"{tag}: cached, skipping")
            continue
        best = best_checkpoint(d)
        if best is None:
            print(f"{tag}: no checkpoint found")
            failed.append(d.name)
            continue
        ckpt, epoch, val = best
        print(f"{tag}: {ckpt.name}", flush=True)
        if evaluator is None:
            evaluator = Evaluator(cfg, args.workers)
        row = {"run": d.name, "model": model, "batch_mode": mode, "modality": mod,
               "best_epoch": epoch, "best_val": val, "checkpoint": str(ckpt),
               "eval": evaluator.evaluate(ckpt, mod == "img")}
        if any("error" in e for e in row["eval"].values()):
            failed.append(d.name)  # kept in the table as "error", re-run with --pattern <name> --rerun
        rows.append(row)
        with open(results_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        write_tables(out_dir, rows)

    if rows:
        write_tables(out_dir, rows)
    print("=" * 60)
    print(f"{len(rows)} runs -> {out_dir}/results_R@10.csv, {out_dir}/results_long.csv")
    if failed:
        print(f"{len(failed)} with problems: " + ", ".join(failed))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
