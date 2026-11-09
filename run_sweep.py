"""Train on every train file in a directory x batch_mode x image_only, one run
after the other, then evaluate each run's best checkpoint on a fixed set of
retrieval benchmarks and collect everything into one results table.

Usage:
    python run_sweep.py INPUT_DIR [--config default] [--pattern '*.json']
        [--out_dir sweep_results] [--gpus 0] [--metric R@10]
        [--batch_modes random video speaker] [--image_only true false]
        [-- extra train.py args, e.g. --epochs=20 --mining_size=32]

For every train file INPUT_DIR/foo.json and every (batch_mode, image_only):
    python train.py <config> --train_ids_path=INPUT_DIR/foo.json --base_exp=foo
        --batch_mode=<mode> --image_only=<bool> --experiment_name=foo_<mode>_<av|img>
The best checkpoint (by the config's checkpoint_metric on val) is then evaluated
on every set in EVAL_SETS (full-corpus audio->video retrieval, like eval.py).

Outputs in --out_dir:
    logs/<experiment_name>.log   train.py stdout/stderr
    results.jsonl                one line per finished run, all metrics
    results.csv / results.md     table: one row per config, <metric> per eval set
The sweep is resumable: runs already in results.jsonl are skipped.
"""
import argparse
import csv
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parent

HOW2_EMBS = "/mnt/private-share/speechDatabases/how2/processed_how2/embeddings/all/"
GABI = "/mnt/private-share/speechDatabases/davis-gabi"
EVAL_SETS = {
    "dev5": (f"{GABI}/how2-spkr-sep/how2_dev5-final.json", HOW2_EMBS),
    "dev5_unfiltered": (f"{GABI}/how2-spkr-sep/how2_dev5-final-no_filter2k.json", HOW2_EMBS),
    "msr-vtt": (f"{GABI}/embeddings/msr-vtt.txt", f"{GABI}/embeddings/msr-vtt"),
    "youcook2": (f"{GABI}/embeddings/youcook2.txt", f"{GABI}/embeddings/youcook2"),
    "flickr8k": (f"{GABI}/embeddings/flickr8k.txt", f"{GABI}/embeddings/flickr8k"),
}


def parse_args():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_dir")
    p.add_argument("--config", default="default", help="config name in configs/")
    p.add_argument("--pattern", default="*", help="glob for train files in input_dir")
    p.add_argument("--out_dir", default="sweep_results")
    p.add_argument("--gpus", default=None, help="CUDA_VISIBLE_DEVICES for train + eval, e.g. '0' or '0,1'")
    p.add_argument("--metric", default="R@10", help="metric shown in the results table")
    p.add_argument("--batch_modes", nargs="+", default=["random", "video", "speaker"])
    p.add_argument("--image_only", nargs="+", default=["false", "true"])
    p.add_argument("--eval_workers", type=int, default=0, help="DataLoader workers for eval")
    args = p.parse_args(argv)
    args.extra = extra
    args.image_only = [s.lower() in ("1", "true", "yes") for s in args.image_only]
    return args


def run_name(stem, mode, image_only):
    return f"{stem}_{mode}_{'img' if image_only else 'av'}"


def best_checkpoint(ckpt_dir, mode):
    """Pick the best of the top-N checkpoints saved by train.py
    (filenames like e7_R_5=41.20pct.pth)."""
    best = None
    for f in Path(ckpt_dir).glob("e*_*=*pct.pth"):
        m = re.match(r"e(\d+)_.*=([-\d.]+)pct\.pth$", f.name)
        if not m:
            continue
        epoch, val = int(m.group(1)), float(m.group(2))
        better = best is None or (val > best[2] if mode == "max" else val < best[2])
        if better:
            best = (f, epoch, val)
    return best


# ---------------------------------------------------------------- evaluation
class Evaluator:
    """Imports torch lazily (it is slow to import here) and caches eval
    datasets per (eval set, image_only) so directory scans happen once."""

    def __init__(self, cfg, workers):
        import torch as th
        from src.utils import build_dataloader, build_model
        self.th, self.build_dataloader, self.build_model = th, build_dataloader, build_model
        self.cfg, self.workers = cfg, workers
        self.loaders = {}

    def loader(self, name, image_only):
        key = (name, image_only)
        if key not in self.loaders:
            ids, embs = EVAL_SETS[name]
            c = self.cfg
            self.loaders[key] = self.build_dataloader(
                ids, None, embs, c["normalize"], image_only, 4096, self.workers,
                shuffle=False, drop_last=False, split=name,
                audio_pool=c["audio_pool"], io_threads=c.get("io_threads", 8),
            )
        return self.loaders[key]

    def model(self, ckpt, image_only):
        c = self.cfg
        net = self.build_model(c["image_dim"], c["video_feat_dim"], c["audio_dim"], c["embd_dim"],
                               image_only, c["audio_pool"], c["normalize"])
        state = self.th.load(ckpt, map_location="cpu")
        net.load_state_dict(state)  # strict: a mismatch here is a real error
        return net.cuda().eval()

    def evaluate(self, ckpt, image_only):
        th = self.th
        net = self.model(ckpt, image_only)
        results = {}
        for name in EVAL_SETS:
            try:
                audio, video = [], []
                with th.no_grad():
                    for data in self.loader(name, image_only):
                        mask = data["audio_mask"].cuda() if "audio_mask" in data else None
                        a, v = net(data["video"].cuda(), data["audio"].cuda(), mask)
                        audio.append(a)
                        video.append(v)
                audio, video = th.cat(audio), th.cat(video)
                # Full-corpus audio -> video ranking (same as eval.py).
                ranks = []
                for i in range(0, len(audio), 4096):
                    logits = audio[i:i + 4096] @ video.t()
                    target = logits[th.arange(len(logits)), th.arange(i, i + len(logits))].unsqueeze(1)
                    ranks.append((logits > target).sum(1) + 1)
                ranks = th.cat(ranks).float()
                results[name] = {
                    "R@1": (ranks <= 1).float().mean().item(),
                    "R@5": (ranks <= 5).float().mean().item(),
                    "R@10": (ranks <= 10).float().mean().item(),
                    "MRR": (1.0 / ranks).mean().item(),
                    "MR": ranks.median().item(),
                    "n": len(ranks),
                }
                print(f"  {name}: " + ", ".join(f"{k}={v:.4f}" for k, v in results[name].items()), flush=True)
            except Exception as e:  # e.g. feature dims don't match this model
                msg = f"{type(e).__name__}: {e}".splitlines()[0][:300]
                results[name] = {"error": msg}
                print(f"  {name}: n/a ({msg})", flush=True)
        del net
        th.cuda.empty_cache()
        return results


# ------------------------------------------------------------------- tables
def write_tables(out_dir, rows, metric):
    cols = ["train_file", "batch_mode", "image_only", "best_epoch", "best_val"] + list(EVAL_SETS)

    def cell(r, name):
        m = r["eval"].get(name, {})
        if metric not in m:
            return "n/a"
        return f"{m[metric]:.1f}" if metric == "MR" else f"{100 * m[metric]:.2f}"

    table = [[r["train_file"], r["batch_mode"], str(r["image_only"]), str(r["best_epoch"]),
              f"{r['best_val']:.2f}"] + [cell(r, n) for n in EVAL_SETS] for r in rows]

    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(table)

    with open(out_dir / "results.md", "w") as f:
        f.write(f"{metric} (%) of the best checkpoint per configuration\n\n")
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "---|" * len(cols) + "\n")
        for t in table:
            f.write("| " + " | ".join(t) + " |\n")


def main():
    args = parse_args()
    os.chdir(REPO)
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    cfg = OmegaConf.to_container(OmegaConf.load(REPO / "configs" / f"{args.config}.yaml"), resolve=True)
    ckpt_root = Path(cfg.get("checkpoint_dir", "checkpoints"))
    ckpt_mode = cfg.get("checkpoint_mode", "max")
    if cfg.get("save_top_n_checkpoints", 0) <= 0:
        sys.exit("save_top_n_checkpoints must be > 0 in the config, otherwise there is no best model to evaluate.")

    input_dir = Path(args.input_dir)
    files = sorted(f for f in input_dir.iterdir() if f.is_file() and fnmatch.fnmatch(f.name, args.pattern))
    if not files:
        sys.exit(f"No files matching '{args.pattern}' in {input_dir}")

    out_dir = Path(args.out_dir)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    rows = [json.loads(l) for l in open(results_path)] if results_path.exists() else []
    done = {r["experiment_name"] for r in rows}

    configs = [(f, m, io) for f in files for m in args.batch_modes for io in args.image_only]
    print(f"{len(files)} train files x {len(args.batch_modes)} batch modes x "
          f"{len(args.image_only)} image_only = {len(configs)} runs ({len(done)} already done)")
    if args.extra:
        print(f"Extra train.py args: {' '.join(args.extra)}")

    evaluator = None
    failed = []
    for k, (f, mode, image_only) in enumerate(configs, 1):
        stem = f.stem
        name = run_name(stem, mode, image_only)
        tag = f"[{k}/{len(configs)}] {name}"
        if name in done:
            print(f"{tag}: already in results, skipping")
            continue

        ckpt_dir = ckpt_root / name
        if ckpt_dir.exists():
            # Leftovers from an interrupted run would be mistaken for this run's best.
            old = ckpt_dir.with_name(f"{name}.old_{int(time.time())}")
            print(f"{tag}: moving stale checkpoints {ckpt_dir} -> {old}")
            shutil.move(str(ckpt_dir), str(old))

        # Train/val/test features come from HOW2_EMBS (same dir as the dev5 eval
        # sets), not from whatever the config says; extra args can still override.
        embs_args = [f"--{k}={HOW2_EMBS}" for k in ("train_embs_path", "val_embs_path", "test_embs_path")
                     if not any(a.startswith(f"--{k}=") for a in args.extra)]
        cmd = [sys.executable, "-u", "train.py", args.config,
               f"--train_ids_path={f}", f"--base_exp={stem}", f"--batch_mode={mode}",
               f"--image_only={image_only}", f"--experiment_name={name}", *embs_args, *args.extra]
        log = out_dir / "logs" / f"{name}.log"
        print(f"{tag}: training -> {log}", flush=True)
        start = time.time()
        with open(log, "w") as lf:
            lf.write("### CMD: " + " ".join(cmd) + "\n\n")
            lf.flush()
            rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)
        elapsed = time.time() - start
        if rc != 0:
            print(f"{tag}: train.py FAILED (exit {rc}) after {elapsed:.0f}s, see {log}", flush=True)
            failed.append(name)
            continue

        best = best_checkpoint(ckpt_dir, ckpt_mode)
        if best is None:
            print(f"{tag}: no checkpoint found in {ckpt_dir}", flush=True)
            failed.append(name)
            continue
        ckpt, epoch, val = best
        print(f"{tag}: trained in {elapsed / 60:.1f} min, best {ckpt.name}; evaluating", flush=True)

        if evaluator is None:
            evaluator = Evaluator(cfg, args.eval_workers)
        row = {
            "experiment_name": name, "train_file": stem, "batch_mode": mode, "image_only": image_only,
            "best_epoch": epoch, "best_val": val, "checkpoint": str(ckpt),
            "train_minutes": round(elapsed / 60, 1), "extra_args": args.extra,
            "eval": evaluator.evaluate(ckpt, image_only),
        }
        rows.append(row)
        with open(results_path, "a") as rf:
            rf.write(json.dumps(row) + "\n")
        write_tables(out_dir, rows, args.metric)

    if rows:
        write_tables(out_dir, rows, args.metric)
    print("=" * 60)
    print(f"Finished. {len(rows)} runs in {out_dir}/results.md, {len(failed)} failed")
    for n in failed:
        print(f"  FAILED: {n} ({out_dir}/logs/{n}.log)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
