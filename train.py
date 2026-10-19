"""Train / evaluate an AVLnet-style audio-video retrieval model on pre-extracted
V-JEPA2 + DINOv2 + WavLM features.

The training procedure -- Gated Embedding Units per modality, dot-product
similarity, MMS contrastive loss, R@1/5/10 + median-rank retrieval evaluation --
is unchanged from the original AVLnet (Rouditchenko et al., Interspeech 2021).
Functional differences from the original repo's train.py:
  - No DAVEnet audio CNN / nframes handling / audio zero-padding, since audio is
    already a single pooled WavLM vector per clip.
  - No text branch, word2vec loading, or tri-modal fusion, since the pipeline in
    this repo (extract_features.py) doesn't produce captions.
  - Train/val/test are each selected by a plain-text id-list file
    (train_ids_path/val_ids_path/test_ids_path) rather than being hardcoded to
    YouCook2/MSR-VTT pickle paths, so you can point this at any dataset whose
    features were extracted with extract_features.py just by swapping those files.
  - Instead of CLI flags, all settings come from a YAML file in `configs/`
    (loaded with OmegaConf, so `${...}` interpolation works -- e.g. an
    experiment_name built from other fields, see configs/default.yaml), and
    training progress/metrics are logged to Weights & Biases.

Example:
    python train.py default  --experiment_name=my_exp # loads configs/default.yaml + other params to override

To try a new setting, copy configs/default.yaml to configs/my_run.yaml, edit it,
then `python train.py my_run`.
"""
from __future__ import absolute_import, division, unicode_literals, print_function

import os
import random
import time
from functools import partial

import numpy as np
import torch as th
import torch.optim as optim
import wandb
from tqdm import tqdm as std_tqdm
from pathlib import Path

tqdm = partial(std_tqdm, dynamic_ncols=True)

th.backends.cudnn.benchmark = True

from src.loss import MMS_loss, Siglip_loss
from src.metrics import AverageMeter, compute_metrics, print_computed_metrics
from src.utils import (
    load_config,
    build_dataloader_negmine,
    build_dataloader,
    build_model,
    checkpoint_filename,
    is_better_metric,
    update_top_checkpoints,
)

CONFIGS_DIR = Path(__file__).parent / "configs"

def train_one_batch(model, opt, data, loss_fun):
    video = data["video"].cuda()
    audio = data["audio"].cuda()

    opt.zero_grad()
    with th.set_grad_enabled(True):
        audio_out, video_out = model(video, audio)
        
        loss = loss_fun(audio_out, video_out)
        loss.backward()
        opt.step()
    return loss.item()

@th.no_grad()
def eval_retrieval(model, 
                   eval_dataloader, 
                   dataset_name,
                   eval_lang_retrieval=False,
                   wandb_log=False, 
                   step=None):

    model.eval()
    mode_str = "audio given input video clips" if eval_lang_retrieval else "video clips given input audio"
    print(f"Evaluating retrieval on {dataset_name} data: retrieving {mode_str}")

    ranks = []

    for data in eval_dataloader:
        video = data["video"].cuda()
        audio = data["audio"].cuda()

        audio_out, video_out = model(video, audio)

        logits = th.matmul(audio_out, video_out.t())
        if eval_lang_retrieval:
            logits = logits.t()

        # The correct match for row i sits on the diagonal, so its rank is the
        # number of distractors scoring strictly higher, plus one.
        target_scores = logits.diagonal().unsqueeze(1)
        ranks.append((logits > target_scores).sum(dim=1) + 1)

    ranks = th.cat(ranks).float()

    metrics = {
        "acc": (ranks <= 1).float().mean().item(),
        "R@5": (ranks <= 5).float().mean().item(),
        "R@10": (ranks <= 10).float().mean().item(),
        "MRR": (1.0 / ranks).mean().item(),
        "MR": ranks.median().item(),   # kept as "MR" for backward compat with V1's key
    }

    print(f"{dataset_name} metrics: " +
          ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    if wandb_log:
        wandb.log(
            {f"{dataset_name}_{k.lower()}": v for k, v in metrics.items()},
            step=step,
        )

    model.train()
    return metrics

def run_training(
    dataset_name: str = None,
    embeddings_path: str = None,
    train_ids_path: str = None,
    val_ids_path: str = None,
    test_ids_path: str = None,
    image_dim: int = 768,
    video_feat_dim: int = 1024,
    audio_dim: int = 768,
    embd_dim: int = 4096,
    normalize: bool = True,
    image_only: bool = False,
    batch_size: int = 128,
    batch_size_val: int = 10000,
    loss: str = "mms",
    lr: float = 1e-4,
    lr_decay: float = 1.0,
    epochs: int = 15,
    seed: int = 1,
    num_thread_reader: int = 8,
    pretrain_path: str = "",
    checkpoint_dir: str = "checkpoints",
    experiment_name: str = None,
    n_display: int = 3,
    verbose: int = 1,
    eval_lang_retrieval: int = 0,
    eval_only: bool = False,
    checkpoint_metric: str = "acc",
    checkpoint_mode: str = "max",
    save_top_n_checkpoints: int = 1,
    early_stopping: bool = False,
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 0.0,
    wandb_project: str = "avlnet",
):
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    run_checkpoint_dir = os.path.join(checkpoint_dir, experiment_name) if experiment_name else checkpoint_dir
    if run_checkpoint_dir and not os.path.isdir(run_checkpoint_dir):
        os.makedirs(run_checkpoint_dir)

    wandb.init(
        project=wandb_project,
        name=experiment_name,
        config={
            "embeddings_path": embeddings_path,
            "train_ids_path": train_ids_path,
            "val_ids_path": val_ids_path,
            "test_ids_path": test_ids_path,
            "image_dim": image_dim,
            "video_feat_dim": video_feat_dim,
            "audio_dim": audio_dim,
            "embd_dim": embd_dim,
            "normalize": normalize,
            "image_only": image_only,
            "batch_size": batch_size,
            "loss": loss,
            "lr": lr,
            "lr_decay": lr_decay,
            "epochs": epochs,
            "checkpoint_metric": checkpoint_metric,
            "checkpoint_mode": checkpoint_mode,
            "save_top_n_checkpoints": save_top_n_checkpoints,
            "early_stopping": early_stopping,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_min_delta": early_stopping_min_delta,
            "seed": seed,
        },
    )

    dataloader = build_dataloader_negmine(train_ids_path, embeddings_path, normalize, image_only, batch_size, num_thread_reader, shuffle=True, drop_last=True, split="train_how2")
    dataloader_val = build_dataloader_negmine(val_ids_path, embeddings_path, normalize, image_only, batch_size_val, num_thread_reader, shuffle=False, drop_last=False, split="val_how2")
    dataloader_test = build_dataloader_negmine(test_ids_path, embeddings_path, normalize, image_only, batch_size_val, num_thread_reader, shuffle=False, drop_last=False, split="dev5_how2")

    net = build_model(image_dim, video_feat_dim, audio_dim, embd_dim, image_only)

    if pretrain_path:
        net.load_checkpoint(pretrain_path)

    net.cuda()
    net = th.nn.DataParallel(net)
    net.train()

    if loss == "mms":
        loss_op = MMS_loss()
    else:
        loss_op = Siglip_loss()
        
    loss_op.cuda()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    global_step = 0

    if eval_only:
        if dataloader_val is not None:
            val_metrics = eval_retrieval(net, dataloader_val, "val", eval_lang_retrieval, wandb_log=True, step=global_step)
        if dataloader_test is not None:
            test_metrics = eval_retrieval(net, dataloader_test, "test", eval_lang_retrieval, wandb_log=True, step=global_step)
            
        print("\n" + "=" * 60)
        print("FINAL RETRIEVAL RESULTS")
        print("=" * 60)

        if val_metrics is not None:
            print(
                f"VAL : "
                f"acc={val_metrics['acc']:.4f} | "
                f"R@5={val_metrics['R@5']:.4f} | "
                f"R@10={val_metrics['R@10']:.4f} | "
                f"MRR={val_metrics['MRR']:.2f} |"
                f"MR={val_metrics['MR']:.2f}"
            )

        if test_metrics is not None:
            print(
                f"TEST: "
                f"acc={test_metrics['acc']:.4f} | "
                f"R@5={test_metrics['R@5']:.4f} | "
                f"R@10={test_metrics['R@10']:.4f} | "
                f"MRR={test_metrics['MRR']:.2f} |"
                f"MR={test_metrics['MR']:.2f}"
            )

        print("=" * 60)


        
        return

    if dataloader is None:
        raise ValueError("train_ids_path is required unless eval_only is set.")

    dataset_size = len(dataloader.dataset)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    best_metric = None
    epochs_without_improvement = 0
    top_checkpoints = []

    if checkpoint_mode not in ("max", "min"):
        raise ValueError("checkpoint_mode must be 'max' or 'min'.")
    if save_top_n_checkpoints < 0:
        raise ValueError("save_top_n_checkpoints must be non-negative.")
    if early_stopping and val_ids_path is None:
        raise ValueError("early_stopping requires val_ids_path to be set.")

    for epoch in range(epochs):
        running_loss = 0.0

        if verbose:
            print("Epoch: %d" % epoch)

        dataloader.batch_sampler.set_epoch(epoch)

        end_time = time.time()
        for i_batch, sample_batch in enumerate(tqdm(dataloader)):
            if i_batch == 0:
                print(sample_batch["video"][0])
                
            data_time.update(time.time() - end_time)

            batch_loss = train_one_batch(net, optimizer, sample_batch, loss_op)
            batch_time.update(time.time() - end_time)
            running_loss += batch_loss
            global_step += 1

            if (i_batch + 1) % n_display == 0:
                avg_loss = running_loss / n_display
                if verbose:
                    print(
                        "Epoch %d, Epoch status: %.4f, Training loss: %.4f"
                        % (epoch + 1, batch_size * float(i_batch) / dataset_size, avg_loss)
                    )
                    print(
                        "Batch load time avg: %.4f, Batch process time avg: %.4f"
                        % (data_time.avg, batch_time.avg)
                    )
                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": avg_loss,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    step=global_step,
                )
                running_loss = 0.0
                batch_time = AverageMeter()
                data_time = AverageMeter()
            end_time = time.time()

        for param_group in optimizer.param_groups:
            param_group["lr"] *= lr_decay

        val_metrics = None
        test_metrics = None
        if dataloader_val is not None:
            val_metrics = eval_retrieval(net, dataloader_val, "val", eval_lang_retrieval, wandb_log=True, step=global_step)
        if dataloader_test is not None:
            test_metrics = eval_retrieval(net, dataloader_test, "test", eval_lang_retrieval, wandb_log=True, step=global_step)

        if dataloader_val is not None:
            if checkpoint_metric not in val_metrics:
                raise ValueError(
                    f"checkpoint_metric '{checkpoint_metric}' not found in validation metrics. "
                    f"Available metrics: {sorted(val_metrics.keys())}"
                )
            current_metric = val_metrics[checkpoint_metric]

            if save_top_n_checkpoints > 0:
                top_checkpoints = update_top_checkpoints(
                    net=net,
                    run_checkpoint_dir=run_checkpoint_dir,
                    epoch=epoch + 1,
                    metric_value=current_metric,
                    metric_name=checkpoint_metric,
                    top_checkpoints=top_checkpoints,
                    top_n=save_top_n_checkpoints,
                    mode=checkpoint_mode,
                )

            if is_better_metric(current_metric, best_metric, mode=checkpoint_mode, min_delta=early_stopping_min_delta):
                best_metric = current_metric
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            wandb.log({f"best_{checkpoint_metric}": best_metric}, step=global_step)

            if early_stopping and epochs_without_improvement >= early_stopping_patience:
                if verbose:
                    print(
                        f"Early stopping after epoch {epoch + 1} "
                        f"(no improvement in {epochs_without_improvement} epochs)."
                    )
                break

    if dataloader_val is not None:
        eval_retrieval(net, dataloader_val, "val", eval_lang_retrieval, wandb_log=True, step=global_step)
    if dataloader_test is not None:
        eval_retrieval(net, dataloader_test, "test", eval_lang_retrieval, wandb_log=True, step=global_step)

    print("\n" + "=" * 60)
    print("FINAL RETRIEVAL RESULTS")
    print("=" * 60)

    if val_metrics is not None:
        print(
            f"VAL : "
            f"R@1={val_metrics['acc']:.4f} | "
            f"R@5={val_metrics['R@5']:.4f} | "
            f"R@10={val_metrics['R@10']:.4f} | "
            f"MRR={val_metrics['MRR']:.2f} |"
            f"MR={val_metrics['MR']:.2f}"
        )

    if test_metrics is not None:
        print(
            f"TEST: "
            f"R@1={test_metrics['acc']:.4f} | "
            f"R@5={test_metrics['R@5']:.4f} | "
            f"R@10={test_metrics['R@10']:.4f} | "
            f"MRR={test_metrics['MRR']:.2f} | "
            f"MR={test_metrics['MR']:.2f}"
        )

    print("=" * 60)

    wandb.finish()


def main(config_name: str = None, **overrides):
    """Run training using `configs/<config_name>.yaml`.

    Example:
        python train.py default
    """
    cfg = dict(load_config(CONFIGS_DIR,config_name)) if config_name else {}
    cfg.update(overrides)
    
    print(f"Running training" + (f" with config `{config_name}`" if config_name else "") + ":")
    for k, v in sorted(cfg.items()):
        print(f"  {k}: {v}")
            
    run_training(**cfg)


if __name__ == "__main__":
    import fire
    fire.Fire(main)