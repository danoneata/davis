"""Train / evaluate an AVLnet-style audio-video retrieval model on pre-extracted
V-JEPA2 + DINOv2 + WavLM features.

The training procedure -- Gated Embedding Units per modality, dot-product
similarity, MMS contrastive loss, R@1/5/10 + median-rank retrieval evaluation --
is unchanged from the original AVLnet (Rouditchenko et al., Interspeech 2021).
The only functional differences from the original repo's train.py:
  - No DAVEnet audio CNN / nframes handling / audio zero-padding, since audio is
    already a single pooled WavLM vector per clip.
  - No text branch, word2vec loading, or tri-modal fusion, since the pipeline in
    this repo (extract_features.py) doesn't produce captions.
  - Train/val/test are each selected by a plain-text id-list file
    (--train_ids_path/--val_ids_path/--test_ids_path) rather than being hardcoded
    to YouCook2/MSR-VTT pickle paths, so you can point this at any dataset whose
    features were extracted with extract_features.py just by swapping those files.

Example:
    python train.py \\
        --embeddings_path data/embeddings/youcook2 \\
        --train_ids_path data/youcook_train_ids.txt \\
        --val_ids_path data/youcook_val_ids.txt \\
        --checkpoint_dir checkpoints/youcook2 \\
        --batch_size 64 --epochs 15 --lr 1e-4

    # Evaluate an existing checkpoint on a held-out test split, no training:
    python train.py \\
        --embeddings_path data/embeddings/youcook2 \\
        --test_ids_path data/youcook_test_ids.txt \\
        --pretrain_path checkpoints/youcook2/e15.pth \\
        --eval_only
"""
from __future__ import absolute_import, division, unicode_literals, print_function

import os
import random
import time
from functools import partial

import numpy as np
import torch as th
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm as std_tqdm


tqdm = partial(std_tqdm, dynamic_ncols=True)

th.backends.cudnn.benchmark = True

from args import get_args
from dataset import Feature_DataLoader
from loss import MMS_loss
from metrics import AverageMeter, compute_metrics, print_computed_metrics
from model import Net
from torchinfo import summary
import wandb

def build_model(args):
    video_dim = args.image_dim + args.video_feat_dim
    return Net(embd_dim=args.embd_dim, video_dim=video_dim, audio_dim=args.audio_dim)


def build_dataloader(ids_path, args, batch_size, shuffle, drop_last, split):
    if not ids_path:
        return None
    normalize = not args.no_normalize
    dataset = Feature_DataLoader(
        ids_path=ids_path,
        embeddings_path=args.embeddings_path,
        split=split,
        normalize=normalize,
    )
    print(f"{ids_path}: {len(dataset)} clips")
    if len(dataset) == 0:
        raise ValueError(f"No embeddings found for any id in {ids_path}.")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=args.num_thread_reader,
        shuffle=shuffle,
        drop_last=drop_last,
    )


def train_one_batch(model, opt, data, loss_fun):
    video = data["video"].cuda()
    audio = data["audio"].cuda()

    opt.zero_grad()
    with th.set_grad_enabled(True):
        audio_out, video_out = model(video, audio)
        sim_matrix = th.matmul(audio_out, video_out.t())
        loss = loss_fun(sim_matrix)
        loss.backward()
        opt.step()
    return loss.item()


def eval_retrieval(model, eval_dataloader, dataset_name, eval_lang_retrieval=False):
    model.eval()
    print("Evaluating retrieval on {} data".format(dataset_name))
    with th.no_grad():
        for data in eval_dataloader:
            video = data["video"].cuda()
            audio = data["audio"].cuda()
            audio_out, video_out = model(video, audio)
            m = th.matmul(audio_out, video_out.t()).cpu().detach().numpy()
            metrics = compute_metrics(m, eval_lang_retrieval=eval_lang_retrieval)
            print_computed_metrics(metrics)
    model.train()


def main():
    args = get_args()
    if args.verbose:
        print(args)
        
    experiment_name = f"how2_siglip_{args.batch_size}_{args.lr}_{args.epochs}_{args.projection_dim}"
    wandb.init(
            project="avlnet",
            name=experiment_name,
            config={
                "train_embeddings": args.train_embeddings,
                "val_embeddings": args.val_embeddings,
                "batch_size": args.batch_size,
                "n_epochs": args.epochs,
                "projection_dim": args.projection_dim,
                "use_video": args.use_video,
                "lr": args.lr,#scale_lr(batch_size, base_lr),
            }
        )

    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.checkpoint_dir and not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    dataloader = build_dataloader(args.train_ids_path, args, args.batch_size, shuffle=True, drop_last=True, split="train")
    dataloader_val = build_dataloader(args.val_ids_path, args, args.batch_size_val, shuffle=False, drop_last=False, split="val")
    dataloader_test = build_dataloader(args.test_ids_path, args, args.batch_size_val, shuffle=False, drop_last=False, split="dev5")

    net = build_model(args)
    
    summary(
        net,
        input_size=[
            (1, args.image_dim + args.video_feat_dim),  # video
            (1, args.audio_dim),                        # audio
        ],
        col_names=[
            "input_size",
            "output_size",
            "num_params",
            "kernel_size",
            "trainable",
        ],
    )

    if args.pretrain_path:
        net.load_checkpoint(args.pretrain_path)

    net.cuda()
    net = th.nn.DataParallel(net)
    net.train()

    loss_op = MMS_loss()
    loss_op.cuda()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    if args.eval_only:
        if dataloader_val is not None:
            eval_retrieval(net, dataloader_val, "val", args.eval_lang_retrieval)
        if dataloader_test is not None:
            eval_retrieval(net, dataloader_test, "test", args.eval_lang_retrieval)
        return

    if dataloader is None:
        raise ValueError("--train_ids_path is required unless --eval_only is set.")

    dataset_size = len(dataloader.dataset)
    batch_time = AverageMeter()
    data_time = AverageMeter()

    for epoch in range(args.epochs):
        running_loss = 0.0

        if dataloader_val is not None:
            eval_retrieval(net, dataloader_val, "val", args.eval_lang_retrieval)
        if dataloader_test is not None:
            eval_retrieval(net, dataloader_test, "test", args.eval_lang_retrieval)

        if args.verbose:
            print("Epoch: %d" % epoch)

        end_time = time.time()
        for i_batch, sample_batch in enumerate(tqdm(dataloader)):
            data_time.update(time.time() - end_time)

            batch_loss = train_one_batch(net, optimizer, sample_batch, loss_op)
            batch_time.update(time.time() - end_time)
            running_loss += batch_loss

            if (i_batch + 1) % args.n_display == 0 and args.verbose:
                print(
                    "Epoch %d, Epoch status: %.4f, Training loss: %.4f"
                    % (epoch + 1, args.batch_size * float(i_batch) / dataset_size,
                       running_loss / args.n_display)
                )
                print(
                    "Batch load time avg: %.4f, Batch process time avg: %.4f"
                    % (data_time.avg, batch_time.avg)
                )
                running_loss = 0.0
                batch_time = AverageMeter()
                data_time = AverageMeter()
            end_time = time.time()

        for param_group in optimizer.param_groups:
            param_group["lr"] *= args.lr_decay

        if args.checkpoint_dir:
            path = os.path.join(args.checkpoint_dir, "e{}.pth".format(epoch + 1))
            net.module.save_checkpoint(path)

    if dataloader_val is not None:
        eval_retrieval(net, dataloader_val, "val", args.eval_lang_retrieval)
    if dataloader_test is not None:
        eval_retrieval(net, dataloader_test, "test", args.eval_lang_retrieval)


if __name__ == "__main__":
    main()
