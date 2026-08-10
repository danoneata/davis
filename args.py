"""Command-line arguments for training/evaluating AVLnet on pre-extracted features.

Trimmed down from the original AVLnet args.py: everything specific to raw
HowTo100M/YouCook2/MSR-VTT loading, the text branch, word2vec, and Apex
mixed-precision has been removed since it no longer applies once features are
pre-extracted. Dataset selection is now just three id-list text files.
"""
from __future__ import absolute_import, division, unicode_literals, print_function

import argparse


def get_args(description="AVLnet on pre-extracted V-JEPA2 / DINOv2 / WavLM features"):
    parser = argparse.ArgumentParser(description=description)

    # ---- data ----
    parser.add_argument("--embeddings_path", type=str, required=True,
                         help="Directory with one <segment_id>.pkl per clip, as produced by extract_features.py.")
    parser.add_argument("--train_ids_path", type=str, default=None,
                         help="Text file with one training segment id per line. Required unless --eval_only.")
    parser.add_argument("--val_ids_path", type=str, default=None,
                         help="Text file with one validation segment id per line. Omit to skip val evaluation.")
    parser.add_argument("--test_ids_path", type=str, default=None,
                         help="Text file with one test segment id per line. Omit to skip test evaluation.")

    # ---- feature dims (defaults match the models used in extract_features.py) ----
    parser.add_argument("--image_dim", type=int, default=768,
                         help="DINOv2 image embedding dim (768 for facebook/dinov2-base).")
    parser.add_argument("--video_feat_dim", type=int, default=1024,
                         help="V-JEPA2 video embedding dim (1024 for facebook/vjepa2-vitl-fpc64-256).")
    parser.add_argument("--audio_dim", type=int, default=768,
                         help="WavLM audio embedding dim (768 for microsoft/wavlm-base-plus).")
    parser.add_argument("--embd_dim", type=int, default=4096,
                         help="Dimension of the shared audio-video embedding space.")
    parser.add_argument("--no_normalize", action="store_true",
                         help="Disable per-modality L2 normalization before concatenation.")

    # ---- optimization ----
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--batch_size_val", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_decay", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_thread_reader", type=int, default=8)

    # ---- misc ----
    parser.add_argument("--pretrain_path", type=str, default="",
                         help="Path to a .pth checkpoint to warm-start the model from.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                         help="Where to save per-epoch checkpoints ('' to disable saving).")
    parser.add_argument("--n_display", type=int, default=20,
                         help="Print training loss every n_display batches.")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--eval_lang_retrieval", type=int, default=0,
                         help="If 1, evaluate video->audio retrieval instead of audio->video.")
    parser.add_argument("--eval_only", action="store_true",
                         help="Skip training; just run evaluation on val/test ids and exit.")

    args = parser.parse_args()
    return args
