# AVLnet on pre-extracted V-JEPA2 / DINOv2 / WavLM features

This is a trimmed-down port of [roudimit/AVLnet](https://github.com/roudimit/AVLnet)
that keeps the **same overall architecture and training procedure**, but assumes
features have already been extracted (with your `extract_features.py`) instead of
computing them on the fly from raw video/audio.

## What's the same as upstream AVLnet

- **Gated Embedding Units** (`Gated_Embedding_Unit` / `Context_Gating` in `model.py`)
  project each modality into a shared embedding space — untouched.
- **MMS contrastive loss** (`loss.py`) — untouched, copied verbatim.
- **Retrieval metrics** (`metrics.py`) — R@1/R@5/R@10 + median rank, same computation.
- **Training loop shape** (`train.py`) — per-epoch eval, Adam, LR decay, checkpointing.
- The "video" modality is still a **concatenation of two visual features** (previously
  ResNet-152 `2d` + ResNeXt-101 `3d`; now DINOv2 `image_embedding` + V-JEPA2
  `video_embedding`), matched against a single "audio" modality with a dot product.

## What's different from upstream AVLnet

- **No DAVEnet audio CNN.** Upstream fed raw spectrograms through a learned conv
  net and mean/adaptive-pooled over time (with `nframes` tracking zero-padding).
  Since WavLM features here are already a single pooled vector per clip, that whole
  pipeline is gone — `GU_audio` now maps the WavLM vector straight to the shared
  embedding space.
- **No text branch / tri-modal fusion / word2vec.** `extract_features.py` doesn't
  produce captions, so `Sentence_Maxpool`, `Fused_Gated_Unit`, and all
  `tri_modal`/`tri_modal_fuse` code paths were removed.
- **Dataset selection is just three id-list text files** (see below) instead of
  hardcoded YouCook2/MSR-VTT pickle paths.

## Expected data layout

Point `extract_features.py` at a dataset (as in your existing script) so you get:

```
<embeddings_path>/<segment_id>.pkl   # one file per clip, each a dict with
                                      # 'video_embedding', 'image_embedding', 'audio_embedding'
```

Then create plain-text id-list files, one id per line (blank lines / `#` comments
ignored):

```
train_ids.txt
val_ids.txt
test_ids.txt
```

Any `segment_id` in these files that doesn't have a matching `.pkl` in
`embeddings_path` is skipped with a warning — so you can freely reuse the same
`selected_files` id lists your `extract_features.py` already reads via `read_file`.

Swapping datasets/splits is then just swapping which files you pass in — no code
changes needed.

## Training

```bash
python train.py \
    --embeddings_path data/embeddings/youcook2 \
    --train_ids_path data/youcook_train_ids.txt \
    --val_ids_path data/youcook_val_ids.txt \
    --checkpoint_dir checkpoints/youcook2 \
    --batch_size 64 --epochs 15 --lr 1e-4 --embd_dim 4096
```

Add `--test_ids_path` to also evaluate on a held-out test split at the end of every
epoch. Feature dimensions default to the models used in `extract_features.py`
(DINOv2-base=768, V-JEPA2-vitl=1024, WavLM-base-plus=768); override with
`--image_dim` / `--video_feat_dim` / `--audio_dim` if you swap backbones.

## Evaluating a checkpoint only

```bash
python train.py \
    --embeddings_path data/embeddings/youcook2 \
    --test_ids_path data/youcook_test_ids.txt \
    --pretrain_path checkpoints/youcook2/e15.pth \
    --eval_only
```

## Files

| File | Role |
|---|---|
| `model.py` | `Net` (Gated Embedding Units for audio + video) |
| `loss.py` | `MMS_loss` — max-margin softmax contrastive loss |
| `metrics.py` | R@1/5/10 + median rank retrieval metrics |
| `dataset.py` | `Feature_DataLoader` — loads pre-extracted `.pkl` features by id list |
| `args.py` | CLI arguments |
| `train.py` | Training / evaluation entry point |
| `utils.py` | `read_file` — id-list file reader (train/val/test splits) |
