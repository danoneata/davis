# AVLNet — SigLIP audio-visual alignment

Small SigLIP-style model that learns a joint audio-visual space on top of frozen
per-segment embeddings (V-JEPA2 video + DINOv2 image + WavLM audio) extracted
from the How2 300h dataset.

## Setup

```bash
uv sync
```

## Data paths

All embeddings and segment videos live under `/data/how2/clean_code/how2/`.

| Split  | Segments (`.mp4`)                                     | Per-segment embeddings                                | Grouped file (used for training)                                              | # segments |
| ------ | ----------------------------------------------------- | ----------------------------------------------------- | ----------------------------------------------------------------------------- | ---------- |
| train  | `/data/how2/clean_code/how2/segments/train/`          | `/data/how2/clean_code/how2/embeddings/train/`        | `/data/how2/clean_code/how2/embeddings/train_grouped_embeddings.pkl`          | 184,924    |
| val    | `/data/how2/clean_code/how2/segments/val/`            | `/data/how2/clean_code/how2/embeddings/val/`          | `/data/how2/clean_code/how2/embeddings/val_grouped_embeddings.pkl`            | 2,022      |
| dev5   | `/data/how2/clean_code/how2/segments/dev5/`           | `/data/how2/clean_code/how2/embeddings/dev5/`         | `/data/how2/clean_code/how2/embeddings/dev5_grouped_embeddings.pkl`           | 2,305      |

Segment IDs are of the form `<youtube_id>_<segment_idx>` and are shared between
the `.mp4` and the corresponding `.pkl` filename.

Per-segment `.pkl` schema:

```python
{
    "video_embedding": np.ndarray,   # (1024,) float32   — V-JEPA2 mean-pooled
    "image_embedding": np.ndarray,   # (768,)  float32   — DINOv2 CLS, 1 fps mean
    "audio_embedding": np.ndarray,   # (768,)  float32   — WavLM
}
```

Grouped `.pkl` schema (produced by `/data/how2/clean_code/segments/group_embeddings.py`):

```python
{
    "video_embeddings": np.ndarray,  # (N, 1024)
    "image_embeddings": np.ndarray,  # (N,  768)
    "audio_embeddings": np.ndarray,  # (N,  768)
}
```

Note: the grouped file drops the segment IDs — order comes from `os.listdir()`
at grouping time. If you need IDs (e.g. for qualitative inspection), reload the
per-segment `.pkl`s in `sorted(os.listdir(...))` order.

Feature-extraction code lives outside this repo at
`/data/how2/clean_code/segments/` (`cache_embeddings.py`, `group_embeddings.py`,
`split_into_segments*.py`, …).

## Training

```bash
uv run siglip.py <root> <experiment_name> [batch_size] [n_epochs]
```

- `<root>` — directory holding the three `*_grouped_embeddings.pkl` files
  (typically `/data/how2/clean_code/how2/embeddings`).
- `<experiment_name>` — used for the wandb run name and the checkpoint
  subdirectory.

Example:

```bash
uv run siglip.py /data/how2/clean_code/how2/embeddings run-00 128 100
```

Outputs:
- Checkpoint of the best-so-far model (by validation R@5) at
  `checkpoints/<experiment_name>/best.pt`. It stores
  `{epoch, model_state_dict, val_r@5, val_r@10, val_accuracy}`.
- Metrics logged to wandb (`train_loss`, `val_accuracy`, `val_r@5`, `val_r@10`,
  `best_val_r@5`, and the test-split equivalents).

## Qualitative retrieval viewer

`show_results.py` is a Streamlit app that displays top-K retrievals for a query
segment against a chosen split's pool.

```bash
uv run streamlit run show_results.py
```

Sidebar controls:
- **Split** — `val`, `dev5`, or `train`.
- **Direction** — `audio→video` or `video→audio`.
- **Checkpoint path** — defaults to `checkpoints/run-00/best.pt`.
- **Pool size** — `full`, or a smaller sample (`500` / `2000` / `10000` /
  custom) for a faster UI. Subsample is deterministic (fixed seed).
- **Top-K** — 1–20.

On first use of a split the app builds a single-file pool cache
(`cache/pool_<split>.pkl`, IDs + stacked embedding arrays); later runs skip
that step. `train` takes a few minutes on cold disk; `val` and `dev5` are ~5 s.
