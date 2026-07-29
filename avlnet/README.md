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

Feature-extraction code for How2 itself lives outside this repo at
`/data/how2/clean_code/segments/` (`cache_embeddings.py`, `group_embeddings.py`,
`split_into_segments*.py`, …). For other datasets we use `extract_features.py`
below.

## Feature extraction (other datasets)

`extract_features.py` runs V-JEPA2 + DINOv2 + WavLM on every video of a
registered dataset and writes one `.pkl` per clip (same schema as the per-segment
files above). Audio is extracted on the fly with `ffmpeg` (mono, 16 kHz) into a
sibling `audios/` cache, so re-runs skip both ffmpeg and any already-processed
clips.

```bash
uv run extract_features.py <dataset_name>
```

Registered datasets (edit the `DATASETS` dict at the top of the file to add more):

| Dataset   | Videos path                          | Output embeddings                          |
| --------- | ------------------------------------ | ------------------------------------------ |
| msr-vtt   | `/data/how2/msr-vtt/TestVideo/`      | `/data/how2/msr-vtt/embeddings/`           |
| youcook2  | `/data/how2/youcook2/val/`           | `/data/how2/youcook2/embeddings/`          |

Video-file extensions handled: `.mp4`, `.mkv`, `.webm`, `.avi`, `.mov`. The
output filename is `<video_stem>.pkl`. Requires **both `ffmpeg` and `sox`** on
`PATH` — audio is extracted as `ffmpeg -vn` → `sox gain -4 channels 1 rate
16000`, matching the upstream How2 preprocessing so the trained model sees
consistent audio at eval time.

## Training

Every run is identified by a config name. The config lives at
`configs/<name>.yaml` and holds the full hyperparameter set. That name is also
used as the wandb run name and the checkpoint subdirectory, so config name and
run parameters stay in sync — to change anything, edit the YAML (or create a
new one) rather than passing CLI flags.

```bash
uv run siglip.py <config_name>
```

Config file schema (`configs/default.yaml`):

```yaml
root: /data/how2/clean_code/how2/embeddings   # dir holding *_grouped_embeddings.pkl
batch_size: 128
n_epochs: 100
projection_dim: 1024   # output dim of both adapters (joint embedding size)
use_video: true        # true: concat V-JEPA2 (1024) + DINOv2 (768) on the video side
                       # false: use DINOv2 (768) only
checkpoint_dir: checkpoints
```

To create a new experiment: drop a new file under `configs/` (e.g.
`configs/proj_512_no_video.yaml`) and reference it by its basename.

Examples:

```bash
# Run with the settings in configs/default.yaml
uv run siglip.py default

# A dedicated ablation config
uv run siglip.py proj_512_no_video
```

Outputs:
- Checkpoint of the best-so-far model (by validation R@5) at
  `checkpoints/<config_name>/best.pt`. It stores
  `{epoch, model_state_dict, config, val_r@5, val_r@10, val_accuracy}`, where
  the embedded `config = {"projection_dim", "use_video"}` is what
  `show_results.py` reads to rebuild the correct architecture on load.
- Metrics logged to wandb (`train_loss`, `val_accuracy`, `val_r@5`, `val_r@10`,
  `best_val_r@5`, and the test-split equivalents). The wandb run config
  captures the hyperparameters loaded from the YAML.

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
