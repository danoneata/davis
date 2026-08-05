# AVLNet — contrastive audio-visual alignment

Small dual-encoder model that learns a joint audio-visual space on top of frozen
per-segment embeddings (V-JEPA2 video + DINOv2 image + WavLM audio) extracted
from the How2 300h dataset. The contrastive objective is configurable: a
SigLIP-style sigmoid pairwise loss (the default) or symmetric InfoNCE.

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

> [!WARNING]
> **The grouped files drop the segment IDs, and their row order is raw
> `os.listdir()` order at grouping time — *not* sorted order.** Reloading the
> per-segment `.pkl`s sorted by name lines up with nothing; this was verified
> against the real data (200/200 spot-checks match `os.listdir`, 0/3 match
> `sorted`). `group_embeddings.py` also `continue`s past unreadable files, so
> the mapping is only recoverable while that directory listing is unchanged.
>
> `make_grouped_ids.py` freezes the ordering into a sidecar
> `<split>_grouped_ids.json` (one ID per row of the grouped array) so it stops
> depending on the live filesystem:
>
> ```bash
> uv run make_grouped_ids.py train
> ```
>
> It refuses to write unless a random sample of per-segment embeddings still
> matches the corresponding grouped rows. `train_grouped_ids.json` already
> exists (184,924 IDs). **Re-run it after regenerating any grouped `.pkl`, or
> after adding/removing/renaming files under `embeddings/<split>/`** — the
> spot-checks will catch a stale sidecar rather than let training silently use
> mislabeled data.

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

| Dataset   | Kind  | Media path                                        | Output embeddings                          |
| --------- | ----- | ------------------------------------------------- | ------------------------------------------ |
| msr-vtt   | video | `/data/how2/msr-vtt/TestVideo/`                   | `/data/how2/msr-vtt/embeddings/`           |
| youcook2  | video | `/data/how2/youcook2/val/`                        | `/data/how2/youcook2/embeddings/`          |
| flickr8k  | image | `/data/how2/flickr8k/flickr8k-images/Flicker8k_Dataset/` | `data/embeddings/flickr8k/`         |

Each dataset's `kind` picks its pipeline: `main` looks it up in the `EXTRACTORS`
table and calls the matching `extract_<kind>_dataset(dataset_name, cfg)`, so
supporting a new modality means writing one function and registering it there.

### Frame decoding

Frames are read by decoding forward from the start (`read_frames`), never by
seeking. Decord's random access (`get_batch`, `vr[i]`) goes through
`seek_accurate`, which on these files is both **wrong and unreliable**: on 15 of
the 3,179 youcook2 segments it spins at 100% CPU indefinitely, and on many others
it silently returns the wrong frame — for `-AwyG1JcMp8_0.mp4`, 57 of 64 sampled
frames disagreed with a sequential read, and cross-checking frame 800 against
`ffmpeg` showed the sequential frame is the correct one. A sequential pass is
also faster at segment length, so there is no tradeoff.

Each item additionally runs under a `time_limit` of `ITEM_TIMEOUT_SECONDS`, so a
pathological file is reported as `[error]` and skipped rather than wedging a long
run. The guard is SIGALRM-based and fires between decoded frames; it cannot
interrupt a single hung call inside a C extension, which is why the seeking was
removed rather than merely wrapped.

> [!NOTE]
> Embeddings extracted before this change came from seek-selected frames, so
> re-extracting a dataset will not reproduce them bit-for-bit **where the seeking
> was wrong**. How2's own `cache_embeddings.py` uses `vr[idx]` too, but its
> segments appear to be clean: on a 40-segment dev5 sample, seeking and reading
> forward return byte-identical frames for all 40, for both the 64-frame V-JEPA2
> set and the 1 fps set. The 184,924 train segments have not been checked.

Re-extracting How2 dev5 with `extract_features.py` reproduces
`/data/how2/clean_code/how2/embeddings/dev5` (see `compare_embeddings.py`):
audio at cosine 1.000000, image at median 1.000000 (minimum 0.999995), video at
median 0.999843 (minimum 0.998392). The wider spread on video is float16 —
V-JEPA2 is the one model loaded in half precision — and is not attributable to
decoding: the frames are identical and this pipeline is bit-deterministic across
reruns, so the residual is the hardware and library versions that produced the
stored set. Note `--rtol/--atol` defaults are tighter than fp16 noise, so
`allclose` reads 0% there while the cosines are ~1.

Video-file extensions handled: `.mp4`, `.mkv`, `.webm`, `.avi`, `.mov`, though
decord in this environment cannot seek or decode VP8/VP9 — all 96 `.webm` files
in youcook2 fail with `DECORDError` and are skipped. The
output filename is `<video_stem>.pkl`. Requires **both `ffmpeg` and `sox`** on
`PATH` — audio is extracted as `ffmpeg -vn` → `sox gain -4 channels 1 rate
16000`, matching the upstream How2 preprocessing so the trained model sees
consistent audio at eval time.

### Image datasets (Flickr8k)

Datasets marked `kind: image` in `DATASETS` are still images paired with spoken
captions, so there is no video track: V-JEPA2 is skipped entirely and only
DINOv2 + WavLM are run. Because a still is a single frame, the DINOv2 CLS
embedding is exactly what the 1 fps max-over-frames pooling would return for a
one-frame video, so it drops into the same 768-d image slot.

Flickr8k specifics — we use the 1,000-image **test** split
(`Flickr_8k.testImages.txt`), each with 5 spoken captions from the Flickr8k
Audio Caption Corpus (`wav2capt.txt` maps wav → image + caption index). The wavs
are already mono 16 kHz, so they are read in place rather than re-encoded; the
`sox gain -4` that `extract_audio` bakes into video-derived wavs is applied
numerically instead (WavLM's feature extractor has `do_normalize=False`, so
level matters in principle — in practice it moves R@10 by <0.1 pt).

Per-image `.pkl` schema (note the plural `audio_embeddings`):

```python
{
    "image_embedding": np.ndarray,   # (768,)    float32 — DINOv2 CLS
    "audio_embeddings": np.ndarray,  # (5, 768)  float32 — WavLM, one per caption
}
```

## Comparing two feature sets

When the same dataset exists in two places — a re-extraction, another machine,
someone else's pipeline — `compare_embeddings.py` reports how closely they agree:

```bash
uv run compare_embeddings.py data/embeddings/youcook2 \
    /data/how2/evaluation_data/youcook2/embeddings
```

Items are matched by filename stem and only shared keys are compared, so feature
sets with different schemas still line up on what they have in common. A 2-D
per-frame array is collapsed with `--pool` (`max`, the default, matching
extraction; `mean`; or `none` to compare per-frame arrays element-wise). Read the
cosine similarity: these embeddings are used for retrieval, where direction
matters and scale does not.

Worth knowing before comparing against `/data/how2/evaluation_data/youcook2`:
that feature set caps the image side at **10 frames** per clip, while extraction
here takes a true 1 fps sample (up to 72 frames on these segments). The audio
sides agree (median cosine 0.999999) but the image and video sides do not
(median 0.97), so the two are not interchangeable.

## Training

Every run is identified by a config name. The config lives at
`configs/<name>.yaml` and holds the full hyperparameter set. That name is also
used as the wandb run name and the checkpoint subdirectory, so config name and
run parameters stay in sync — to change anything, edit the YAML (or create a
new one) rather than passing CLI flags.

```bash
uv run train.py <config_name>
```

Config file schema (`configs/default.yaml`):

```yaml
root: /data/how2/clean_code/how2/embeddings   # dir holding *_grouped_embeddings.pkl
batch_size: 128
n_epochs: 100
projection_dim: 1024   # output dim of both adapters (joint embedding size)
use_video: true        # true: concat V-JEPA2 (1024) + DINOv2 (768) on the video side
                       # false: use DINOv2 (768) only
loss: siglip           # siglip (sigmoid pairwise) or infonce (symmetric CLIP)
checkpoint_dir: checkpoints
```

### Objectives (`loss`)

The training objective is a config key, resolved through the `LOSSES` table in
`train.py`; adding one means writing a
`<name>_loss(model, video_embeddings, audio_embeddings)` and registering it there.

| `loss`    | Definition                                                              |
| --------- | ----------------------------------------------------------------------- |
| `siglip`  | Sigmoid pairwise loss: each pair is an independent binary decision, with a learnable temperature and bias. Default. |
| `infonce` | Symmetric InfoNCE (CLIP): softmax cross-entropy over rows and columns, averaged. Uses the temperature; the bias cancels in a softmax and stays unused. |

Both share the same model — the loss is a training-time choice only, so a
checkpoint can be evaluated without knowing which one produced it. They also
share the temperature parameterisation and its initialisation (`exp(t_prime)`,
starting at 10), so two runs differing only in `loss` start from the same place.

The one thing to hold fixed when comparing them is **batch size**: InfoNCE
normalises over the batch and the sigmoid loss does not, which is precisely what
SigLIP claims as its advantage at scale. `run-03-full` (siglip) and
`run-04-full` (infonce) are such a pair, both at `batch_size: 128`; a conclusion
drawn there does not transfer to other batch sizes without re-running.

### Training on a subset (`filter`)

An optional `filter` key selects which training segments to keep. It applies to
the **training set only** — validation and test stay whole, so runs remain
comparable. It needs `<root>/train_grouped_ids.json` (see the warning above).

Two filter types. `random` takes a uniform sample:

```yaml
filter:
  type: random
  n: 9829
  seed: 42      # optional, defaults to 42
```

`from_list` keeps the segments whose record in a JSONL scores file satisfies a
predicate. YAML has no lambdas, so predicates are data: a leaf is
`{field, op, value}` with `op` one of `> >= < <= == != in`, and leaves nest
under `all` / `any` / `not`. Nothing is `eval`'d.

```yaml
filter:
  type: from_list
  path: data/filelists/how2-train-alignment-scores.jsonl
  predicate:
    field: denseav_score
    op: ">"
    value: 3.0

# compound:
#   predicate:
#     all:
#       - {field: denseav_score, op: ">", value: 3.0}
#       - {field: lm_score, op: ">=", value: 7}
```

Records are keyed by the stem of their `video_path`; the selection is the
intersection with the embeddings, so scored segments without embeddings (25 of
them in the alignment-scores file) are ignored. An empty selection, or one
smaller than `batch_size` under `drop_last`, is an error rather than a silent
no-op. See `filtering.py`.

### Training budget (`max_steps`)

Filtering shrinks the training set, so at fixed `n_epochs` it also shortens the
run — `denseav_score > 3` keeps 9,829 segments (5.3%), which at `batch_size:
128` is 7,600 steps against the full set's 144,400. That would confound data
quality with training length. `max_steps` overrides the
`n_epochs * steps_per_epoch` budget so both runs train equally long, with epochs
simply repeating the smaller set:

```yaml
n_epochs: 100
max_steps: 144400   # optional; when set, n_epochs only sets the epoch length
```

`max_steps` feeds the warmup and cosine schedule too, so the LR curve stretches
to match. Also set `eval_every` explicitly when filtering: it defaults to one
epoch, which on a small subset means far more frequent evaluations and hence a
much tighter effective `patience`.

The shipped `run-03` configs are a three-arm comparison — identical
architecture (`use_video: false`, `projection_dim: 128`), batch size, eval
cadence and 144,400-step budget, differing only in which segments they train on:

| Config                        | Training set                  | Segments |
| ----------------------------- | ----------------------------- | -------- |
| `run-03-full`                 | everything (no filter)        | 184,924  |
| `run-03-denseav3`             | `denseav_score > 3`           | 9,829    |
| `run-03-random9829`           | random, size-matched, seed 42 | 9,829    |

`run-03-full` needs no `max_steps`: its natural budget already *is* 144,400.

The wandb run config records `filter`, `n_train` (segments actually trained on)
and `max_steps`.

To create a new experiment: drop a new file under `configs/` (e.g.
`configs/proj_512_no_video.yaml`) and reference it by its basename.

Examples:

```bash
# Run with the settings in configs/default.yaml
uv run train.py default

# A dedicated ablation config
uv run train.py proj_512_no_video
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

## Evaluation

```bash
uv run evaluate.py <config_name> [--datasets a,b,c]
```

Reports R@10 and MRR (video→audio retrieval, ranked within a single batch that
holds the whole pool). Dataset names are either a How2 split
(`how2-train`/`how2-val`/`how2-dev5`) or a key of `extract_features.DATASETS`;
the default set is `how2-dev5,msr-vtt,youcook2`.

Image datasets carry several spoken captions per image. Rather than picking one
caption or stuffing all five into the same pool (which would put five copies of
each image in the gallery and make the diagonal ambiguous), the evaluation runs
one full retrieval per caption index and averages the metrics — so all 5,000
Flickr8k captions are used, each in a clean 1,000-way task.

They also have no video track, so evaluating a `use_video: true` checkpoint on
them is rejected with an explicit error; use a `use_video: false` config such as
`run-01-novideo`.

```bash
uv run evaluate.py run-01-novideo --datasets flickr8k
```

## Qualitative retrieval viewer

`show_results.py` is a Streamlit app that displays top-K retrievals for a query
segment against a chosen split's pool.

```bash
uv run streamlit run show_results.py
```

Sidebar controls:
- **Model** — a subdirectory of `checkpoints/`; the architecture is rebuilt from
  the config stored inside `best.pt`.
- **Dataset** — a How2 split (`how2-val`, `how2-dev5`, `how2-train`) or a key of
  `extract_features.DATASETS` (`msr-vtt`, `youcook2`, `flickr8k`).
- **Direction** — `audio→video` or `video→audio` (shown as `audio→image` /
  `image→audio` for image datasets).
- **Caption** (image datasets only) — `all`, or a single caption index.
- **Pool size** — `full`, or a smaller sample (`500` / `2000` / `10000` /
  custom) for a faster UI. Subsample is deterministic (fixed seed).
- **Top-K** — 1–20.

On first use of a dataset the app builds a single-file pool cache
(`cache/pool_<tag>.pkl`, IDs + stacked embedding arrays; `_img` suffix for image
datasets, whose payload has a different shape); later runs skip that step.
`train` takes a few minutes on cold disk; `val` and `dev5` are ~5 s.

### Image datasets in the viewer

Flickr8k renders as JPEGs and audio players, with each clip's transcript (from
`Flickr8k.token.txt`) printed underneath. As in the evaluation, it needs a
`use_video: false` checkpoint; picking one trained with video shows an error
instead of results.

The **Caption** control decides how the five captions per image are used:

- A single index gives a clean 1,000-way task and reproduces exactly one of the
  trials `evaluate.py` averages — the viewer's R@10 for caption `c` equals that
  trial's, and the mean over the five equals the reported number.
- `all` uses every caption at once: `audio→image` runs 5,000 queries against the
  1,000-image gallery (one correct answer each), and `image→audio` runs 1,000
  queries against a 5,000-clip gallery where **all five** captions of the query
  image count as correct. The reported ground-truth rank is the rank of the
  best-scoring correct answer, and every correct clip is marked ✅ in the results.

## Alignment-scores viewer

`data/filelists/how2-train-alignment-scores.jsonl` holds one record per train
segment (184,949 lines) with an `lm_score` (integer 1–10) and a
`denseav_score` (float), alongside the ASR transcript (`text`) and
`video_caption`. It is what `filter: {type: from_list}` reads.

`show_alignment_scores.py` browses it — video, transcript, caption and both
scores per segment, sorted by either score, with range filters, a text search
and score histograms.

```bash
uv run streamlit run show_alignment_scores.py
```
