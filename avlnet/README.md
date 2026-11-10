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
> matches the corresponding grouped rows. The sidecars already exist for all
> three splits. **Re-run it after regenerating any grouped `.pkl`, or
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

Optimisation keys (all optional; the defaults are what every config before
`run-08` used):

```yaml
lr: 3.0e-4          # default: 3e-4 * sqrt(batch_size / 128) (`scale_lr`)
weight_decay: 0.01  # AdamW, decoupled
warmup_steps: 1000  # linear warmup, then cosine to 0; default: 10% of the steps
max_steps: 30000    # overrides n_epochs * steps_per_epoch
eval_every: 500     # in steps; default: one epoch
patience: 10        # evaluations without val gain before stopping; 0 disables
min_steps: 3000     # no early stop before this step (e.g. during warmup)
seed: 1             # seeds torch/numpy (init and data order); default: unseeded
```

Write floats with a dot (`3.0e-4`): PyYAML reads `3e-4` as a string. `train.py`
casts `lr` and `weight_decay`, but other keys get no such help.

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

### Speaker-separated splits (`split_tag`)

The original How2 splits share speakers across folds. `split_tag` replaces all three
with the filelists `data/filelists/how2_{train,val,dev5}-<tag>.json`; `final` is the
speaker-separated split:

```yaml
split_tag: final
```

| Split | Segments | Speakers | From original train / val / dev5 |
| ----- | -------- | -------- | -------------------------------- |
| train | 152,218  | 1,698    | 148,785 / 1,519 / 1,914          |
| val   | 2,009    | 127      | 1,966 / 27 / 16                  |
| dev5  | 2,001    | 145      | 1,958 / 18 / 25                  |

The 25 segments of `c0eUfSt8Dro` in the train filelist have no embeddings and are
dropped. The splits cut across the original ones, so rows are looked up in the grouped
pickles through the sidecars of **all three**; run `make_grouped_ids.py` for `train`,
`val` and `dev5` if any is regenerated. See `how2_splits.py`.

> [!WARNING]
> Original and tagged splits overlap: most of `how2-dev5` is in `how2_train-final`, and
> most of `how2-dev5-final` is in the original train. `evaluate.py` defaults to the
> test split matching the config's `split_tag` and warns on any crossed How2 dataset;
> treat a crossed number as contaminated.

The tagged val and dev5 folds are themselves filtered to `lm_score >= 8`, while train
is not (41.6% of it passes). For an unfiltered test fold, use
`how2-dev5-final-no_filter2k`: 2,000 segments from the same speaker-separated pool,
with the train distribution of `lm_score`, and no speakers shared with train or val.

`filter` applies on top of the tagged train split. Note that
`how2-train-alignment-scores.jsonl` covers the original train only, so `from_list`
with it drops the tagged segments from the original val and dev5.

### Choosing positives (`positive_filter`)

`filter` removes segments entirely. `positive_filter` instead only stops them from
being *positive pairs*: a poorly aligned clip's own (video, audio) pair is a wrong
positive, but paired with any other clip it is still a valid negative.

```yaml
split_tag: final          # required: the predicate reads the filelist records
positive_filter:
  predicate: {field: lm_score, op: ">=", value: 8}   # same grammar as `filter`
  extra_negatives: 128
```

Each batch is `batch_size` segments passing the predicate plus `extra_negatives`
drawn at random from the rest, and an epoch is one pass over the passing segments.
The number of positive pairs per step therefore matches a run that `filter`s on the
same predicate, so the two differ only in the added negatives. Each loss drops the
matching-pair term of the negative-only items: `siglip` zeroes their diagonal
weight and normalises by the number of positives; `infonce` keeps only the
positive rows and columns as queries, with the rest left in every denominator. When
every item is positive, both are exactly the plain losses.

The `run-06` configs compare the options at a fixed 144,400-step budget, with no
early stopping (`patience: 0`):

| Config             | Training signal                                                |
| ------------------ | -------------------------------------------------------------- |
| `run-06-all`       | every segment                                                  |
| `run-06-lm8`       | only `lm_score >= 8` (63,261 segments)                         |
| `run-06-random63k` | random 63,261 segments (size control for `run-06-lm8`)         |
| `run-06-lm8-negs`  | `lm_score >= 8` as positives, plus 128 others per batch as negatives |

### Same-speaker negatives (`group_negatives`)

With random batches nearly every negative has a different speaker, so pairs can be
told apart by voice ↔ appearance alone, a cue that does not transfer to the
speaker-disjoint `final` folds. `group_negatives` builds each batch from a few
groups instead, so some negatives always share the speaker:

```yaml
split_tag: final          # required: the groups come from the filelist records
group_negatives: {by: speaker_id, per_group: 8}   # any record field, e.g. original_video_id
```

A batch of 128 is about 16 speakers × 8 segments. Speakers are drawn with
probability proportional to their segment count, and smaller ones give all they
have. Cross-speaker pairs are still negatives. With `positive_filter`, only
positives fill the groups, and the `extra_negatives` come from the same speakers
where possible. Batches are drawn independently, so an epoch is nominal and
`max_steps` sets the length. The `run-07-spk-*` configs are the four `run-06` arms
with this added.

### Replacing the audio side (`audio_features`)

`audio_features` swaps the WavLM audio embedding of every split (train, val, dev5)
for external per-segment features. Everything else stays the same, so a text-vision
model can be trained as a direct control for a speech-vision one:

```yaml
audio_features:
  path: data/embeddings/how2-clip-vitl14/text-final.pkl   # from `clip_baseline.py extract_text`
  key: text_embeddings                                    # or video_caption_embeddings
  l2_normalize: true
```

The pickle holds `ids` plus one or more `(n, 768)` arrays, and rows are matched **by
segment ID** (`how2_splits.with_audio_features`), never by position. A segment with
no feature raises an error instead of being dropped. `evaluate.py` applies the same
swap to How2 datasets and refuses non-How2 ones, which have no such features.
`show_results.py` loads checkpoints directly and does not apply it, so do not use
the viewer for these runs. `run-10-text-asr-8x16` is `run-09-spk-all-8x16` with the
CLIP text of the transcript as the audio side.

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
uv run evaluate.py <config_name> [--datasets a,b,c] [--direction v2a|a2v|both]
```

Reports R@10, MRR and `below`, ranked against the whole pool. `below` (in %) is the
share of wrong gallery items scoring below the correct one, averaged over queries:
one minus a normalised mean rank, i.e. per-query ROC AUC, with 50 as chance. R@10 and
MRR measure the head of the ranking, `below` the whole of it, so the two can disagree:
a model can keep the correct item out of the bottom of the list without bringing it
into the top 10. Dataset names are either a
How2 split (`how2-train`/`how2-val`/`how2-dev5`, optionally with a filelist tag, as
in `how2-dev5-final`) or a key of `extract_features.DATASETS`; the default set is
`how2-dev5,msr-vtt,youcook2`, with `how2-dev5-<tag>` for a config with a `split_tag`.

`--direction` picks which side asks the questions: `v2a` (the default, and what
training reports), `a2v`, or `both`. With a single direction the CSV keeps its
two-level metric/dataset header; `both` adds a third header row for the
direction.

Image datasets carry several spoken captions per image, and the two directions
are then genuinely different problems rather than transposes:

- `v2a` — 1,000 Flickr8k images queried against all 5,000 clips, with **all
  five** captions of the query image counting as correct. The reported rank is
  that of the best-scoring correct caption.
- `a2v` — each of the 5,000 clips queried against the 1,000 images, one correct
  answer each.

This is the pooled protocol Flickr8k numbers are normally reported under. It
replaced an earlier scheme that ran one 1,000-way retrieval per caption index
and averaged the five, so flickr8k numbers recorded before that change are not
comparable.

`retrieval.py` holds the scoring itself — `build_problem` (which gallery items
count as correct, per direction) and `score_problem` (rank of the best-scoring
positive). `show_results.py` scores through the same functions, so the viewer's
R@K and the table agree exactly.

They also have no video track, so evaluating a `use_video: true` checkpoint on
them is rejected with an explicit error; use a `use_video: false` config such as
`run-01-novideo`.

```bash
uv run evaluate.py run-01-novideo --datasets flickr8k --direction both
```

One caveat on the How2 splits: ranking is now against the whole pool, where it
used to be against 5,000-row chunks. Only `how2-train` (~185k segments) is large
enough for that to have mattered, and its numbers drop accordingly — the earlier
ones were flattered by the smaller effective gallery. `how2-val`, `how2-dev5`,
`msr-vtt` and `youcook2` all fit in one chunk and are unchanged.

### CLIP text baseline (`clip_baseline.py`)

A reference point for the speech model on the How2 dev5 `final` folds: the audio
side is replaced by text — the transcript (`text`) or the generated video
description (`video_caption`) — and both sides are embedded by CLIP ViT-L/14
(`openai/clip-vit-large-patch14`), the visual side from the segment's central frame.

```bash
uv run clip_baseline.py extract    # -> data/embeddings/how2-clip-vitl14/dev5-final-no_filter.pkl
uv run clip_baseline.py evaluate   # CLIP and SPEECH_MODELS, R@1/5/10, MRR, MedR, % below per fold
uv run clip_baseline.py extract_text   # CLIP text of train/val/dev5 for `audio_features` runs
```

`extract` covers all three dev5 filelists at once (`final` and
`final-no_filter2k` are subsets of `final-no_filter`) and, unlike the grouped
pickles, stores the segment IDs. Text longer than CLIP's 77 tokens is truncated and
flagged (`text_truncated`). `evaluate` scores every system on the same pool — the
fold's segments that have speech embeddings, via `how2_splits.load_split_ids` — through
`retrieval.py`, so its speech-model numbers match `evaluate.py`.

## Qualitative retrieval viewer

`show_results.py` is a Streamlit app that displays top-K retrievals for a query
segment against a chosen split's pool.

```bash
uv run streamlit run show_results.py
```

Sidebar controls:
- **Model** — a subdirectory of `checkpoints/`; the architecture is rebuilt from
  the config stored inside `best.pt`.
- **Dataset** — a How2 split (`how2-val`, `how2-dev5`, `how2-train`, or the
  speaker-separated `how2-val-final`, `how2-dev5-final`, `how2-train-final`) or a key of
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

- `all` uses every caption at once, which is exactly what `evaluate.py` scores:
  `audio→image` runs 5,000 queries against the 1,000-image gallery (one correct
  answer each), and `image→audio` runs 1,000 queries against a 5,000-clip gallery
  where **all five** captions of the query image count as correct. The viewer's
  R@10 at pool `full` equals the reported number for the matching direction.
- A single index narrows the pool to one caption per image, giving a clean
  1,000-way task in both directions. Useful for isolating a caption, but not a
  number `evaluate.py` reports.

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
