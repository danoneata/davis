# Same-speaker negatives (`group_negatives`)

**Summary.** Building each batch from 16 speakers × 8 segments makes the models
better at telling a clip from other clips of the same speaker, in every arm and on
both folds. Retrieval on the full pool barely changes: `run-07-spk-all` is level
with or slightly above `run-06-all`, and the smaller arms are level or worse. Few
gallery items share the query's speaker, so the gain there barely moves R@10. The
peak comes a little later, but the smaller arms overfit more after it.

## Motivation

The run-06 report (`2026-11-06-lm-score-positives.md`) found that generalising to
unseen speakers is the bottleneck. With random batches nearly every negative has a
different speaker, so a pair can be matched by voice ↔ appearance alone, a cue that
does not carry over to the speaker-disjoint `final` folds. If a batch holds several
segments per speaker, matching by speaker gives high logits to those same-speaker
negatives, and SigLIP penalises them.

We group by speaker rather than by video. A same-video negative also shares the
scene, topic and recording conditions, so speaker would be mixed up with all of
these, and more of those negatives would be false.

## Setup

- **Same as run-06:** the `final` split, `use_video: false`, `projection_dim: 128`,
  `loss: siglip`, `batch_size: 128`, 144,400 steps, no early stopping, and the best
  checkpoint chosen by val R@5.
- **The only change:** `group_negatives: {by: speaker_id, per_group: 8}`.
  - Each batch is about 16 speakers × 8 segments, so every item gets 7 same-speaker
    negatives. The pairs across speakers still count as negatives (about 120 per item).
  - Speakers are drawn in proportion to their segment count, so each segment is
    drawn about equally often. A speaker with fewer than 8 segments contributes all
    of them.
  - In `lm8-negs`, only positives fill the speaker groups. The 128 extra negatives
    come from the batch's speakers where possible (97% of them share a speaker with
    the batch).
- **Code:** commit `af89d14` (`GroupedBatchSampler` in `train.py`). The configs are
  `run-07-spk-{all,lm8,random63k,lm8-negs}`, each a copy of its run-06 config plus
  the key above.

## Results

R@10 (%), ranked against the full pool (`uv run evaluate.py <config> --direction
both`). Each cell is run-06 → run-07. One standard error is about 0.8 points at 15%
R@10 on 2,000 segments, so differences under about 1.5 points are noise.

**Video → audio**

| | `all` | `lm8` | `random63k` | `lm8-negs` |
| --- | --- | --- | --- | --- |
| `how2-val-final` | 15.2 → 16.9 | 15.5 → 14.8 | 14.7 → 15.0 | 14.9 → 15.5 |
| `how2-dev5-final` | 14.9 → 14.9 | 14.5 → 13.9 | 14.4 → 15.4 | 14.2 → 13.8 |
| `how2-dev5-final-no_filter2k` | 17.0 → 17.1 | 15.9 → 13.5 | 16.1 → 15.9 | 15.9 → 15.6 |
| `msr-vtt` | 7.2 → 7.3 | 7.3 → 6.2 | 8.5 → 8.6 | 8.6 → 7.7 |
| `youcook2` | 1.4 → 1.6 | 1.4 → 1.5 | 1.8 → 1.2 | 1.2 → 1.4 |

**Audio → video**

| | `all` | `lm8` | `random63k` | `lm8-negs` |
| --- | --- | --- | --- | --- |
| `how2-val-final` | 14.7 → 15.8 | 15.5 → 13.9 | 16.1 → 13.3 | 15.8 → 15.7 |
| `how2-dev5-final` | 14.2 → 15.7 | 13.7 → 12.5 | 14.9 → 15.1 | 14.8 → 13.9 |
| `how2-dev5-final-no_filter2k` | 17.0 → 16.5 | 15.5 → 12.7 | 16.8 → 16.2 | 15.8 → 15.0 |
| `msr-vtt` | 4.5 → 5.9 | 5.1 → 4.2 | 4.7 → 5.2 | 5.1 → 5.0 |
| `youcook2` | 1.6 → 2.1 | 1.9 → 1.5 | 2.1 → 2.1 | 1.8 → 1.5 |

**Training dynamics.** Val R@10 here is the training-time number (`val_r@10` in
wandb), and train loss is the last logged value.

| | `all` | `lm8` | `random63k` | `lm8-negs` |
| --- | --- | --- | --- | --- |
| Best step | 9,000 → 10,500 | 7,500 → 15,000 | 5,500 → 6,000 | 5,000 → 6,500 |
| Val R@10 peak | 15.2 → 16.9 | 15.6 → 17.3 | 14.8 → 15.0 | 15.1 → 15.5 |
| Val R@10 at end | 13.7 → 13.8 | 11.1 → 9.7 | 10.6 → 8.4 | 11.9 → 8.5 |
| Final train loss | 0.63 → 1.90 | 0.24 → 0.82 | 0.23 → 0.84 | 0.37 → 1.34 |

The training-time val peak is above the evaluated val R@10 (`lm8`: 17.3 vs 14.8)
because the checkpoint is chosen by R@5, not R@10.

## Does the speaker cue weaken?

On each speaker-disjoint fold, every video query is scored against all audio clips.
For each query we compute how often its positive scores above the distractors from
the same speaker, and separately above the distractors from other speakers. The
average distractor from the same speaker is also compared with the average
distractor from other speakers: a smaller similarity gap means less clustering by
speaker. Same-speaker distractors are about 5% of the gallery (103 of 2,009 per
query on val, 55 of 2,001 on dev5); about 3 of them come from the same video.

| Fold | Run | Pos > same-spk | Pos > other-spk | Sim gap |
| --- | --- | --- | --- | --- |
| val-final | `all` | 65.7 → 68.3 | 89.8 → 90.5 | 0.217 → 0.208 |
| | `lm8` | 66.3 → 68.6 | 89.9 → 89.5 | 0.231 → 0.158 |
| | `random63k` | 66.3 → 68.6 | 89.9 → 91.4 | 0.256 → 0.231 |
| | `lm8-negs` | 66.3 → 67.8 | 90.1 → 90.6 | 0.251 → 0.213 |
| dev5-final | `all` | 63.5 → 66.4 | 89.5 → 89.8 | 0.223 → 0.210 |
| | `lm8` | 64.0 → 66.8 | 89.2 → 88.9 | 0.228 → 0.157 |
| | `random63k` | 64.5 → 66.4 | 89.7 → 90.4 | 0.256 → 0.228 |
| | `lm8-negs` | 63.7 → 65.5 | 90.1 → 90.6 | 0.252 → 0.215 |

## Findings

- **The grouping does what it was meant to.** Discrimination against same-speaker
  distractors improves by 1.5–3 points in all 8 arm × fold cells. Discrimination
  against other speakers stays at about 90%, and same-speaker clips cluster less.
- **It hardly moves full-pool retrieval.** Same-speaker distractors are only about
  5% of the gallery, so a 3-point gain on them changes few top-10 lists.
  - `all` is the best arm. Its clearest gain is on val (+1.7 / +1.1), but val also
    selects the checkpoint, so that gain is inflated. On the held-out folds it is
    within about two standard errors (dev5 +0.0 / +1.5, no_filter2k +0.1 / −0.5).
  - `lm8` is worse on no_filter2k (−2.4 / −2.8); `random63k` and `lm8-negs` are
    within noise.
- **The speaker cue is still the main difficulty.** The positive beats a
  same-speaker distractor only about 66% of the time, against 90% for other
  speakers.
- **Overfitting moves, it doesn't go away.** The peak comes later in every arm. But
  the three 63k-segment arms end 1.4–3.4 points lower than in run-06. The harder
  task, with 2–4× higher final train loss, lets the model fit more of the
  per-speaker detail of a small set.

## Next steps

- **Stronger grouping:** keep only same-speaker pairs as negatives (mask the
  cross-speaker pairs), or use more segments per speaker (e.g. 4 × 32).
- **Grouping by video** (`by: original_video_id`), to separate the effect of
  matching the speaker from matching the whole video.
- **Combine with the regularisation sweep:** the overfitting after the peak is still
  the main limit, and the grouping makes it worse on the smaller sets.
