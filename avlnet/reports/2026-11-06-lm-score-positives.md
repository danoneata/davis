# Using `lm_score` to select positive pairs

**Summary.** On the speaker-separated How2 split, choosing training pairs by alignment
score (`lm_score >= 8`) makes no measurable difference, including when the poorly aligned
clips are kept as negatives. Every arm peaks early (steps 5,000–9,000) and then overfits,
so the bottleneck is generalisation to unseen speakers, not noisy positives.

## Setup

- **Data:** speaker-separated split (`split_tag: final`; filelists
  `data/filelists/how2_{train,val,dev5}-final.json`).
  - Train: 152,218 segments with embeddings, unfiltered. 63,261 of them (41.6%) have
    `lm_score >= 8`.
  - Val and dev5: about 2,000 segments each, all with `lm_score >= 8`.
  - `how2-dev5-final-no_filter2k`: an unfiltered, speaker-disjoint test fold with the
    train distribution of `lm_score`.
- **Model:** `use_video: false`, `projection_dim: 128`, `loss: siglip`,
  `batch_size: 128`.
- **Schedule:** 144,400 steps, no early stopping (`patience: 0`), with the best
  checkpoint chosen by val R@5.
- **Code:** commit `abde30e` (`run-06-all` started on `ae406c9`, which has the same
  code path for that arm).

| Config | Training signal |
| --- | --- |
| `run-06-all` | every segment |
| `run-06-lm8` | only `lm_score >= 8` (63,261 segments) |
| `run-06-random63k` | random 63,261 segments (size control) |
| `run-06-lm8-negs` | `lm_score >= 8` as positives, plus 128 other segments per batch as negatives only (`positive_filter`) |

`run-06-lm8-negs` has the same number of positives per step as `run-06-lm8`, so the
two differ only in the extra negatives.

## Results

R@10 (%) shown as video→audio / audio→video, ranked against the full pool
(`uv run evaluate.py <config> --direction both`). One standard error is about 0.8
points at 15% R@10 on 2,000 segments.

| | `all` | `lm8` | `random63k` | `lm8-negs` |
| --- | --- | --- | --- | --- |
| `how2-val-final` | 15.2 / 14.7 | 15.5 / 15.5 | 14.7 / 16.1 | 14.9 / 15.8 |
| `how2-dev5-final` | 14.9 / 14.2 | 14.5 / 13.7 | 14.4 / 14.9 | 14.2 / 14.8 |
| `how2-dev5-final-no_filter2k` | 17.0 / 17.0 | 15.9 / 15.5 | 16.1 / 16.8 | 15.9 / 15.8 |
| `msr-vtt` | 7.2 / 4.5 | 7.3 / 5.1 | 8.5 / 4.7 | 8.6 / 5.1 |
| `youcook2` | 1.4 / 1.6 | 1.4 / 1.9 | 1.8 / 2.1 | 1.2 / 1.8 |
| Best step | 9,000 | 7,500 | 5,500 | 5,000 |
| Val R@10, peak → end | 15.2 → 13.7 | 15.6 → 11.1 | 14.8 → 10.6 | 15.1 → 11.9 |
| Final train loss | 0.63 | 0.24 | 0.23 | 0.37 |

## Findings

- **Alignment quality matters little.** `lm8` does not beat its size-matched random
  control, even on the filtered folds, which were selected with the same score.
  Its gain over random data there is about one point at most.
- **Extra negatives don't help.** `lm8-negs` vs `lm8` is within noise on every fold.
- **Overfitting limits the models.** Validation peaks during LR warmup, while train
  loss keeps falling. The smaller training sets fall furthest after the peak.
- **Training longer doesn't help.** `run-06-all` (14.9) matches
  `run-05-speaker-full` (15.2), which stopped early at step 11,500.
- **The filtered folds aren't easier.** The unfiltered `no_filter2k` scores as high
  or higher, so the LM's idea of "well aligned" doesn't match what this model finds
  easy to retrieve.

## Why 41% of the data costs nothing

Subsampling *segments* barely reduces diversity. Segments of the same video share
speaker, voice, recording conditions, scene and topic, so a random 41% of them still
covers almost every video and speaker:

| Subset | Segments | Videos | Speakers | Median segments per speaker |
| --- | --- | --- | --- | --- |
| all | 152,218 | 10,824 | 1,698 | 45 |
| `random63k` | 63,261 | 10,647 (98%) | 1,653 (97%) | 19 |
| `lm8` | 63,261 | 9,857 (91%) | 1,601 (94%) | 16 |
| random 15k | 15,000 | 7,668 (71%) | 1,496 (88%) | 5 |
| random 5k | 5,000 | 3,840 (35%) | 1,162 (68%) | 2 |

The last two rows are coverage counts only; no models were trained on them.

- **Data size is closer to the number of videos and speakers than of segments.** For
  generalising to unseen speakers, the useful data is closer to about 10.8k videos and
  1.7k speakers, and `random63k` keeps nearly all of them.
- **Extra segments delay overfitting without raising the peak.** All arms peak at
  14.8–15.6% val R@10, and the smaller sets just decline faster after it.
- **`lm8` drops more videos than random does** (91% vs 98% kept), which may offset any
  gain from better alignment.
- **Untested hypothesis:** the frozen features with a 128-d adapter may saturate on a
  few tens of thousands of varied pairs.

Keep this in mind for any data-selection experiment: a subset that keeps segment
count but not video or speaker coverage is a different (likely harder) condition. To
test it, train on 41% of the *speakers* with all their segments (about 63k segments)
and compare with `random63k`. Random 15k and 5k subsets would show where performance
starts to fall; from coverage alone, likely somewhere around 5–15k.

## Context: speaker separation

On the original How2 split, the same architecture (`run-03-full`) reaches 47.3% R@10
on dev5, against about 15% here. The original folds share no videos with train but do
share speakers. On `msr-vtt` and `youcook2`, which neither split contains, the
speaker-separated models are as good or better. Most of the drop therefore looks like
the old split overstating generalisation, but seen and unseen speakers have not yet
been compared directly within one model.

## Next steps

- Regularisation sweep: weight decay, adapter dropout, lower or constant LR, smaller
  adapters.
- Seen vs unseen speaker evaluation of `run-03-full` on the original folds.
- Speaker-subset vs segment-subset runs at equal size, plus random 15k and 5k runs.
