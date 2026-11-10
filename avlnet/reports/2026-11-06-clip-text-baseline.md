# CLIP text baselines on the dev5 `final` folds

**Summary.** CLIP ViT-L/14, used zero-shot with the central frame on one side and the
transcript on the other ("CLIP-ASR"), is about level with the speech model
(`run-09-spk-all-8x16`) on the unfiltered `no_filter2k` fold: 18.6 vs 19.1 R@10 video →
text/audio, 20.8 vs 18.3 the other way. On the `final` fold it is far ahead (28.5 vs
16.1), but that fold is filtered to `lm_score >= 8`, which selects segments whose
transcript matches the video and so favours a text model. Retrieving the generated video
description ("CLIP-caption") reaches 46–48 R@10 on both folds. That is close to an upper
bound, not a baseline: the captions were generated from the video itself. The useful
finding is that **CLIP-ASR and the speech model get mostly different queries right.**
Summing their scores, with no tuning, takes `no_filter2k` R@10 from 19.1 to 29.8.
Several diagnostics show why. The speech model matches mostly on non-lexical cues,
such as the speaker's gender against the person on screen, and little on what is said.
A text-vision control (`run-10-text-asr-8x16`: same model, but the audio side is the
CLIP text of the transcript) confirms both halves of this. The words are learnable:
28.5 vs 16.1 R@10 on `final`, and 28.0 vs 15.7 on val. But on unfiltered data they
carry little: 21.1 vs 19.1 on `no_filter2k`. Speech and text models fused reach 31.0.

## Motivation

The speech model has never been compared with a reference outside its own family. CLIP
on the transcript is the natural cascade (speech → text → image–text model). It sees
the words only: no voice, no speaker identity, no acoustics. Comparing it with the
speech model shows how much of the speech model's retrieval comes from what is said,
and how much from other cues.

## Setup

- **Model:** `openai/clip-vit-large-patch14`, zero-shot, 768-d projections,
  L2-normalised.
- **Visual side:** one central frame per segment (`len(vr) // 2`), decoded sequentially
  with `extract_features.read_frames`. 20,273 / 20,273 segments decoded, 0 failures.
- **Text side:** `text` (the transcript) or `video_caption` from the filelist. CLIP
  truncates text at 77 tokens. 0.8% of transcripts were truncated (2.3% on `final`) and
  no captions were.
- **Pools:** exactly the segments the speech model is scored on
  (`how2_splits.load_split_ids`): `final` 2,001, `final-no_filter2k` 2,000,
  `final-no_filter` 20,273. Scoring goes through `retrieval.py`, so the speech-model
  numbers below reproduce the run-07/run-09 report.
- **Speech models:** `run-09-spk-all-8x16` (8 speakers × 16 segments per batch, the best
  held-out arm) and `run-07-spk-all` (16 × 8) as the reference. Both use DINOv2 + WavLM
  with no V-JEPA2 video (`use_video: false`).
- **Code:** `clip_baseline.py` (`extract`, then `evaluate`). Features are in
  `data/embeddings/how2-clip-vitl14/dev5-final-no_filter.pkl`.
- **Where the numbers come from:** the main table and the length breakdown come from
  `clip_baseline.py evaluate`. The `lm_score`, fusion and tie numbers, and the
  diagnostics in "What the speech model matches on" and in the run-10 breakdowns, came
  from one-off analyses over the same features and checkpoints. They are not in the
  repo. Run-10's main R@10 rows are from `clip_baseline.py evaluate --speech_models
  run-09-spk-all-8x16,run-10-text-asr-8x16`, and its val row from `evaluate.py`. The
  % below table comes from the same two commands (`retrieval.metrics_from_ranks`,
  `below`), with all seven speech/text configs. The rank-distribution table is
  one-off.
- **Noise:** one standard error is about 0.9 points at 19% R@10 on 2,000 segments, and
  about 1.1 points at 47%.

`v2a` is visual query → text/audio gallery, and `a2v` is the reverse.

## Results

R@10 (%) against the full pool.

| Direction | System | `final` | `no_filter2k` | `no_filter` (20k) |
| --- | --- | --- | --- | --- |
| v2a | CLIP-ASR | **28.5** | 18.6 | **4.8** |
| v2a | `run-07-spk-all` | 14.9 | 17.1 | 3.0 |
| v2a | `run-09-spk-all-8x16` | 16.1 | **19.1** | 3.3 |
| v2a | *CLIP-caption* | *48.2* | *46.0* | *15.2* |
| a2v | CLIP-ASR | **31.8** | **20.8** | **5.3** |
| a2v | `run-07-spk-all` | 15.7 | 16.4 | 3.0 |
| a2v | `run-09-spk-all-8x16` | 15.7 | 18.3 | 3.1 |
| a2v | *CLIP-caption* | *48.4* | *47.1* | *15.1* |

**% below** against the full pool, v2a / a2v. This is the share of wrong gallery items
scoring below the correct one, averaged over queries: 1 − (rank − 1) / (n − 1), i.e.
one minus a normalised mean rank, or per-query ROC AUC; 50 is chance. R@10 measures
the head of the ranking, and % below the whole of it, so the two can disagree.
Normalising by pool size also makes it comparable across folds: the 20k `no_filter`
pool gives about the same value as `no_filter2k`, while its R@10 is six times lower.
"Text" is run-10, described below.

| System | `how2-val-final` | `final` | `no_filter2k` | `no_filter` (20k) |
| --- | --- | --- | --- | --- |
| CLIP-ASR | – | 94.9 / 95.2 | 82.8 / 83.4 | 82.9 / 83.5 |
| Speech, random batches (`run-06-all`) | 88.7 / 89.1 | 88.9 / 88.6 | 90.3 / 90.3 | 90.0 / 90.1 |
| Speech, 16 × 8 (`run-07-spk-all`) | 89.5 / 89.5 | 89.2 / 88.6 | 90.8 / 90.6 | 90.6 / 90.4 |
| Speech, 8 × 16 (`run-09-spk-all-8x16`) | 89.1 / 89.4 | 90.7 / 90.5 | **92.2 / 92.2** | **91.8** / 91.7 |
| Speech, 4 × 32 | 89.8 / 89.8 | 89.9 / 89.3 | 91.8 / 91.9 | 91.7 / **91.8** |
| Speech, 2 × 64 | 88.9 / 89.4 | 89.0 / 89.0 | 91.5 / 91.4 | 91.3 / 91.2 |
| Speech, 1 × 128 | 86.3 / 84.8 | 86.8 / 86.1 | 89.6 / 89.8 | 89.4 / 89.7 |
| Text (`run-10-text-asr-8x16`) | **96.4 / 96.3** | **96.0 / 95.5** | 91.2 / 90.7 | 91.5 / 90.9 |
| *CLIP-caption* | – | *98.5 / 98.5* | *97.4 / 97.7* | *97.6 / 97.7* |

- **On the `lm_score`-filtered folds (val, `final`), the text models lead on both
  metrics.**
- **On the unfiltered folds the order flips:** the 8 × 16 speech model has the best
  % below (92.2 vs 91.2 for text, 82.8 for CLIP-ASR), even though its R@10 is lower
  (19.1 vs 21.1). The rank distribution on `no_filter2k`, v2a, shows why:

  | | Rank 1–10 | 11–100 | 101–500 | 501–1000 | > 1000 | Mean rank |
  | --- | --- | --- | --- | --- | --- | --- |
  | CLIP-ASR | 18.6 | 27.8 | 27.4 | 14.8 | 11.3 | 345 |
  | Speech, 8 × 16 | 19.1 | 40.0 | 33.0 | 6.7 | **1.4** | **156** |
  | Text (run-10) | **21.1** | 38.9 | 29.6 | 7.7 | 2.7 | 177 |

  Words are all-or-nothing: an uninformative transcript ("Okay.") leaves the correct
  clip anywhere in the list. The speech model's coarse cues (gender, voice) are always
  there, so it rarely fails badly, but they seldom pick out the one right clip.
- **Across batch compositions, % below agrees with R@10:** 8 × 16 or 4 × 32 is best,
  and 1 × 128 is worst on every fold. The gain from same-speaker negatives is small on
  both metrics, because other-speaker distractors make up 97% of the pool.

R@1 / MRR / median rank, `no_filter2k`, v2a:

| System | R@1 | R@5 | MRR | MedR |
| --- | --- | --- | --- | --- |
| CLIP-ASR | 4.6 | 13.0 | 0.095 | 128 |
| `run-09-spk-all-8x16` | 3.4 | 11.3 | 0.084 | 66 |
| *CLIP-caption* | *13.7* | *34.0* | *0.241* | *13* |

CLIP-ASR wins at the top of the list (R@1, R@5, MRR), and the speech model wins in the
tail (median rank 66 vs 128). When the transcript doesn't help, CLIP-ASR has nothing
else to fall back on, while the speech model still places the clip in the right area.

### Breakdown by `lm_score` (`no_filter2k`, v2a R@10)

| `lm_score` | n | CLIP-ASR | `run-09` 8×16 | CLIP-ASR + `run-09` | *CLIP-caption* |
| --- | --- | --- | --- | --- | --- |
| 1–4 | 390 | 17.9 | 17.4 | 29.2 | *38.2* |
| 5–6 | 316 | 14.2 | 20.6 | 23.4 | *36.1* |
| 7 | 473 | 16.7 | 18.4 | 26.2 | *43.1* |
| 8–10 | 821 | 21.8 | 19.6 | 34.5 | *55.2* |

On `no_filter2k`, CLIP-ASR gains 4–7 points from the `lm_score >= 8` segments, and the
speech model gains nothing. On `final` the same high-score segments give CLIP-ASR 28.5
but the speech model only 16.1, against 19.6 on `no_filter2k`. So `lm_score` alone
does not explain the speech model's lower `final` number. The folds also differ in
speakers and in the distractors each query is ranked against.

### By transcript length (`no_filter2k`, v2a R@10)

| Length (words) | n | CLIP-ASR | `run-09` 8×16 |
| --- | --- | --- | --- |
| 1–10 | 685 | 9.3 | 19.1 |
| 11–20 | 687 | 22.7 | 15.3 |
| 21–117 | 628 | 24.4 | 23.1 |

On short utterances ("Good.", "Okay.") CLIP-ASR collapses, while the speech model does
not. It must be using non-lexical cues there: voice, recording conditions, speaker.
Truncation at 77 tokens affects too few segments (24 on this fold) to read anything into.

### Complementarity and fusion (v2a R@10)

"Fusion" is the sum of the two systems' similarity matrices, each standardised per
query. It uses equal weights and no tuning.

| Fold | CLIP-ASR | `run-09` | Both hit | Only CLIP-ASR | Only `run-09` | Fusion |
| --- | --- | --- | --- | --- | --- | --- |
| `final` | 28.5 | 16.1 | 4.8 | 23.6 | 11.3 | **38.1** |
| `no_filter2k` | 18.6 | 19.1 | 3.4 | 15.2 | 15.6 | **29.8** |
| `no_filter` | 4.8 | 3.3 | 0.3 | 4.5 | 3.0 | **8.2** |

The overlap is what you would expect if the two systems were independent
(0.186 × 0.190 ≈ 3.5%, measured 3.4%). The two systems pick up almost entirely
different signals.

## What the speech model matches on

CLIP-ASR has the words and nothing else, so it is a natural control. Wherever the
speech model behaves differently from it, the speech model is using something other
than the words. Five pieces of evidence point the same way: the speech model matches
**mostly on non-lexical cues** (voice attributes such as gender, and probably
recording conditions), and very little on what is said.

**1. It ignores whether the words describe the scene.** Across `lm_score` bins on
`no_filter2k` the speech model stays at 17.4–20.6 R@10, about the noise level, while
CLIP-ASR rises by 4–7 points on the `lm_score >= 8` segments (see the breakdown above).
A model that matched content would gain on the segments whose content matches.

**2. It does not need words.** On utterances of 10 words or fewer, CLIP-ASR falls to
9.3 R@10, but the speech model holds at 19.1, its average.

**3. It is unaffected by talking-head shots.** About 23% of `no_filter2k` captions
mention talking, speaking or the camera. There the transcript says little about the
picture ("today I'll show you…"), and CLIP-ASR drops. The speech model does not:

| R@10, v2a | `no_filter2k` talking-head (n=462) | `no_filter2k` other | `final` talking-head (n=175) | `final` other |
| --- | --- | --- | --- | --- |
| CLIP-ASR | 12.8 | 20.4 | 41.1 | 27.3 |
| `run-09` 8×16 | 18.2 | 19.3 | 20.0 | 15.8 |

(The talking-head shots on `final` passed the `lm_score` filter, so their transcripts
match the picture, which is why CLIP-ASR is high there.)

**4. Its mistakes have the query's gender.** For each query, we take its top-10 wrong
answers and compare their gender with the query's. Gender is read from the caption
("a man" / "a woman" and similar), available for 75% of `no_filter2k` and 69% of
`final`. "Other speaker" drops distractors from the query's own speaker, so this is not
a single voice being recognised.

| Wrong answer in top-10 with query's gender | `no_filter2k` | `no_filter2k`, other speaker | `final` | `final`, other speaker |
| --- | --- | --- | --- | --- |
| Chance | 50.5% | 50.5% | 58.3% | 58.3% |
| CLIP-ASR | 74.8% | 63.3% | 84.0% | 72.1% |
| `run-09` 8×16 | **92.7%** | **90.1%** | **91.3%** | **89.2%** |

The speech model's wrong answers almost always have the right gender, across
speakers it has never seen. That fits matching voice to the person on screen at the
level of attributes. The caption label is noisy: the visible person need not be the
speaker, and some segments are voice-over. That noise pushes agreement towards
chance, so the real figure is probably higher. CLIP-ASR is also above chance, because
speech content correlates with gender ("my wife", topic).

**5. Its hits are independent of the words'.** The two systems get the same queries
right no more often than chance would predict (see the fusion table above). If the
speech model had learned even a noisy version of the content, its hits would
concentrate on the queries CLIP-ASR gets right.

**Speaker and topic are tangled.** In How2 a speaker usually records a whole series on
one topic in one place, so "same speaker" also means same topic and same setting.
Speaker clustering alone therefore does not prove a voice cue:

| Top-10 wrong answers from the query's speaker | `no_filter2k` | `final` |
| --- | --- | --- |
| Chance | 3.4% | 2.7% |
| CLIP-ASR | 33.2% | 51.2% |
| `run-09` 8×16 | 22.9% | 23.3% |

CLIP-ASR has no access to the voice at all, and it clusters by speaker more than the
speech model, through topic. (`run-09` was trained with same-speaker negatives, which
pushes those clips apart.) This is why the gender test above, restricted to other
speakers, is the cleaner evidence for a voice cue.

**Why it is plausible.** The audio side is one mean-pooled WavLM-base+ vector per
segment. Pooled self-supervised speech representations mostly encode who is speaking
and in what acoustic conditions, and keep little of the word sequence. The adapter can
only use what is in that vector, so speaker and channel attributes are the easiest
signal for it to learn.

**What this does not show.** None of the tests removes the words from the audio, so we
cannot say the lexical contribution is zero, only that it is small next to the other
cues. The direct test is in Next steps.

## Text-vision control: training on transcripts (run-10)

Is the speech model relying on voice cues because the words in How2 carry little, or
because its features don't let it reach them? To separate the two, we trained the
same model with the audio side replaced by the CLIP text embedding of the transcript.

- **Config:** `run-10-text-asr-8x16` is `run-09-spk-all-8x16` with one added key,
  `audio_features` (CLIP ViT-L/14 text of `text`, L2-normalised, matched to rows by
  segment ID). The visual side (DINOv2), loss, size, speaker-grouped 8 × 16 batches,
  early stopping and LR are all unchanged.
- **Training:** stopped early at step 13,500, with the best checkpoint at step 8,500
  (val R@5 18.0). It is a single seed, like run-09.
- **What it measures:** this model gets the words perfectly and nothing else. It sets
  how much the lexical channel alone supports *with this architecture and data*.

R@10 (%), v2a / a2v:

| | `how2-val-final` | `final` | `no_filter2k` | `no_filter` (20k) |
| --- | --- | --- | --- | --- |
| CLIP-ASR (zero-shot) | – | 28.5 / 31.8 | 18.6 / 20.8 | 4.8 / 5.3 |
| `run-09-spk-all-8x16` (speech) | 15.7 / 13.8 | 16.1 / 15.7 | 19.1 / 18.3 | 3.3 / 3.1 |
| `run-10-text-asr-8x16` (text) | **28.0 / 29.3** | **28.5 / 28.4** | **21.1 / 21.7** | **4.6 / 4.4** |

MRR on `no_filter2k`: 0.107 / 0.104 for text, against 0.084 / 0.082 for speech. On
% below, the text model leads on val and `final` (96.4 and 96.0, against 89.1 and 90.7
for speech) but trails on `no_filter2k` (91.2 vs 92.2); see the table under Results.

**Breakdowns on `no_filter2k`, v2a R@10.** "Fusion" is run-09 + run-10, scored as in
the fusion section above.

| Subset | n | CLIP-ASR | run-09 (speech) | run-10 (text) | Fusion |
| --- | --- | --- | --- | --- | --- |
| `lm_score` 1–4 | 390 | 17.9 | 17.4 | 17.4 | 29.2 |
| `lm_score` 5–6 | 316 | 14.2 | 20.6 | 18.4 | 26.3 |
| `lm_score` 7 | 473 | 16.7 | 18.4 | 19.9 | 31.9 |
| `lm_score` 8–10 | 821 | 21.8 | 19.6 | 24.7 | 33.3 |
| ≤ 10 words | 685 | 9.3 | 19.1 | 15.5 | 26.7 |
| Talking-head | 462 | 12.8 | 18.2 | 13.2 | 22.9 |
| Other shots | 1,538 | 20.4 | 19.3 | 23.5 | 33.5 |
| All | 2,000 | 18.6 | 19.1 | 21.2 | **31.0** |

(The analysis recomputes ranks itself, which is why "All" for run-10 reads 21.2
rather than the table's 21.1.)

**The cue tests, now with a control.** These are top-10 wrong answers, v2a, scored as
in "What the speech model matches on":

| | Same gender, other speaker (`no_filter2k` / `final`) | Same speaker (`no_filter2k` / `final`) | Hits shared with CLIP-ASR (`no_filter2k`), measured vs. if independent |
| --- | --- | --- | --- |
| Chance | 50.5% / 58.3% | 3.4% / 2.7% | – |
| CLIP-ASR | 63.3% / 72.1% | 33.2% / 51.2% | – |
| run-09 (speech) | **90.1% / 89.2%** | 22.9% / 23.3% | 3.4% vs 3.5% |
| run-10 (text) | 58.2% / 74.8% | 30.7% / 44.6% | 8.3% vs 3.9% |

The text model behaves like CLIP-ASR on every test, which is what a model that uses
only the words should do. Its wrong answers have the query's gender barely above
chance across speakers (58% vs 51%). They cluster by speaker through topic. Its hits
overlap with CLIP-ASR's twice as often as independence predicts. The speech model
differs from both on all three tests. This confirms the earlier diagnostics: the
tests do separate lexical from voice cues, and the speech model's 90% gender
agreement comes from the voice.

**What run-10 shows:**

- **The words are learnable, and the speech model is not using them.** On val and
  `final` the text model is 12–15 R@10 points ahead of the speech model, with the same
  visual side, data and training. The lexical signal is there, and the pooled-WavLM
  model leaves it unused.
- **On unfiltered data the lexical signal really is weak.** On
  `no_filter2k` the text model, with perfect transcripts, beats the speech model by
  only 2.0 / 3.4 points (about 2–3 standard errors, one seed). On `lm_score` 1–4 the
  two tie at 17.4. Where the speech does not describe the scene, perfect words are no
  better than voice cues. That is a large part of why the speech model can afford to
  ignore them.
- **The two signals add.** Fusing run-09 with run-10 gives 31.0 R@10 on `no_filter2k`,
  against 19.1 and 21.2 alone. Their hits overlap only a little more than independence
  predicts (5.5% vs 4.0%). Voice cues and words are close to orthogonal, and a speech
  model that used both would have room to gain about 10 points.
- **Voice cues win where words fail.** On utterances of 10 words or fewer and on
  talking-head shots, the speech model beats the trained text model (19.1 vs 15.5;
  18.2 vs 13.2). Where it loses is on segments whose words describe the scene.
- **Training on How2 adds little over zero-shot CLIP on high-`lm_score` data** (28.5
  vs 28.5 on `final`), and 2.5 points on `no_filter2k`. This is not a like-for-like
  comparison: zero-shot CLIP pairs its text with CLIP image features, and run-10 pairs
  it with DINOv2.

**What it does not show yet.** That the speech model *could* learn the lexical
signal. Run-10 gets the words for free. Whether the words survive in one mean-pooled
WavLM-base+ vector is the open question, and the WavLM-to-CLIP-text probe (Next
steps) answers it.

## Findings

- **On an unfiltered fold, a zero-shot text cascade matches the trained speech model.**
  On `no_filter2k` the two are within noise in both directions. CLIP-ASR has the
  better R@1/MRR, and the speech model has half the median rank.
- **`final` is a biased test fold for text baselines.** Its `lm_score >= 8` filter
  selects segments where the words describe the scene, which is where CLIP-ASR does
  best. Speech-vs-text comparisons should be made on `no_filter2k`, or reported on
  both.
- **The speech model matches mostly on who is speaking, not what is said.** It is flat
  across `lm_score`, holds up on utterances of 10 words or fewer and on talking-head
  shots, and its hits overlap with CLIP-ASR's no more than chance would predict. 90% of
  its top-10 wrong answers from *other* speakers have the query's gender (chance 50%).
  This fits the run-06/07 finding that speaker cues dominate, and says what the cue
  is: attributes of the voice (at least gender) matched to the person on screen, which
  carry over to unseen speakers.
- **The lexical channel is the obvious missing piece.** Untuned late fusion gains about
  11 R@10 points on `no_filter2k` and 22 on `final`.
- **CLIP-caption is a ceiling, not a competitor.** The caption was produced by a VLM
  looking at the same video, so 46–48 R@10 measures how well CLIP can recover a
  description of what it sees. It does not measure speech.

## Caveats

- **The transcript is not raw speech.** CLIP-ASR gets clean text for free, so it
  measures a cascade with a perfect front end.
- **Ties:** 440 of the 2,000 `no_filter2k` segments (593 on `final`) share an identical
  caption with another segment ("A woman is sitting in front of a camera and talking."
  appears 28 times). `score_problem` breaks ties in favour of the positive, which
  flatters CLIP-caption v2a by 0.8–1.5 R@10 points (`no_filter2k`: 46.0 optimistic,
  44.9 pessimistic). Identical transcripts are rare, and CLIP-ASR is unaffected (no
  change at one decimal place).
- **One frame per segment:** segments can run past 20 s, so a single central frame
  understates CLIP's visual side. Multi-frame pooling would raise both CLIP rows.
- **The fusion is evaluated on the test folds.** It has no tuned weights, but it has
  not been checked on val.
- **Caption-derived labels are rough.** Gender comes from keywords in a generated
  caption and describes the visible person. The talking-head subset comes from the
  keywords talk/speak/camera. Both are proxies, good enough to show a large effect but
  not to measure it precisely.

## Next steps

- **Probe pooled WavLM for words:** train a small map from the WavLM segment vector to
  the CLIP text embedding of its transcript (train fold), and measure audio →
  transcript retrieval on dev5. If it fails, the lexical signal is lost in the
  features, and the fix is the front end (Whisper encoder, later HuBERT/WavLM layers,
  attention pooling over frames). If it works, the content is there, and the fix is
  training (distil from run-10, or same-gender negatives to remove the shortcut).
- **Caption-vision run** (`key: video_caption_embeddings`), the upper bound from the
  original plan. The features are already in `text-final.pkl`, so it needs only a
  config.
- **Seeds for run-10**, since the `no_filter2k` gap over run-09 is 2–3 standard errors.
- **Remove the words, keep the voice:** re-extract WavLM for `no_filter2k` on
  time-reversed audio, or on audio whose ~200 ms chunks are shuffled, and re-score
  `run-09-spk-all-8x16`. If R@10 stays near 19, the lexical contribution is about zero.
- **Proper gender labels:** label the audio with a voice gender classifier. Then
  measure what a gender-only matcher scores, and how far the speech model gets beyond
  it, for example by ranking only against same-gender clips.
- **Fuse properly:** choose the fusion weight on `how2-val-final`. A further step is to
  feed CLIP text features of an ASR transcript into training as a third tower, or as a
  distillation target for the audio side.
- **Use real ASR** (e.g. Whisper on the segment audio) in place of the provided
  transcript, to price in recognition errors.
- **Multi-frame CLIP** (mean over 1 fps frames) for both CLIP rows.
- **Report `no_filter2k` as the primary test fold** in future comparisons. `final` is
  biased towards lexical alignment.
