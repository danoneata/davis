# Optimisation hyperparameter sweep (`run-08`)

**Summary.** Tuning learning rate, weight decay and the schedule does not move the
held-out numbers: every arm lands within about half a point of `run-06-all` on
`how2-dev5-final`. The best setting, `lr: 3.0e-5, weight_decay: 0.1` on a 30k-step
schedule with early stopping, is level with or slightly ahead of `run-06-all` on every
fold (largest gain: MSR-VTT, +1.3 / +1.2 R@10), at about a fifth of the compute. Very
high weight decay (1.0) is the only clearly harmful setting. This supports the lm_score
report's conclusion that the limit is generalisation to unseen speakers, not
optimisation.

## Setup

- **Base:** `run-06-all` (`split_tag: final`, `use_video: false`, `projection_dim: 128`,
  `loss: siglip`, `batch_size: 128`, all 152,218 training segments).
- **New config keys** (`train.py`; defaults reproduce every earlier config): `lr`,
  `weight_decay`, `warmup_steps`, `min_steps` (no early stop before this step),
  `seed`.
- **Sweep schedule:** `max_steps: 30000`, `warmup_steps: 1000`, cosine to 0,
  `eval_every: 500`, `patience: 10`, `min_steps: 3000`. Every run stopped early, at
  7,500–26,500 steps. `run-06-all` used 144,400 steps with a 14,440-step warmup and
  peaked at step 9,000, while still in warmup at LR ≈ 1.9e-4.
- **Selection** used training-time val R@5/R@10 only. Test folds were evaluated once,
  for the finalists, after selection.
- **Code:** branch `antonio` on top of `af89d14`, with the config keys above added.

## Stage A: LR × weight decay

Val, the metric logged during training (%). `run-06-all` scores 9.8 / 15.2 on this
metric.

| lr \ wd | 0.01 (R@5 / R@10, best step) | 0.1 |
| --- | --- | --- |
| 3e-5 | 9.7 / 14.7, 12,000 | **10.2 / 16.0**, 20,000 |
| 1e-4 | 9.2 / 14.4, 5,000 | **10.7 / 15.8**, 15,000 |
| 3e-4 | 9.3 / 14.4, 9,000 | 9.5 / 14.4, 5,500 |
| 1e-3 | 9.8 / 15.1, 13,500 | 9.9 / 14.5, 3,500 |
| 3e-3 | 9.0 / 13.7, 6,000 | 9.4 / 14.0, 13,000 |

## Stage B: around `lr 1e-4, wd 0.1`, one change at a time

| Change | Val R@5 / R@10 | Best step |
| --- | --- | --- |
| none (stage A) | 10.7 / 15.8 | 15,000 |
| wd 0.3 | 9.7 / 14.3 | 5,500 |
| wd 1.0 | 8.6 / 13.9 | 4,500 |
| lr 3e-5, wd 0.3 | 10.2 / 15.6 | 20,500 |
| lr 3e-5, wd 1.0 | 8.3 / 12.8 | 10,000 |
| lr 5e-5 | 10.0 / 15.3 | 13,000 |
| lr 2e-4 | 9.0 / 14.2 | 2,500 |
| 10k-step schedule | 9.4 / 15.1 | 8,000 |
| 60k-step schedule | 10.1 / 16.2 | 10,000 |
| no warmup | 9.7 / 14.8 | 12,000 |
| 5k warmup | 9.6 / 15.6 | 7,000 |

Neighbouring arms often differ by 1.5 points (e.g. lr 1e-4 vs 2e-4), which is more
than seems plausible from the hyperparameter alone. Stage C measures how much is seed
noise.

## Stage C: seed replicates

Two leaders plus a control (old LR and weight decay, on the same short schedule), each
run with seeds 1–3 alongside the unseeded run. That gives 4 runs per arm; the ± is the
standard error over runs. Val is the training-time metric.

| Arm | Val R@5 | Val R@10 |
| --- | --- | --- |
| lr 1e-4, wd 0.1 | 10.2 ± 0.2 | 15.2 ± 0.2 |
| lr 3e-5, wd 0.1 | 9.9 ± 0.1 | 15.3 ± 0.3 |
| lr 3e-4, wd 0.01 (control) | 9.5 ± 0.1 | 14.3 ± 0.3 |

The single-run winners regress to the mean: the stage-A 15.8 and 16.0 were lucky seeds.

## Finalists on all folds

R@10 (%), video→audio / audio→video, full pool (`uv run evaluate.py <config>
--direction both`), mean of the 4 runs per arm. Seed-to-seed standard error is
0.1–0.5 points.

| | `run-06-all` (1 run) | lr 3e-5, wd 0.1 | lr 1e-4, wd 0.1 | lr 3e-4, wd 0.01 |
| --- | --- | --- | --- | --- |
| `how2-val-final` | 15.2 / 14.7 | 15.2 / 15.5 | 15.2 / 14.7 | 14.3 / 14.2 |
| `how2-dev5-final` | 14.9 / 14.2 | 15.2 / 15.2 | 14.8 / 14.8 | 14.7 / 14.7 |
| `how2-dev5-final-no_filter2k` | 17.0 / 16.9 | 17.4 / 17.4 | 16.8 / 16.8 | 16.9 / 16.3 |
| `msr-vtt` | 7.2 / 4.5 | 8.5 / 5.7 | 7.4 / 5.2 | 7.3 / 5.0 |
| `youcook2` | 1.4 / 1.6 | 1.3 / 2.4 | 1.3 / 2.1 | 1.3 / 1.7 |

## Findings

- **On How2 test folds the optimiser settings barely matter.** All arms sit within
  about 0.5 points on `how2-dev5-final`, which is within single-run noise (about 0.8).
  The val gap between the control and the leaders (about 1 point) mostly doesn't carry
  over to dev5, so part of it is selection on val.
- **`lr 3e-5, wd 0.1` is the best default.** It ties or beats `run-06-all` on every
  fold. It is clearly ahead only on MSR-VTT (+1.3 / +1.2), and on YouCook2 a2v by a
  smaller margin.
- **Lower LR and moderate weight decay help a little; large weight decay hurts.**
  wd 0.1 beats 0.01 at most LRs, and wd 1.0 costs 2–3 points. Low-LR runs peak later
  (13k–21k steps) and overfit more slowly.
- **The schedule shape is unimportant.** Warmup length (0 / 1k / 5k) and cosine length
  (10k / 30k / 60k) change little. `run-06-all`'s long warmup acted as a low-LR phase,
  which is why it does about as well as the tuned runs.
- **Short runs with early stopping are reliable.** They reach `run-06-all`'s numbers
  in 8–27k steps instead of 144k.

## Recommendation

Use `lr: 3.0e-5, weight_decay: 0.1, max_steps: 30000, warmup_steps: 1000, patience: 10,
min_steps: 3000` for future runs on this setup. It costs about a fifth of a
`run-06-all` run. Use `seed` and at least 2–3 replicates for any comparison under about
1.5 points. Next to test are architecture-side regularisation (adapter dropout, smaller
adapters) and data diversity (speakers, videos), not the optimiser.
