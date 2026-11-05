# CLAUDE.md

Guidance for Claude Code when working in `avlnet/`. `README.md` is the reference for
what everything does — this file only covers what is easy to get wrong.

## Project

Contrastive adapters learning a joint audio-visual space on top of frozen per-segment
embeddings (V-JEPA2 + DINOv2 + WavLM) from How2 300h. The objective is a config key
(`loss:`): `siglip` (sigmoid pairwise) or `infonce` (symmetric CLIP).

## Conventions

- **Run everything through `uv run`** (`uv run train.py <config>`, `uv run python -c ...`).
  Bare `python` has no numpy/torch. Setup is `uv sync`.
- **Configs are the interface.** `run_training` is driven by `configs/<name>.yaml`; the
  config name is also the wandb run name and the checkpoint subdir. To change a
  hyperparameter, edit the YAML or add a new one — do not add CLI flags.
- **Set `WANDB_MODE=offline` for smoke runs**, so throwaway runs stay out of the real
  wandb project. Point `checkpoint_dir` at a scratch path too.
- **Streamlit apps can be checked headlessly** with `streamlit.testing.v1.AppTest`
  (`AppTest.from_file("show_results.py").run()`, then inspect `at.exception`) — faster
  and more conclusive than starting a server.

## Data

Lives on the host under `/data/how2/`, not in the repo; `data/` and `cache/` are
gitignored. Nothing here is reproducible from a fresh clone without that host state.

**Before touching the grouped embedding pickles, read the warning in `README.md`
("Data paths").** They store no segment IDs and their row order is raw `os.listdir()`
order, not sorted order. Anything joining scores or IDs onto training rows must go
through the `<split>_grouped_ids.json` sidecar written by `make_grouped_ids.py`;
getting this wrong mislabels training data silently instead of erroring.

## Layout

- `train.py` — `DualEncoder`, the losses (`LOSSES`), training loop, `Dataset`;
  `evaluate.py` — retrieval metrics.
- `retrieval.py` — how a retrieval problem is scored: `build_problem` (which gallery
  items count as correct, given a direction `v2a`/`a2v`) and `score_problem` (rank of
  the best-scoring positive). Both `evaluate.py` and `show_results.py` go through it,
  so a change here moves the reported metrics *and* the viewer together — which is the
  point. `train.evaluate` deliberately does not: it keeps its own diagonal-only
  ranking so training-time numbers stay comparable across old runs.
- `filtering.py` — the config-driven training-subset filters (`filter:` key).
- `how2_splits.py` — loads How2 as originally split or through a filelist (`split_tag:`
  key, e.g. the speaker-separated `final`). Tagged splits cut across the original ones,
  so results must not be crossed between the two: most of `how2-dev5` is training data
  for a `split_tag: final` model.
- `extract_features.py` — feature extraction for the non-How2 datasets (`DATASETS`).
- `show_results.py`, `show_alignment_scores.py` — Streamlit viewers.
