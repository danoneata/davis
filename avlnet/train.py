import torch
import os
import glob
import numpy as np
from tqdm import tqdm
import fire
import sys
import wandb
import math
import yaml
from pathlib import Path

from filtering import evaluate_predicate, select_indices
from how2_splits import filelist_path, load_split_ids, load_split_with_features, read_records


CONFIGS_DIR = Path(__file__).parent / "configs"


def load_config(config_name: str) -> dict:
    path = CONFIGS_DIR / f"{config_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Config `{config_name}` not found at {path}. "
            f"Available: {sorted(p.stem for p in CONFIGS_DIR.glob('*.yaml'))}"
        )
    with open(path) as f:
        return yaml.safe_load(f) or {}



class Dataset(torch.utils.data.Dataset):
    def __init__(self, embeddings: dict, indices=None, positive=None):
        """`indices` restricts the dataset to a subset of rows; None uses all of them.

        `positive[k]` says whether item `k` (a position in the dataset, not a row)
        may be used as a positive pair; None means every item may.
        """
        self.embeddings = embeddings
        n = len(self.embeddings["video_embeddings"])
        self.indices = np.arange(n) if indices is None else np.asarray(indices)
        self.positive = (
            np.ones(len(self.indices), dtype=bool) if positive is None else np.asarray(positive)
        )
        assert len(self.positive) == len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return (
            self.embeddings["video_embeddings"][i],
            self.embeddings["image_embeddings"][i],
            self.embeddings["audio_embeddings"][i],
            self.positive[idx],
        )

    def __len__(self):
        return len(self.indices)


class PositivesWithNegativesSampler(torch.utils.data.Sampler):
    """Batches of `batch_size` positive items plus `extra_negatives` other items.

    An epoch is one shuffled pass over the positives (the tail is dropped); the extra
    items are drawn at random from the non-positives and serve only as negatives. This
    keeps the number of positive pairs per step equal to a run that trains on the
    positives alone, so the two differ only in the added negatives.
    """

    def __init__(self, positive: np.ndarray, batch_size: int, extra_negatives: int, seed: int = 0):
        self.positives = np.flatnonzero(positive)
        self.negatives = np.flatnonzero(~positive)
        self.batch_size = batch_size
        self.extra_negatives = extra_negatives
        self.rng = np.random.default_rng(seed)
        if len(self.positives) < batch_size:
            raise ValueError(
                f"{len(self.positives)} positives, fewer than batch_size={batch_size}."
            )
        if extra_negatives > len(self.negatives):
            raise ValueError(
                f"extra_negatives={extra_negatives} but only {len(self.negatives)} "
                f"non-positive segments."
            )

    def __iter__(self):
        order = self.rng.permutation(self.positives)
        for b in range(len(self)):
            batch = order[b * self.batch_size : (b + 1) * self.batch_size]
            if self.extra_negatives:
                extra = self.rng.choice(self.negatives, size=self.extra_negatives, replace=False)
                batch = np.concatenate([batch, extra])
            yield batch.tolist()

    def __len__(self):
        return len(self.positives) // self.batch_size


class GroupedBatchSampler(torch.utils.data.Sampler):
    """Batches made of a few groups (e.g. speakers), `per_group` items from each.

    Each batch draws groups without replacement, with probability proportional to
    their number of eligible items (so every segment is drawn about equally often),
    and takes `min(per_group, size)` random items from each until `batch_size` items
    are collected; the last group is cut short. Pairs across groups still count as
    negatives -- grouping only guarantees that some negatives share the group.

    `eligible` restricts which items fill the batch (the positives under a
    `positive_filter`). `extra_negatives` then adds that many non-eligible items:
    up to `extra_negatives // n_groups` from each group in the batch, topped up at
    random from all non-eligible items, so the negatives share groups too.

    An epoch is nominal (`n_eligible // batch_size` batches): batches are drawn
    independently, so the training length is set by `max_steps`.
    """

    def __init__(
        self,
        groups: np.ndarray,
        batch_size: int,
        per_group: int,
        seed: int = 0,
        eligible: np.ndarray = None,
        extra_negatives: int = 0,
    ):
        groups = np.asarray(groups)
        eligible = np.ones(len(groups), dtype=bool) if eligible is None else np.asarray(eligible)
        self.batch_size = batch_size
        self.per_group = per_group
        self.extra_negatives = extra_negatives
        self.rng = np.random.default_rng(seed)
        self.n_eligible = int(eligible.sum())
        if self.n_eligible < batch_size:
            raise ValueError(
                f"{self.n_eligible} eligible items, fewer than batch_size={batch_size}."
            )

        def by_group(mask):
            out = {}
            for i in np.flatnonzero(mask):
                out.setdefault(groups[i], []).append(i)
            return {g: np.array(v) for g, v in out.items()}

        self.members = by_group(eligible)
        self.group_names = list(self.members)
        sizes = np.array([len(self.members[g]) for g in self.group_names], dtype=float)
        self.group_probs = sizes / sizes.sum()
        self.others = by_group(~eligible)
        self.all_others = np.flatnonzero(~eligible)
        if extra_negatives > len(self.all_others):
            raise ValueError(
                f"extra_negatives={extra_negatives} but only {len(self.all_others)} "
                f"non-eligible items."
            )

    @property
    def n_groups(self) -> int:
        return len(self.group_names)

    def _take(self, pool: np.ndarray, k: int) -> np.ndarray:
        return self.rng.choice(pool, size=min(k, len(pool)), replace=False)

    def _batch(self) -> list[int]:
        # Enough groups to fill the batch even if every group is small; the greedy
        # loop stops as soon as `batch_size` items are in.
        n_draw = min(self.n_groups, self.batch_size)
        order = self.rng.choice(self.n_groups, size=n_draw, replace=False, p=self.group_probs)
        batch, used = [], []
        for gi in order:
            g = self.group_names[gi]
            take = self._take(self.members[g], min(self.per_group, self.batch_size - len(batch)))
            batch.extend(take.tolist())
            used.append(g)
            if len(batch) == self.batch_size:
                break

        if self.extra_negatives:
            per = self.extra_negatives // len(used)
            extra = []
            for g in used:
                if g in self.others:
                    extra.extend(self._take(self.others[g], per).tolist())
            extra = extra[: self.extra_negatives]
            if len(extra) < self.extra_negatives:
                pool = np.setdiff1d(self.all_others, extra, assume_unique=True)
                extra.extend(self._take(pool, self.extra_negatives - len(extra)).tolist())
            batch.extend(extra)
        return batch

    def __iter__(self):
        for _ in range(len(self)):
            yield self._batch()

    def __len__(self):
        return self.n_eligible // self.batch_size


def group_labels(spec: dict, ids: list[str], split_tag: str) -> np.ndarray:
    """The group of each of `ids`: field `spec["by"]` of its filelist record."""
    if split_tag is None:
        raise ValueError("`group_negatives` needs `split_tag`: it reads the filelist records.")
    records = read_records("train", split_tag)
    return np.array([records[i][spec["by"]] for i in ids])


def select_positives(spec: dict, ids: list[str], split_tag: str) -> np.ndarray:
    """Which of `ids` may be positive pairs, by `spec["predicate"]` on their records."""
    if split_tag is None:
        raise ValueError("`positive_filter` needs `split_tag`: it reads the filelist scores.")
    records = read_records("train", split_tag)
    return np.array([evaluate_predicate(spec["predicate"], records[i]) for i in ids])


class Adapter(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, output_dim),
            torch.nn.GELU(),
            torch.nn.Linear(output_dim, output_dim),
        )

    def forward(self, x):
        return self.linear(x) + self.mlp(x)


VIDEO_DIM = 1024
IMAGE_DIM = 768
AUDIO_DIM = 768

BASE_LR = 3e-4
BASE_BATCH_SIZE = 128


def scale_lr(batch_size: int) -> float:
    """Learning rate for `batch_size`, anchored at BASE_LR for BASE_BATCH_SIZE.

    Square-root rather than linear scaling, which is the rule of thumb for
    adaptive optimisers such as AdamW; linear scaling is an SGD rule and tends
    to overshoot here.
    """
    return BASE_LR * math.sqrt(batch_size / BASE_BATCH_SIZE)


class DualEncoder(torch.nn.Module):
    """Two adapters projecting the video and audio sides into a shared space.

    The objective is not part of the model: `t_prime` (a log-temperature) and `b`
    (a logit bias) are always present, and each loss in `LOSSES` uses the ones it
    needs -- InfoNCE leaves `b` untouched.
    """

    def __init__(self, projection_dim: int = 1024, use_video: bool = True):
        super().__init__()
        self.projection_dim = projection_dim
        self.use_video = use_video
        video_input_dim = (VIDEO_DIM + IMAGE_DIM) if use_video else IMAGE_DIM
        self.video_adapter = Adapter(video_input_dim, projection_dim)
        self.audio_adapter = Adapter(AUDIO_DIM, projection_dim)
        self.t_prime = torch.nn.Parameter(torch.tensor(2.3026))
        self.b = torch.nn.Parameter(torch.tensor(-10.0))

    def compose_video(self, video_embeddings, image_embeddings):
        if self.use_video:
            return torch.cat([video_embeddings, image_embeddings], dim=1)
        return image_embeddings

    def forward(self, video_embeddings, audio_embeddings, normalize=True):
        video_embeddings = self.video_adapter(video_embeddings)
        audio_embeddings = self.audio_adapter(audio_embeddings)
        if normalize:
            video_embeddings = torch.nn.functional.normalize(video_embeddings, p=2, dim=1)
            audio_embeddings = torch.nn.functional.normalize(audio_embeddings, p=2, dim=1)
        return video_embeddings, audio_embeddings


def siglip_loss(model, video_embeddings, audio_embeddings, positive=None):
    """Sigmoid pairwise loss: every pair is an independent binary decision.

    Nothing is normalised across the batch, so each pair contributes on its own --
    which is what makes this loss tolerant of very large batches.

    `positive` (bool, one per item) drops the matching-pair term of the items that are
    False; their pairings with other items still count as negatives. The sum is
    normalised by the number of positives, so all-True reproduces the plain loss.
    """
    zvideo, zaudio = model(video_embeddings, audio_embeddings, normalize=True)
    logits = torch.matmul(zvideo, zaudio.T) * torch.exp(model.t_prime) + model.b
    n = logits.shape[0]
    eye = torch.eye(n, device=logits.device)
    # +1 on the diagonal (the matching pair), -1 everywhere else.
    labels = 2 * eye - 1
    terms = torch.nn.functional.logsigmoid(labels * logits)
    if positive is None:
        return -torch.sum(terms) / n
    positive = positive.to(logits.device, dtype=logits.dtype)
    weights = 1 - eye + torch.diag(positive)
    return -torch.sum(weights * terms) / positive.sum().clamp(min=1)


def infonce_loss(model, video_embeddings, audio_embeddings, positive=None):
    """Symmetric InfoNCE (CLIP): a softmax over each row and each column.

    The two directions are averaged, so video->audio and audio->video are weighted
    equally. `model.b` is unused: a per-logit bias cancels in a softmax.

    `positive` (bool, one per item) keeps only the rows and columns of the True items
    as queries; the others stay in every softmax denominator, as negatives.
    """
    zvideo, zaudio = model(video_embeddings, audio_embeddings, normalize=True)
    logits = torch.matmul(zvideo, zaudio.T) * torch.exp(model.t_prime)
    labels = torch.arange(logits.shape[0], device=logits.device)
    cross_entropy = torch.nn.functional.cross_entropy
    if positive is None:
        return 0.5 * (cross_entropy(logits, labels) + cross_entropy(logits.T, labels))
    positive = positive.to(logits.device, dtype=logits.dtype)
    per_item = cross_entropy(logits, labels, reduction="none") + cross_entropy(
        logits.T, labels, reduction="none"
    )
    return 0.5 * torch.sum(positive * per_item) / positive.sum().clamp(min=1)


# The training objective, selected by the `loss` config key; adding one means
# writing a `<name>_loss(model, video_embeddings, audio_embeddings, positive=None)`
# and registering it here (`positive` comes from `positive_filter`).
LOSSES = {
    "siglip": siglip_loss,
    "infonce": infonce_loss,
}


@torch.no_grad()
def evaluate(model, val_dataloader):
    """Video->audio retrieval metrics, computed within each batch.

    Returns a dict with R@1/5/10 (fractions in [0, 1]), MRR (mean reciprocal
    rank) and MedR (median rank). All are derived from the rank of the matching
    audio for each video: rank 1 means retrieved first.
    """
    model.eval()
    ranks = []
    for batch in val_dataloader:
        video_embeddings, image_embeddings, audio_embeddings = batch[:3]
        video_embeddings = video_embeddings.cuda()
        image_embeddings = image_embeddings.cuda()
        video_embeddings = model.compose_video(video_embeddings, image_embeddings)
        audio_embeddings = audio_embeddings.cuda()

        video_embeddings, audio_embeddings = model(video_embeddings, audio_embeddings, normalize=True)
        logits = torch.matmul(video_embeddings, audio_embeddings.T)

        # The correct audio for video i sits on the diagonal, so its rank is the
        # number of distractors scoring strictly higher, plus one.
        target_scores = logits.diagonal().unsqueeze(1)
        ranks.append((logits > target_scores).sum(dim=1) + 1)

    ranks = torch.cat(ranks).float()
    return {
        "r1": (ranks <= 1).float().mean().item(),
        "r5": (ranks <= 5).float().mean().item(),
        "r10": (ranks <= 10).float().mean().item(),
        "mrr": (1.0 / ranks).mean().item(),
        "medr": ranks.median().item(),
    }


def run_training(
    experiment_name: str,
    root: str,
    batch_size: int = 2,
    n_epochs: int = 200,
    projection_dim: int = 1024,
    use_video: bool = True,
    loss: str = "siglip",
    checkpoint_dir: str = "checkpoints",
    log_every: int = 50,
    eval_every: int = None,
    patience: int = 10,
    select_metric: str = "r5",
    drop_last: bool = True,
    filter: dict = None,
    max_steps: int = None,
    split_tag: str = None,
    positive_filter: dict = None,
    group_negatives: dict = None,
    lr: float = None,
    weight_decay: float = 0.01,
    warmup_steps: int = None,
    min_steps: int = 0,
    seed: int = None,
    audio_features: dict = None,
):
    """Train the adapters on precomputed How2 embeddings.

    Everything is indexed by optimizer step rather than epoch, so runs with
    different `batch_size` remain comparable on the same x-axis.

    `loss` names an objective in `LOSSES`. It is a training-time choice only: both
    objectives share the same model, so a checkpoint can be evaluated without
    knowing which one produced it. Note that `infonce` normalises over the batch
    and `siglip` does not, which makes InfoNCE the more batch-size-sensitive of
    the two -- comparisons between them hold at the batch size they were run at.

    By default the learning rate is derived from `batch_size` as
    BASE_LR * sqrt(batch_size / BASE_BATCH_SIZE) (see `scale_lr`); an explicit `lr`
    overrides that (and then does not follow `batch_size`). With the default, when
    scaling `batch_size`, keep `n_epochs` fixed -- the number of data passes then
    stays constant, and the learning rate follows automatically. Holding the step
    count constant instead would also multiply the number of data passes,
    confounding batch size with training length.

    `eval_every` is in steps and defaults to one epoch. `patience` is counted in
    evaluations without improvement on `select_metric` (0 disables early
    stopping). The best checkpoint is always kept, so early stopping only saves
    compute -- it cannot change the reported numbers.

    `filter` selects which training segments to keep (see `filtering.py`); it
    applies to the training set only, so validation and test stay comparable
    across runs. `max_steps` overrides the `n_epochs * steps_per_epoch` budget,
    which is what keeps a filtered run as long as an unfiltered one instead of
    confounding data quality with training length. Since a filtered set also
    makes epochs shorter, set `eval_every` explicitly when filtering, or the
    evaluation cadence -- and with it the effective patience -- shifts too.

    `weight_decay` is AdamW's (decoupled) decay. `warmup_steps` is the length of the
    linear warmup before the cosine decay; None means 10% of the total steps.
    `min_steps` blocks early stopping before that step (e.g. during warmup, where
    run-05-speaker-full stopped); evaluations still count towards `patience`.
    `seed`, if set, seeds torch and numpy before the model and data order are built.

    `split_tag` trains, validates and tests on the filelists
    `data/filelists/how2_{train,val,dev5}-<split_tag>.json` instead of the original
    How2 splits (see `how2_splits.py`).

    `positive_filter` (`{predicate, extra_negatives}`, needs `split_tag`) uses only the
    training segments whose filelist record satisfies `predicate` as positive pairs.
    Each batch is `batch_size` of them plus `extra_negatives` of the rest, which count
    only as negatives (see `PositivesWithNegativesSampler`).

    `group_negatives` (`{by, per_group}`, needs `split_tag`) builds each batch from a
    few groups -- segments sharing the filelist field `by`, e.g. `speaker_id` -- with
    `per_group` segments from each, so some negatives always share the group (see
    `GroupedBatchSampler`). With `positive_filter` too, only positives fill the groups
    and the extra negatives are drawn from the same groups where possible.

    `audio_features` (`{path, key, l2_normalize}`) replaces the audio side of every
    split with external per-segment features, matched by segment id (see
    `how2_splits.with_audio_features`) -- e.g. CLIP text embeddings of the
    transcripts, to train a text-vision model with everything else unchanged. The
    features must be `AUDIO_DIM`-dimensional.
    """
    if loss not in LOSSES:
        raise ValueError(f"Unknown loss `{loss}`. Available: {sorted(LOSSES)}")
    loss_fn = LOSSES[loss]
    # float() because PyYAML (YAML 1.1) reads `3e-4` without a dot as a string.
    lr = scale_lr(batch_size) if lr is None else float(lr)
    weight_decay = float(weight_decay)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    def split_source(split: str) -> str:
        if split_tag is None:
            return os.path.join(root, f"{split}_grouped_embeddings.pkl")
        return str(filelist_path(split, split_tag))

    checkpoint_path = os.path.join(checkpoint_dir, experiment_name, "best.pt")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # Resolved before wandb.init so the selected size can be logged with the config.
    train_indices = None
    if filter is not None:
        train_indices = select_indices(
            filter,
            load_split_ids(root, "train", split_tag),
            min_size=batch_size if drop_last else 1,
        )

    # Ids of the dataset's items (after `filter`), in dataset order: the positive mask
    # and the group labels are both positional, so they must come from this list.
    train_ids = None
    if positive_filter is not None or group_negatives is not None:
        train_ids = load_split_ids(root, "train", split_tag)
        if train_indices is not None:
            train_ids = [train_ids[i] for i in train_indices]

    positive = None
    if positive_filter is not None:
        positive = select_positives(positive_filter, train_ids, split_tag)
        print(f"Positive filter: {positive.sum()} of {len(positive)} segments are positives.")

    groups = None
    if group_negatives is not None:
        groups = group_labels(group_negatives, train_ids, split_tag)
        print(f"Group negatives: {len(set(groups))} groups by `{group_negatives['by']}`.")

    dataset = Dataset(
        load_split_with_features(root, "train", split_tag, audio_features),
        indices=train_indices,
        positive=positive,
    )

    wandb.init(
        project="avlnet",
        name=experiment_name,
        config={
            "train_embeddings": split_source("train"),
            "val_embeddings": split_source("val"),
            "split_tag": split_tag,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "projection_dim": projection_dim,
            "use_video": use_video,
            "loss": loss,
            "lr": lr,
            "weight_decay": weight_decay,
            "warmup_steps": warmup_steps,
            "min_steps": min_steps,
            "seed": seed,
            "patience": patience,
            "select_metric": select_metric,
            "filter": filter,
            "n_train": len(dataset),
            "max_steps": max_steps,
            "positive_filter": positive_filter,
            "n_positive": int(dataset.positive.sum()),
            "group_negatives": group_negatives,
            "n_groups": None if groups is None else len(set(groups)),
            "audio_features": audio_features,
        }
    )

    if group_negatives is not None:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=GroupedBatchSampler(
                groups,
                batch_size,
                group_negatives["per_group"],
                eligible=dataset.positive,
                extra_negatives=0 if positive_filter is None
                else positive_filter.get("extra_negatives", 0),
            ),
            num_workers=0,
        )
    elif positive_filter is None:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            # A short tail batch is a much easier contrastive task (fewer negatives),
            # so it would inject noise into the loss curve at large batch sizes.
            drop_last=drop_last,
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=PositivesWithNegativesSampler(
                dataset.positive, batch_size, positive_filter.get("extra_negatives", 0)
            ),
            num_workers=0,
        )
    val_dataset = Dataset(load_split_with_features(root, "val", split_tag, audio_features))
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=5000,
        shuffle=False,
        num_workers=0,
    )
    test_dataset = Dataset(load_split_with_features(root, "dev5", split_tag, audio_features))
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=5000,
        shuffle=False,
        num_workers=0,
    )

    model = DualEncoder(projection_dim=projection_dim, use_video=use_video)
    model.cuda()

    # muon_params = [
    #     p for p in model.parameters() if p.ndim == 2
    # ]
    # other_params = [
    #     p for p in model.parameters() if p.ndim != 2
    # ]
    # optim_muon = torch.optim.Muon(
    #     muon_params, lr=0.02, momentum=0.95
    # )
    # optim_adamw = torch.optim.AdamW(
    #     other_params, lr=3e-4, weight_decay=0.01
    # )
    # optim_muon.zero_grad()
    # optim_adamw.zero_grad()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    optimizer.zero_grad()

    steps_per_epoch = len(dataloader)
    if steps_per_epoch == 0:
        raise ValueError(
            f"Training set has {len(dataset)} segments, fewer than batch_size="
            f"{batch_size} with drop_last={drop_last}: no steps would run."
        )
    # `max_steps` decouples the training length from the dataset size, so a filtered
    # run can be as long as an unfiltered one; epochs then just repeat the subset.
    total_steps = max_steps if max_steps is not None else n_epochs * steps_per_epoch
    n_epochs_effective = math.ceil(total_steps / steps_per_epoch)
    if warmup_steps is None:
        warmup_steps = int(0.1 * total_steps)
    print(f"lr={lr:g} weight_decay={weight_decay:g} warmup_steps={warmup_steps} total_steps={total_steps}")
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    # scheduler_muon = torch.optim.lr_scheduler.LambdaLR(optim_muon, lr_lambda)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if eval_every is None:
        eval_every = steps_per_epoch

    best_val = -float("inf")
    evals_without_improvement = 0
    global_step = 0
    recent_losses = []
    stop_early = False

    def run_evaluation(epoch: int):
        nonlocal best_val, evals_without_improvement

        val_metrics = evaluate(model, val_dataloader)
        test_metrics = evaluate(model, test_dataloader)

        if val_metrics[select_metric] > best_val:
            best_val = val_metrics[select_metric]
            evals_without_improvement = 0
            torch.save({
                "epoch": epoch,
                "step": global_step,
                "model_state_dict": model.state_dict(),
                # `config` is exactly the DualEncoder constructor kwargs, so it can
                # be splatted straight back in; `loss` sits outside it because the
                # objective is not part of the model.
                "config": {"projection_dim": projection_dim, "use_video": use_video},
                "loss": loss,
                "val_r@5": val_metrics["r5"],
                "val_r@10": val_metrics["r10"],
                "val_accuracy": val_metrics["r1"],
                "val_mrr": val_metrics["mrr"],
            }, checkpoint_path)
        else:
            evals_without_improvement += 1

        wandb.log({
            "epoch": epoch,
            "val_accuracy": val_metrics["r1"],
            "val_r@5": val_metrics["r5"],
            "val_r@10": val_metrics["r10"],
            "val_mrr": val_metrics["mrr"],
            "val_medr": val_metrics["medr"],
            f"best_val_{select_metric}": best_val,
            "test_accuracy": test_metrics["r1"],
            "test_r@5": test_metrics["r5"],
            "test_r@10": test_metrics["r10"],
            "test_mrr": test_metrics["mrr"],
            "test_medr": test_metrics["medr"],
        }, step=global_step)

        model.train()

    for epoch in tqdm(range(n_epochs_effective), total=n_epochs_effective, desc="Training"):
        model.train()
        for batch in dataloader:
            if global_step >= total_steps:
                break

            video_embeddings, image_embeddings, audio_embeddings, positive = batch
            video_embeddings = video_embeddings.cuda()
            image_embeddings = image_embeddings.cuda()
            video_embeddings = model.compose_video(video_embeddings, image_embeddings)
            audio_embeddings = audio_embeddings.cuda()

            batch_loss = loss_fn(
                model,
                video_embeddings,
                audio_embeddings,
                positive=None if positive_filter is None else positive,
            )
            recent_losses.append(batch_loss.item())

            batch_loss.backward()
            # optim_muon.step()
            # optim_adamw.step()
            # scheduler_muon.step()
            # scheduler_adamw.step()
            # optim_muon.zero_grad()
            # optim_adamw.zero_grad()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % log_every == 0:
                wandb.log({
                    "train_loss": sum(recent_losses) / len(recent_losses),
                    "lr": scheduler.get_last_lr()[0],
                    # The learnable temperature is worth watching: it drifts a lot
                    # with the batch size. `bias` is logged for every run to keep
                    # the metric schema stable, but only `siglip` uses it -- under
                    # `infonce` it stays flat at its initial value.
                    "temperature": torch.exp(model.t_prime).item(),
                    "bias": model.b.item(),
                }, step=global_step)
                recent_losses = []

            if global_step % eval_every == 0:
                run_evaluation(epoch)
                if (
                    patience
                    and evals_without_improvement >= patience
                    and global_step >= min_steps
                ):
                    print(
                        f"Early stopping at step {global_step} (epoch {epoch}): "
                        f"val {select_metric} did not improve on the last "
                        f"{patience} evaluations (best={best_val:.4f})."
                    )
                    stop_early = True
                    break

        if stop_early or global_step >= total_steps:
            break

    # Make sure the final state is scored even if the run does not end on an
    # evaluation boundary.
    if global_step % eval_every != 0 and not stop_early:
        run_evaluation(n_epochs_effective - 1)

    wandb.summary[f"best_val_{select_metric}"] = best_val
    wandb.summary["stopped_early"] = stop_early
    wandb.summary["total_steps"] = global_step



def main(config_name: str):
    """Run training using `configs/<config_name>.yaml`.

    Example:
        uv run train.py default
    """
    cfg = load_config(config_name)
    print(f"Running config `{config_name}`:")
    for k, v in sorted(cfg.items()):
        print(f"  {k}: {v}")
    run_training(experiment_name=config_name, **cfg)


if __name__ == "__main__":
    fire.Fire(main)