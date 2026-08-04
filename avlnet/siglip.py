import torch
import os
import pickle
import glob
import numpy as np
from tqdm import tqdm
import fire
import sys
import wandb
import math
import yaml
from pathlib import Path

from filtering import select_indices
from make_grouped_ids import load_grouped_ids


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
    def __init__(self, embeddings_path: str, indices=None):
        """`indices` restricts the dataset to a subset of rows; None uses all of them."""
        with open(embeddings_path, "rb") as f:
            self.embeddings = pickle.load(f)
        n = len(self.embeddings["video_embeddings"])
        self.indices = np.arange(n) if indices is None else np.asarray(indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return (
            self.embeddings["video_embeddings"][i],
            self.embeddings["image_embeddings"][i],
            self.embeddings["audio_embeddings"][i]
        )

    def __len__(self):
        return len(self.indices)


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


class SigLIP(torch.nn.Module):
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

    def loss(self, video_embeddings, audio_embeddings, normalize=True):
        video_embeddings = self.video_adapter(video_embeddings)
        audio_embeddings = self.audio_adapter(audio_embeddings)
        if normalize:
            video_embeddings = torch.nn.functional.normalize(video_embeddings, p=2, dim=1)
            audio_embeddings = torch.nn.functional.normalize(audio_embeddings, p=2, dim=1)
        t = torch.exp(self.t_prime)
        zvideo = torch.nn.functional.normalize(video_embeddings, p=2, dim=1)
        zaudio = torch.nn.functional.normalize(audio_embeddings, p=2, dim=1)
        logits = torch.matmul(zvideo, zaudio.T) * t + self.b
        labels = 2 * torch.eye(video_embeddings.shape[0]) - torch.ones(video_embeddings.shape[0])
        labels = labels.cuda()
        loss = -torch.sum(torch.nn.functional.logsigmoid(labels * logits)) / video_embeddings.shape[0]
        return loss
        

@torch.no_grad()
def evaluate(siglip, val_dataloader):
    """Video->audio retrieval metrics, computed within each batch.

    Returns a dict with R@1/5/10 (fractions in [0, 1]), MRR (mean reciprocal
    rank) and MedR (median rank). All are derived from the rank of the matching
    audio for each video: rank 1 means retrieved first.
    """
    siglip.eval()
    ranks = []
    for batch in val_dataloader:
        video_embeddings, image_embeddings, audio_embeddings = batch
        video_embeddings = video_embeddings.cuda()
        image_embeddings = image_embeddings.cuda()
        video_embeddings = siglip.compose_video(video_embeddings, image_embeddings)
        audio_embeddings = audio_embeddings.cuda()

        video_embeddings, audio_embeddings = siglip(video_embeddings, audio_embeddings, normalize=True)
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
    checkpoint_dir: str = "checkpoints",
    log_every: int = 50,
    eval_every: int = None,
    patience: int = 10,
    select_metric: str = "r5",
    drop_last: bool = True,
    filter: dict = None,
    max_steps: int = None,
):
    """Train the SigLIP adapters on precomputed How2 embeddings.

    Everything is indexed by optimizer step rather than epoch, so runs with
    different `batch_size` remain comparable on the same x-axis.

    The learning rate is not a free parameter: it is derived from `batch_size` as
    BASE_LR * sqrt(batch_size / BASE_BATCH_SIZE) (see `scale_lr`). So when
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
    """
    train_embeddings = os.path.join(root, "train_grouped_embeddings.pkl")
    val_embeddings = os.path.join(root, "val_grouped_embeddings.pkl")
    test_embeddings = os.path.join(root, "dev5_grouped_embeddings.pkl")

    checkpoint_path = os.path.join(checkpoint_dir, experiment_name, "best.pt")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # Resolved before wandb.init so the selected size can be logged with the config.
    train_indices = None
    if filter is not None:
        train_indices = select_indices(
            filter,
            load_grouped_ids(root, "train"),
            min_size=batch_size if drop_last else 1,
        )

    dataset = Dataset(train_embeddings, indices=train_indices)

    wandb.init(
        project="avlnet",
        name=experiment_name,
        config={
            "train_embeddings": train_embeddings,
            "val_embeddings": val_embeddings,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "projection_dim": projection_dim,
            "use_video": use_video,
            "lr": scale_lr(batch_size),
            "patience": patience,
            "select_metric": select_metric,
            "filter": filter,
            "n_train": len(dataset),
            "max_steps": max_steps,
        }
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        # A short tail batch is a much easier contrastive task (fewer negatives),
        # so it would inject noise into the loss curve at large batch sizes.
        drop_last=drop_last,
    )
    val_dataset = Dataset(val_embeddings)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=5000,
        shuffle=False,
        num_workers=0,
    )
    test_dataset = Dataset(test_embeddings)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=5000,
        shuffle=False,
        num_workers=0,
    )

    siglip = SigLIP(projection_dim=projection_dim, use_video=use_video)
    siglip.cuda()

    # muon_params = [
    #     p for p in siglip.parameters() if p.ndim == 2
    # ]
    # other_params = [
    #     p for p in siglip.parameters() if p.ndim != 2
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
        siglip.parameters(), lr=scale_lr(batch_size), weight_decay=0.01
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
    warmup_steps = int(0.1 * total_steps)
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

        val_metrics = evaluate(siglip, val_dataloader)
        test_metrics = evaluate(siglip, test_dataloader)

        if val_metrics[select_metric] > best_val:
            best_val = val_metrics[select_metric]
            evals_without_improvement = 0
            torch.save({
                "epoch": epoch,
                "step": global_step,
                "model_state_dict": siglip.state_dict(),
                "config": {"projection_dim": projection_dim, "use_video": use_video},
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

        siglip.train()

    for epoch in tqdm(range(n_epochs_effective), total=n_epochs_effective, desc="Training"):
        siglip.train()
        for batch in dataloader:
            if global_step >= total_steps:
                break

            video_embeddings, image_embeddings, audio_embeddings = batch
            video_embeddings = video_embeddings.cuda()
            image_embeddings = image_embeddings.cuda()
            video_embeddings = siglip.compose_video(video_embeddings, image_embeddings)
            audio_embeddings = audio_embeddings.cuda()

            loss = siglip.loss(video_embeddings, audio_embeddings)
            recent_losses.append(loss.item())

            loss.backward()
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
                    # The learnable temperature and bias of the sigmoid loss are
                    # worth watching: they drift a lot with the batch size.
                    "temperature": torch.exp(siglip.t_prime).item(),
                    "bias": siglip.b.item(),
                }, step=global_step)
                recent_losses = []

            if global_step % eval_every == 0:
                run_evaluation(epoch)
                if patience and evals_without_improvement >= patience:
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
        uv run siglip.py default
    """
    cfg = load_config(config_name)
    print(f"Running config `{config_name}`:")
    for k, v in sorted(cfg.items()):
        print(f"  {k}: {v}")
    run_training(experiment_name=config_name, **cfg)


if __name__ == "__main__":
    fire.Fire(main)