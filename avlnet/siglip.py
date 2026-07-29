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
    def __init__(self, embeddings_path: str):
        with open(embeddings_path, "rb") as f:
            self.embeddings = pickle.load(f)

    def __getitem__(self, idx):
        return (
            self.embeddings["video_embeddings"][idx],
            self.embeddings["image_embeddings"][idx],
            self.embeddings["audio_embeddings"][idx]
        )

    def __len__(self):
        return len(self.embeddings["video_embeddings"])


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
        

def evaluate(siglip, val_dataloader):
    siglip.eval()
    val_accuracy = 0
    r_at_5 = 0
    r_at_10 = 0
    n_samples = 0
    for batch in val_dataloader:
        video_embeddings, image_embeddings, audio_embeddings = batch
        video_embeddings = video_embeddings.cuda()
        image_embeddings = image_embeddings.cuda()
        video_embeddings = siglip.compose_video(video_embeddings, image_embeddings)
        audio_embeddings = audio_embeddings.cuda()

        video_embeddings, audio_embeddings = siglip(video_embeddings, audio_embeddings, normalize=True)
        labels = torch.eye(video_embeddings.shape[0])
        labels = labels.cuda()
        logits = torch.matmul(video_embeddings, audio_embeddings.T)
        acc = torch.sum(torch.argmax(logits, dim=1) == torch.argmax(labels, dim=1))# / video_embeddings.shape[0]
        val_accuracy += acc.item()
        
        # compute r@5 and r@10
        targets = torch.argmax(labels, dim=1).unsqueeze(1)
        _, top10_indices = torch.topk(logits, k=min(10, logits.shape[1]), dim=1)
        correct_topk = (top10_indices == targets)
        r_at_5 += correct_topk[:, :5].sum().item()
        r_at_10 += correct_topk.sum().item()
        
        n_samples += video_embeddings.shape[0]
    val_accuracy /= n_samples
    r_at_5 /= n_samples
    r_at_10 /= n_samples
    return val_accuracy, r_at_5, r_at_10


def run_training(
    experiment_name: str,
    root: str,
    batch_size: int = 2,
    n_epochs: int = 200,
    projection_dim: int = 1024,
    use_video: bool = True,
    checkpoint_dir: str = "checkpoints",
):

    train_embeddings = os.path.join(root, "train_grouped_embeddings.pkl")
    val_embeddings = os.path.join(root, "val_grouped_embeddings.pkl")
    test_embeddings = os.path.join(root, "dev5_grouped_embeddings.pkl")

    checkpoint_path = os.path.join(checkpoint_dir, experiment_name, "best.pt")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

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
        }
    )

    dataset = Dataset(train_embeddings)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
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
    optimizer = torch.optim.AdamW(siglip.parameters(), lr=3e-4, weight_decay=0.01)
    optimizer.zero_grad()

    total_steps = n_epochs * len(dataloader)
    warmup_steps = int(0.1 * total_steps)
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    # scheduler_muon = torch.optim.lr_scheduler.LambdaLR(optim_muon, lr_lambda)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_r5 = -float("inf")

    for epoch in tqdm(range(n_epochs), total=n_epochs, desc="Training"):
        siglip.train()
        train_loss = 0.
        for batch in dataloader:
            video_embeddings, image_embeddings, audio_embeddings = batch
            video_embeddings = video_embeddings.cuda()
            image_embeddings = image_embeddings.cuda()
            video_embeddings = siglip.compose_video(video_embeddings, image_embeddings)
            audio_embeddings = audio_embeddings.cuda()

            loss = siglip.loss(video_embeddings, audio_embeddings)
            train_loss += loss.item()

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
        train_loss /= len(dataloader)

        val_accuracy, val_r5, val_r10 = evaluate(siglip, val_dataloader)
        test_accuracy, test_r5, test_r10 = evaluate(siglip, test_dataloader)

        if val_r5 > best_val_r5:
            best_val_r5 = val_r5
            torch.save({
                "epoch": epoch,
                "model_state_dict": siglip.state_dict(),
                "config": {"projection_dim": projection_dim, "use_video": use_video},
                "val_r@5": val_r5,
                "val_r@10": val_r10,
                "val_accuracy": val_accuracy,
            }, checkpoint_path)

        wandb.log({
            "train_loss": train_loss,
            "val_accuracy": val_accuracy,
            "val_r@5": val_r5,
            "val_r@10": val_r10,
            "best_val_r@5": best_val_r5,
            "test_accuracy": test_accuracy,
            "test_r@5": test_r5,
            "test_r@10": test_r10
        })



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