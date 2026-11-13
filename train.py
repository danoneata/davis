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
from perceiver import Perceiver
from torch.nn.utils.rnn import pad_sequence


def pad(seqs):
    lengths = torch.tensor([x.shape[0] for x in seqs])
    x = pad_sequence(seqs, batch_first=True)                      # zero-padded to the longest in this list
    pad_mask = torch.arange(x.shape[1])[None] >= lengths[:, None]  # True = padding
    return x, pad_mask

def collate(batch):                                    # list of (audio, video) pairs
    audios, videos = zip(*batch)
    audio, audio_mask = pad(list(audios))              # (B, T_audio_max, 1280)
    video, video_mask = pad(list(videos))              # (B, T_video_max, 1280)
    return audio, audio_mask, video, video_mask

class Dataset(torch.utils.data.Dataset):
    def __init__(self, audio_embeddings_dir, visual_embeddings_dir):
        self.filenames = os.listdir(audio_embeddings_dir)
        self.audio_embeddings_dir = audio_embeddings_dir
        self.visual_embeddings_dir = visual_embeddings_dir

    def __getitem__(self, sample_n):
        # audio_embedding = torch.load(os.path.join(self.audio_embeddings_dir, self.filenames[sample_n]))
        # video_embedding = torch.load(os.path.join(self.visual_embeddings_dir, self.filenames[sample_n]))
        # print(audio_embedding.shape, video_embedding.shape)
        return (
            torch.load(os.path.join(self.audio_embeddings_dir, self.filenames[sample_n])),
            torch.load(os.path.join(self.visual_embeddings_dir, self.filenames[sample_n])),
        )

    def __len__(self):
        return len(self.filenames)


class SigLIP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_adapter = Perceiver(
            num_register_tokens = 8,
            input_dim=1280, use_positional_embeddings=False, input_max_length=1500,
            dim=1024, depth=2, heads=10, dim_head=64, mlp_dim=4096,
            dropout = 0., emb_dropout = 0., 
        )
        self.video_adapter = Perceiver(
            num_register_tokens = 8,
            input_dim=1280, use_positional_embeddings=True, input_max_length=600,
            dim=1024, depth=2, heads=10, dim_head=64, mlp_dim=4096,
            dropout = 0., emb_dropout = 0., 
        )
        self.t_prime = torch.nn.Parameter(torch.tensor(2.3026))
        self.b = torch.nn.Parameter(torch.tensor(-10.0))

    def forward(self, audio_input, video_input, audio_mask=None, video_mask=None, normalize=True):
        audio_embeddings = self.audio_adapter(audio_input, audio_mask)
        video_embeddings = self.video_adapter(video_input, video_mask)
        if normalize:
            audio_embeddings = torch.nn.functional.normalize(audio_embeddings, p=2, dim=1)
            video_embeddings = torch.nn.functional.normalize(video_embeddings, p=2, dim=1)
        return audio_embeddings, video_embeddings

    def loss(self, audio_input, video_input, audio_mask=None, video_mask=None, normalize=True):
        audio_embeddings = self.audio_adapter(audio_input, audio_mask)
        video_embeddings = self.video_adapter(video_input, video_mask)
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
    siglip.eval()
    audio_embeddings = []
    video_embeddings = []
    for batch in val_dataloader:
        audio_input, audio_mask, video_input, video_mask = batch
        audio_input = audio_input.to(torch.float32).cuda()
        video_input = video_input.to(torch.float32).cuda()
        audio_mask = audio_mask.cuda()
        video_mask = video_mask.cuda()

        audio_embedding, video_embedding = siglip(audio_input, video_input, audio_mask, video_mask, normalize=True)
        # fp32 so that close scores among thousands of candidates don't tie after bf16 rounding
        audio_embeddings.append(audio_embedding.float())
        video_embeddings.append(video_embedding.float())
    audio_embeddings = torch.cat(audio_embeddings)
    video_embeddings = torch.cat(video_embeddings)
    n_samples = video_embeddings.shape[0]

    logits = torch.matmul(audio_embeddings, video_embeddings.T)
    targets = torch.arange(n_samples).cuda().unsqueeze(1)

    # compute r@1, r@5 and r@10
    _, top10_indices = torch.topk(logits, k=min(10, n_samples), dim=1)
    correct_topk = (top10_indices == targets)
    val_accuracy = correct_topk[:, 0].sum().item() / n_samples
    r_at_5 = correct_topk[:, :5].sum().item() / n_samples
    r_at_10 = correct_topk.sum().item() / n_samples
    return val_accuracy, r_at_5, r_at_10


def param_groups(model, weight_decay):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        # ndim < 2 covers biases, LayerNorm weights, t_prime and b
        if param.ndim < 2 or name.endswith(('cls_token', 'register_tokens', 'pos_embedding')):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.},
    ]


def main(
    experiment_name='perceiver',
    batch_size=128,
    n_epochs=100,
):
    wandb.init(
        project="avlnet",
        name=experiment_name,
    )

    dataset = Dataset(
        audio_embeddings_dir='/root/davis/whisper/embeddings/audio/train',
        visual_embeddings_dir='/root/davis/whisper/embeddings/visual/train',
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        collate_fn=collate,
    )
    val_dataset = Dataset(
        audio_embeddings_dir='/root/davis/whisper/embeddings/audio/val',
        visual_embeddings_dir='/root/davis/whisper/embeddings/visual/val',
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=16,
        collate_fn=collate,
    )
    test_dataset = Dataset(
        audio_embeddings_dir='/root/davis/whisper/embeddings/audio/dev5',
        visual_embeddings_dir='/root/davis/whisper/embeddings/visual/dev5',
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=16,
        collate_fn=collate,
    )

    siglip = SigLIP().cuda()
    optimizer = torch.optim.AdamW(param_groups(siglip, weight_decay=0.01), lr=1e-3)
    optimizer.zero_grad()

    total_steps = n_epochs * len(dataloader)
    warmup_steps = int(0.05 * total_steps)
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    

    # for epoch in tqdm(range(n_epochs), total=n_epochs, desc="Training"):
    for epoch in range(n_epochs):
        siglip.train()
        train_loss = 0.
        for batch in tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch}"):
            audio_input, audio_mask, video_input, video_mask = batch
            audio_input = audio_input.to(torch.float32).cuda()
            video_input = video_input.to(torch.float32).cuda()
            audio_mask = audio_mask.cuda()
            video_mask = video_mask.cuda()

            loss = siglip.loss(audio_input, video_input, audio_mask, video_mask)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        train_loss /= len(dataloader)

        val_accuracy, val_r5, val_r10 = evaluate(siglip, val_dataloader)
        test_accuracy, test_r5, test_r10 = evaluate(siglip, test_dataloader)

        wandb.log({
            "train_loss": train_loss,
            "val_accuracy": val_accuracy,
            "val_r@5": val_r5,
            "val_r@10": val_r10,
            "test_accuracy": test_accuracy,
            "test_r@5": test_r5,
            "test_r@10": test_r10
        })
    torch.save(siglip, f"./ckpt/siglip.pt")



if __name__ == "__main__":
    fire.Fire(main)