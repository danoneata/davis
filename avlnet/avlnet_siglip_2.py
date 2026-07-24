import torch
import fire
import pickle
import glob
import sys
import librosa
import numpy as np
import scipy
import math
import torch.nn as nn
import wandb
from tqdm import tqdm



def conv1x9(in_planes, out_planes, stride=1):
    """1x9 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1,9), stride=stride, padding=(0,4), bias=False)

def conv1d(in_planes, out_planes, width=9, stride=1, bias=False):
    """1xd convolution with padding"""
    if width % 2 == 0:
        pad_amt = int(width / 2)
    else:
        pad_amt = int((width - 1) / 2)
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1,width), stride=stride, padding=(0,pad_amt), bias=bias)

class SpeechBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, width=9, stride=1, downsample=None):
        super(SpeechBasicBlock, self).__init__()
        self.conv1 = conv1d(inplanes, planes, width=width, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1d(planes, planes, width=width)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResDavenet(nn.Module):
    def __init__(self, feat_dim=40, block=SpeechBasicBlock, layers=[2, 2, 2, 2], layer_widths=[128, 128, 256, 512, 1024], convsize=9):
        super(ResDavenet, self).__init__()
        self.feat_dim = feat_dim
        self.inplanes = layer_widths[0]
        self.batchnorm1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=(self.feat_dim,1), stride=1, padding=(0,0), bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, layer_widths[1], layers[0], width=convsize, stride=2)
        self.layer2 = self._make_layer(block, layer_widths[2], layers[1], width=convsize, stride=2)
        self.layer3 = self._make_layer(block, layer_widths[3], layers[2], width=convsize, stride=2)
        self.layer4 = self._make_layer(block, layer_widths[4], layers[3], width=convsize, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, width=9, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )       
        layers = []
        layers.append(block(self.inplanes, planes, width=width, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, width=width, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.squeeze(2)
        return x

def load_DAVEnet():
    layer_widths = [128,128,256,512,1024]
    layer_depths = [2,2,2,2]
    audio_model = ResDavenet(feat_dim=40, layers=layer_depths, convsize=9, layer_widths=layer_widths)

    return audio_model


class AVLNet(torch.nn.Module):   
    def __init__(
        self,
        embd_dim=3072,
        video_dim=3072,
    ):
        super(AVLNet, self).__init__()
        self.DAVEnet = load_DAVEnet()
        self.DAVEnet_projection = nn.Linear(1024, embd_dim)
        self.GU_audio = Gated_Embedding_Unit(1024, 1024)
        self.GU_video = Gated_Embedding_Unit(video_dim, embd_dim)

        #siglip
        self.t_prime = torch.nn.Parameter(torch.tensor(2.3026))
        self.b = torch.nn.Parameter(torch.tensor(-10.0))

    def forward(self, video, audio_input, nframes, normalize=True):
        video = self.GU_video(video)
        audio = self.DAVEnet(audio_input)
        if not self.training: # controlled by net.train() / net.eval() (use for downstream tasks) 
            # Mean-pool audio embeddings and disregard embeddings from input 0 padding
            pooling_ratio = round(audio_input.size(-1) / audio.size(-1))
            # nframes.div_(pooling_ratio)
            nframes = nframes // pooling_ratio
            audioPoolfunc = torch.nn.AdaptiveAvgPool2d((1, 1))
            audio_outputs = audio.unsqueeze(2)
            pooled_audio_outputs_list = []
            for idx in range(audio.shape[0]):
                nF = max(1, nframes[idx])
                pooled_audio_outputs_list.append(audioPoolfunc(audio_outputs[idx][:, :, 0:nF]).unsqueeze(0))
            audio = torch.cat(pooled_audio_outputs_list).squeeze(3).squeeze(2)
        else:
            audio = audio.mean(dim=2) # this averages features from 0 padding too

        # Gating in lower embedding dimension (1024 vs 4096) for stability with mixed-precision training        
        audio = self.GU_audio(audio) 
        audio = self.DAVEnet_projection(audio)

        if normalize:
            video = torch.nn.functional.normalize(video, p=2, dim=1)
            audio = torch.nn.functional.normalize(audio, p=2, dim=1)

        return video, audio

    def loss(self, video_embeddings, audio_embeddings, normalize=True):
        t = torch.exp(self.t_prime)
        logits = torch.matmul(video_embeddings, audio_embeddings.T) * t + self.b
        labels = 2 * torch.eye(video_embeddings.shape[0]) - torch.ones(video_embeddings.shape[0])
        labels = labels.cuda()
        loss = -torch.sum(torch.nn.functional.logsigmoid(labels * logits)) / video_embeddings.shape[0]
        return loss

class Gated_Embedding_Unit(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(Gated_Embedding_Unit, self).__init__()
        self.fc = nn.Linear(input_dimension, output_dimension)
        self.cg = Context_Gating(output_dimension)

    def forward(self, x):
        x = self.fc(x)
        x = self.cg(x)
        return x

class Context_Gating(nn.Module):
    def __init__(self, dimension):
        super(Context_Gating, self).__init__()
        self.fc = nn.Linear(dimension, dimension)

    def forward(self, x):
        x1 = self.fc(x)
        x = torch.cat((x, x1), 1)
        return torch.nn.functional.glu(x, 1)



class Dataset(torch.utils.data.Dataset):
    def __init__(self, embeddings_path: str | list):
        if type(embeddings_path) == list:
            self.embeddings = {
                "video_embeddings": [],
                "audio_embeddings": [],
                "n_audio_frames": [],
                "audio_paths": [],
            }
            for path in embeddings_path:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                    self.embeddings["video_embeddings"].extend(data["video_embeddings"])
                    self.embeddings["audio_embeddings"].extend(data["audio_embeddings"])
                    self.embeddings["n_audio_frames"].extend(data["n_audio_frames"])
                    self.embeddings["audio_paths"].extend(data["audio_paths"])
        else:
            with open(embeddings_path, "rb") as f:
                self.embeddings = pickle.load(f)

    def __len__(self):
        return len(self.embeddings["audio_embeddings"])
    
    def __getitem__(self, idx):
        return (
            self.embeddings["video_embeddings"][idx],
            self.embeddings["audio_embeddings"][idx],
            int(self.embeddings["n_audio_frames"][idx]),
        )


class Trainer:
    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Dataset,
        batch_size: int,
        model: AVLNet,
        n_epochs: int,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        self.test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        self.n_epochs = n_epochs

    def train(self):
        self.model.cuda()
        self.model.train()
        # val_accuracy, val_r5, val_r10 = self.evaluate(self.val_dataloader)

        muon_params = [
            p for p in self.model.parameters() if p.ndim == 2
        ]
        other_params = [
            p for p in self.model.parameters() if p.ndim != 2
        ]
        optim_muon = torch.optim.Muon(
            muon_params, lr=0.02, momentum=0.95
        )
        optim_adamw = torch.optim.AdamW(
            other_params, lr=3e-4, weight_decay=0.01
        )
        optim_muon.zero_grad()
        optim_adamw.zero_grad()

        total_steps = self.n_epochs * len(self.train_dataloader)
        warmup_steps = int(0.1 * total_steps)
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler_muon = torch.optim.lr_scheduler.LambdaLR(optim_muon, lr_lambda)
        scheduler_adamw = torch.optim.lr_scheduler.LambdaLR(optim_adamw, lr_lambda)

        for epoch in range(self.n_epochs): #tqdm(range(self.n_epochs), total=self.n_epochs, desc="Training"):
            self.model.train()
            train_loss = 0.
            for batch in tqdm(self.train_dataloader, total=len(self.train_dataloader), desc="Training"):
                video_embeddings, audio_embeddings, n_frames = batch
                video_embeddings = video_embeddings.cuda()
                audio_embeddings = audio_embeddings.cuda()
                z_video, z_audio = self.model(video_embeddings, audio_embeddings, n_frames)
                loss = self.model.loss(z_video, z_audio)
                train_loss += loss.item()

                loss.backward()
                optim_muon.step()
                optim_adamw.step()
                scheduler_muon.step()
                scheduler_adamw.step()
                optim_muon.zero_grad()
                optim_adamw.zero_grad()
            train_loss /= len(self.train_dataloader)

            val_accuracy, val_r5, val_r10 = self.evaluate(self.val_dataloader)
            test_accuracy, test_r5, test_r10 = self.evaluate(self.test_dataloader)

            wandb.log({
                "train_loss": train_loss,
                "val_accuracy": val_accuracy,
                "val_r@5": val_r5,
                "val_r@10": val_r10,
                "test_accuracy": test_accuracy,
                "test_r@5": test_r5,
                "test_r@10": test_r10
            })

    def evaluate(self, val_dataloader):
        self.model.eval()
        val_accuracy = 0
        r_at_5 = 0
        r_at_10 = 0
        n_samples = 0
        for batch in val_dataloader:
            video_embeddings, audio_embeddings, n_frames = batch
            video_embeddings = video_embeddings.cuda()
            audio_embeddings = audio_embeddings.cuda()
            with torch.no_grad():
                video_embeddings, audio_embeddings = self.model(video_embeddings, audio_embeddings, n_frames)
            
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



def main(
    batch_size = 2,
    experiment_name = "avlnet_siglip_bs_2",
    n_epochs = 100,
):
    wandb.init(
        project="avlnet",
        name=experiment_name,
        config={
            "batch_size": batch_size,
            "n_epochs": n_epochs,
        }
    )

    avlnet = AVLNet()
    train_dataset = Dataset(glob.glob("/root/how2/avlnet_embeddings/train_with_audio_embeddings/*.pkl"))
    val_dataset = Dataset("/root/how2/avlnet_embeddings/val.pkl")
    test_dataset = Dataset("/root/how2/avlnet_embeddings/dev5.pkl")

    trainer = Trainer(
        model=avlnet,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        n_epochs=n_epochs,
    )
    trainer.train()



if __name__ == "__main__":
    fire.Fire(main)