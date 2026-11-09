"""AVLnet-style audio-video retrieval model, adapted for pre-extracted SSL features.

The overall architecture is unchanged from the original AVLnet (Rouditchenko et al.,
"AVLnet: Learning Audio-Visual Language Representations from Instructional Videos",
Interspeech 2021): each modality is projected into a shared embedding space with a
Gated Embedding Unit (linear layer + context gating), and audio/video embeddings are
compared with a dot product for retrieval, trained with the max-margin softmax (MMS)
loss in loss.py.

The only structural change from the original ``model.py`` is that there is no more
DAVEnet audio CNN: the original model consumed *raw* audio spectrograms and pooled
DAVEnet's output over time (handling zero-padding via ``nframes``). Here the audio
branch instead consumes an already pooled SSL audio embedding (WavLM), and the video
branch consumes concatenated, already pooled DINOv2 (image) + V-JEPA2 (video)
embeddings -- exactly the format written out by extract_features.py -- so there's
nothing left to pool over time and the CNN/nframes machinery is removed. The
Gated_Embedding_Unit / Context_Gating building blocks themselves are untouched.

Optionally (``audio_pool=False``) the audio branch takes frame-level features
[B, T, D] plus a padding mask and pools them with ``AttentivePooling`` before
GU_audio. It is initialized to exact masked mean pooling, so at init it matches
the ``audio_pool=True`` model and only learns to reweight frames from there.
"""
from __future__ import absolute_import, division, unicode_literals, print_function

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, embd_dim=4096, video_dim=1792, audio_dim=768,
                 audio_pool=True, normalize=True, audio_pool_heads=8):
        """
        Args:
            embd_dim: dimension of the shared audio-video embedding space.
            video_dim: dimension of the concatenated video feature, i.e.
                DINOv2 image-embedding dim + V-JEPA2 video-embedding dim
                (e.g. 1536 + 1024 for dinov2 CLS+pool + vjepa2-vitl-fpc64-256).
            audio_dim: dimension of the audio embedding
                (1280 for SPEAR, 768 for microsoft/wavlm-base-plus).
            audio_pool: if True, audio is already pooled to [B, D] by the
                dataset. If False, audio is [B, T, D] + mask and is pooled
                here with AttentivePooling.
            normalize: L2-normalize the attentively pooled audio (matches the
                dataset's normalization in the audio_pool=True case).
            audio_pool_heads: number of attention heads of AttentivePooling.
        """
        super(Net, self).__init__()
        self.normalize = normalize
        self.audio_temporal_pool = (
            None if audio_pool else AttentivePooling(audio_dim, audio_pool_heads)
        )
        self.GU_video = Gated_Embedding_Unit(video_dim, embd_dim)
        self.GU_audio = Gated_Embedding_Unit(audio_dim, embd_dim)

    def save_checkpoint(self, path):
        th.save(self.state_dict(), path)

    def load_checkpoint(self, path):
        try:
            self.load_state_dict(th.load(path, map_location="cpu"))
        except Exception as e:
            print(e)
            print("IGNORING ERROR, LOADING MODEL USING STRICT=FALSE")
            self.load_state_dict(th.load(path, map_location="cpu"), strict=False)
        print("Loaded model checkpoint from {}".format(path))

    def forward(self, video, audio, audio_mask=None):
        if audio.dim() == 3:
            audio = self.audio_temporal_pool(audio, audio_mask)
            if self.normalize:
                audio = F.normalize(audio, dim=-1)
        video = self.GU_video(video)
        audio = self.GU_audio(audio)
        return audio, video


class AttentivePooling(nn.Module):
    """Masked multi-head attention pooling over time: [B, T, D] -> [B, D].

    Each head scores every frame with a linear layer, softmaxes the scores
    over the valid (non-padded) frames, and takes the weighted sum of its own
    D / num_heads slice of the features. The scorer is zero-initialized, so
    at init all weights are uniform, i.e. exact masked mean pooling.
    """

    def __init__(self, dimension, num_heads=8):
        super(AttentivePooling, self).__init__()
        if dimension % num_heads:
            raise ValueError(f"dimension {dimension} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.score = nn.Linear(dimension, num_heads)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        scores = self.score(x)  # [B, T, H]
        if mask is not None:
            scores = scores.masked_fill(~mask[:, :, None], float("-inf"))
        weights = th.softmax(scores, dim=1)  # [B, T, H]
        x = x.view(B, T, self.num_heads, D // self.num_heads)
        return (weights[..., None] * x).sum(dim=1).reshape(B, D)


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
        x = th.cat((x, x1), 1)
        return F.glu(x, 1)
