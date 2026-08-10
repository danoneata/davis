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
"""
from __future__ import absolute_import, division, unicode_literals, print_function

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, embd_dim=4096, video_dim=1792, audio_dim=768):
        """
        Args:
            embd_dim: dimension of the shared audio-video embedding space.
            video_dim: dimension of the concatenated video feature, i.e.
                DINOv2 image-embedding dim + V-JEPA2 video-embedding dim
                (768 + 1024 = 1792 for dinov2-base + vjepa2-vitl-fpc64-256).
            audio_dim: dimension of the WavLM audio embedding
                (768 for microsoft/wavlm-base-plus).
        """
        super(Net, self).__init__()
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

    def forward(self, video, audio):
        video = self.GU_video(video)
        audio = self.GU_audio(audio)
        return audio, video


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
