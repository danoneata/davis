"""Max-margin softmax (MMS) contrastive loss.

Unchanged from the original AVLnet loss.py (Rouditchenko et al., Interspeech 2021).
"""
from __future__ import absolute_import, division, unicode_literals, print_function

import torch as th
import torch.nn.functional as F

class MMS_loss(th.nn.Module):
    def __init__(self):
        super(MMS_loss, self).__init__()

    def forward(self, audio_out, video_out, margin=0.001):
        sim_matrix = th.matmul(audio_out, video_out.t())
        deltas = margin * th.eye(sim_matrix.size(0)).to(sim_matrix.device)
        S = sim_matrix - deltas
        target = th.LongTensor(list(range(S.size(0)))).to(S.device)
        I2C_loss = F.nll_loss(F.log_softmax(S, dim=1), target)
        C2I_loss = F.nll_loss(F.log_softmax(S.t(), dim=1), target)
        loss = I2C_loss + C2I_loss
        return loss

class Siglip_loss(th.nn.Module):
    def __init__(self):
        super(Siglip_loss, self).__init__()
        self.t_prime = th.nn.Parameter(th.tensor(2.3026))
        self.b = th.nn.Parameter(th.tensor(-10.0))
    
    def forward(self, video_embeddings, audio_embeddings, normalize=True):
        t = th.exp(self.t_prime)
        #zvideo = th.nn.functional.normalize(video_embeddings, p=2, dim=1)
        #zaudio = th.nn.functional.normalize(audio_embeddings, p=2, dim=1)
        logits = th.matmul(video_embeddings, audio_embeddings.T) * t + self.b
        labels = 2 * th.eye(video_embeddings.shape[0]) - th.ones(video_embeddings.shape[0])
        labels = labels.cuda()
        loss = -th.sum(th.nn.functional.logsigmoid(labels * logits)) / video_embeddings.shape[0]
        return loss