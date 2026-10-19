"""Dataset loader for pre-extracted V-JEPA2 + DINOv2 + WavLM features.

Expects the file layout produced by ``extract_features.py``:

    <embeddings_path>/<segment_id>.pkl

where each pickle is a dict with keys ``video_embedding`` (V-JEPA2),
``image_embedding`` (DINOv2), and ``audio_embedding`` (WavLM) -- exactly the
dict ``extract_features.py`` writes out.

Which ids are used for train / val / test is controlled entirely by a plain text
file (one id per line, blank lines and '#' comments ignored), e.g.
``train_ids.txt``, ``val_ids.txt``, ``test_ids.txt``. This means you can point at
a different dataset, or a different subset of the same dataset, without touching
any code -- just pass a different ``--*_ids_path``.
"""
from __future__ import absolute_import, division, unicode_literals, print_function

import pickle
from pathlib import Path

import numpy as np
import torch as th
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from src.utils import read_file


class Feature_DataLoader(Dataset):
    """Loads pre-extracted per-clip audio/image/video embeddings for retrieval."""

    def __init__(self, 
                 ids_path, 
                 embeddings_path, 
                 split, 
                 normalize=True, 
                 image_only=False):
        """
        Args:
            ids_path: text file with one segment id per line (train/val/test split).
            embeddings_path: directory containing '<segment_id>.pkl' files, as
                produced by extract_features.py.
            normalize: if True, L2-normalize each modality's feature vector before
                concatenation/use, matching how the original AVLnet dataloader
                normalizes its '2d'/'3d' video features.
        """
        if split in ["train_how2", "val_how2", "dev5_how2"]:
            self.embeddings_path = Path(embeddings_path) / "all" #split
        else:
            self.embeddings_path = Path(embeddings_path) / split
        self.normalize = normalize
        self.image_only = image_only

        wanted_ids = read_file(ids_path)
        self.ids = [
            seg_id for seg_id in wanted_ids
            if (self.embeddings_path / f"{seg_id}.pkl").exists()
        ]
        missing = len(wanted_ids) - len(self.ids)
        if missing:
            print(
                f"[Feature_DataLoader] {missing} id(s) from {ids_path} have no "
                f"embeddings in {embeddings_path} and will be skipped."
            )

    def __len__(self):
        return len(self.ids)

    def custom_collate(self, batch):
        return default_collate(batch)

    def _load(self, seg_id):
        with open(self.embeddings_path / f"{seg_id}.pkl", "rb") as f:
            return pickle.load(f)

    def __getitem__(self, idx):
        seg_id = self.ids[idx]
        data = self._load(seg_id)

        image_embedding = th.from_numpy(np.asarray(data["image_embedding"])).float()
        if not self.image_only:
            video_embedding = th.from_numpy(np.asarray(data["video_embedding"])).float()     
            audio_embedding = th.from_numpy(np.asarray(data["audio_embedding"])).float()       
        else:
            data["audio_embedding"] = data.pop("audio_embeddings")[0] # some problem in the Flick8k extracted features

        if self.normalize:
            if not self.image_only:
                video_embedding = F.normalize(video_embedding, dim=0)
            image_embedding = F.normalize(image_embedding, dim=0)
            audio_embedding = F.normalize(audio_embedding, dim=0)

        # Concatenated video representation: DINOv2 image embedding + V-JEPA2 video
        # embedding, mirroring the original AVLnet '2d' + '3d' concatenation into a
        # single "video" feature vector.
        if self.image_only:
            video = image_embedding
        else:  
            video = th.cat((image_embedding, video_embedding))

        return {
            "video": video,
            "audio": audio_embedding,
            "video_id": seg_id,
        }