"""Dataset loader for pre-extracted V-JEPA2 + DINOv2 + WavLM features."""

from __future__ import absolute_import, division, unicode_literals, print_function

import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch as th
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from src.utils import read_file


class Feature_DataLoader(Dataset):
    """Loads pre-extracted per-clip audio/image/video embeddings."""

    def __init__(
        self,
        ids_path,
        embeddings_path,
        split,
        normalize=True,
        image_only=False,
    ):
        if "train" in split or "val" in split or "dev5" in split or "how2" in split:
            self.embeddings_path = Path(embeddings_path) / "all"
        else:
            self.embeddings_path = Path(embeddings_path) / split
            
        self.normalize = normalize
        self.image_only = image_only

        wanted_ids = read_file(ids_path)

        self.ids = [
            seg_id
            for seg_id in wanted_ids
            if (self.embeddings_path / f"{seg_id}.pkl").exists()
        ]

        missing = len(wanted_ids) - len(self.ids)
        if missing:
            print(
                f"[Feature_DataLoader] {missing} id(s) from {ids_path} have no "
                f"embeddings in {self.embeddings_path} and will be skipped."
            )

        # ------------------------------------------------------------
        # Group clips by root video.
        #
        # Example:
        #   001evy9D-mw_10 -> 001evy9D-mw
        #   001evy9D-mw_1  -> 001evy9D-mw
        #
        # IMPORTANT:
        # Keep the original dataset indices, because the BatchSampler
        # operates on indices.
        # ------------------------------------------------------------
        self.root_to_indices = defaultdict(list)

        for idx, seg_id in enumerate(self.ids):
            root_id = self._get_root_id(seg_id)
            self.root_to_indices[root_id].append(idx)

        self.roots = list(self.root_to_indices.keys())

        print(
            f"[Feature_DataLoader] {len(self.ids)} clips "
            f"from {len(self.roots)} videos"
        )

    @staticmethod
    def _get_root_id(seg_id):
        """
        Convert:

            001evy9D-mw_10 -> 001evy9D-mw
            002pHlZGEQ8_0  -> 002pHlZGEQ8

        We split only on the LAST underscore, since the segment number
        is the suffix.
        """
        return seg_id.rsplit("_", 1)[0]

    def get_root_indices(self, root_id):
        """Return all dataset indices belonging to one video."""
        return self.root_to_indices[root_id]

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

        image_embedding = th.from_numpy(
            np.asarray(data["image_embedding"])
        ).float()

        if self.image_only:
            # Some Flick8k extracted features use "audio_embeddings"
            # rather than "audio_embedding".
            if "audio_embedding" in data:
                audio_embedding = th.from_numpy(
                    np.asarray(data["audio_embedding"])
                ).float()
            elif "audio_embeddings" in data:
                audio_embedding = th.from_numpy(
                    np.asarray(data["audio_embeddings"][0])
                ).float()
            else:
                audio_embedding = None

            if self.normalize:
                image_embedding = F.normalize(image_embedding, dim=0)

            video = image_embedding

            # Keep the output structure consistent.
            return {
                "video": video,
                "audio": audio_embedding,
                "video_id": seg_id,
            }

        video_embedding = th.from_numpy(
            np.asarray(data["video_embedding"])
        ).float()

        audio_embedding = th.from_numpy(
            np.asarray(data["audio_embedding"])
        ).float()

        if self.normalize:
            video_embedding = F.normalize(video_embedding, dim=0)
            image_embedding = F.normalize(image_embedding, dim=0)
            audio_embedding = F.normalize(audio_embedding, dim=0)

        # DINOv2 + V-JEPA2
        video = th.cat((image_embedding, video_embedding))

        return {
            "video": video,
            "audio": audio_embedding,
            "video_id": seg_id,
        }
        
import random

from torch.utils.data import Sampler


class VideoGroupedBatchSampler(Sampler):
    """
    Batch sampler that keeps ALL clips belonging to the same root video
    together whenever possible.

    Example:

        video A -> [0, 1, 2, 3]
        video B -> [4, 5]
        video C -> [6, 7, 8]

    With batch_size=6:

        batch 1 -> [A clips + B clips] = 6 clips
        batch 2 -> [C clips]           = 3 clips

    A video is never split merely because it does not fit in the current
    batch. Instead, it starts the next batch.

    This guarantees:
        - random video ordering every epoch
        - all clips of a video stay together
        - complete traversal of the dataset
        - no duplicated clips
        - no permanently fixed batches
    """

    def __init__(
        self,
        dataset,
        batch_size,
        shuffle=True,
        drop_last=False,
        seed=0,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        # Store the root video names.
        self.roots = list(dataset.roots)
        
        print(len(self.roots))

        # Validate groups.
        oversized = []

        for root in self.roots:
            n_clips = len(dataset.get_root_indices(root))

            if n_clips > batch_size:
                oversized.append((root, n_clips))

        if oversized:
            examples = ", ".join(
                f"{root}: {n}"
                for root, n in oversized[:5]
            )

            raise ValueError(
                "Some videos contain more clips than batch_size, so their "
                "clips cannot all fit into one batch while preserving the "
                "whole-video constraint. "
                f"Examples: {examples}"
            )

    def set_epoch(self, epoch):
        """
        Call this at the beginning of every epoch when using a custom
        training loop.

        The DataLoader itself does not automatically call set_epoch()
        for a generic BatchSampler.
        """
        self.epoch = epoch

    def __iter__(self):
        roots = list(self.roots)

        if self.shuffle:
            # Different random ordering every epoch, but reproducible
            # given the same seed.
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(roots)

        batch = []
        batch_size = 0

        for root in roots:
            indices = list(self.dataset.get_root_indices(root))
            n_clips = len(indices)

            if self.shuffle:
                rng.shuffle(indices)

            # --------------------------------------------------------
            # If this whole video does not fit, emit the current batch
            # and start a new one with this video.
            # --------------------------------------------------------
            if batch and batch_size + n_clips > self.batch_size:
                yield batch

                batch = []
                batch_size = 0

            # Add ALL clips from this video.
            batch.extend(indices)
            batch_size += n_clips

            # Exactly full.
            if batch_size == self.batch_size:
                yield batch
                batch = []
                batch_size = 0

        # ------------------------------------------------------------
        # Remaining videos/clips.
        #
        # drop_last=False is what allows us to traverse the entire
        # dataset, including the final incomplete batch.
        # ------------------------------------------------------------
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        """
        Number of batches.

        Because batches are packed using complete videos, this is not
        simply ceil(len(dataset) / batch_size).
        """
        n_batches = 0
        current_size = 0

        for root in self.roots:
            n_clips = len(self.dataset.get_root_indices(root))

            if current_size and current_size + n_clips > self.batch_size:
                n_batches += 1
                current_size = 0

            current_size += n_clips

            if current_size == self.batch_size:
                n_batches += 1
                current_size = 0

        if current_size and not self.drop_last:
            n_batches += 1

        return n_batches