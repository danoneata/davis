"""Dataset loader for pre-extracted V-JEPA2 + DINOv2 + audio (SPEAR / WavLM) features.

Expects the file layout produced by ``extract_features.py``:

    <embeddings_path>/<segment_id>.pkl

where each pickle is a dict with keys ``video_embedding`` (V-JEPA2),
``image_embedding`` (DINOv2), and ``audio_embedding`` (SPEAR [T, 1280] frame
features; already-pooled [D] features also work).

With ``audio_pool=True`` the precomputed ``audio_embedding_pooled`` [D] vector
is used if the pickle has it (see add_pooled_audio.py), otherwise the frames are
mean-pooled to [D] here; with
``audio_pool=False`` they are kept as [T, D], zero-padded per batch, and pooled
by the model (see ``AttentivePooling`` in model.py).

Which ids are used for train / val / test is controlled by a manifest file:
either a JSONL manifest (one record per line, with ``video_path`` and optionally
``original_video_id`` / ``speaker_id``) or a plain text file with one id per line.

Training batches can be built in three ways (``batch_mode``):
    - "random":  plain shuffled batches over the whole dataset.
    - "video":   clips of the same ``original_video_id`` are packed together, so
                 in-batch negatives come from the same source video.
    - "speaker": clips of the same ``speaker_id`` are packed together, so in-batch
                 negatives come from the same speaker.
"""
from __future__ import absolute_import, division, unicode_literals, print_function

import os
import pickle
import random
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch as th
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler
from torch.utils.data.dataloader import default_collate

from src.utils import read_manifest

BATCH_MODES = ("random", "video", "speaker")
NULL_GROUP = ""  # key shared by all clips with a null video / speaker id


class Feature_DataLoader(Dataset):
    """Loads pre-extracted per-clip audio/image/video embeddings."""

    def __init__(
        self,
        ids_path,
        embeddings_path,
        embeddings_path_split,
        split,
        normalize=True,
        image_only=False,
        audio_pool=True,
        io_threads=8,
    ):
        if embeddings_path_split is not None:
            self.embeddings_path = Path(embeddings_path_split)
        else:
            self.embeddings_path = Path(embeddings_path) / split

        self.normalize = normalize
        self.image_only = image_only
        self.audio_pool = audio_pool
        self.io_threads = io_threads
        self._pool = None  # per-process thread pool, created lazily

        # One directory listing instead of one stat() per id: on NFS a
        # stat per file costs a network round trip (minutes for 1M+ ids).
        if self.embeddings_path.is_dir():
            with os.scandir(self.embeddings_path) as it:
                available = {e.name[:-4] for e in it if e.name.endswith(".pkl")}
        else:
            available = set()

        wanted = read_manifest(ids_path)
        records = [r for r in wanted if r["id"] in available]
        del available

        missing = len(wanted) - len(records)
        if missing:
            print(
                f"[Feature_DataLoader] {missing} id(s) from {ids_path} have no "
                f"embeddings in {self.embeddings_path} and will be skipped."
            )

        # numpy string arrays instead of lists of Python str: reading Python
        # objects in forked workers touches their refcounts, which slowly
        # copies the parent's memory into every worker.
        # Null video / speaker ids all map to NULL_GROUP, so they form a
        # single group in group_indices.
        self.ids = np.array([r["id"] for r in records], dtype=str)
        self.video_ids = np.array(
            [NULL_GROUP if r["original_video_id"] is None else str(r["original_video_id"]) for r in records],
            dtype=str,
        )
        self.speaker_ids = np.array(
            [NULL_GROUP if r["speaker_id"] is None else str(r["speaker_id"]) for r in records],
            dtype=str,
        )
        del records, wanted

    def __getstate__(self):
        # Thread pools can't be pickled / don't survive fork; each worker makes its own.
        state = self.__dict__.copy()
        state["_pool"] = None
        return state

    def group_indices(self, batch_mode):
        """Return a list of index groups (one per video / speaker)."""
        if batch_mode == "video":
            keys = self.video_ids
        elif batch_mode == "speaker":
            keys = self.speaker_ids
        else:
            raise ValueError(f"No grouping for batch_mode '{batch_mode}'.")

        groups = defaultdict(list)
        for idx, key in enumerate(keys.tolist()):
            groups[key].append(idx)

        print(
            f"[Feature_DataLoader] {len(self.ids)} clips "
            f"from {len(groups)} {batch_mode} groups"
            + (f" ({len(groups[NULL_GROUP])} clips with a null {batch_mode} id grouped together)"
               if NULL_GROUP in groups else "")
        )
        return list(groups.values())

    def __len__(self):
        return len(self.ids)

    def custom_collate(self, batch):
        if self.audio_pool:
            return default_collate(batch)

        # Temporal audio: zero-pad [T_i, D] to [B, T_max, D] and add a
        # [B, T_max] bool mask (True = real frame) for the model's pooling.
        audios = [b["audio"] for b in batch]
        lengths = th.tensor([a.shape[0] for a in audios])
        out = default_collate([{k: v for k, v in b.items() if k != "audio"} for b in batch])
        out["audio"] = pad_sequence(audios, batch_first=True)
        out["audio_mask"] = th.arange(out["audio"].shape[1])[None, :] < lengths[:, None]
        return out

    def _load_audio(self, audio_embedding):
        """[T, D] frame features -> [D] (audio_pool) or [T, D] (temporal).

        Already-pooled [D] features are used as-is, or as a single frame.
        """
        audio_embedding = th.from_numpy(np.asarray(audio_embedding)).float()
        if self.audio_pool:
            if audio_embedding.dim() == 2:
                audio_embedding = audio_embedding.mean(dim=0)
            if self.normalize:
                audio_embedding = F.normalize(audio_embedding, dim=0)
        elif audio_embedding.dim() == 1:
            audio_embedding = audio_embedding[None, :]
        # In temporal mode, normalization happens in the model after pooling.
        return audio_embedding

    def _audio_feats(self, data, key):
        """With audio_pool, prefer the precomputed "<key>_pooled" [D] vector
        (see add_pooled_audio.py) if present; otherwise return the raw [T, D]
        frames and let _load_audio pool them."""
        if self.audio_pool:
            pooled = data.get(f"{key}_pooled")
            if pooled is not None:
                return pooled
        return data[key]

    def _load(self, seg_id):
        with open(self.embeddings_path / f"{seg_id}.pkl", "rb") as f:
            return pickle.load(f)

    def __getitem__(self, idx):
        return self._read(idx)

    def __getitems__(self, indices):
        """Fetch a whole batch (DataLoader calls this instead of __getitem__).

        Clips are read with a thread pool: each read mostly
        waits on the (NFS) disk, so a few reads in flight hide that latency
        instead of one worker reading its batch file by file.
        """
        if self.io_threads <= 1 or len(indices) <= 1:
            return [self[i] for i in indices]
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self.io_threads)
        return list(self._pool.map(self.__getitem__, indices))

    def _read(self, idx):
        """Load one clip from disk and build its sample."""
        seg_id = str(self.ids[idx])
        data = self._load(seg_id)

        image_embedding = th.from_numpy(
            np.asarray(data["image_embedding"])
        ).float()

        if self.image_only:
            # Some Flick8k extracted features use "audio_embeddings"
            # rather than "audio_embedding".
            if "audio_embedding" in data:
                audio_embedding = self._load_audio(self._audio_feats(data, "audio_embedding"))
            elif "audio_embeddings" in data:
                audio_embedding = self._load_audio(data["audio_embeddings"][0])
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
                "speaker_id": str(self.speaker_ids[idx]),
            }

        video_embedding = th.from_numpy(
            np.asarray(data["video_embedding"])
        ).float()

        audio_embedding = self._load_audio(self._audio_feats(data, "audio_embedding"))

        if self.normalize:
            video_embedding = F.normalize(video_embedding, dim=0)
            image_embedding = F.normalize(image_embedding, dim=0)

        # DINOv2 + V-JEPA2
        video = th.cat((image_embedding, video_embedding))

        return {
            "video": video,
            "audio": audio_embedding,
            "video_id": seg_id,
            "speaker_id": str(self.speaker_ids[idx]),
        }


class GroupedBatchSampler(Sampler):
    """
    Batch sampler that keeps clips of the same group (video or speaker)
    together, so in-batch negatives are hard negatives.

    Each group is shuffled and cut into chunks of at most
    ``min(mining_size, batch_size)`` clips (``mining_size=-1`` -> chunks of up
    to batch_size, i.e. no per-group cap). The leftover clips of a large group
    are NOT dropped: they form further chunks that end up in other batches, so
    every clip is used once per epoch.

    Chunks are shuffled and packed so that every batch is exactly batch_size:
    if a chunk does not fit, it is split, its first part fills the batch and
    the rest opens the next batch. A batch never holds two chunks of the same
    group (that would exceed the cap); such a chunk is deferred to the next
    batch instead. Batches are smaller than batch_size only at the very end of
    the epoch, when the remaining clips come from too few distinct groups.

    Example with batch_size=6, mining_size=2:

        video A -> [0, 1, 2, 3, 4]  -> chunks [0, 1] [2, 3] [4]
        video B -> [5, 6]           -> chunk  [5, 6]
        video C -> [7, 8, 9]        -> chunks [7, 8] [9]

        batch 1 -> [A0 A1 | B5 B6 | C7 C8]
        batch 2 -> [A2 A3 | C9]
        batch 3 -> [A4]                      (only A left; dropped if drop_last)
    """

    def __init__(
        self,
        groups,
        batch_size,
        shuffle=True,
        drop_last=False,
        seed=0,
        mining_size=-1,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if mining_size == 0 or mining_size < -1:
            raise ValueError("mining_size must be > 0, or -1 for no cap")

        self.groups = [list(g) for g in groups]
        self.batch_size = batch_size
        self.chunk_size = batch_size if mining_size == -1 else min(mining_size, batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        """
        Call this at the beginning of every epoch when using a custom
        training loop, to get a different (reproducible) ordering.
        """
        self.epoch = epoch

    def _batches(self):
        rng = random.Random(self.seed + self.epoch)

        # (group_id, chunk) pairs; group_id lets us enforce the per-batch cap.
        chunks = []
        for gid, group in enumerate(self.groups):
            indices = list(group)
            if self.shuffle:
                rng.shuffle(indices)
            chunks.extend(
                (gid, indices[i:i + self.chunk_size])
                for i in range(0, len(indices), self.chunk_size)
            )

        if self.shuffle:
            rng.shuffle(chunks)

        queue = deque(chunks)
        deferred = []  # chunks whose group is already in the current batch
        batches = []
        batch = []
        groups_in_batch = set()

        def close_batch():
            nonlocal batch, groups_in_batch, deferred
            if batch:
                batches.append(batch)
            batch = []
            groups_in_batch = set()
            # Deferred chunks go first into the next batch.
            queue.extendleft(reversed(deferred))
            deferred = []

        while queue or deferred:
            if not queue:
                # Only same-group leftovers remain: start a new batch for them.
                close_batch()
                continue

            gid, chunk = queue.popleft()

            if gid in groups_in_batch:
                deferred.append((gid, chunk))
                continue

            # If the chunk does not fit, split it: fill the batch up to
            # batch_size and send the rest to the front of the next batch.
            room = self.batch_size - len(batch)
            if len(chunk) > room:
                chunk, rest = chunk[:room], chunk[room:]
                deferred.insert(0, (gid, rest))

            batch.extend(chunk)
            groups_in_batch.add(gid)

            if len(batch) == self.batch_size:
                close_batch()

        if batch and not self.drop_last:
            batches.append(batch)

        return batches

    def __iter__(self):
        return iter(self._batches())

    def __len__(self):
        # Packing depends on the chunk order, so count the batches of the
        # current epoch exactly.
        return len(self._batches())
