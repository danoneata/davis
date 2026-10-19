import os
import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from tqdm import tqdm

embeddings_path = "/mnt/private-share/speechDatabases/how2/processed_how2/embeddings/all/"
outfile = "/mnt/private-share/speechDatabases/how2/processed_how2/embeddings/all_grouped_embeddings.pkl"
NUM_WORKERS = 32  # I/O-bound on a network share; raise or lower to taste

KEYS = ("video_embedding", "image_embedding", "audio_embedding")


def to_np(x):
    """Convert a torch tensor or array-like to a float32 numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_one(name):
    """Load one file and keep only the three embeddings (drops everything else)."""
    try:
        with open(os.path.join(embeddings_path, name), "rb") as f:
            data = f.read()          # one big read is faster than pickle streaming over NFS/SMB
        e = pickle.loads(data)
        return name, tuple(to_np(e[k]) for k in KEYS)
    except Exception as ex:
        return name, ex


# os.scandir is faster than os.listdir on large dirs; sort for a deterministic row order
files = sorted(entry.name for entry in os.scandir(embeddings_path) if entry.is_file())

names, video, image, audio = [], [], [], []
with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
    # map() keeps results in the same order as `files`
    for name, result in tqdm(pool.map(load_one, files, chunksize=64),
                             total=len(files), desc="Loading embeddings"):
        if isinstance(result, Exception):
            print(f"Error loading {name}: {result}")
            continue
        v, i, a = result
        names.append(name)
        video.append(v)
        image.append(i)
        audio.append(a)

# np.stack on a list of numpy arrays is much faster than np.array on a list of tensors
grouped_embeddings = {
    "video_embeddings": np.stack(video),
    "image_embeddings": np.stack(image),
    "audio_embeddings": np.stack(audio),
    "file_names": names,  # row i of each array comes from file_names[i]
}

with open(outfile, "wb") as f:
    pickle.dump(grouped_embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Saved {len(names)} embeddings to {outfile}")
