from tqdm import tqdm
import os
import pickle
import sys
import numpy as np
import torch

embeddings_path = '/mnt/private-share/speechDatabases/how2/processed_how2/embeddings/all/'
embeddings = []
outfile = "/mnt/private-share/speechDatabases/how2/processed_how2/embeddings/all_grouped_embeddings.pkl"
for embedding_file in tqdm(os.listdir(embeddings_path), total=len(os.listdir(embeddings_path)), desc="Loading embeddings"):
    try:
        with open(os.path.join(embeddings_path, embedding_file), "rb") as f:
            embeddings.append(pickle.load(f))
    except Exception as e:
        print(f"Error loading {embedding_file}: {e}")
        continue

grouped_embeddings = {
    "video_embeddings": np.array([e["video_embedding"] for e in embeddings]),
    "image_embeddings": np.array([e["image_embedding"] for e in embeddings]),
    "audio_embeddings": np.array([e["audio_embedding"] for e in embeddings]),
}

with open(outfile, "wb") as f:
    pickle.dump(grouped_embeddings, f)