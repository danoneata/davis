import pickle
import numpy as np



with open("/root/how2/avlnet_embeddings/train_with_audio_embeddings.pkl", "rb") as f:
    data = pickle.load(f)

print(len(data["video_embeddings"]))
print(len(data["audio_embeddings"]))
print(len(data["n_audio_frames"]))