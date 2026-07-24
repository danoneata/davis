import pickle
import librosa
import scipy
import numpy as np
from tqdm import tqdm
import sys
import os


def load_audio(path, target_length=2048, use_raw_length=False):
    """ Step 3
        Convert audio wav file to mel spec feats
        target_length is the maximum number of frames stored (disable with use_raw_length)
        # NOTE: assumes audio in 16 kHz wav file
    """
    audio_type = 'melspectrogram'
    preemph_coef = 0.97
    sample_rate = 16000
    window_size = 0.025
    window_stride = 0.01
    window_type = 'hamming'
    num_mel_bins = 40
    padval = 0
    fmin = 20
    n_fft = int(sample_rate * window_size)
    win_length = int(sample_rate * window_size)
    hop_length = int(sample_rate * window_stride)
    windows = {'hamming': scipy.signal.windows.hamming}
    # load audio, subtract DC, preemphasis
    # NOTE: sr=None to avoid resampling (assuming audio already at 16 kHz sr
    y, sr = librosa.load(path, sr=None)
    if y.size == 0:
        y = np.zeros(200)
    y = y - y.mean()
    y = preemphasis(y, preemph_coef)
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
        win_length=win_length,
        window=windows[window_type])
    spec = np.abs(stft)**2
    if audio_type == 'melspectrogram':
        mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=num_mel_bins, fmin=fmin)
        melspec = np.dot(mel_basis, spec)
        feats = librosa.power_to_db(melspec, ref=np.max)
    n_frames = feats.shape[1]
    if use_raw_length:
        target_length = n_frames
    p = target_length - n_frames
    if p > 0:
        feats = np.pad(feats, ((0,0),(0,p)), 'constant',
            constant_values=(padval,padval))
    elif p < 0:
        feats = feats[:,0:p]
        n_frames = target_length
    return feats, n_frames


def preemphasis(signal,coeff=0.97):
    """perform preemphasis on the input signal.

    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])


with open("/root/how2/avlnet_embeddings/train.pkl", "rb") as f:
    embeddings = pickle.load(f)

audio_embeddings = []
n_audio_frames = []
audio_paths = []
n_shards = 8
shard_n = int(sys.argv[1])
shard_len = int(len(embeddings["audio_paths"]) / n_shards)
start_idx = shard_n * shard_len
end_idx = start_idx + shard_len
if shard_n == n_shards - 1:
    end_idx = len(embeddings["audio_paths"])
for idx in tqdm(list(range(start_idx, end_idx))):
    try:
        audio_embedding, n_frames = load_audio(embeddings["audio_paths"][idx])
        audio_embeddings.append(audio_embedding)
        n_audio_frames.append(n_frames)
        audio_paths.append(embeddings["audio_paths"][idx])
    except Exception as e:
        print('\n\n\n\n\n\n')
        print(f"Error processing {embeddings['audio_paths'][idx]}: {e}")
        print('\n\n\n\n\n\n')
        continue

if not os.path.exists(os.path.dirname("/root/how2/avlnet_embeddings/train_with_audio_embeddings/")):
    os.makedirs(os.path.dirname("/root/how2/avlnet_embeddings/train_with_audio_embeddings/"))
with open("/root/how2/avlnet_embeddings/train_with_audio_embeddings/shard_{}.pkl".format(shard_n), "wb") as f:
    pickle.dump({
        "video_embeddings": embeddings["embeddings"],
        "audio_embeddings": audio_embeddings,
        "n_audio_frames": n_audio_frames,
        "audio_paths": audio_paths,
    }, f)