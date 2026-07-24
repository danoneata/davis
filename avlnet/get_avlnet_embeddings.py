import os
import pickle
from tqdm import tqdm
import fire
import torch
import torchvision.models as models
import numpy as np
from decord import VideoReader
from decord import cpu
from PIL import Image
import sys
from transformers import VideoMAEModel, VideoMAEImageProcessor
import librosa
import scipy



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



class ImageProcessor:
    def __init__(self):
        self.weights = models.ResNet152_Weights.DEFAULT
        self.model = models.resnet152(weights=self.weights)
        self.model.fc = torch.nn.Identity()
        self.model.eval().cuda()
        self.preprocess = self.weights.transforms()

    def __call__(self, segment_path):
        vr = VideoReader(segment_path, ctx=cpu(0))
        num_frames = len(vr)
        fps = float(vr.get_avg_fps())
        frame_indices = np.arange(0, num_frames, int(round(fps)))
        embeddings = []

        frames = []
        for idx in frame_indices:
            frame = vr[idx].asnumpy()  # H, W, C in RGB for decord
            image = Image.fromarray(frame)
            image = self.preprocess(image)
            frames.append(image)
        frames = torch.stack(frames)
        frames = frames.cuda()
        embeddings = self.model(frames)
        embedding = embeddings.max(dim=0)[0]
        return embedding.detach().cpu().numpy()


class VideoProcessor:
    def __init__(self):
        # Using VideoMAE Large finetuned on Kinetics for better feature extraction
        model_id = "MCG-NJU/videomae-large-finetuned-kinetics"
        self.processor = VideoMAEImageProcessor.from_pretrained(model_id)
        self.model = VideoMAEModel.from_pretrained(model_id)
        self.model.eval().cuda()

    def __call__(self, segment_path):
        vr = VideoReader(segment_path, ctx=cpu(0))
        num_frames = len(vr)
        
        # VideoMAE expects exactly 16 frames per clip
        clip_len = 16
        
        # We sample exactly 16 frames evenly spaced across the entire segment
        frame_indices = np.linspace(0, num_frames - 1, clip_len, dtype=int)
        
        frames = vr.get_batch(frame_indices).asnumpy()  # T, H, W, C
        
        # VideoMAEImageProcessor expects a list of numpy arrays
        frames_list = [frame for frame in frames]
        
        # Processor handles resizing, normalization, and adds batch dimension
        inputs = self.processor([frames_list], return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Max pool the patches to get a single embedding for the whole segment
            embedding = outputs.last_hidden_state.max(dim=1)[0]
            
        return embedding.squeeze(0).detach().cpu().numpy()


def main(
    split = "train",
):
    config = {
        "segments_path": f"/root/how2/segments/{split}",
        "audios_path": f"/root/how2/audios/{split}",
        "embeddings_path": f"/root/how2/avlnet_embeddings/{split}.pkl",
    }
    if not os.path.exists(os.path.dirname(config["embeddings_path"])):
        os.makedirs(os.path.dirname(config["embeddings_path"]))

    embeddings = []
    audio_embeddings = []
    n_audio_frames = []
    audio_paths = []
    image_processor = ImageProcessor()
    video_processor = VideoProcessor()

    segment_names = sorted(os.listdir(config['segments_path']))
    for segment_name in tqdm(segment_names):
        segment_path = os.path.join(config["segments_path"], segment_name)
        audio_path = os.path.join(config["audios_path"], segment_name.replace(".mp4", ".wav"))

        try:
            image_embedding = image_processor(segment_path)
            video_embedding = video_processor(segment_path)
            audio_embedding, n_frames = load_audio(audio_path)
        except Exception as e:
            print(f"Error processing {segment_name}: {e}")
            continue

        embeddings.append(np.concatenate([image_embedding, video_embedding], axis=0))
        audio_embeddings.append(audio_embedding)
        n_audio_frames.append(n_frames)
        audio_paths.append(audio_path)
    with open(config["embeddings_path"], "wb") as f:
        pickle.dump({
            "video_embeddings": embeddings,
            "audio_embeddings": audio_embeddings,
            "n_audio_frames": n_audio_frames,
            "audio_paths": audio_paths,
        }, f)



if __name__ == "__main__":
    fire.Fire(main)