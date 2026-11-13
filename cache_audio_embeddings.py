import torch, librosa, glob, os
from transformers import WhisperFeatureExtractor, WhisperModel
from tqdm import tqdm

fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
encoder = WhisperModel.from_pretrained("openai/whisper-large-v3").get_encoder().cuda().eval()

def whisper_encode(path):
    audio, sr = librosa.load(path, sr=16000)
    duration = len(audio) / sr 
    feats = fe(audio, sampling_rate=16000, return_tensors="pt").input_features
    feats = feats.to(device=encoder.device, dtype=encoder.dtype)
    with torch.no_grad():
        embeddings = encoder(feats).last_hidden_state[0].detach().cpu().to(torch.bfloat16)
    embeddings = embeddings[:int(duration * 50), :]
    return embeddings
    
for split in ['train', 'val', 'dev5']:
    audio_paths = sorted(glob.glob(f'/root/how2/audios/{split}/*'))
    os.makedirs(f'./embeddings/audio/{split}', exist_ok=True)
    for audio_path in tqdm(audio_paths):
        embedding = whisper_encode(audio_path)
        embedding_filename = os.path.basename(audio_path.replace('.wav', '.pt'))
        torch.save(embedding.clone(), f'./embeddings/audio/{split}/{embedding_filename}')