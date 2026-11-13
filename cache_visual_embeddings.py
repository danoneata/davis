import av, torch, glob, os, sys
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

name = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
# name = "facebook/dinov2-large"
processor = AutoImageProcessor.from_pretrained(name)
model = AutoModel.from_pretrained(name, torch_dtype=torch.bfloat16).to("cuda").eval()

def dino_encode(path):
    frames, next_t = [], 0
    with av.open(path) as container:
        for frame in container.decode(video=0):
            if frame.time >= next_t:                  # first frame past the middle of each second
                frames.append(frame.to_image())
                next_t += 0.2
    pixels = processor(images=frames, return_tensors="pt").pixel_values.to("cuda", torch.bfloat16)
    with torch.no_grad():
        return model(pixel_values=pixels).last_hidden_state[:, 0].detach().cpu().to(torch.bfloat16)  # (T, 768) CLS per second

for split in ['train', 'val', 'dev5']:
    os.makedirs(f'./embeddings/visual/{split}', exist_ok=True)
    video_paths = sorted(glob.glob(f'/root/how2/segments/{split}/*'))
    for video_path in tqdm(video_paths):
        embedding = dino_encode(video_path)
        embedding_filename = os.path.basename(video_path.replace('.mp4', '.pt'))
        torch.save(embedding.clone(), f'./embeddings/visual/{split}/{embedding_filename}')
