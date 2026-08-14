"""
Tag2Text video segmentation (1 frame/sec)

Improvements:
- Merge similar consecutive captions
- Select BEST representative caption per segment
  (highest average similarity to others)
"""

import argparse
import cv2
import torch
from PIL import Image
from difflib import SequenceMatcher

from ram.models import tag2text
from ram import inference_tag2text as inference
from ram import get_transform


def sim(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def best_caption(captions):
    """
    Choose caption that best represents the group:
    highest average similarity to all others.
    """
    if len(captions) == 1:
        return captions[0]

    best = None
    best_score = -1

    for i, c1 in enumerate(captions):
        score = 0.0
        for j, c2 in enumerate(captions):
            if i != j:
                score += sim(c1, c2)

        score /= (len(captions) - 1)

        if score > best_score:
            best_score = score
            best = c1

    return best


parser = argparse.ArgumentParser()
parser.add_argument('--video', required=True)
parser.add_argument('--pretrained', default='pretrained/tag2text_swin_14m.pth')
parser.add_argument('--image-size', type=int, default=384)
parser.add_argument('--thre', type=float, default=0.68)
parser.add_argument('--similarity-threshold', type=float, default=0.5)
parser.add_argument('--specified-tags', default='None')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

delete_tag_index = [127, 2961, 3351, 3265, 3338, 3355, 3359]

model = tag2text(
    pretrained=args.pretrained,
    image_size=args.image_size,
    vit='swin_b',
    delete_tag_index=delete_tag_index
)

model.threshold = args.thre
model.eval().to(device)

transform = get_transform(args.image_size)

cap = cv2.VideoCapture(args.video)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps*0.5)

segments = []

buffer_captions = []
buffer_times = []

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % frame_interval == 0:

        t = frame_idx / fps

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        x = transform(image).unsqueeze(0).to(device)

        res = inference(x, model, args.specified_tags)
        caption = str(res[2]).strip()

        print(f"[{t:.2f}s] {caption}")

        # first frame
        if not buffer_captions:
            buffer_captions.append(caption)
            buffer_times.append(t)

        else:
            prev = buffer_captions[-1]
            s = sim(prev, caption)

            if s >= args.similarity_threshold:
                buffer_captions.append(caption)
                buffer_times.append(t)

            else:
                # finalize previous segment
                best = best_caption(buffer_captions)

                segments.append(
                    (
                        round(buffer_times[0], 2),
                        round(buffer_times[-1], 2),
                        best
                    )
                )

                # start new segment
                buffer_captions = [caption]
                buffer_times = [t]

    frame_idx += 1

cap.release()

# flush last segment
if buffer_captions:
    best = best_caption(buffer_captions)
    segments.append(
        (
            round(buffer_times[0], 2),
            round(buffer_times[-1], 2),
            best
        )
    )

print("\n================ FINAL SEGMENTS ================\n")

for s in segments:
    print(f"({s[0]:.2f}s, {s[1]:.2f}s) -> {s[2]}")

print("\n================================================\n")