"""
Run Tag2Text on video frames sampled every 1 second.
Prints tags + caption for each frame.
"""

import argparse
import cv2
import torch
from PIL import Image

from ram.models import tag2text
from ram import inference_tag2text as inference
from ram import get_transform


parser = argparse.ArgumentParser(
    description='Tag2Text inference on video frames'
)

parser.add_argument(
    '--video',
    required=True,
    help='Path to video'
)

parser.add_argument(
    '--pretrained',
    default='pretrained/tag2text_swin_14m.pth',
    help='Path to pretrained model'
)

parser.add_argument(
    '--image-size',
    default=384,
    type=int,
    help='Input image size'
)

parser.add_argument(
    '--thre',
    default=0.68,
    type=float,
    help='Threshold value'
)

parser.add_argument(
    '--specified-tags',
    default='None',
    help='User specified tags'
)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# tags to suppress
delete_tag_index = [
    127,
    2961,
    3351,
    3265,
    3338,
    3355,
    3359
]

# load model
model = tag2text(
    pretrained=args.pretrained,
    image_size=args.image_size,
    vit='swin_b',
    delete_tag_index=delete_tag_index
)

model.threshold = args.thre
model.eval()
model = model.to(device)

transform = get_transform(image_size=args.image_size)

# open video
cap = cv2.VideoCapture(args.video)

if not cap.isOpened():
    raise Exception(f"Cannot open video: {args.video}")

fps = cap.get(cv2.CAP_PROP_FPS)

print(f"\nVideo FPS: {fps}")

frame_interval = int(fps * 0.5)  # every 1 second

frame_idx = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # process one frame every 1 second
    if frame_idx % frame_interval == 0:

        timestamp = frame_idx / fps

        print(f"\n========== FRAME @ {timestamp:.2f}s ==========")

        # BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # numpy -> PIL
        image = Image.fromarray(frame_rgb)

        # preprocess
        image_tensor = transform(image).unsqueeze(0).to(device)

        # inference
        res = inference(
            image_tensor,
            model,
            args.specified_tags
        )

        print("\nRAW RESULT:")
        print(res)

        print("\nTAGS:")
        print(res[0])

        print("\nCAPTION:")
        print(res[2])

    frame_idx += 1

cap.release()

print("\nDone.")
