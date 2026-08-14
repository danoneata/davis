import argparse
import json
import cv2
import torch

from PIL import Image
from difflib import SequenceMatcher
from tqdm import tqdm

from ram.models import tag2text
from ram import inference_tag2text as inference
from ram import get_transform


def sim(a, b):
    return SequenceMatcher(
        None,
        a.lower(),
        b.lower()
    ).ratio()


def best_caption(captions):

    if len(captions) == 1:
        return captions[0]

    best = None
    best_score = -1

    for i, c1 in enumerate(captions):

        score = sum(
            sim(c1, c2)
            for j, c2 in enumerate(captions)
            if i != j
        )

        score /= (len(captions) - 1)

        if score > best_score:
            best_score = score
            best = c1

    return best


def process_video(
    video_path,
    model,
    transform,
    device,
    spec_tags,
    sim_th
):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)

    # sample every 1 second
    step = max(int(fps), 0.5)

    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    # true video duration
    video_duration = frame_count / fps

    total_steps = max(frame_count // step, 1)

    buffer_caps = []
    buffer_times = []

    segments = []

    frame_idx = 0

    pbar = tqdm(
        total=total_steps,
        desc="Frames",
        leave=False
    )

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_idx % step == 0:

            t = frame_idx / fps

            img = Image.fromarray(
                cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )
            )

            x = transform(img)\
                .unsqueeze(0)\
                .to(device)

            res = inference(
                x,
                model,
                spec_tags
            )

            caption = str(res[2]).strip()

            # first caption
            if not buffer_caps:

                buffer_caps = [caption]
                buffer_times = [t]

            else:

                similarity = sim(
                    buffer_caps[-1],
                    caption
                )

                # merge similar captions
                if similarity >= sim_th:

                    buffer_caps.append(caption)
                    buffer_times.append(t)

                else:

                    segments.append([

                        # start
                        round(buffer_times[0], 2),

                        # end at new caption start
                        round(t, 2),

                        # best representative caption
                        best_caption(buffer_caps)
                    ])

                    # new segment
                    buffer_caps = [caption]
                    buffer_times = [t]

            pbar.update(1)

        frame_idx += 1

    cap.release()
    pbar.close()

    # flush final segment
    if buffer_caps:

        segments.append([

            # segment start
            round(buffer_times[0], 2),

            # extend to actual end of video
            round(video_duration, 2),

            # representative caption
            best_caption(buffer_caps)
        ])

    return segments


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input_jsonl',
        required=True
    )

    parser.add_argument(
        '--output_jsonl',
        required=True
    )

    parser.add_argument(
        '--pretrained',
        default='pretrained/tag2text_swin_14m.pth'
    )

    parser.add_argument(
        '--image-size',
        type=int,
        default=384
    )

    parser.add_argument(
        '--thre',
        type=float,
        default=0.68
    )

    parser.add_argument(
        '--similarity-threshold',
        type=float,
        default=0.5
    )

    parser.add_argument(
        '--specified-tags',
        default='None'
    )

    args = parser.parse_args()

    device = torch.device(
        'cuda'
        if torch.cuda.is_available()
        else 'cpu'
    )

    delete_tag_index = [
        127,
        2961,
        3351,
        3265,
        3338,
        3355,
        3359
    ]

    model = tag2text(
        pretrained=args.pretrained,
        image_size=args.image_size,
        vit='swin_b',
        delete_tag_index=delete_tag_index
    ).to(device).eval()

    model.threshold = args.thre

    transform = get_transform(
        args.image_size
    )

    with open(args.input_jsonl) as fin, \
         open(args.output_jsonl, "w") as fout:

        lines = fin.readlines()

        for line in tqdm(
            lines,
            desc="Videos"
        ):

            item = json.loads(line)

            item["captions"] = process_video(
                item["video_segment"],
                model,
                transform,
                device,
                args.specified_tags,
                args.similarity_threshold
            )

            fout.write(
                json.dumps(item) + "\n"
            )


if __name__ == "__main__":
    main()