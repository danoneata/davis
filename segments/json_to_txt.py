import json
import os



with open('./how2_train-aligned_score-threshold8.json', 'r') as f:
    lines = f.readlines()
    lines = [json.loads(line) for line in lines]

with open('./how2_train-aligned_score-threshold8.txt', 'w') as f:
    for line in lines:
        print(line)
        segment_name = os.path.basename(line['video_path']).split('.')[0]
        video_name = line['original_video_id']
        start = line['start']
        end = line['end']
        f.write(f'{segment_name} {video_name} {start} {end}\n')