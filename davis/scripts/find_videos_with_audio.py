import os
import subprocess

import click
from toolz import first

from davis.scripts.show_data import VIDEO_DIRS

PATHS_IDS = {
    "msr-vtt": "msrvtt_val_ids.txt",
    "youcook2": "youcook_val_ids_common.txt",
}


def read_file(path, parse_line=lambda x: x.strip()):
    with open(path) as f:
        return [parse_line(line) for line in f]


def get_path_video(dataset, filename):
    drop_ext = lambda f: os.path.splitext(f)[0]
    folder = VIDEO_DIRS[dataset]
    files = os.listdir(folder)
    file = [f for f in files if drop_ext(f) == filename]
    try:
        file = first(file)
        return os.path.join(folder, file)
    except StopIteration:
        raise StopIteration("Failed to find video for id: {}".format(filename))


def has_audio(path_video):
    # Check if the file is empty
    if os.path.getsize(path_video) == 0:
        return False

    # Use ffprobe to check for audio streams
    command = ["ffprobe", "-i", path_video, "-show_streams", "-select_streams", "a", "-loglevel", "error"]
    result = subprocess.run(command, capture_output=True, text=True)
    return bool(result.stdout.strip())


@click.command()
@click.argument("dataset", type=click.Choice(VIDEO_DIRS.keys()))
def main(dataset):
    path_ids = "/data/how2/avlnet-data/{}".format(PATHS_IDS[dataset])
    filenames = read_file(path_ids)
    for filename in filenames:
        path_video = get_path_video(dataset, filename)
        has_audio_str = "✓" if has_audio(path_video) else "✗"
        print(f"{has_audio_str} {path_video}")


if __name__ == "__main__":
    main()
