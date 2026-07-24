import subprocess
import glob
import hydra
from tqdm import tqdm
import os
import sys
import pathlib



def extract_audio(input_file, output_file):
    """ Step 1
        Extracts audio at the native sampling rate into a separate wav file. 
    """
    subprocess.call(['ffmpeg', '-i', input_file, '-vn', output_file])


def stereo_to_mono_downsample(input_file, output_file, sample_rate=16000):
    """ Step 2
        Resamples wav file (we use 16 kHz).
        Convert from stereo to mono.
        Apply a gain of -4 to avoid clipping for mono to stereo conversion.
    """
    subprocess.call(['sox', input_file, output_file, 'gain', '-4', 'channels', '1', 'rate', str(sample_rate)])


def main() -> None:
    split = sys.argv[1]
    config = {
        "segments_path": f"/root/how2/segments/{split}",
        "audios_path": f"/root/how2/audios/{split}",
    }
    os.makedirs(config["audios_path"], exist_ok=True)
    for video_path in tqdm(glob.glob(config["segments_path"] + "/*.mp4")):
        video_name = os.path.basename(video_path)
        audio_name = video_name.replace(".mp4", ".wav")
        audio_path = os.path.join(config["audios_path"], audio_name)
        if os.path.exists(audio_path):
            continue
        temp_audio_path = audio_path.replace(".wav", "_temp.wav")
        extract_audio(video_path, temp_audio_path)
        stereo_to_mono_downsample(temp_audio_path, audio_path)
        os.remove(temp_audio_path)


if __name__ == "__main__":
    main()