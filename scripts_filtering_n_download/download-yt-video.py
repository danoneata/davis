import csv
import logging
import os
import re
import shutil
import sys
import subprocess
import tempfile

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Matches the line yt-dlp prints once per video, right as it starts
# processing that URL, e.g. "[youtube] -2yHu5qgTzM: Downloading webpage"
VIDEO_START_RE = re.compile(r"\[youtube\]\s+([\w-]{11}):")

YT_DLP_SCRIPT = "yt-dlp/yt-dlp.sh"

COOKIES_MASTER_PATH = "cookies_master3.txt"
COOKIES_FILE_PATH = "cookies3.txt"

VIDEO_STORAGE = "/mnt/private-share/speechDatabases/Panda70M/panda2m_training_part3"
DOWNLOAD_ARCHIVE = "panda2m_part3_downloaded.txt"


def refresh_working_cookies() -> None:
    """
    Copies the master cookies file into the working cookies file before
    each run.
    """
    if not os.path.exists(COOKIES_MASTER_PATH):
        raise FileNotFoundError(
            f"Master cookies file not found: {COOKIES_MASTER_PATH}. "
            "Export fresh cookies and save them there first."
        )

    shutil.copyfile(
        COOKIES_MASTER_PATH,
        COOKIES_FILE_PATH,
    )

    logger.info(
        "Refreshed working cookies from %s",
        COOKIES_MASTER_PATH,
    )


def read_rows(input_path: str) -> list[tuple[str, str]]:
    """Returns a list of (videoID, url) tuples from the input CSV/TSV."""

    with open(
        input_path,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        first_line = f.readline()

    delimiter = "\t" if "\t" in first_line else ","

    with open(
        input_path,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        reader = csv.DictReader(
            f,
            delimiter=delimiter,
        )

        if (
            not reader.fieldnames
            or "url" not in reader.fieldnames
            or "videoID" not in reader.fieldnames
        ):
            raise ValueError(
                f"Expected 'videoID' and 'url' columns, "
                f"got: {reader.fieldnames}"
            )

        rows = [
            (
                row["videoID"].strip(),
                row["url"].strip(),
            )
            for row in reader
            if row.get("videoID", "").strip()
            and row.get("url", "").strip()
        ]

    return rows


def already_downloaded(video_id: str) -> bool:
    return os.path.exists(
        os.path.join(
            VIDEO_STORAGE,
            f"{video_id}.mp4",
        )
    )


def download_all(urls: list[str]) -> int:
    os.makedirs(
        VIDEO_STORAGE,
        exist_ok=True,
    )

    # yt-dlp wants a plain text file with one URL per line
    # for --batch-file.
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        delete=False,
        encoding="utf-8",
    ) as batch_file:

        batch_file.write(
            "\n".join(urls)
        )

        batch_file_path = batch_file.name

    output_template = os.path.join(
        VIDEO_STORAGE,
        "%(id)s.%(ext)s",
    )

    command = [
        "sh",
        YT_DLP_SCRIPT,

        "--batch-file",
        batch_file_path,

        "--download-archive",
        DOWNLOAD_ARCHIVE,

        "--no-abort-on-error",

        # Request throttling
        "--sleep-requests",
        "1",

        "--sleep-interval",
        "4",

        "--max-sleep-interval",
        "6",

        # Retry behavior
        "--retries",
        "1",

        "--fragment-retries",
        "1",

        "--extractor-retries",
        "1",

        "--retry-sleep",
        "http:exp=2:5",

        "--retry-sleep",
        "fragment:exp=2:5",

        "--retry-sleep",
        "extractor:exp=2:5",

        # Download multiple video fragments concurrently
        "--concurrent-fragments",
        "8",

        # ---------------------------------------------------------
        # MAX VIDEO HEIGHT = 360p
        #
        # Prefer MP4 video + M4A audio.
        # Only select video streams with height <= 360.
        #
        # If separate video/audio isn't available, fall back
        # to a single MP4 <= 360p.
        # ---------------------------------------------------------
        "-f",
        (
            "bestvideo[height<=360][ext=mp4]"
            "+bestaudio[ext=m4a]"
            "/best[height<=360][ext=mp4]"
            "/best[height<=360]"
        ),

        "--merge-output-format",
        "mp4",

        # YouTube JS challenge solving
        "--remote-components",
        "ejs:github",

        "--js-runtimes",
        "deno",

        # Cookies
        "--cookies",
        COOKIES_FILE_PATH,

        # Output
        "--output",
        output_template,
    ]

    logger.info(
        "Starting batch download of %d URLs",
        len(urls),
    )

    logger.info(
        "Command: %s",
        command,
    )

    log_file_path = os.path.join(
        VIDEO_STORAGE,
        "yt-dlp.log",
    )

    seen_ids = set()

    try:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        with (
            open(
                log_file_path,
                "a",
                encoding="utf-8",
            ) as log_file,

            tqdm(
                total=len(urls),
                desc="Downloading videos",
                unit="video",
            ) as pbar
        ):

            # Read yt-dlp output line-by-line.
            # Everything is:
            #   1. written to yt-dlp.log
            #   2. displayed in the terminal
            for line in proc.stdout:
                line = line.rstrip()

                # Always preserve complete yt-dlp output.
                log_file.write(
                    line + "\n"
                )
                log_file.flush()

                # Display yt-dlp output while keeping tqdm usable.
                tqdm.write(line)

                # Detect start of a new YouTube video.
                match = VIDEO_START_RE.search(line)

                if match:
                    video_id = match.group(1)

                    if video_id not in seen_ids:
                        seen_ids.add(video_id)

                        pbar.set_postfix_str(
                            video_id
                        )

                        pbar.update(1)

            # Wait for yt-dlp to finish.
            return_code = proc.wait()

            # If yt-dlp skipped URLs because they were already in
            # the download archive, the "[youtube] ID:" line might
            # not have appeared. Fill the progress bar at the end.
            if pbar.n < pbar.total:
                pbar.update(
                    pbar.total - pbar.n
                )

        return return_code

    finally:
        if os.path.exists(batch_file_path):
            os.remove(batch_file_path)


def main():
    if len(sys.argv) < 2:
        print(
            f"Usage: python {os.path.basename(__file__)} "
            "<input_csv_or_tsv>"
        )
        sys.exit(1)

    input_path = sys.argv[1]

    if not os.path.exists(input_path):
        print(
            f"Input file not found: {input_path}"
        )
        sys.exit(1)

    rows = read_rows(input_path)

    if not rows:
        print(
            "No rows found in input file."
        )
        sys.exit(1)

    os.makedirs(
        VIDEO_STORAGE,
        exist_ok=True,
    )

    to_download = []
    skipped = 0

    for video_id, url in rows:
        if already_downloaded(video_id):
            skipped += 1
        else:
            to_download.append(url)

    logger.info(
        "%d total, %d already downloaded (skipped), "
        "%d to download",
        len(rows),
        skipped,
        len(to_download),
    )

    if not to_download:
        logger.info(
            "Nothing left to download."
        )
        sys.exit(0)

    refresh_working_cookies()

    return_code = download_all(
        to_download
    )

    if return_code == 0:
        logger.info(
            "All downloads completed successfully."
        )
    else:
        logger.warning(
            "yt-dlp exited with code %s. "
            "Some videos may have failed "
            "(--no-abort-on-error means the rest "
            "were still attempted). "
            "Check %s for what succeeded, or "
            "re-run this script later to retry "
            "the failures.",
            return_code,
            DOWNLOAD_ARCHIVE,
        )

    sys.exit(return_code)


if __name__ == "__main__":
    main()