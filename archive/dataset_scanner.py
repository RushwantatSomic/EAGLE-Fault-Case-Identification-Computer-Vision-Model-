import argparse
import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".m4v",
    ".wmv",
    ".flv",
    ".ts",
    ".mts",
    ".m2ts",
}


def run_ffprobe(video_path):
    """
    Read video metadata using FFprobe.
    Returns a dictionary containing metadata or an error.
    """

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,size,format_name:"
        "stream=index,codec_name,codec_type,width,height,"
        "r_frame_rate,avg_frame_rate,nb_frames",
        "-of",
        "json",
        str(video_path),
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )

        if result.returncode != 0:
            return {
                "readable": False,
                "error": result.stderr.strip(),
            }

        data = json.loads(result.stdout)

        streams = data.get("streams", [])
        video_stream = next(
            (s for s in streams if s.get("codec_type") == "video"),
            None,
        )

        if video_stream is None:
            return {
                "readable": False,
                "error": "No video stream found",
            }

        format_data = data.get("format", {})

        duration = None

        if format_data.get("duration"):
            duration = float(format_data["duration"])

        frame_count = video_stream.get("nb_frames")

        if frame_count is not None:
            try:
                frame_count = int(frame_count)
            except ValueError:
                frame_count = None

        fps = parse_fps(video_stream.get("avg_frame_rate"))

        if fps is None:
            fps = parse_fps(video_stream.get("r_frame_rate"))

        return {
            "readable": True,
            "error": None,
            "duration": duration,
            "fps": fps,
            "frame_count": frame_count,
            "width": video_stream.get("width"),
            "height": video_stream.get("height"),
            "codec": video_stream.get("codec_name"),
            "container": format_data.get("format_name"),
        }

    except subprocess.TimeoutExpired:
        return {
            "readable": False,
            "error": "FFprobe timeout",
        }

    except Exception as exc:
        return {
            "readable": False,
            "error": str(exc),
        }


def parse_fps(value):
    """
    Convert an FFmpeg frame-rate fraction such as 93/2 into FPS.
    """

    if not value or value == "0/0":
        return None

    try:
        if "/" in value:
            numerator, denominator = value.split("/")
            return float(numerator) / float(denominator)

        return float(value)

    except (ValueError, ZeroDivisionError):
        return None


def create_database(db_path):
    connection = sqlite3.connect(db_path)

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            full_path TEXT UNIQUE NOT NULL,
            filename TEXT NOT NULL,
            extension TEXT,

            relative_path TEXT,
            source_folder TEXT,

            size_bytes INTEGER,
            modified_time TEXT,

            duration_seconds REAL,
            fps REAL,
            frame_count INTEGER,

            width INTEGER,
            height INTEGER,

            codec TEXT,
            container TEXT,

            readable INTEGER,
            scan_error TEXT,

            scanned_at TEXT
        )
        """
    )

    connection.commit()

    return connection


def find_videos(root):
    """
    Recursively find all supported video files.
    """

    for path in root.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def format_hours(seconds):
    if not seconds:
        return 0.0

    return seconds / 3600.0


def main():

    parser = argparse.ArgumentParser(
        description="EAGLE-MARS recursive video dataset scanner"
    )

    parser.add_argument(
        "--root",
        required=True,
        help="Root directory containing the EAGLE videos",
    )

    parser.add_argument(
        "--database",
        default="database/eagle_dataset.db",
        help="SQLite database path",
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()
    db_path = Path(args.database).resolve()

    if not root.exists():
        print(f"ERROR: Dataset root does not exist:")
        print(root)
        sys.exit(1)

    if not root.is_dir():
        print(f"ERROR: Dataset root is not a directory:")
        print(root)
        sys.exit(1)

    db_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 70)
    print("EAGLE-MARS DATASET SCANNER")
    print("=" * 70)

    print(f"Dataset root : {root}")
    print(f"Database     : {db_path}")

    print()
    print("Scanning for video files...")
    print("The source dataset will NOT be modified.")
    print()

    connection = create_database(db_path)
    cursor = connection.cursor()

    videos = list(find_videos(root))

    total = len(videos)

    print(f"Videos discovered: {total}")
    print()

    total_size = 0
    total_duration = 0.0

    readable_count = 0
    unreadable_count = 0

    codec_counts = {}
    resolution_counts = {}
    fps_counts = {}

    for index, video_path in enumerate(videos, start=1):

        print(
            f"[{index}/{total}] "
            f"{video_path.name}",
            flush=True,
        )

        try:
            stat = video_path.stat()

            size_bytes = stat.st_size

            modified_time = datetime.fromtimestamp(
                stat.st_mtime
            ).isoformat()

        except OSError as exc:

            print(f"  File error: {exc}")

            size_bytes = None
            modified_time = None

        metadata = run_ffprobe(video_path)

        readable = metadata["readable"]

        if readable:
            readable_count += 1

            duration = metadata["duration"] or 0
            total_duration += duration

            codec = metadata["codec"]

            if codec:
                codec_counts[codec] = (
                    codec_counts.get(codec, 0) + 1
                )

            resolution = (
                f"{metadata['width']}x"
                f"{metadata['height']}"
            )

            resolution_counts[resolution] = (
                resolution_counts.get(resolution, 0) + 1
            )

            fps = metadata["fps"]

            if fps is not None:

                fps_key = round(fps, 2)

                fps_counts[fps_key] = (
                    fps_counts.get(fps_key, 0) + 1
                )

        else:
            unreadable_count += 1

            duration = None

        if size_bytes:
            total_size += size_bytes

        relative_path = str(
            video_path.relative_to(root)
        )

        source_folder = str(
            video_path.parent.relative_to(root)
        )

        cursor.execute(
            """
            INSERT OR REPLACE INTO videos (
                full_path,
                filename,
                extension,
                relative_path,
                source_folder,
                size_bytes,
                modified_time,
                duration_seconds,
                fps,
                frame_count,
                width,
                height,
                codec,
                container,
                readable,
                scan_error,
                scanned_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(video_path),
                video_path.name,
                video_path.suffix.lower(),
                relative_path,
                source_folder,
                size_bytes,
                modified_time,
                metadata.get("duration"),
                metadata.get("fps"),
                metadata.get("frame_count"),
                metadata.get("width"),
                metadata.get("height"),
                metadata.get("codec"),
                metadata.get("container"),
                1 if readable else 0,
                metadata.get("error"),
                datetime.now().isoformat(),
            ),
        )

        connection.commit()

    connection.close()

    print()
    print("=" * 70)
    print("SCAN COMPLETE")
    print("=" * 70)

    print(f"Videos found       : {total}")
    print(f"Readable           : {readable_count}")
    print(f"Unreadable         : {unreadable_count}")

    print(
        f"Total size         : "
        f"{total_size / (1024 ** 3):.2f} GB"
    )

    print(
        f"Total duration     : "
        f"{format_hours(total_duration):.2f} hours"
    )

    print()
    print("CODECS")
    print("-" * 70)

    for codec, count in sorted(
        codec_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(f"{codec:15} : {count}")

    print()
    print("RESOLUTIONS")
    print("-" * 70)

    for resolution, count in sorted(
        resolution_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(f"{resolution:15} : {count}")

    print()
    print("FRAME RATES")
    print("-" * 70)

    for fps, count in sorted(
        fps_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(f"{fps:15} : {count}")

    print()
    print(f"Database saved to:")
    print(db_path)

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()