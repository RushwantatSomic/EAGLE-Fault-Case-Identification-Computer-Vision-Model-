import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

XML_PATH = Path("Annotations/cvat_export.xml")
VIDEO_DIR = Path("data/annotated_videos")
OUTPUT_DIR = Path("outputs/dataset")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# CVAT task -> recovered video filename
TASK_VIDEO = {
    "2523742": "Lamella Chain 2.mp4",
    "2523747": "Lamella Chain 3.mp4",
    "2523748": "Lamella Chain 4.mp4",
    "2524033": "Lamella Chain.mp4",

    "2524047": "Magazine.mp4",
    "2524064": "Magazine 2.mp4",
    "2524065": "Magazine 3.mp4",
    "2524068": "Magazine 4.mp4",

    "2524049": "Collecting Table.mp4",
    "2524050": "Collecting Table 2.mp4",
    "2524051": "Collecting Table 3.mp4",
    "2524052": "Collecting Table 4.mp4",
    "2524059": "Collecting Table 5.mp4",

    "2524053": "Cartoning.mp4",
    "2524054": "Cartoning 2.mp4",
    "2524056": "Cartoning 3.mp4",
    "2524057": "Cartoning 4.mp4",

    "2524061": "Covering 2.mp4",
    "2524062": "Covering 3.mp4",
    "2524063": "Covering 4.mp4",
    "2526254": "covering.mp4",
}

TASK_ENV = {
    "2523742": "Lamella Chain",
    "2523747": "Lamella Chain",
    "2523748": "Lamella Chain",
    "2524033": "Lamella Chain",

    "2524047": "Magazine",
    "2524064": "Magazine",
    "2524065": "Magazine",
    "2524068": "Magazine",

    "2524049": "Collecting Table",
    "2524050": "Collecting Table",
    "2524051": "Collecting Table",
    "2524052": "Collecting Table",
    "2524059": "Collecting Table",

    "2524053": "Cartoning",
    "2524054": "Cartoning",
    "2524056": "Cartoning",
    "2524057": "Cartoning",

    "2524061": "Covering",
    "2524062": "Covering",
    "2524063": "Covering",
    "2526254": "Covering",
}

# Labels that represent the machine state.
NORMAL_LABELS = {"Normal"}
FAULT_LABELS = {"Fault Start", "Product on Floor"}

# These are boundary/context labels, not independent classes.
BOUNDARY_LABELS = {"Fault End", "Cause Normal", "Cause Fault"}


def get_track_interval(track):
    """
    Return the first and last annotated CVAT frame for a track.
    """
    boxes = track.findall("./box")

    if not boxes:
        return None

    frames = [int(b.attrib["frame"]) for b in boxes]

    return min(frames), max(frames)


def main():
    print("=" * 100)
    print("EAGLE-MARS TRAINING MANIFEST BUILDER")
    print("=" * 100)

    if not XML_PATH.exists():
        raise FileNotFoundError(XML_PATH)

    root = ET.parse(XML_PATH).getroot()

    tracks = root.findall(".//track")

    print(f"CVAT tracks: {len(tracks)}")
    print(f"Recovered videos: {len(list(VIDEO_DIR.glob('*.mp4')))}")
    print()

    rows = []

    # Group annotations by task.
    task_tracks = defaultdict(list)

    for track in tracks:
        task_id = track.attrib.get("task_id")
        task_tracks[task_id].append(track)

    for task_id, video_name in TASK_VIDEO.items():

        video_path = VIDEO_DIR / video_name

        if not video_path.exists():
            print(f"WARNING: missing video: {video_name}")
            continue

        environment = TASK_ENV[task_id]

        print(f"{task_id}: {video_name}")

        tracks_for_task = task_tracks.get(task_id, [])

        state_intervals = []

        for track in tracks_for_task:

            label = track.attrib.get("label")

            interval = get_track_interval(track)

            if interval is None:
                continue

            start_frame, end_frame = interval

            if label in NORMAL_LABELS:
                state = "NORMAL"

            elif label in FAULT_LABELS:
                state = "FAULT"

            else:
                continue

            state_intervals.append(
                {
                    "label": label,
                    "state": state,
                    "start_frame_original": start_frame,
                    "end_frame_original": end_frame,
                }
            )

        for item in state_intervals:

            rows.append(
                {
                    "task_id": task_id,
                    "video": video_name,
                    "video_path": str(video_path),
                    "environment": environment,
                    "label": item["state"],
                    "source_label": item["label"],
                    "start_frame_original": item["start_frame_original"],
                    "end_frame_original": item["end_frame_original"],
                    # These will be calculated after timing alignment.
                    "start_frame_recovered": "",
                    "end_frame_recovered": "",
                    "start_time_recovered": "",
                    "end_time_recovered": "",
                }
            )

        print(f"  Tracks used: {len(state_intervals)}")

    output = OUTPUT_DIR / "annotation_intervals.csv"

    fields = [
        "task_id",
        "video",
        "video_path",
        "environment",
        "label",
        "source_label",
        "start_frame_original",
        "end_frame_original",
        "start_frame_recovered",
        "end_frame_recovered",
        "start_time_recovered",
        "end_time_recovered",
    ]

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 100)
    print(f"Rows written: {len(rows)}")
    print(f"Output: {output}")
    print("=" * 100)

    print()
    print("IMPORTANT:")
    print("Recovered-frame timing is intentionally NOT filled yet.")
    print("We must align original CVAT frames to the 30-FPS screen recordings before training.")


if __name__ == "__main__":
    main()