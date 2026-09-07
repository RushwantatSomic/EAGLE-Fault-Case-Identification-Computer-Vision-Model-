import os
import cv2
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from collections import defaultdict


# ============================================================
# EAGLE-MARS EXPERIMENT 4 DATASET BUILDER
# ============================================================

XML_PATH = r"Annotations\annotations2.xml"

VIDEO_DIR = r"data\annotated_videos"

OUTPUT_DIR = r"outputs\dataset_exp4"

SEGMENTS_CSV = os.path.join(
    OUTPUT_DIR,
    "segments.csv"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# NEW CVAT TASK -> VIDEO MAPPING
# ============================================================

TASK_TO_VIDEO = {

    "2531473": "Carton1.mp4",
    "2531477": "Carton2.mp4",
    "2531479": "Carton3.mp4",

    "2531523": "Cover1.mp4",
    "2531524": "Cover2.mp4",
    "2531525": "Cover3.mp4",

    "2531526": "LamChain.mp4",
    "2531527": "LamChain2.mp4",
    "2531528": "LamChain3.mp4",

    "2531532": "Mag1.mp4",
    "2531538": "Mag2.mp4",
    "2531539": "Mag3.mp4",

    "2531543": "Table1.mp4",
    "2531545": "Table2.mp4",
    "2531547": "Table3.mp4",
}


# ============================================================
# LOAD XML
# ============================================================

print("=" * 100)
print("EAGLE-MARS EXPERIMENT 4 DATASET BUILDER")
print("=" * 100)

root = ET.parse(XML_PATH).getroot()

tracks = root.findall(".//track")

print()
print(f"XML tracks: {len(tracks)}")
print(f"New tasks:  {len(TASK_TO_VIDEO)}")


# ============================================================
# GROUP TRACKS BY TASK
# ============================================================

task_tracks = defaultdict(list)

for track in tracks:

    task_id = track.attrib.get(
        "task_id"
    )

    if task_id in TASK_TO_VIDEO:

        task_tracks[task_id].append(
            track
        )


# ============================================================
# VERIFY VIDEOS
# ============================================================

print()
print("VIDEO VERIFICATION")
print("-" * 100)

video_info = {}

for task_id, video_name in TASK_TO_VIDEO.items():

    path = os.path.join(
        VIDEO_DIR,
        video_name
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Missing video: {path}"
        )

    cap = cv2.VideoCapture(path)

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open: {path}"
        )

    frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    cap.release()

    video_info[video_name] = {
        "frames": frames,
        "fps": fps,
        "width": width,
        "height": height,
    }

    print(
        f"{video_name:20s} "
        f"{frames:4d} frames | "
        f"{fps:5.2f} FPS | "
        f"{width}x{height} | "
        f"task={task_id} | "
        f"tracks={len(task_tracks[task_id])}"
    )


# ============================================================
# EXTRACT FRAME LABELS
# ============================================================

def collect_frame_labels(task_id):

    """
    Convert CVAT temporal annotations into frame-level states.

    Normal       -> NORMAL
    Fault Start  -> FAULT
    Fault End    -> NORMAL
    """

    frame_labels = defaultdict(list)

    for track in task_tracks[task_id]:

        label = track.attrib.get(
            "label",
            ""
        )

        if label not in (
            "Normal",
            "Fault Start",
            "Fault End",
        ):

            continue

        for box in track.findall("./box"):

            try:

                frame = int(
                    box.attrib["frame"]
                )

            except (
                KeyError,
                ValueError,
            ):

                continue

            if label == "Normal":

                state = "NORMAL"

            elif label == "Fault Start":

                state = "FAULT"

            elif label == "Fault End":

                state = "NORMAL"

            else:

                continue

            frame_labels[
                frame
            ].append(state)

    # --------------------------------------------------------
    # Resolve multiple annotations on the same frame
    #
    # FAULT takes priority because we do not want to
    # accidentally train a fault frame as NORMAL.
    # --------------------------------------------------------

    resolved = {}

    for frame, labels in frame_labels.items():

        if "FAULT" in labels:

            resolved[frame] = "FAULT"

        elif "NORMAL" in labels:

            resolved[frame] = "NORMAL"

    return resolved


# ============================================================
# BUILD CONTIGUOUS SEGMENTS
# ============================================================

def build_segments(frame_labels):

    if not frame_labels:

        return []

    frames = sorted(
        frame_labels.keys()
    )

    segments = []

    start = frames[0]

    previous = frames[0]

    current_label = frame_labels[
        frames[0]
    ]

    for frame in frames[1:]:

        label = frame_labels[
            frame
        ]

        contiguous = (
            frame == previous + 1
        )

        if (
            not contiguous
            or label != current_label
        ):

            segments.append({
                "start_frame": start,
                "end_frame": previous,
                "label": current_label,
            })

            start = frame
            current_label = label

        previous = frame

    segments.append({
        "start_frame": start,
        "end_frame": previous,
        "label": current_label,
    })

    return segments


# ============================================================
# PROCESS 15 VIDEOS
# ============================================================

rows = []

print()
print("BUILDING TEMPORAL SEGMENTS")
print("-" * 100)

for task_id, video_name in TASK_TO_VIDEO.items():

    labels = collect_frame_labels(
        task_id
    )

    segments = build_segments(
        labels
    )

    print()
    print(
        f"{video_name}"
    )

    print(
        f"  Annotated frames: "
        f"{len(labels)}"
    )

    print(
        f"  Segments: "
        f"{len(segments)}"
    )

    for segment in segments:

        rows.append({
            "video": video_name,
            "task_id": task_id,
            "label": segment["label"],
            "start_frame": segment["start_frame"],
            "end_frame": segment["end_frame"],
            "frames": (
                segment["end_frame"]
                - segment["start_frame"]
                + 1
            ),
        })

        print(
            f"  "
            f"{segment['label']:6s} "
            f"{segment['start_frame']:4d}"
            f" -> "
            f"{segment['end_frame']:4d} "
            f"("
            f"{segment['end_frame'] - segment['start_frame'] + 1}"
            f" frames)"
        )


# ============================================================
# SAVE
# ============================================================

df = pd.DataFrame(
    rows
)

df.to_csv(
    SEGMENTS_CSV,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 100)
print("EXPERIMENT 4 DATASET COMPLETE")
print("=" * 100)

print()
print(
    f"Videos   : "
    f"{df.video.nunique()}"
)

print(
    f"Segments : "
    f"{len(df)}"
)

print()

if len(df):

    print(
        df.groupby(
            "label"
        ).size()
    )

print()
print(
    f"Saved: {SEGMENTS_CSV}"
)

print("=" * 100)