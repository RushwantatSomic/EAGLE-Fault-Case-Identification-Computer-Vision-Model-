import os
import xml.etree.ElementTree as ET
from collections import defaultdict

import cv2
import pandas as pd


# ============================================================
# EAGLE-MARS EXPERIMENT 4
# TEMPORAL ANNOTATION CLEANER
# ============================================================

XML_PATH = r"Annotations\annotations2.xml"
VIDEO_DIR = r"data\annotated_videos"
OUTPUT_DIR = r"outputs\dataset_exp4"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "segments_exp4.csv",
)

# Minimum useful VideoMAE clip length.
MIN_CLIP_FRAMES = 16

# Small gaps are bridged when they occur between fault
# annotations. This handles sparse CVAT tracking.
MAX_FAULT_GAP = 12

# Very short isolated fault regions are retained only when
# they are close to another fault region.
MIN_FAULT_FRAMES = 4


# ============================================================
# TASK -> VIDEO
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
# HELPERS
# ============================================================

def get_video_info(path):
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    cap.release()

    return frames, fps


def build_frame_annotations(task_tracks):
    """
    Extract frame-level annotations.

    Normal      -> NORMAL
    Fault Start -> FAULT
    Fault End   -> NORMAL

    FAULT has priority if multiple annotations exist on
    the same frame.
    """

    frame_states = defaultdict(set)

    for track in task_tracks:

        label = track.attrib.get("label", "")

        if label not in {
            "Normal",
            "Fault Start",
            "Fault End",
        }:
            continue

        for box in track.findall("./box"):

            try:
                frame = int(box.attrib["frame"])
            except (KeyError, ValueError):
                continue

            if label == "Normal":
                frame_states[frame].add("NORMAL")

            elif label == "Fault Start":
                frame_states[frame].add("FAULT")

            elif label == "Fault End":
                frame_states[frame].add("NORMAL")

    resolved = {}

    for frame, states in frame_states.items():

        if "FAULT" in states:
            resolved[frame] = "FAULT"

        elif "NORMAL" in states:
            resolved[frame] = "NORMAL"

    return resolved


def convert_to_local_frames(frames, video_frame_count):
    """
    The XML frame numbers are global across the CVAT export.

    Convert them to local frame indices by subtracting the
    first annotated frame.
    """

    if not frames:
        return {}

    first = min(frames.keys())

    local = {}

    for frame, label in frames.items():

        local_frame = frame - first

        if 0 <= local_frame < video_frame_count:
            local[local_frame] = label

    return local


def fill_fault_gaps(frame_labels):
    """
    Fill short gaps inside a fault event.

    Example:

        FAULT FAULT ? ? FAULT FAULT

    becomes:

        FAULT FAULT FAULT FAULT FAULT FAULT

    only when the gap is <= MAX_FAULT_GAP.
    """

    if not frame_labels:
        return frame_labels

    labels = dict(frame_labels)

    fault_frames = sorted(
        f for f, label in labels.items()
        if label == "FAULT"
    )

    for a, b in zip(fault_frames, fault_frames[1:]):

        gap = b - a - 1

        if 0 < gap <= MAX_FAULT_GAP:

            # Only fill frames that have no explicit NORMAL
            # annotation.
            for frame in range(a + 1, b):

                if frame not in labels:
                    labels[frame] = "FAULT"

    return labels


def build_segments(frame_labels, video_frame_count):
    """
    Convert frame labels into contiguous state segments.

    Unannotated frames are NOT automatically labelled NORMAL.

    This is important because we don't want to manufacture
    training labels from missing CVAT annotations.
    """

    if not frame_labels:
        return []

    frames = sorted(frame_labels)

    raw_segments = []

    start = frames[0]
    previous = frames[0]
    current = frame_labels[start]

    for frame in frames[1:]:

        label = frame_labels[frame]

        if (
            frame != previous + 1
            or label != current
        ):

            raw_segments.append(
                (
                    start,
                    previous,
                    current,
                )
            )

            start = frame
            current = label

        previous = frame

    raw_segments.append(
        (
            start,
            previous,
            current,
        )
    )

    # --------------------------------------------------------
    # Remove tiny isolated fault fragments.
    #
    # If a tiny fault fragment is directly adjacent to another
    # fault segment, it will already have been merged by the
    # gap-filling stage.
    # --------------------------------------------------------

    cleaned = []

    for start, end, label in raw_segments:

        length = end - start + 1

        if (
            label == "FAULT"
            and length < MIN_FAULT_FRAMES
        ):
            continue

        if (
            label == "NORMAL"
            and length < MIN_CLIP_FRAMES
        ):
            continue

        cleaned.append(
            (
                start,
                end,
                label,
            )
        )

    # --------------------------------------------------------
    # Merge neighbouring same-state segments.
    # --------------------------------------------------------

    merged = []

    for start, end, label in cleaned:

        if merged:

            p_start, p_end, p_label = merged[-1]

            if (
                p_label == label
                and start <= p_end + 1
            ):

                merged[-1] = (
                    p_start,
                    max(p_end, end),
                    label,
                )

                continue

        merged.append(
            (
                start,
                end,
                label,
            )
        )

    return merged


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("EAGLE-MARS EXPERIMENT 4 TEMPORAL CLEANER")
    print("=" * 100)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load XML
    # --------------------------------------------------------

    root = ET.parse(XML_PATH).getroot()

    tracks = root.findall(".//track")

    print()
    print(f"XML tracks : {len(tracks)}")
    print(f"New videos : {len(TASK_TO_VIDEO)}")

    # --------------------------------------------------------
    # Group tracks
    # --------------------------------------------------------

    task_tracks = defaultdict(list)

    for track in tracks:

        task_id = track.attrib.get(
            "task_id"
        )

        if task_id in TASK_TO_VIDEO:
            task_tracks[task_id].append(
                track
            )

    rows = []

    print()
    print("CLEANING TEMPORAL ANNOTATIONS")
    print("-" * 100)

    # --------------------------------------------------------
    # Process each video
    # --------------------------------------------------------

    for task_id, video_name in TASK_TO_VIDEO.items():

        video_path = os.path.join(
            VIDEO_DIR,
            video_name,
        )

        if not os.path.exists(video_path):
            raise FileNotFoundError(
                f"Missing video: {video_path}"
            )

        video_frames, fps = get_video_info(
            video_path
        )

        absolute_labels = build_frame_annotations(
            task_tracks[task_id]
        )

        local_labels = convert_to_local_frames(
            absolute_labels,
            video_frames,
        )

        local_labels = fill_fault_gaps(
            local_labels
        )

        segments = build_segments(
            local_labels,
            video_frames,
        )

        print()
        print(
            f"{video_name}"
        )

        print(
            f"  Video frames : {video_frames}"
        )

        print(
            f"  FPS          : {fps:.2f}"
        )

        print(
            f"  Annotated    : {len(absolute_labels)}"
        )

        print(
            f"  Clean labels : {len(local_labels)}"
        )

        print(
            f"  Segments     : {len(segments)}"
        )

        for start, end, label in segments:

            length = end - start + 1

            print(
                f"    {label:6s} "
                f"{start:4d} -> "
                f"{end:4d} "
                f"({length:4d} frames)"
            )

            rows.append({
                "video": video_name,
                "task_id": task_id,
                "label": label,
                "start_frame_local": start,
                "end_frame_local": end,
                "frames": length,
                "fps": fps,
            })

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df = pd.DataFrame(rows)

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("EXP4 TEMPORAL CLEANING COMPLETE")
    print("=" * 100)

    print()
    print(
        f"Videos   : {df.video.nunique()}"
    )

    print(
        f"Segments : {len(df)}"
    )

    if len(df):

        print()
        print("LABEL COUNTS")
        print("-" * 100)

        print(
            df.groupby("label").size()
        )

        print()
        print("SEGMENT LENGTHS")
        print("-" * 100)

        print(
            df.groupby("label")["frames"]
            .agg(
                [
                    "count",
                    "min",
                    "median",
                    "max",
                ]
            )
        )

    print()
    print(
        f"Saved: {OUTPUT_FILE}"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()