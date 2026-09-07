import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
import pandas as pd


# =============================================================================
# EAGLE-MARS
# CVAT -> CONTINUAL LEARNING DATASET BUILDER
#
# Converts CVAT temporal annotations into fixed 16-frame learning windows.
#
# Sources:
#   1. cvat_export.xml
#   2. annotations2.xml
#
# Output:
#   outputs/unified_dataset/cvat_continual/
# =============================================================================


ROOT = Path(__file__).resolve().parents[1]

ORIGINAL_XML = (
    ROOT
    / "Annotations"
    / "cvat_export.xml"
)

NEW_XML = (
    ROOT
    / "Annotations"
    / "annotations2.xml"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "unified_dataset"
    / "cvat_continual"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "cvat_continual_segments.csv"
)


# =============================================================================
# CONFIGURATION
# =============================================================================

WINDOW_SIZE = 16

# Window movement.
#
# 8 means 50% overlap:
#
# 0-15
# 8-23
# 16-31
# ...
#
STRIDE = 8


# =============================================================================
# VIDEO LENGTHS
#
# These are the recovered physical MP4 lengths.
#
# Used to prevent generated windows from extending outside the real video.
# =============================================================================

VIDEO_LENGTHS = {

    "Carton1.mp4": 557,
    "Carton2.mp4": 557,
    "Carton3.mp4": 557,

    "Cartoning.mp4": 622,
    "Cartoning 2.mp4": 628,
    "Cartoning 3.mp4": 640,
    "Cartoning 4.mp4": 642,

    "Collecting Table 2.mp4": 636,
    "Collecting Table 3.mp4": 630,
    "Collecting Table 4.mp4": 788,
    "Collecting Table 5.mp4": 638,
    "Collecting Table.mp4": 635,

    "Cover1.mp4": 557,
    "Cover2.mp4": 557,
    "Cover3.mp4": 557,

    "Covering 2.mp4": 625,
    "Covering 3.mp4": 632,
    "Covering 4.mp4": 647,
    "covering.mp4": 512,

    "LamChain.mp4": 557,
    "LamChain2.mp4": 557,
    "LamChain3.mp4": 557,

    "Lamella Chain 2.mp4": 595,
    "Lamella Chain 3.mp4": 632,
    "Lamella Chain 4.mp4": 641,
    "Lamella Chain.mp4": 644,

    "Mag1.mp4": 371,
    "Mag2.mp4": 371,
    "Mag3.mp4": 371,

    "Magazine 2.mp4": 387,
    "Magazine 3.mp4": 384,
    "Magazine 4.mp4": 371,
    "Magazine.mp4": 423,

    "Table1.mp4": 557,
    "Table2.mp4": 557,
    "Table3.mp4": 557,
}


# =============================================================================
# CVAT GLOBAL START FRAMES
#
# These correspond to the global frame numbering used in the XML exports.
#
# If the CSV already contains a valid offset, that value is preferred.
# =============================================================================

KNOWN_OFFSETS = {

    # Original CVAT
    "Carton1.mp4": 0,
    "Carton2.mp4": 557,
    "Carton3.mp4": 1114,

    "Cartoning.mp4": 1671,
    "Cartoning 2.mp4": 1671,
    "Cartoning 3.mp4": 2228,
    "Cartoning 4.mp4": 2599,

    "Collecting Table 2.mp4": 3713,
    "Collecting Table 3.mp4": 4270,
    "Collecting Table 4.mp4": 4827,
    "Collecting Table 5.mp4": 5384,
    "Collecting Table.mp4": 5941,

    "Covering 2.mp4": 6498,
    "Covering 3.mp4": 7055,
    "Covering 4.mp4": 7612,
    "covering.mp4": 7979,

    "Lamella Chain 2.mp4": 8169,
    "Lamella Chain 3.mp4": 8726,
    "Lamella Chain 4.mp4": 9283,
    "Lamella Chain.mp4": 9654,

    "Magazine 2.mp4": 9654,
    "Magazine 3.mp4": 10025,
    "Magazine 4.mp4": 10437,
    "Magazine.mp4": 10025,

    # annotations2.xml
    "Cover1.mp4": 12624,
    "Cover2.mp4": 13021,
    "Cover3.mp4": 13374,

    "LamChain.mp4": 13737,
    "LamChain2.mp4": 14294,
    "LamChain3.mp4": 14851,

    "Mag1.mp4": 15408,
    "Mag2.mp4": 15965,
    "Mag3.mp4": 16535,

    "Table1.mp4": 17079,
    "Table2.mp4": 17695,
    "Table3.mp4": 18193,
}


# =============================================================================
# LABEL MAPPING
# =============================================================================
#
# We only care about the machine state:
#
# Fault Start / Fault End -> FAULT
# Normal                   -> NORMAL
#
# "Cause Fault", "Cause Normal", and "Product on Floor" are not used as
# separate classes.
# =============================================================================

FAULT_LABELS = {
    "Fault Start",
    "Fault End",
}

NORMAL_LABELS = {
    "Normal",
}


# =============================================================================
# LOAD XML
# =============================================================================

def load_xml(path):

    print(
        f"Loading XML: {path}"
    )

    tree = ET.parse(
        path
    )

    return tree.getroot()


# =============================================================================
# GET VIDEO NAME
# =============================================================================
#
# CVAT exports do not always contain the actual MP4 filename in a convenient
# field. We therefore use task_id -> video mapping based on the track metadata
# and known frame ranges.
#
# The annotations2.xml videos are identifiable directly from the task IDs.
# =============================================================================

ORIGINAL_TASK_TO_VIDEO = {

    "2523742": "Carton1.mp4",
    "2523747": "Carton2.mp4",
    "2523748": "Carton3.mp4",

    "2524033": "Cartoning.mp4",

    "2524047": "Cartoning 2.mp4",
    "2524049": "Covering 2.mp4",
    "2524050": "Covering 3.mp4",
    "2524051": "Covering 4.mp4",

    "2524052": "Collecting Table 2.mp4",
    "2524053": "Collecting Table 3.mp4",
    "2524054": "Collecting Table 4.mp4",
    "2524056": "Collecting Table 5.mp4",
    "2524057": "Collecting Table.mp4",

    "2524059": "Lamella Chain 2.mp4",
    "2524061": "Lamella Chain 3.mp4",
    "2524062": "Lamella Chain 4.mp4",
    "2524063": "Lamella Chain.mp4",

    "2524064": "Magazine 2.mp4",
    "2524065": "Magazine 3.mp4",
    "2524068": "Magazine 4.mp4",

    "2526254": "Magazine.mp4",
}


NEW_TASK_TO_VIDEO = {

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


# =============================================================================
# EXTRACT TASKS
# =============================================================================

def extract_tasks(
    root,
    source,
    task_to_video,
):

    tasks = defaultdict(list)

    for track in root.findall(
        ".//track"
    ):

        task_id = track.attrib.get(
            "task_id",
            "",
        )

        label = track.attrib.get(
            "label",
            "",
        )

        video = task_to_video.get(
            task_id
        )

        if video is None:
            continue

        boxes = track.findall(
            "./box"
        )

        for box in boxes:

            frame = box.attrib.get(
                "frame"
            )

            if frame is None:
                continue

            try:
                frame = int(frame)
            except ValueError:
                continue

            tasks[
                (
                    task_id,
                    video,
                )
            ].append(
                (
                    frame,
                    label,
                )
            )

    return tasks


# =============================================================================
# GLOBAL -> LOCAL
# =============================================================================

def get_local_frame(
    global_frame,
    video,
):

    offset = KNOWN_OFFSETS.get(
        video
    )

    if offset is None:
        return None

    return (
        global_frame
        - offset
    )


# =============================================================================
# BUILD FRAME LABEL MAP
# =============================================================================

def build_frame_labels(
    task_data,
    video,
):

    labels = {}

    for (
        task_id,
        video_name,
    ), entries in task_data.items():

        if video_name != video:
            continue

        for (
            global_frame,
            label,
        ) in entries:

            local_frame = get_local_frame(
                global_frame,
                video,
            )

            if local_frame is None:
                continue

            # -------------------------------------------------------------
            # Only machine-state labels
            # -------------------------------------------------------------

            if label in FAULT_LABELS:

                labels[
                    local_frame
                ] = "FAULT"

            elif label in NORMAL_LABELS:

                # Do not overwrite an existing FAULT.
                if (
                    local_frame
                    not in labels
                ):

                    labels[
                        local_frame
                    ] = "NORMAL"

    return labels


# =============================================================================
# CONTIGUOUS SEGMENTS
# =============================================================================

def build_segments(
    frame_labels,
):

    if not frame_labels:
        return []

    frames = sorted(
        frame_labels
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

        if (
            frame != previous + 1
            or label != current_label
        ):

            segments.append(
                {
                    "start": start,
                    "end": previous,
                    "label": current_label,
                }
            )

            start = frame
            current_label = label

        previous = frame

    segments.append(
        {
            "start": start,
            "end": previous,
            "label": current_label,
        }
    )

    return segments


# =============================================================================
# GENERATE FIXED WINDOWS
# =============================================================================

def generate_windows(
    start,
    end,
    video_length,
):

    # Clip to physical video
    start = max(
        0,
        start,
    )

    end = min(
        video_length - 1,
        end,
    )

    if end < start:
        return []

    length = (
        end
        - start
        + 1
    )

    if length < WINDOW_SIZE:
        return []

    windows = []

    current = start

    while (
        current + WINDOW_SIZE - 1
        <= end
    ):

        windows.append(
            (
                current,
                current + WINDOW_SIZE - 1,
            )
        )

        current += STRIDE

    # Make sure the tail of a segment is not discarded.
    final_start = (
        end
        - WINDOW_SIZE
        + 1
    )

    if final_start >= start:

        final_window = (
            final_start,
            end,
        )

        if final_window not in windows:

            windows.append(
                final_window
            )

    return windows


# =============================================================================
# PROCESS ONE SOURCE
# =============================================================================

def process_source(
    xml_path,
    source_name,
    task_to_video,
):

    root = load_xml(
        xml_path
    )

    task_data = extract_tasks(
        root,
        source_name,
        task_to_video,
    )

    print()
    print(
        f"{source_name.upper()} TASKS"
    )
    print("-" * 90)

    videos = sorted(
        set(
            video
            for (
                _,
                video,
            ) in task_data
        )
    )

    print(
        f"Videos: {len(videos)}"
    )

    output = []

    for video in videos:

        if video not in VIDEO_LENGTHS:

            print(
                f"WARNING: no video length "
                f"for {video}"
            )

            continue

        frame_labels = build_frame_labels(
            task_data,
            video,
        )

        segments = build_segments(
            frame_labels
        )

        print()
        print(
            f"{video}"
        )

        print(
            f"  labelled frames : "
            f"{len(frame_labels)}"
        )

        print(
            f"  segments        : "
            f"{len(segments)}"
        )

        video_length = VIDEO_LENGTHS[
            video
        ]

        video_windows = 0

        for segment in segments:

            windows = generate_windows(
                segment["start"],
                segment["end"],
                video_length,
            )

            for (
                clip_start,
                clip_end,
            ) in windows:

                output.append(
                    {
                        "video": video,
                        "source": (
                            f"{source_name}_continual"
                        ),
                        "label": segment[
                            "label"
                        ],

                        "start_frame_local":
                            clip_start,

                        "end_frame_local":
                            clip_end,

                        "frames":
                            WINDOW_SIZE,

                        "fps_recovered":
                            None,

                        "task_id":
                            "",

                        "split":
                            None,

                        "original_segment_start":
                            segment["start"],

                        "original_segment_end":
                            segment["end"],
                    }
                )

                video_windows += 1

        print(
            f"  continual windows: "
            f"{video_windows}"
        )

    return output


# =============================================================================
# ASSIGN VIDEO-LEVEL SPLITS
# =============================================================================
#
# IMPORTANT:
# We use the same video-level split already used by the unified dataset.
#
# This prevents frames from the same video appearing in both train and test.
# =============================================================================

def load_existing_splits():

    unified_csv = (
        ROOT
        / "outputs"
        / "unified_dataset"
        / "segments_unified.csv"
    )

    if not unified_csv.exists():

        return {}

    df = pd.read_csv(
        unified_csv
    )

    result = {}

    for _, row in df.iterrows():

        video = row[
            "video"
        ]

        split = row[
            "split"
        ]

        if video not in result:

            result[
                video
            ] = split

    return result


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 90)
    print(
        "EAGLE-MARS CVAT CONTINUAL "
        "LEARNING DATASET BUILDER"
    )
    print("=" * 90)

    print()
    print("CONFIGURATION")
    print("-" * 90)

    print(
        f"Original XML : {ORIGINAL_XML}"
    )

    print(
        f"New XML      : {NEW_XML}"
    )

    print(
        f"Window       : {WINDOW_SIZE} frames"
    )

    print(
        f"Stride       : {STRIDE} frames"
    )

    print(
        f"Output       : {OUTPUT_DIR}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Process original CVAT
    # -------------------------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "PROCESSING ORIGINAL CVAT"
    )
    print("=" * 90)

    original = process_source(
        ORIGINAL_XML,
        "cvat_original",
        ORIGINAL_TASK_TO_VIDEO,
    )

    print()
    print(
        f"Original continual windows: "
        f"{len(original)}"
    )

    # -------------------------------------------------------------------------
    # Process annotations2
    # -------------------------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "PROCESSING annotations2.xml"
    )
    print("=" * 90)

    new = process_source(
        NEW_XML,
        "cvat_annotations2",
        NEW_TASK_TO_VIDEO,
    )

    print()
    print(
        f"New CVAT continual windows: "
        f"{len(new)}"
    )

    # -------------------------------------------------------------------------
    # Combine
    # -------------------------------------------------------------------------

    rows = (
        original
        + new
    )

    df = pd.DataFrame(
        rows
    )

    if df.empty:

        raise RuntimeError(
            "No CVAT continual windows "
            "were generated."
        )

    # -------------------------------------------------------------------------
    # Existing video-level split
    # -------------------------------------------------------------------------

    splits = load_existing_splits()

    print()
    print("=" * 90)
    print(
        "ASSIGNING VIDEO-LEVEL SPLITS"
    )
    print("=" * 90)

    missing_split_videos = []

    for video in df[
        "video"
    ].unique():

        if video in splits:

            df.loc[
                df["video"] == video,
                "split",
            ] = splits[
                video
            ]

        else:

            missing_split_videos.append(
                video
            )

    if missing_split_videos:

        print()
        print(
            "WARNING: videos without "
            "existing split:"
        )

        for video in missing_split_videos:

            print(
                f"  {video}"
            )

        # Put unknown videos in train.
        for video in missing_split_videos:

            df.loc[
                df["video"] == video,
                "split",
            ] = "train"

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # =========================================================================
    # STATISTICS
    # =========================================================================

    print()
    print("=" * 90)
    print(
        "CVAT CONTINUAL DATASET COMPLETE"
    )
    print("=" * 90)

    print()
    print("TOTAL WINDOWS")
    print("-" * 90)

    print(
        len(df)
    )

    print()
    print("BY SOURCE")
    print("-" * 90)

    print(
        df.groupby(
            "source"
        ).size()
    )

    print()
    print("SOURCE × LABEL")
    print("-" * 90)

    print(
        df.groupby(
            [
                "source",
                "label",
            ]
        ).size()
    )

    print()
    print("SPLIT × LABEL")
    print("-" * 90)

    print(
        df.groupby(
            [
                "split",
                "label",
            ]
        ).size()
    )

    print()
    print("VIDEOS")
    print("-" * 90)

    print(
        f"Total videos : "
        f"{df['video'].nunique()}"
    )

    print(
        f"Train videos : "
        f"{df.loc[df.split == 'train', 'video'].nunique()}"
    )

    print(
        f"Val videos   : "
        f"{df.loc[df.split == 'val', 'video'].nunique()}"
    )

    print(
        f"Test videos  : "
        f"{df.loc[df.split == 'test', 'video'].nunique()}"
    )

    print()
    print("OUTPUT")
    print("-" * 90)

    print(
        f"Saved: {OUTPUT_CSV}"
    )

    print()
    print("=" * 90)


if __name__ == "__main__":
    main()