import json
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

ORIGINAL_XML = ROOT / "Annotations" / "cvat_export.xml"
NEW_XML = ROOT / "Annotations" / "annotations2.xml"
CONTINUAL_JSON = (
    ROOT
    / "outputs"
    / "continual_learning"
    / "continual_annotations.json"
)

OUTPUT_DIR = ROOT / "outputs" / "unified_dataset"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "segments_unified.csv"


# =============================================================================
# ORIGINAL CVAT TASK -> VIDEO
# =============================================================================

ORIGINAL_TASKS = [
    ("2523742", "Carton1.mp4"),
    ("2523747", "Carton2.mp4"),
    ("2523748", "Carton3.mp4"),
    ("2524033", "Cartoning 2.mp4"),
    ("2524047", "Cartoning 3.mp4"),
    ("2524049", "Cartoning 4.mp4"),
    ("2524050", "Cartoning.mp4"),
    ("2524051", "Collecting Table 2.mp4"),
    ("2524052", "Collecting Table 3.mp4"),
    ("2524053", "Collecting Table 4.mp4"),
    ("2524054", "Collecting Table 5.mp4"),
    ("2524056", "Collecting Table.mp4"),
    ("2524057", "Covering 2.mp4"),
    ("2524059", "Covering 3.mp4"),
    ("2524061", "Covering 4.mp4"),
    ("2524062", "Lamella Chain 2.mp4"),
    ("2524063", "Lamella Chain 3.mp4"),
    ("2524064", "Lamella Chain 4.mp4"),
    ("2524065", "Magazine 2.mp4"),
    ("2524068", "Magazine 3.mp4"),
    ("2526254", "Magazine 4.mp4"),
]


# =============================================================================
# NEW CVAT TASK -> VIDEO
# =============================================================================

NEW_TASKS = [
    ("2531473", "Carton1.mp4"),
    ("2531477", "Carton2.mp4"),
    ("2531479", "Carton3.mp4"),
    ("2531523", "Cover1.mp4"),
    ("2531524", "Cover2.mp4"),
    ("2531525", "Cover3.mp4"),
    ("2531526", "LamChain.mp4"),
    ("2531527", "LamChain2.mp4"),
    ("2531528", "LamChain3.mp4"),
    ("2531532", "Mag1.mp4"),
    ("2531538", "Mag2.mp4"),
    ("2531539", "Mag3.mp4"),
    ("2531543", "Table1.mp4"),
    ("2531545", "Table2.mp4"),
    ("2531547", "Table3.mp4"),
]


# =============================================================================
# FIXED SPLITS FOR CVAT VIDEOS
# =============================================================================

TRAIN_VIDEOS = {
    "Carton1.mp4",
    "Carton2.mp4",
    "Cover1.mp4",
    "Cover2.mp4",
    "LamChain.mp4",
    "LamChain2.mp4",
    "Mag1.mp4",
    "Table1.mp4",

    "Cartoning 3.mp4",
    "Collecting Table 2.mp4",
    "Collecting Table 3.mp4",
    "Collecting Table 4.mp4",
    "Covering 2.mp4",
    "Covering 3.mp4",
    "Lamella Chain 2.mp4",
    "Lamella Chain 3.mp4",
    "Magazine 2.mp4",
    "Magazine 3.mp4",
}

VAL_VIDEOS = {
    "Carton3.mp4",
    "Cover3.mp4",
    "LamChain3.mp4",
    "Mag2.mp4",

    "Cartoning 4.mp4",
    "Collecting Table 5.mp4",
    "Collecting Table.mp4",
    "Covering 4.mp4",
    "Magazine 4.mp4",
}

TEST_VIDEOS = {
    "Mag3.mp4",
    "Table2.mp4",
    "Table3.mp4",

    "Cartoning 2.mp4",
    "Cartoning.mp4",
    "covering.mp4",
}


# =============================================================================
# SPLIT ASSIGNMENT
# =============================================================================

def assign_split(video_name):

    if video_name in TEST_VIDEOS:
        return "test"

    if video_name in VAL_VIDEOS:
        return "val"

    if video_name in TRAIN_VIDEOS:
        return "train"

    # -------------------------------------------------------------------------
    # Continual-learning videos.
    #
    # Deterministic assignment based on video filename.
    # -------------------------------------------------------------------------

    digest = hashlib.md5(
        video_name.encode("utf-8")
    ).hexdigest()

    value = int(digest[:8], 16) / 0xFFFFFFFF

    if value < 0.15:
        return "test"

    if value < 0.30:
        return "val"

    return "train"


# =============================================================================
# XML LOADING
# =============================================================================

def load_xml(path):

    print(f"Loading XML: {path}")

    return ET.parse(path).getroot()


# =============================================================================
# GROUP XML TRACKS BY TASK
# =============================================================================

def get_task_tracks(root):

    result = defaultdict(list)

    for track in root.findall(".//track"):

        task_id = track.attrib.get("task_id")

        if task_id:
            result[task_id].append(track)

    return result


# =============================================================================
# CVAT TEMPORAL SEGMENT RECONSTRUCTION
# =============================================================================

def reconstruct_cvat_segments(
    root,
    task_mapping,
    source_name,
):

    task_tracks = get_task_tracks(root)

    segments = []

    for task_id, video_name in task_mapping:

        tracks = task_tracks.get(task_id, [])

        if not tracks:
            print(
                f"WARNING: task {task_id} "
                f"not found in XML"
            )
            continue

        frame_labels = defaultdict(set)

        for track in tracks:

            label = track.attrib.get(
                "label",
                "",
            ).strip()

            for box in track.findall("./box"):

                try:
                    frame = int(
                        box.attrib["frame"]
                    )
                except Exception:
                    continue

                frame_labels[frame].add(label)

        if not frame_labels:
            continue

        frames = sorted(frame_labels)

        current_state = None
        segment_start = None
        previous_frame = None

        for frame in frames:

            labels = frame_labels[frame]

            # -------------------------------------------------------------
            # Determine state
            # -------------------------------------------------------------

            if "Fault Start" in labels:
                state = "FAULT"

            elif "Fault End" in labels:
                state = "NORMAL"

            elif "Normal" in labels:
                state = "NORMAL"

            else:
                state = current_state

            # -------------------------------------------------------------
            # First frame
            # -------------------------------------------------------------

            if current_state is None:

                current_state = state
                segment_start = frame
                previous_frame = frame

                continue

            # -------------------------------------------------------------
            # Annotation gap
            # -------------------------------------------------------------

            if frame != previous_frame + 1:

                if (
                    segment_start is not None
                    and current_state in {
                        "NORMAL",
                        "FAULT",
                    }
                ):

                    segments.append({
                        "video": video_name,
                        "task_id": task_id,
                        "label": current_state,
                        "start_frame_global": segment_start,
                        "end_frame_global": previous_frame,
                        "frames": (
                            previous_frame
                            - segment_start
                            + 1
                        ),
                        "source": source_name,
                    })

                segment_start = frame
                current_state = state

            # -------------------------------------------------------------
            # State change
            # -------------------------------------------------------------

            elif state != current_state:

                if (
                    segment_start is not None
                    and current_state in {
                        "NORMAL",
                        "FAULT",
                    }
                ):

                    segments.append({
                        "video": video_name,
                        "task_id": task_id,
                        "label": current_state,
                        "start_frame_global": segment_start,
                        "end_frame_global": previous_frame,
                        "frames": (
                            previous_frame
                            - segment_start
                            + 1
                        ),
                        "source": source_name,
                    })

                segment_start = frame
                current_state = state

            previous_frame = frame

        # -------------------------------------------------------------
        # Final segment
        # -------------------------------------------------------------

        if (
            segment_start is not None
            and previous_frame is not None
            and current_state in {
                "NORMAL",
                "FAULT",
            }
        ):

            segments.append({
                "video": video_name,
                "task_id": task_id,
                "label": current_state,
                "start_frame_global": segment_start,
                "end_frame_global": previous_frame,
                "frames": (
                    previous_frame
                    - segment_start
                    + 1
                ),
                "source": source_name,
            })

    return segments


# =============================================================================
# CONTINUAL LEARNING
#
# IMPORTANT:
#
# We intentionally DO NOT use:
#
#     fault_intervals
#
# because the previous unified builder showed that those records are not
# sufficient for reconstructing the labels.
#
# Instead we use:
#
#     windows[].label
#
# label 0 = NORMAL
# label 1 = FAULT
#
# The overlapping 16-frame windows are converted into per-frame votes.
# Then the votes are converted into continuous NORMAL/FAULT segments.
# =============================================================================

def load_continual_segments():

    print(
        f"Loading continual annotations: "
        f"{CONTINUAL_JSON}"
    )

    with open(
        CONTINUAL_JSON,
        "r",
        encoding="utf-8",
    ) as f:

        records = json.load(f)

    if not isinstance(records, list):

        raise RuntimeError(
            "continual_annotations.json "
            "must contain a list."
        )

    segments = []

    total_videos = 0
    total_windows = 0
    total_normal_windows = 0
    total_fault_windows = 0

    for record in records:

        video_path = record.get("video")

        if not video_path:
            continue

        video_name = Path(
            video_path
        ).name

        frame_count = int(
            record.get(
                "frame_count",
                0,
            )
        )

        windows = record.get(
            "windows",
            [],
        )

        if frame_count <= 0:
            continue

        if not windows:
            print(
                f"WARNING: no windows for "
                f"{video_name}"
            )
            continue

        total_videos += 1

        total_windows += len(windows)

        # ---------------------------------------------------------------------
        # Per-frame voting
        #
        # Every 16-frame window votes on each frame it covers.
        #
        # label:
        #   0 = NORMAL
        #   1 = FAULT
        #
        # Fault wins ties.
        # ---------------------------------------------------------------------

        normal_votes = [0] * frame_count
        fault_votes = [0] * frame_count

        for window in windows:

            try:
                start = int(
                    window["start_frame"]
                )

                end = int(
                    window["end_frame"]
                )

                label = int(
                    window["label"]
                )

            except (
                KeyError,
                TypeError,
                ValueError,
            ):
                continue

            start = max(
                0,
                start,
            )

            end = min(
                frame_count - 1,
                end,
            )

            if end < start:
                continue

            if label == 1:

                total_fault_windows += 1

                for frame in range(
                    start,
                    end + 1,
                ):
                    fault_votes[frame] += 1

            else:

                total_normal_windows += 1

                for frame in range(
                    start,
                    end + 1,
                ):
                    normal_votes[frame] += 1

        # ---------------------------------------------------------------------
        # Determine frame labels
        # ---------------------------------------------------------------------

        frame_states = []

        for frame in range(
            frame_count
        ):

            normal = normal_votes[frame]
            fault = fault_votes[frame]

            # No annotation covering this frame.
            if normal == 0 and fault == 0:

                state = None

            elif fault >= normal:

                state = "FAULT"

            else:

                state = "NORMAL"

            frame_states.append(state)

        # ---------------------------------------------------------------------
        # Fill small unlabeled gaps.
        #
        # This prevents isolated gaps between overlapping windows from
        # producing many artificial segments.
        # ---------------------------------------------------------------------

        for i in range(
            1,
            len(frame_states) - 1,
        ):

            if frame_states[i] is None:

                if (
                    frame_states[i - 1]
                    is not None
                    and frame_states[i + 1]
                    is not None
                    and frame_states[i - 1]
                    == frame_states[i + 1]
                ):

                    frame_states[i] = (
                        frame_states[i - 1]
                    )

        # ---------------------------------------------------------------------
        # Build continuous segments
        # ---------------------------------------------------------------------

        current_state = None
        start_frame = None

        for frame, state in enumerate(
            frame_states
        ):

            # -------------------------------------------------------------
            # Ignore uncovered frames
            # -------------------------------------------------------------

            if state is None:

                if (
                    current_state is not None
                    and start_frame is not None
                ):

                    segments.append({
                        "video": video_name,
                        "task_id": "CONTINUAL",
                        "label": current_state,
                        "start_frame_global": start_frame,
                        "end_frame_global": frame - 1,
                        "frames": (
                            frame
                            - start_frame
                        ),
                        "source": "continual",
                    })

                current_state = None
                start_frame = None

                continue

            # -------------------------------------------------------------
            # Start first segment
            # -------------------------------------------------------------

            if current_state is None:

                current_state = state
                start_frame = frame

                continue

            # -------------------------------------------------------------
            # State change
            # -------------------------------------------------------------

            if state != current_state:

                segments.append({
                    "video": video_name,
                    "task_id": "CONTINUAL",
                    "label": current_state,
                    "start_frame_global": start_frame,
                    "end_frame_global": frame - 1,
                    "frames": (
                        frame
                        - start_frame
                    ),
                    "source": "continual",
                })

                current_state = state
                start_frame = frame

        # ---------------------------------------------------------------------
        # Final segment
        # ---------------------------------------------------------------------

        if (
            current_state is not None
            and start_frame is not None
        ):

            segments.append({
                "video": video_name,
                "task_id": "CONTINUAL",
                "label": current_state,
                "start_frame_global": start_frame,
                "end_frame_global": frame_count - 1,
                "frames": (
                    frame_count
                    - start_frame
                ),
                "source": "continual",
            })

    # -------------------------------------------------------------------------
    # Print continual statistics
    # -------------------------------------------------------------------------

    print()
    print(
        "CONTINUAL WINDOW COUNTS"
    )
    print("-" * 90)

    print(
        f"Videos processed       : {total_videos}"
    )

    print(
        f"Windows processed      : {total_windows}"
    )

    print(
        f"NORMAL windows         : "
        f"{total_normal_windows}"
    )

    print(
        f"FAULT windows          : "
        f"{total_fault_windows}"
    )

    print()
    print(
        "CONTINUAL SEGMENTS"
    )
    print("-" * 90)

    continual_df = pd.DataFrame(
        segments
    )

    if not continual_df.empty:

        print(
            continual_df.groupby(
                "label"
            ).size()
        )

        print()

        print(
            "FAULT frames:",
            int(
                continual_df.loc[
                    continual_df["label"]
                    == "FAULT",
                    "frames",
                ].sum()
            ),
        )

        print(
            "NORMAL frames:",
            int(
                continual_df.loc[
                    continual_df["label"]
                    == "NORMAL",
                    "frames",
                ].sum()
            ),
        )

    return segments


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 90)
    print(
        "EAGLE-MARS UNIFIED DATASET BUILDER"
    )
    print("=" * 90)

    # -------------------------------------------------------------------------
    # Verify input files
    # -------------------------------------------------------------------------

    required_files = [
        ORIGINAL_XML,
        NEW_XML,
        CONTINUAL_JSON,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    # -------------------------------------------------------------------------
    # Load XML
    # -------------------------------------------------------------------------

    original_root = load_xml(
        ORIGINAL_XML
    )

    new_root = load_xml(
        NEW_XML
    )

    # -------------------------------------------------------------------------
    # Original CVAT
    # -------------------------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "PROCESSING ORIGINAL CVAT"
    )
    print("=" * 90)

    original_segments = (
        reconstruct_cvat_segments(
            original_root,
            ORIGINAL_TASKS,
            "cvat_original",
        )
    )

    print(
        f"Segments generated: "
        f"{len(original_segments)}"
    )

    # -------------------------------------------------------------------------
    # New CVAT
    # -------------------------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "PROCESSING annotations2.xml"
    )
    print("=" * 90)

    new_segments = (
        reconstruct_cvat_segments(
            new_root,
            NEW_TASKS,
            "cvat_annotations2",
        )
    )

    print(
        f"Segments generated: "
        f"{len(new_segments)}"
    )

    # -------------------------------------------------------------------------
    # Continual learning
    # -------------------------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "PROCESSING 50 CONTINUAL VIDEOS"
    )
    print("=" * 90)

    continual_segments = (
        load_continual_segments()
    )

    print()
    print(
        f"Segments generated: "
        f"{len(continual_segments)}"
    )

    # -------------------------------------------------------------------------
    # Combine
    # -------------------------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "COMBINING DATASETS"
    )
    print("=" * 90)

    print(
        f"Original CVAT segments : "
        f"{len(original_segments)}"
    )

    print(
        f"New CVAT segments      : "
        f"{len(new_segments)}"
    )

    print(
        f"Continual segments     : "
        f"{len(continual_segments)}"
    )

    all_segments = (
        original_segments
        + new_segments
        + continual_segments
    )

    print(
        f"TOTAL                  : "
        f"{len(all_segments)}"
    )

    # -------------------------------------------------------------------------
    # DataFrame
    # -------------------------------------------------------------------------

    df = pd.DataFrame(
        all_segments
    )

    if df.empty:

        raise RuntimeError(
            "No segments were generated."
        )

    # -------------------------------------------------------------------------
    # Split
    # -------------------------------------------------------------------------

    df["split"] = df[
        "video"
    ].apply(assign_split)

    # -------------------------------------------------------------------------
    # Remove invalid segments
    # -------------------------------------------------------------------------

    df = df[
        df["frames"] > 0
    ].copy()

    # -------------------------------------------------------------------------
    # Remove exact duplicates
    # -------------------------------------------------------------------------

    df = df.drop_duplicates(
        subset=[
            "video",
            "task_id",
            "label",
            "start_frame_global",
            "end_frame_global",
            "source",
        ]
    ).reset_index(
        drop=True
    )

    # -------------------------------------------------------------------------
    # Sort
    # -------------------------------------------------------------------------

    df = df.sort_values(
        [
            "split",
            "video",
            "start_frame_global",
        ]
    ).reset_index(
        drop=True
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    df = df[
        [
            "video",
            "task_id",
            "label",
            "start_frame_global",
            "end_frame_global",
            "frames",
            "source",
            "split",
        ]
    ]

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "UNIFIED DATASET STATISTICS"
    )
    print("=" * 90)

    print()
    print(
        "SOURCE × LABEL"
    )
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
    print(
        "SPLIT × LABEL"
    )
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
    print(
        "SEGMENTS PER SOURCE"
    )
    print("-" * 90)

    print(
        df.groupby(
            "source"
        ).size()
    )

    print()
    print(
        "VIDEOS"
    )
    print("-" * 90)

    print(
        f"Total videos : "
        f"{df['video'].nunique()}"
    )

    print(
        f"Train videos : "
        f"{df.loc[df['split'] == 'train', 'video'].nunique()}"
    )

    print(
        f"Val videos   : "
        f"{df.loc[df['split'] == 'val', 'video'].nunique()}"
    )

    print(
        f"Test videos  : "
        f"{df.loc[df['split'] == 'test', 'video'].nunique()}"
    )

    print()
    print(
        "TOTAL"
    )
    print("-" * 90)

    print(
        f"Segments : {len(df)}"
    )

    print(
        f"Videos   : {df['video'].nunique()}"
    )

    print()
    print(
        "Saved:",
        OUTPUT_FILE,
    )

    print()
    print("=" * 90)
    print(
        "UNIFIED DATASET COMPLETE"
    )
    print("=" * 90)


if __name__ == "__main__":
    main()