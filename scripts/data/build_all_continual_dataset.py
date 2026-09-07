from pathlib import Path
import json
import hashlib

import pandas as pd


# =============================================================================
# EAGLE-MARS
# FINAL THREE-SOURCE CONTINUAL-LEARNING DATASET BUILDER
# =============================================================================
#
# Combines:
#
#   A. Original CVAT videos with manually entered timeframes
#   B. annotations2 CVAT videos with manually entered timeframes
#   C. Existing 50-video continual-learning annotations
#
# IMPORTANT
# ---------
# CVAT bounding-box labels are NOT used here.
#
# For the CVAT timeframe datasets:
#     fault_intervals -> FAULT/NORMAL windows
#
# For the 50-video continual dataset:
#     existing "windows" are preserved exactly.
#
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

TIMEFRAME_DIR = (
    ROOT
    / "outputs"
    / "unified_dataset"
    / "timeframe_annotations"
)

CVAT_ORIGINAL_JSON = (
    TIMEFRAME_DIR
    / "cvat_original_timeframes.json"
)

CVAT_ANNOTATIONS2_JSON = (
    TIMEFRAME_DIR
    / "cvat_annotations2_timeframes.json"
)

CONTINUAL_50_JSON = (
    ROOT
    / "outputs"
    / "continual_learning"
    / "continual_annotations.json"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "unified_dataset"
    / "all_continual"
)

OUTPUT_MANIFEST = (
    OUTPUT_DIR
    / "all_continual_manifest.csv"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "all_continual_dataset.json"
)


# =============================================================================
# TEMPORAL CONFIGURATION
# =============================================================================

WINDOW_FRAMES = 16
WINDOW_STRIDE = 8

# -------------------------------------------------------------------------
# Maximum windows retained per video.
#
# This prevents a CVAT video with thousands of overlapping windows from
# dominating the training dataset.
#
# Set to None to keep ALL windows.
# -------------------------------------------------------------------------

MAX_WINDOWS_PER_VIDEO = 32


# =============================================================================
# LABEL CONFIGURATION
# =============================================================================

NORMAL = "NORMAL"
FAULT = "FAULT"


# =============================================================================
# SPLIT CONFIGURATION
# =============================================================================
#
# Splits are VIDEO LEVEL.
#
# Existing split information is preserved when present.
#
# For datasets without splits:
#
#       70% train
#       15% val
#       15% test
#
# =============================================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SPLIT_SEED = (
    "EAGLE_MARS_FINAL_ALL_CONTINUAL_V1"
)


# =============================================================================
# PRINT HELPERS
# =============================================================================

def line():
    print("=" * 90)


def subline():
    print("-" * 90)


# =============================================================================
# LOAD JSON
# =============================================================================

def load_json(path):

    print(
        f"Loading:\n  {path}"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"\nRequired file does not exist:\n"
            f"{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    if not isinstance(data, list):

        raise RuntimeError(
            f"Expected a JSON list:\n{path}"
        )

    print(
        f"  Records: {len(data)}"
    )

    return data


# =============================================================================
# LABEL NORMALIZATION
# =============================================================================

def normalize_label(value):

    # Existing continual-learning JSON uses:
    #
    #   0 = NORMAL
    #   1 = FAULT

    if isinstance(
        value,
        bool
    ):

        return FAULT if value else NORMAL

    if isinstance(
        value,
        (int, float)
    ):

        if int(value) == 1:
            return FAULT

        if int(value) == 0:
            return NORMAL

    if value is None:

        return None

    text = str(
        value
    ).strip().upper()

    if text in (
        "0",
        "NORMAL",
        "CAUSE NORMAL",
    ):

        return NORMAL

    if text in (
        "1",
        "FAULT",
        "FAULT START",
        "CAUSE FAULT",
    ):

        return FAULT

    return text


# =============================================================================
# VIDEO ID
# =============================================================================

def make_video_id(
    source,
    task_id,
    video
):

    raw = (
        f"{source}|"
        f"{task_id}|"
        f"{video}"
    )

    digest = hashlib.md5(
        raw.encode(
            "utf-8"
        )
    ).hexdigest()[:12]

    return (
        f"{source}__"
        f"{digest}"
    )


# =============================================================================
# DETERMINISTIC SPLIT
# =============================================================================

def deterministic_split(
    video_id
):

    raw = (
        f"{SPLIT_SEED}|"
        f"{video_id}"
    )

    digest = hashlib.md5(
        raw.encode(
            "utf-8"
        )
    ).hexdigest()

    value = (
        int(
            digest[:8],
            16
        )
        / 0xFFFFFFFF
    )

    if value < TRAIN_RATIO:

        return "train"

    if value < (
        TRAIN_RATIO
        + VAL_RATIO
    ):

        return "val"

    return "test"


# =============================================================================
# SELECT WINDOWS
# =============================================================================
#
# We want temporal diversity rather than simply taking the first 32 windows.
#
# Therefore, if a video has more than MAX_WINDOWS_PER_VIDEO windows,
# the windows are sampled approximately uniformly through the video.
#
# FAULT windows are not intentionally removed.
#
# =============================================================================

def select_windows(
    windows,
    max_windows
):

    if (
        max_windows is None
        or len(windows)
        <= max_windows
    ):

        return list(
            windows
        )

    if max_windows <= 0:

        return []

    if max_windows == 1:

        return [
            windows[0]
        ]

    # Uniformly distributed indices.

    indices = []

    for i in range(
        max_windows
    ):

        position = (
            i
            * (len(windows) - 1)
            / (max_windows - 1)
        )

        indices.append(
            int(
                round(position)
            )
        )

    selected = []

    seen = set()

    for index in indices:

        if index in seen:
            continue

        seen.add(index)

        selected.append(
            windows[index]
        )

    return selected


# =============================================================================
# CREATE CVAT WINDOWS
# =============================================================================

def create_cvat_windows(
    item
):

    fps = float(
        item.get(
            "fps",
            30.0
        )
    )

    total_frames = int(
        item.get(
            "total_frames",
            0
        )
    )

    if total_frames <= 0:

        # Some records may use frame_count.

        total_frames = int(
            item.get(
                "frame_count",
                0
            )
        )

    if total_frames <= 0:

        raise RuntimeError(
            f"Invalid frame count for "
            f"{item.get('video')}"
        )

    fault_intervals = (
        item.get(
            "fault_intervals",
            []
        )
        or []
    )

    def is_fault(
        frame
    ):

        for interval in fault_intervals:

            start = int(
                interval.get(
                    "start_frame",
                    0
                )
            )

            end = int(
                interval.get(
                    "end_frame",
                    -1
                )
            )

            if (
                start
                <= frame
                <= end
            ):

                return True

        return False

    windows = []

    start = 0

    while (
        start + WINDOW_FRAMES
        <= total_frames
    ):

        end = (
            start
            + WINDOW_FRAMES
            - 1
        )

        fault_frames = sum(
            1
            for frame in range(
                start,
                end + 1
            )
            if is_fault(
                frame
            )
        )

        fault_ratio = (
            fault_frames
            / WINDOW_FRAMES
        )

        # -----------------------------------------------------------------
        # A window is FAULT when at least half of its frames are inside
        # a manually annotated fault interval.
        # -----------------------------------------------------------------

        if fault_ratio >= 0.50:

            label = FAULT

        else:

            label = NORMAL

        windows.append(
            {
                "start_frame":
                    start,

                "end_frame":
                    end,

                "start_seconds":
                    start / fps,

                "end_seconds":
                    (end + 1) / fps,

                "fault_frames":
                    fault_frames,

                "fault_ratio":
                    fault_ratio,

                "label":
                    label,
            }
        )

        start += WINDOW_STRIDE

    return windows


# =============================================================================
# EXPAND CVAT TIMEFRAME RECORD
# =============================================================================

def expand_cvat_record(
    item,
    source
):

    video = item.get(
        "video"
    )

    if not video:

        raise RuntimeError(
            "CVAT timeframe record "
            "has no video name."
        )

    task_id = str(
        item.get(
            "task_id",
            "NO_TASK"
        )
    )

    fps = float(
        item.get(
            "fps",
            30.0
        )
    )

    total_frames = int(
        item.get(
            "total_frames",
            item.get(
                "frame_count",
                0
            )
        )
    )

    video_path = item.get(
        "video_path"
    )

    video_id = make_video_id(
        source,
        task_id,
        video
    )

    windows = create_cvat_windows(
        item
    )

    original_window_count = (
        len(windows)
    )

    windows = select_windows(
        windows,
        MAX_WINDOWS_PER_VIDEO
    )

    rows = []

    for index, window in enumerate(
        windows
    ):

        label = normalize_label(
            window.get(
                "label"
            )
        )

        if label not in (
            NORMAL,
            FAULT
        ):

            continue

        rows.append(
            {
                "video_id":
                    video_id,

                "video":
                    video,

                "video_path":
                    video_path,

                "source":
                    source,

                "task_id":
                    task_id,

                "split":
                    item.get(
                        "split"
                    ),

                "label":
                    label,

                "start_frame":
                    int(
                        window[
                            "start_frame"
                        ]
                    ),

                "end_frame":
                    int(
                        window[
                            "end_frame"
                        ]
                    ),

                "start_seconds":
                    float(
                        window[
                            "start_seconds"
                        ]
                    ),

                "end_seconds":
                    float(
                        window[
                            "end_seconds"
                        ]
                    ),

                "fault_frames":
                    int(
                        window.get(
                            "fault_frames",
                            0
                        )
                    ),

                "fault_ratio":
                    float(
                        window.get(
                            "fault_ratio",
                            0.0
                        )
                    ),

                "window_index":
                    index,

                "frames":
                    WINDOW_FRAMES,

                "fps":
                    fps,

                "total_frames":
                    total_frames,

                "width":
                    item.get(
                        "width"
                    ),

                "height":
                    item.get(
                        "height"
                    ),

                "original_windows":
                    original_window_count,
            }
        )

    return rows


# =============================================================================
# EXPAND 50-VIDEO CONTINUAL RECORD
# =============================================================================
#
# THIS IS THE IMPORTANT CORRECTION.
#
# We directly consume:
#
#     item["windows"]
#
# from continual_annotations.json.
#
# Existing labels:
#
#     0 -> NORMAL
#     1 -> FAULT
#
# are preserved.
#
# =============================================================================

def expand_continual_record(
    item,
    source
):

    video_path = item.get(
        "video"
    )

    if not video_path:

        raise RuntimeError(
            "Continual record has no video path."
        )

    video = Path(
        video_path
    ).name

    fps = float(
        item.get(
            "fps",
            30.0
        )
    )

    total_frames = int(
        item.get(
            "frame_count",
            0
        )
    )

    if total_frames <= 0:

        total_frames = int(
            item.get(
                "total_frames",
                0
            )
        )

    video_number = item.get(
        "video_number"
    )

    task_id = (
        f"continual_{video_number}"
        if video_number is not None
        else "continual"
    )

    video_id = make_video_id(
        source,
        task_id,
        video_path
    )

    windows = item.get(
        "windows",
        []
    )

    if not windows:

        print(
            f"WARNING: no windows for "
            f"{video}"
        )

        return []

    # -------------------------------------------------------------------------
    # PRESERVE THE EXISTING CONTINUAL WINDOWS.
    #
    # Only limit their number if MAX_WINDOWS_PER_VIDEO is configured.
    # -------------------------------------------------------------------------

    original_window_count = (
        len(windows)
    )

    windows = select_windows(
        windows,
        MAX_WINDOWS_PER_VIDEO
    )

    rows = []

    for index, window in enumerate(
        windows
    ):

        label = normalize_label(
            window.get(
                "label"
            )
        )

        if label not in (
            NORMAL,
            FAULT
        ):

            continue

        start_frame = int(
            window.get(
                "start_frame"
            )
        )

        end_frame = int(
            window.get(
                "end_frame"
            )
        )

        rows.append(
            {
                "video_id":
                    video_id,

                "video":
                    video,

                "video_path":
                    video_path,

                "source":
                    source,

                "task_id":
                    task_id,

                "split":
                    None,

                "label":
                    label,

                "start_frame":
                    start_frame,

                "end_frame":
                    end_frame,

                "start_seconds":
                    start_frame / fps,

                "end_seconds":
                    (
                        end_frame + 1
                    ) / fps,

                "fault_frames":
                    None,

                "fault_ratio":
                    None,

                "window_index":
                    index,

                "frames":
                    end_frame
                    - start_frame
                    + 1,

                "fps":
                    fps,

                "total_frames":
                    total_frames,

                "width":
                    None,

                "height":
                    None,

                "original_windows":
                    original_window_count,
            }
        )

    return rows


# =============================================================================
# ASSIGN VIDEO-LEVEL SPLITS
# =============================================================================

def assign_splits(
    df
):

    print()
    line()

    print(
        "ASSIGNING VIDEO-LEVEL SPLITS"
    )

    line()

    assignments = {}

    for video_id, group in df.groupby(
        "video_id"
    ):

        known_splits = set(
            group[
                "split"
            ]
            .dropna()
            .astype(str)
        )

        # -------------------------------------------------------------
        # If source already has one split, preserve it.
        # -------------------------------------------------------------

        if len(known_splits) == 1:

            assignments[
                video_id
            ] = next(
                iter(
                    known_splits
                )
            )

            continue

        # -------------------------------------------------------------
        # Conflicting splits for the same video = error.
        # -------------------------------------------------------------

        if len(known_splits) > 1:

            raise RuntimeError(
                "\nConflicting splits for video:\n"
                f"{video_id}\n"
                f"{sorted(known_splits)}"
            )

        # -------------------------------------------------------------
        # No split -> deterministic assignment.
        # -------------------------------------------------------------

        assignments[
            video_id
        ] = deterministic_split(
            video_id
        )

    df[
        "split"
    ] = df.apply(
        lambda row:
            assignments[
                row[
                    "video_id"
                ]
            ],
        axis=1
    )

    return df


# =============================================================================
# CHECK VIDEO-LEVEL LEAKAGE
# =============================================================================

def check_video_leakage(
    df
):

    print()
    print(
        "CHECKING VIDEO-LEVEL SPLIT LEAKAGE"
    )

    subline()

    problems = []

    for video_id, group in df.groupby(
        "video_id"
    ):

        splits = set(
            group[
                "split"
            ]
        )

        if len(splits) > 1:

            problems.append(
                (
                    video_id,
                    sorted(splits)
                )
            )

    if problems:

        print(
            "LEAKAGE FOUND:"
        )

        for video_id, splits in problems:

            print(
                f"  {video_id}: "
                f"{splits}"
            )

        raise RuntimeError(
            "Video-level leakage detected."
        )

    print(
        "No video-level split leakage detected."
    )


# =============================================================================
# CHECK SOURCE STATISTICS
# =============================================================================

def source_statistics(
    df
):

    print()
    print(
        "SOURCE × LABEL"
    )

    subline()

    print(
        df.groupby(
            [
                "source",
                "label"
            ]
        ).size()
    )

    print()
    print(
        "SOURCE × SPLIT × LABEL"
    )

    subline()

    print(
        df.groupby(
            [
                "source",
                "split",
                "label"
            ]
        ).size()
    )


# =============================================================================
# CHECK VIDEO STATISTICS
# =============================================================================

def video_statistics(
    df
):

    print()
    print(
        "VIDEO STATISTICS"
    )

    subline()

    for split in (
        "train",
        "val",
        "test"
    ):

        subset = df[
            df[
                "split"
            ]
            == split
        ]

        videos = (
            subset[
                "video_id"
            ]
            .nunique()
        )

        clips = len(
            subset
        )

        faults = (
            subset[
                "label"
            ]
            == FAULT
        ).sum()

        normals = (
            subset[
                "label"
            ]
            == NORMAL
        ).sum()

        print()
        print(
            f"{split.upper()}"
        )

        print(
            f"  Videos : {videos}"
        )

        print(
            f"  Clips  : {clips}"
        )

        print(
            f"  FAULT  : {faults}"
        )

        print(
            f"  NORMAL : {normals}"
        )


# =============================================================================
# DEDUPLICATE
# =============================================================================

def deduplicate(
    df
):

    before = len(
        df
    )

    df = (
        df
        .drop_duplicates(
            subset=[
                "video_id",
                "start_frame",
                "end_frame",
                "label",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    after = len(
        df
    )

    print()
    print(
        "EXACT WINDOW DEDUPLICATION"
    )

    subline()

    print(
        f"Before : {before}"
    )

    print(
        f"After  : {after}"
    )

    print(
        f"Removed: {before - after}"
    )

    return df


# =============================================================================
# SAVE
# =============================================================================

def save_outputs(
    df
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_MANIFEST,
        index=False
    )

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            df.to_dict(
                orient="records"
            ),
            f,
            indent=2
        )

    print()
    print(
        "OUTPUT"
    )

    subline()

    print(
        f"CSV:"
    )

    print(
        OUTPUT_MANIFEST
    )

    print()

    print(
        f"JSON:"
    )

    print(
        OUTPUT_JSON
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    line()

    print(
        "EAGLE-MARS FINAL ALL-CONTINUAL "
        "DATASET BUILDER"
    )

    line()

    print()
    print(
        "THREE SOURCES"
    )

    subline()

    print(
        "1. Original CVAT manual timeframes"
    )

    print(
        f"   {CVAT_ORIGINAL_JSON}"
    )

    print()

    print(
        "2. annotations2 manual timeframes"
    )

    print(
        f"   {CVAT_ANNOTATIONS2_JSON}"
    )

    print()

    print(
        "3. Existing 50-video continual learning"
    )

    print(
        f"   {CONTINUAL_50_JSON}"
    )

    print()

    print(
        "TEMPORAL CONFIGURATION"
    )

    subline()

    print(
        f"Window size : "
        f"{WINDOW_FRAMES} frames"
    )

    print(
        f"Stride      : "
        f"{WINDOW_STRIDE} frames"
    )

    print(
        f"Max/video  : "
        f"{MAX_WINDOWS_PER_VIDEO}"
    )

    # =========================================================================
    # LOAD
    # =========================================================================

    original_data = load_json(
        CVAT_ORIGINAL_JSON
    )

    annotations2_data = load_json(
        CVAT_ANNOTATIONS2_JSON
    )

    continual_data = load_json(
        CONTINUAL_50_JSON
    )

    # =========================================================================
    # ORIGINAL CVAT
    # =========================================================================

    print()
    line()

    print(
        "PROCESSING ORIGINAL CVAT TIMEFRAMES"
    )

    line()

    original_rows = []

    for item in original_data:

        rows = expand_cvat_record(
            item,
            "cvat_original_timeframe"
        )

        original_rows.extend(
            rows
        )

    print(
        f"Videos processed : "
        f"{len(original_data)}"
    )

    print(
        f"Windows retained : "
        f"{len(original_rows)}"
    )

    # =========================================================================
    # ANNOTATIONS2
    # =========================================================================

    print()
    line()

    print(
        "PROCESSING annotations2 TIMEFRAMES"
    )

    line()

    annotations2_rows = []

    for item in annotations2_data:

        rows = expand_cvat_record(
            item,
            "cvat_annotations2_timeframe"
        )

        annotations2_rows.extend(
            rows
        )

    print(
        f"Videos processed : "
        f"{len(annotations2_data)}"
    )

    print(
        f"Windows retained : "
        f"{len(annotations2_rows)}"
    )

    # =========================================================================
    # 50 VIDEO CONTINUAL
    # =========================================================================

    print()
    line()

    print(
        "PROCESSING 50-VIDEO CONTINUAL LEARNING"
    )

    line()

    continual_rows = []

    total_input_windows = 0

    for item in continual_data:

        total_input_windows += len(
            item.get(
                "windows",
                []
            )
        )

        rows = expand_continual_record(
            item,
            "continual_50_videos"
        )

        continual_rows.extend(
            rows
        )

    print(
        f"Videos processed : "
        f"{len(continual_data)}"
    )

    print(
        f"Input windows    : "
        f"{total_input_windows}"
    )

    print(
        f"Windows retained : "
        f"{len(continual_rows)}"
    )

    # =========================================================================
    # COMBINE
    # =========================================================================

    print()
    line()

    print(
        "COMBINING ALL THREE SOURCES"
    )

    line()

    print()
    print(
        f"Original CVAT : "
        f"{len(original_rows)}"
    )

    print(
        f"annotations2  : "
        f"{len(annotations2_rows)}"
    )

    print(
        f"Continual 50  : "
        f"{len(continual_rows)}"
    )

    all_rows = (
        original_rows
        + annotations2_rows
        + continual_rows
    )

    print()
    print(
        f"Combined      : "
        f"{len(all_rows)}"
    )

    if not all_rows:

        raise RuntimeError(
            "No training windows generated."
        )

    df = pd.DataFrame(
        all_rows
    )

    # =========================================================================
    # DEDUP
    # =========================================================================

    df = deduplicate(
        df
    )

    # =========================================================================
    # SPLITS
    # =========================================================================

    df = assign_splits(
        df
    )

    # =========================================================================
    # LEAKAGE
    # =========================================================================

    check_video_leakage(
        df
    )

    # =========================================================================
    # SORT
    # =========================================================================

    split_order = {
        "train": 0,
        "val": 1,
        "test": 2,
    }

    label_order = {
        FAULT: 0,
        NORMAL: 1,
    }

    df[
        "_split_order"
    ] = df[
        "split"
    ].map(
        split_order
    )

    df[
        "_label_order"
    ] = df[
        "label"
    ].map(
        label_order
    )

    df = (
        df
        .sort_values(
            [
                "_split_order",
                "video_id",
                "start_frame",
                "_label_order",
            ]
        )
        .drop(
            columns=[
                "_split_order",
                "_label_order",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # =========================================================================
    # STATISTICS
    # =========================================================================

    print()
    line()

    print(
        "FINAL DATASET STATISTICS"
    )

    line()

    source_statistics(
        df
    )

    video_statistics(
        df
    )

    # =========================================================================
    # FINAL TOTALS
    # =========================================================================

    print()
    print(
        "TOTAL"
    )

    subline()

    print(
        f"Windows : {len(df)}"
    )

    print(
        f"Videos  : "
        f"{df['video_id'].nunique()}"
    )

    print(
        f"FAULT   : "
        f"{(df['label'] == FAULT).sum()}"
    )

    print(
        f"NORMAL  : "
        f"{(df['label'] == NORMAL).sum()}"
    )

    # =========================================================================
    # SAVE
    # =========================================================================

    save_outputs(
        df
    )

    # =========================================================================
    # SOURCE COUNTS
    # =========================================================================

    print()
    print(
        "SOURCE VIDEO COUNTS"
    )

    subline()

    print(
        df.groupby(
            "source"
        )[
            "video_id"
        ]
        .nunique()
    )

    # =========================================================================
    # FINAL
    # =========================================================================

    print()
    line()

    print(
        "ALL CONTINUAL DATASET COMPLETE"
    )

    line()

    print()
    print(
        "Included:"
    )

    print(
        "  [1] Original CVAT manual timeframes"
    )

    print(
        "  [2] annotations2 manual timeframes"
    )

    print(
        "  [3] 50-video continual-learning windows"
    )

    print()
    print(
        "All sources are now represented in ONE "
        "temporal dataset."
    )

    print()
    print(
        "Ready for clip generation."
    )

    line()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()