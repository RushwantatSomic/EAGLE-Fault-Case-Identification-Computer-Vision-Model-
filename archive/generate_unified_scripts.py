import cv2
import numpy as np
import pandas as pd

from pathlib import Path


# =============================================================================
# EAGLE-MARS UNIFIED VIDEO CLIP GENERATOR
# FINAL CVAT OFFSET CORRECTION
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

SEGMENTS_FILE = (
    ROOT
    / "outputs"
    / "unified_dataset"
    / "segments_unified.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "unified_dataset"
    / "clips"
)

MANIFEST_FILE = (
    ROOT
    / "outputs"
    / "unified_dataset"
    / "clips_manifest.csv"
)

FRAME_COUNT = 16
IMAGE_SIZE = 224
STRIDE = 8


VIDEO_ROOTS = [
    ROOT / "data" / "continual_videos",
    ROOT / "data" / "annotated_videos",
    ROOT / "videos",
    ROOT / "Video",
    ROOT / "Videos",
    ROOT / "data",
    ROOT,
]


# =============================================================================
# KNOWN CVAT GLOBAL STARTS
#
# Used only when the complete annotation range fits the physical video.
# For problematic videos, the script derives the offset automatically.
# =============================================================================

CVAT_GLOBAL_STARTS = {
    "Carton1.mp4": 10953,
    "Carton2.mp4": 11510,
    "Carton3.mp4": 12067,

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
# PROBLEM VIDEOS
#
# These videos contain CVAT global ranges that cannot be mapped using the
# original global start because the annotation range extends beyond the
# physical video.
#
# For these, derive the offset from the annotation coverage.
# =============================================================================

PROBLEM_VIDEOS = {
    "Cartoning.mp4",
    "Mag1.mp4",
    "Mag2.mp4",
    "Mag3.mp4",
    "Magazine 4.mp4",
}


# =============================================================================
# FIND VIDEO
# =============================================================================

def find_video(video_name):

    for root in VIDEO_ROOTS:

        candidate = root / video_name

        if candidate.exists():
            return candidate

    for root in VIDEO_ROOTS:

        if not root.exists():
            continue

        try:
            matches = list(
                root.rglob(video_name)
            )
        except Exception:
            continue

        if matches:
            return matches[0]

    return None


# =============================================================================
# VIDEO INFO
# =============================================================================

def get_video_info(video_path):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video:\n{video_path}"
        )

    frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    cap.release()

    return frames, fps, width, height


# =============================================================================
# READ FRAMES
# =============================================================================

def read_frames(
    video_path,
    start_frame,
    end_frame,
):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open:\n{video_path}"
        )

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame,
    )

    frames = []

    expected = (
        end_frame
        - start_frame
        + 1
    )

    for _ in range(expected):

        ok, frame = cap.read()

        if not ok:
            break

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        frame = cv2.resize(
            frame,
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
            interpolation=cv2.INTER_AREA,
        )

        frames.append(frame)

    cap.release()

    if len(frames) != expected:

        raise RuntimeError(
            f"Could not read requested frames\n"
            f"Requested: {start_frame}-{end_frame}\n"
            f"Read: {len(frames)}"
        )

    return np.asarray(
        frames,
        dtype=np.uint8,
    )


# =============================================================================
# WINDOWS
# =============================================================================

def generate_windows(
    start_frame,
    end_frame,
):

    length = (
        end_frame
        - start_frame
        + 1
    )

    if length < FRAME_COUNT:
        return []

    windows = []

    current = start_frame

    while (
        current + FRAME_COUNT - 1
        <= end_frame
    ):

        windows.append(
            (
                current,
                current + FRAME_COUNT - 1,
            )
        )

        current += STRIDE

    final_start = (
        end_frame
        - FRAME_COUNT
        + 1
    )

    if final_start >= start_frame:

        final_window = (
            final_start,
            end_frame,
        )

        if final_window not in windows:
            windows.append(
                final_window
            )

    return windows


# =============================================================================
# DETERMINE OFFSET
# =============================================================================

def determine_offset(
    video_name,
    video_frames,
    video_df,
    source,
):
    """
    Determine the global -> local CVAT offset.

    Continual videos:
        annotations are already local.

    Normal CVAT:
        use the known global start if it produces a valid range.

    Problematic videos:
        infer the offset by fitting the annotated global span into
        the physical video frame range.
    """

    if source == "continual":
        return 0, "continual-local"

    global_min = int(
        video_df[
            "start_frame_global"
        ].min()
    )

    global_max = int(
        video_df[
            "end_frame_global"
        ].max()
    )

    # -------------------------------------------------------------------------
    # Try known offset first
    # -------------------------------------------------------------------------

    if video_name in CVAT_GLOBAL_STARTS:

        known = CVAT_GLOBAL_STARTS[
            video_name
        ]

        local_min = (
            global_min - known
        )

        local_max = (
            global_max - known
        )

        if (
            local_min >= 0
            and local_max < video_frames
        ):

            return (
                known,
                "known-valid",
            )

    # -------------------------------------------------------------------------
    # Automatic fitting
    #
    # If annotations span approximately the entire video, align the earliest
    # annotated frame with local frame 0.
    # -------------------------------------------------------------------------

    annotation_span = (
        global_max
        - global_min
        + 1
    )

    if annotation_span <= video_frames:

        inferred = global_min

        local_max = (
            global_max
            - inferred
        )

        if (
            local_max
            < video_frames
        ):

            return (
                inferred,
                "inferred-min",
            )

    # -------------------------------------------------------------------------
    # If annotation span is larger than the video, calculate a shift that
    # keeps the maximum amount inside the physical video.
    # -------------------------------------------------------------------------

    possible_offsets = [
        global_min,
        global_max - video_frames + 1,
    ]

    best_offset = None
    best_valid = -1

    for offset in possible_offsets:

        valid_start = max(
            0,
            global_min - offset,
        )

        valid_end = min(
            video_frames - 1,
            global_max - offset,
        )

        if valid_end < valid_start:
            continue

        valid_count = (
            valid_end
            - valid_start
            + 1
        )

        if valid_count > best_valid:

            best_valid = valid_count
            best_offset = offset

    if best_offset is not None:

        return (
            best_offset,
            "best-fit",
        )

    return None, "failed"


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 90)
    print(
        "EAGLE-MARS UNIFIED VIDEO CLIP GENERATOR"
    )
    print(
        "FINAL CVAT OFFSET CORRECTION"
    )
    print("=" * 90)

    if not SEGMENTS_FILE.exists():

        raise FileNotFoundError(
            f"Missing:\n{SEGMENTS_FILE}"
        )

    df = pd.read_csv(
        SEGMENTS_FILE
    )

    print()
    print("INPUT DATASET")
    print("-" * 90)

    print(
        f"Segments : {len(df)}"
    )

    print(
        f"Videos   : {df['video'].nunique()}"
    )

    print()
    print(
        df.groupby(
            ["split", "label"]
        ).size()
    )

    # -------------------------------------------------------------------------
    # Clear old clips
    # -------------------------------------------------------------------------

    if OUTPUT_DIR.exists():

        old_files = (
            list(
                OUTPUT_DIR.glob("*.npz")
            )
            +
            list(
                OUTPUT_DIR.glob("*.npy")
            )
        )

        print()
        print(
            f"Removing old clips: "
            f"{len(old_files)}"
        )

        for file in old_files:

            try:
                file.unlink()
            except Exception:
                pass

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Process videos
    # -------------------------------------------------------------------------

    manifest_rows = []

    grouped = list(
        df.groupby(
            "video",
            sort=False,
        )
    )

    print()
    print("=" * 90)
    print("GENERATING CLIPS")
    print("=" * 90)

    for number, (
        video_name,
        video_df,
    ) in enumerate(
        grouped,
        start=1,
    ):

        print()
        print(
            f"[{number}/{len(grouped)}] "
            f"{video_name}"
        )

        video_path = find_video(
            video_name
        )

        if video_path is None:

            print(
                "  ERROR: video not found"
            )

            continue

        try:

            (
                video_frames,
                fps,
                width,
                height,
            ) = get_video_info(
                video_path
            )

        except Exception as exc:

            print(
                f"  ERROR: {exc}"
            )

            continue

        print(
            f"  video : "
            f"{video_frames} frames | "
            f"{fps:.2f} FPS | "
            f"{width}x{height}"
        )

        source_values = (
            video_df[
                "source"
            ]
            .astype(str)
            .unique()
        )

        # A video should normally belong to one source.
        source = source_values[0]

        offset, offset_method = (
            determine_offset(
                video_name,
                video_frames,
                video_df,
                source,
            )
        )

        if offset is None:

            print(
                "  ERROR: could not determine "
                "frame offset"
            )

            continue

        print(
            f"  source : {source}"
        )

        print(
            f"  offset : {offset} "
            f"({offset_method})"
        )

        video_clips = 0

        # ---------------------------------------------------------------------
        # Segments
        # ---------------------------------------------------------------------

        for _, row in video_df.iterrows():

            label = str(
                row["label"]
            ).upper()

            split = str(
                row["split"]
            ).lower()

            source = str(
                row["source"]
            )

            task_id = str(
                row.get(
                    "task_id",
                    "",
                )
            )

            global_start = int(
                row[
                    "start_frame_global"
                ]
            )

            global_end = int(
                row[
                    "end_frame_global"
                ]
            )

            # -----------------------------------------------------------------
            # GLOBAL -> LOCAL
            # -----------------------------------------------------------------

            local_start = (
                global_start
                - offset
            )

            local_end = (
                global_end
                - offset
            )

            # -----------------------------------------------------------------
            # Clip to physical video.
            #
            # This is deliberate. If an annotation extends a few frames beyond
            # the recovered MP4, we keep the valid portion instead of losing
            # the entire annotation.
            # -----------------------------------------------------------------

            local_start = max(
                0,
                local_start,
            )

            local_end = min(
                video_frames - 1,
                local_end,
            )

            if local_end < local_start:

                continue

            windows = generate_windows(
                local_start,
                local_end,
            )

            for clip_index, (
                clip_start,
                clip_end,
            ) in enumerate(
                windows
            ):

                try:

                    frames = read_frames(
                        video_path,
                        clip_start,
                        clip_end,
                    )

                except Exception as exc:

                    print(
                        f"  WARNING: "
                        f"{video_name} "
                        f"{clip_start}-"
                        f"{clip_end}: "
                        f"{exc}"
                    )

                    continue

                if frames.shape != (
                    FRAME_COUNT,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                    3,
                ):
                    continue

                safe_name = (
                    Path(video_name)
                    .stem
                    .replace(
                        " ",
                        "_",
                    )
                    .replace(
                        "-",
                        "_",
                    )
                )

                filename = (
                    f"{split}_"
                    f"{label}_"
                    f"{safe_name}_"
                    f"{clip_start:06d}_"
                    f"{clip_end:06d}_"
                    f"{clip_index:04d}.npz"
                )

                output_path = (
                    OUTPUT_DIR
                    / filename
                )

                save_clip(
                    frames,
                    output_path,
                )

                manifest_rows.append(
                    {
                        "file": str(
                            output_path.resolve()
                        ),
                        "video": video_name,
                        "split": split,
                        "label": label,
                        "source": source,
                        "task_id": task_id,

                        "start_frame_global":
                            global_start,

                        "end_frame_global":
                            global_end,

                        "start_frame_local":
                            local_start,

                        "end_frame_local":
                            local_end,

                        "start_frame":
                            clip_start,

                        "end_frame":
                            clip_end,

                        "frames":
                            FRAME_COUNT,

                        "fps_recovered":
                            fps,

                        "width":
                            width,

                        "height":
                            height,

                        "offset_used":
                            offset,

                        "offset_method":
                            offset_method,
                    }
                )

                video_clips += 1

        print(
            f"  clips : {video_clips}"
        )

    # =========================================================================
    # MANIFEST
    # =========================================================================

    manifest = pd.DataFrame(
        manifest_rows
    )

    if manifest.empty:

        raise RuntimeError(
            "No clips were generated."
        )

    manifest.to_csv(
        MANIFEST_FILE,
        index=False,
    )

    # =========================================================================
    # STATISTICS
    # =========================================================================

    print()
    print("=" * 90)
    print(
        "UNIFIED CLIP GENERATION COMPLETE"
    )
    print("=" * 90)

    print()
    print(
        f"Total clips: {len(manifest)}"
    )

    print()
    print("BY SOURCE")
    print("-" * 90)

    print(
        manifest.groupby(
            "source"
        ).size()
    )

    print()
    print("SOURCE × LABEL")
    print("-" * 90)

    print(
        manifest.groupby(
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
        manifest.groupby(
            [
                "split",
                "label",
            ]
        ).size()
    )

    print()
    print("OFFSET METHODS")
    print("-" * 90)

    print(
        manifest.groupby(
            [
                "source",
                "offset_method",
            ]
        ).size()
    )

    print()
    print("VIDEOS")
    print("-" * 90)

    print(
        f"Total videos : "
        f"{manifest['video'].nunique()}"
    )

    print(
        f"Train videos : "
        f"{manifest.loc[manifest.split == 'train', 'video'].nunique()}"
    )

    print(
        f"Val videos   : "
        f"{manifest.loc[manifest.split == 'val', 'video'].nunique()}"
    )

    print(
        f"Test videos  : "
        f"{manifest.loc[manifest.split == 'test', 'video'].nunique()}"
    )

    print()
    print("CLIP FORMAT")
    print("-" * 90)

    print(
        f"Frames : {FRAME_COUNT}"
    )

    print(
        f"Size   : {IMAGE_SIZE} x {IMAGE_SIZE}"
    )

    print(
        "Format : RGB uint8"
    )

    print()
    print("OUTPUT")
    print("-" * 90)

    print(
        f"Clips    : {OUTPUT_DIR}"
    )

    print(
        f"Manifest : {MANIFEST_FILE}"
    )

    print()
    print("=" * 90)


def save_clip(
    frames,
    output_path,
):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_path,
        frames=frames,
    )


if __name__ == "__main__":
    main()