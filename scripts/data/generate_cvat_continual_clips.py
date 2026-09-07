from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import shutil


# =============================================================================
# EAGLE-MARS
# CVAT CONTINUAL LEARNING CLIP GENERATOR
#
# Input:
#   outputs/unified_dataset/cvat_continual/cvat_continual_segments.csv
#
# Output:
#   outputs/unified_dataset/cvat_continual/clips/
#   outputs/unified_dataset/cvat_continual/cvat_continual_manifest.csv
#
# Clip format:
#   (16, 224, 224, 3)
#   uint8
# =============================================================================


ROOT = Path(__file__).resolve().parents[1]

SEGMENTS_CSV = (
    ROOT
    / "outputs"
    / "unified_dataset"
    / "cvat_continual"
    / "cvat_continual_segments.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "unified_dataset"
    / "cvat_continual"
    / "clips"
)

MANIFEST = (
    ROOT
    / "outputs"
    / "unified_dataset"
    / "cvat_continual"
    / "cvat_continual_manifest.csv"
)


# =============================================================================
# VIDEO SEARCH DIRECTORIES
# =============================================================================
#
# The script searches these locations recursively for the recovered videos.
#
# Add another directory here if your videos are stored somewhere else.
# =============================================================================

VIDEO_ROOTS = [

    ROOT / "Videos",

    ROOT / "videos",

    ROOT / "RecoveredVideos",

    ROOT / "recovered_videos",

    ROOT / "Data",

    ROOT / "data",

    ROOT,
]


# =============================================================================
# CONFIGURATION
# =============================================================================

CLIP_LENGTH = 16

IMAGE_SIZE = 224

# Save as .npy rather than .npz.
#
# This is faster during training and matches your existing dataset:
#
# (16, 224, 224, 3) uint8
# =============================================================================


# =============================================================================
# VIDEO INDEX
# =============================================================================

def build_video_index():

    print()
    print("=" * 90)
    print("BUILDING VIDEO INDEX")
    print("=" * 90)

    video_index = {}

    extensions = {
        ".mp4",
        ".MP4",
        ".avi",
        ".AVI",
        ".mov",
        ".MOV",
        ".mkv",
        ".MKV",
    }

    for root in VIDEO_ROOTS:

        if not root.exists():
            continue

        print(
            f"Searching: {root}"
        )

        try:

            for path in root.rglob("*"):

                if not path.is_file():
                    continue

                if path.suffix not in extensions:
                    continue

                name = path.name

                # Prefer the first occurrence.
                if name not in video_index:

                    video_index[
                        name
                    ] = path

        except PermissionError:

            print(
                f"Permission denied: {root}"
            )

    print()
    print(
        f"Videos indexed: {len(video_index)}"
    )

    return video_index


# =============================================================================
# VIDEO FINDER
# =============================================================================

def find_video(
    video_name,
    video_index,
):

    if video_name in video_index:

        return video_index[
            video_name
        ]

    # Case-insensitive fallback.

    target = video_name.lower()

    for name, path in video_index.items():

        if name.lower() == target:

            return path

    return None


# =============================================================================
# OPEN VIDEO
# =============================================================================

def open_video(
    path,
):

    cap = cv2.VideoCapture(
        str(path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video:\n{path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
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

    return (
        cap,
        fps,
        frame_count,
        width,
        height,
    )


# =============================================================================
# READ CLIP
# =============================================================================

def read_clip(
    cap,
    start_frame,
):

    frames = []

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        int(start_frame),
    )

    for _ in range(
        CLIP_LENGTH
    ):

        ok, frame = cap.read()

        if not ok:

            return None

        # OpenCV = BGR
        # Model dataset = RGB

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

        frames.append(
            frame
        )

    return np.stack(
        frames,
        axis=0,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 90)
    print(
        "EAGLE-MARS CVAT CONTINUAL "
        "LEARNING CLIP GENERATOR"
    )
    print("=" * 90)

    # -------------------------------------------------------------------------
    # Check input
    # -------------------------------------------------------------------------

    if not SEGMENTS_CSV.exists():

        raise FileNotFoundError(
            f"\nCVAT continual CSV not found:\n"
            f"{SEGMENTS_CSV}"
        )

    print()
    print("INPUT")
    print("-" * 90)

    print(
        f"Segments : {SEGMENTS_CSV}"
    )

    df = pd.read_csv(
        SEGMENTS_CSV
    )

    print(
        f"Windows  : {len(df)}"
    )

    # -------------------------------------------------------------------------
    # Validate columns
    # -------------------------------------------------------------------------

    required = {
        "video",
        "label",
        "start_frame_local",
        "end_frame_local",
        "frames",
        "split",
        "source",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise RuntimeError(
            "Missing required columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    # -------------------------------------------------------------------------
    # Remove invalid rows
    # -------------------------------------------------------------------------

    df = df[
        df["frames"] >= CLIP_LENGTH
    ].copy()

    df = df[
        df["label"].isin(
            [
                "NORMAL",
                "FAULT",
            ]
        )
    ].copy()

    df[
        "start_frame_local"
    ] = pd.to_numeric(
        df["start_frame_local"],
        errors="coerce",
    )

    df[
        "end_frame_local"
    ] = pd.to_numeric(
        df["end_frame_local"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "start_frame_local",
            "end_frame_local",
        ]
    )

    df[
        "start_frame_local"
    ] = df[
        "start_frame_local"
    ].astype(int)

    df[
        "end_frame_local"
    ] = df[
        "end_frame_local"
    ].astype(int)

    print()
    print(
        f"Usable windows: {len(df)}"
    )

    # -------------------------------------------------------------------------
    # Recreate output directory
    # -------------------------------------------------------------------------

    if OUTPUT_DIR.exists():

        print()
        print(
            "Removing existing clip directory..."
        )

        shutil.rmtree(
            OUTPUT_DIR
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Build video index
    # -------------------------------------------------------------------------

    video_index = build_video_index()

    # -------------------------------------------------------------------------
    # Verify videos before writing anything
    # -------------------------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "VERIFYING SOURCE VIDEOS"
    )
    print("=" * 90)

    required_videos = sorted(
        df["video"].unique()
    )

    missing_videos = []

    for video in required_videos:

        path = find_video(
            video,
            video_index,
        )

        if path is None:

            missing_videos.append(
                video
            )

            print(
                f"MISSING: {video}"
            )

        else:

            print(
                f"FOUND  : {video}"
            )
            print(
                f"         {path}"
            )

    if missing_videos:

        print()
        print("=" * 90)
        print(
            "ERROR: SOURCE VIDEOS MISSING"
        )
        print("=" * 90)

        for video in missing_videos:

            print(
                f"  {video}"
            )

        print()
        print(
            "Add the directory containing the "
            "missing MP4 files to VIDEO_ROOTS "
            "at the top of this script."
        )

        raise FileNotFoundError(
            "One or more source videos were not found."
        )

    # -------------------------------------------------------------------------
    # Group by video
    # -------------------------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "GENERATING CLIPS"
    )
    print("=" * 90)

    records = []

    total = 0
    failed = 0

    for video in required_videos:

        video_df = df[
            df["video"] == video
        ].copy()

        path = find_video(
            video,
            video_index,
        )

        print()
        print(
            video
        )
        print("-" * 90)

        (
            cap,
            fps,
            frame_count,
            width,
            height,
        ) = open_video(
            path
        )

        print(
            f"Source : {path}"
        )

        print(
            f"Video  : {frame_count} frames"
        )

        print(
            f"FPS    : {fps:.3f}"
        )

        print(
            f"Size   : {width}x{height}"
        )

        print(
            f"Windows: {len(video_df)}"
        )

        video_success = 0

        # ---------------------------------------------------------------------
        # Process windows
        # ---------------------------------------------------------------------

        for row_number, (_, row) in enumerate(
            video_df.iterrows()
        ):

            start = int(
                row[
                    "start_frame_local"
                ]
            )

            end = int(
                row[
                    "end_frame_local"
                ]
            )

            # -------------------------------------------------------------
            # Sanity checks
            # -------------------------------------------------------------

            if start < 0:

                failed += 1

                print(
                    f"  WARNING: negative start "
                    f"frame {start}"
                )

                continue

            if (
                start + CLIP_LENGTH
                > frame_count
            ):

                failed += 1

                print(
                    f"  WARNING: window outside "
                    f"video: {start}-{end}"
                )

                continue

            # -----------------------------------------------------------------
            # Read clip
            # -----------------------------------------------------------------

            clip = read_clip(
                cap,
                start,
            )

            if clip is None:

                failed += 1

                print(
                    f"  WARNING: could not read "
                    f"frames {start}-{end}"
                )

                continue

            # -----------------------------------------------------------------
            # Verify shape
            # -----------------------------------------------------------------

            expected_shape = (
                CLIP_LENGTH,
                IMAGE_SIZE,
                IMAGE_SIZE,
                3,
            )

            if clip.shape != expected_shape:

                failed += 1

                print(
                    f"  WARNING: bad shape "
                    f"{clip.shape}"
                )

                continue

            # -----------------------------------------------------------------
            # Filename
            # -----------------------------------------------------------------

            safe_video = (
                Path(video)
                .stem
                .replace(" ", "_")
                .replace("/", "_")
                .replace("\\", "_")
            )

            split = str(
                row["split"]
            )

            label = str(
                row["label"]
            )

            source = str(
                row["source"]
            )

            filename = (
                f"{source}__"
                f"{safe_video}__"
                f"{split}__"
                f"{label}__"
                f"{start:06d}.npy"
            )

            output_path = (
                OUTPUT_DIR
                / filename
            )

            # -----------------------------------------------------------------
            # Save
            # -----------------------------------------------------------------

            np.save(
                output_path,
                clip,
            )

            # -----------------------------------------------------------------
            # Manifest record
            # -----------------------------------------------------------------

            records.append(
                {
                    "file":
                        str(
                            output_path
                        ),

                    "video":
                        video,

                    "source":
                        source,

                    "label":
                        label,

                    "split":
                        split,

                    "start_frame_local":
                        start,

                    "end_frame_local":
                        start
                        + CLIP_LENGTH
                        - 1,

                    "frames":
                        CLIP_LENGTH,

                    "width":
                        IMAGE_SIZE,

                    "height":
                        IMAGE_SIZE,

                    "channels":
                        3,

                    "fps_recovered":
                        fps,
                }
            )

            total += 1
            video_success += 1

        cap.release()

        print(
            f"Generated: {video_success}"
        )

    # =========================================================================
    # SAVE MANIFEST
    # =========================================================================

    manifest_df = pd.DataFrame(
        records
    )

    manifest_df.to_csv(
        MANIFEST,
        index=False,
    )

    # =========================================================================
    # STATISTICS
    # =========================================================================

    print()
    print("=" * 90)
    print(
        "CLIP GENERATION COMPLETE"
    )
    print("=" * 90)

    print()
    print(
        f"Requested windows : {len(df)}"
    )

    print(
        f"Generated clips   : {total}"
    )

    print(
        f"Failed windows    : {failed}"
    )

    print()

    if not manifest_df.empty:

        print(
            "SOURCE × LABEL"
        )

        print("-" * 90)

        print(
            manifest_df.groupby(
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
            manifest_df.groupby(
                [
                    "split",
                    "label",
                ]
            ).size()
        )

        print()
        print(
            "CLIPS PER VIDEO"
        )

        print("-" * 90)

        print(
            manifest_df.groupby(
                [
                    "video",
                    "label",
                ]
            ).size()
        )

        print()
        print(
            "CLIP SHAPE"
        )

        print("-" * 90)

        sample_file = manifest_df.iloc[
            0
        ]["file"]

        sample = np.load(
            sample_file
        )

        print(
            sample.shape
        )

        print(
            sample.dtype
        )

    print()
    print(
        f"Output : {OUTPUT_DIR}"
    )

    print(
        f"Manifest: {MANIFEST}"
    )

    print()
    print("=" * 90)


if __name__ == "__main__":
    main()