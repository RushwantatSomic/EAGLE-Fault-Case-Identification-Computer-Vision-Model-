from pathlib import Path
import cv2
import numpy as np
import pandas as pd


# ============================================================
# EAGLE-MARS EXPERIMENT 4
# TRAINING CLIP GENERATOR
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

SEGMENTS_FILE = (
    ROOT / "outputs" / "dataset_exp4" / "segments_exp4.csv"
)

VIDEO_DIR = (
    ROOT / "data" / "recovered_videos"
)

OUTPUT_DIR = (
    ROOT / "outputs" / "dataset_exp4" / "clips"
)

MANIFEST_FILE = (
    ROOT / "outputs" / "dataset_exp4" / "clips_manifest.csv"
)


# ============================================================
# VIDEO MAE SETTINGS
# ============================================================

CLIP_LENGTH = 16
FRAME_SIZE = (224, 224)


# ============================================================
# VIDEO-LEVEL SPLIT
#
# IMPORTANT:
# The same video must never appear in multiple splits.
# ============================================================

TRAIN_VIDEOS = {
    "Carton1.mp4",
    "Carton2.mp4",
    "Cover1.mp4",
    "Cover2.mp4",
    "LamChain.mp4",
    "LamChain2.mp4",
    "Mag1.mp4",
    "Table1.mp4",
}

VAL_VIDEOS = {
    "Carton3.mp4",
    "Cover3.mp4",
    "LamChain3.mp4",
    "Mag2.mp4",
}

TEST_VIDEOS = {
    "Mag3.mp4",
    "Table2.mp4",
    "Table3.mp4",
}


# ============================================================
# FIND VIDEO
# ============================================================

def get_video_path(video_name):

    direct = VIDEO_DIR / video_name

    if direct.exists():
        return direct

    matches = list(ROOT.rglob(video_name))

    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"\nCould not find video:\n"
        f"  {video_name}\n"
        f"\nExpected directory:\n"
        f"  {VIDEO_DIR}\n"
    )


# ============================================================
# DETERMINE SPLIT
# ============================================================

def determine_split(video):

    if video in TRAIN_VIDEOS:
        return "train"

    if video in VAL_VIDEOS:
        return "val"

    if video in TEST_VIDEOS:
        return "test"

    raise ValueError(
        f"Video has no split assignment: {video}"
    )


# ============================================================
# OPEN VIDEO
# ============================================================

def open_video(video_path):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video:\n{video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    return cap, fps, frame_count


# ============================================================
# READ 16 FRAMES
# ============================================================

def read_frames(
    cap,
    start_frame,
    end_frame
):

    expected_count = (
        end_frame -
        start_frame +
        1
    )

    if expected_count != CLIP_LENGTH:
        raise ValueError(
            f"Expected {CLIP_LENGTH} frames, "
            f"got {expected_count}"
        )

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame
    )

    frames = []

    for frame_number in range(
        start_frame,
        end_frame + 1
    ):

        ok, frame = cap.read()

        if not ok:
            raise RuntimeError(
                f"Failed to read frame "
                f"{frame_number}"
            )

        # BGR -> RGB
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Resize BEFORE saving.
        #
        # Original:
        # 1280 x 720 x 3
        #
        # New:
        # 224 x 224 x 3
        #
        # This dramatically reduces disk usage.
        # ----------------------------------------------------

        frame = cv2.resize(
            frame,
            FRAME_SIZE,
            interpolation=cv2.INTER_AREA
        )

        frames.append(frame)

    array = np.stack(
        frames,
        axis=0
    )

    return array.astype(
        np.uint8
    )


# ============================================================
# GENERATE CLIP RANGES
# ============================================================

def generate_clip_ranges(
    start_frame,
    end_frame
):

    ranges = []

    current = start_frame

    while (
        current +
        CLIP_LENGTH -
        1
        <= end_frame
    ):

        clip_start = current

        clip_end = (
            current +
            CLIP_LENGTH -
            1
        )

        ranges.append(
            (
                clip_start,
                clip_end
            )
        )

        # Non-overlapping clips
        current += CLIP_LENGTH

    return ranges


# ============================================================
# SAVE CLIP
# ============================================================

def save_clip(
    array,
    output_path
):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        output_path,
        array,
        allow_pickle=False
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print(
        "EAGLE-MARS EXPERIMENT 4 "
        "TRAINING CLIP GENERATOR"
    )
    print("=" * 90)

    # --------------------------------------------------------
    # Check segments file
    # --------------------------------------------------------

    if not SEGMENTS_FILE.exists():

        raise FileNotFoundError(
            f"\nMissing segments file:\n"
            f"{SEGMENTS_FILE}"
        )

    df = pd.read_csv(
        SEGMENTS_FILE
    )

    required_columns = {
        "video",
        "task_id",
        "label",
        "start_frame_local",
        "end_frame_local",
        "frames",
        "fps",
    }

    missing = (
        required_columns -
        set(df.columns)
    )

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    print()
    print(
        f"Segments file : "
        f"{SEGMENTS_FILE}"
    )

    print(
        f"Input segments: "
        f"{len(df)}"
    )

    # --------------------------------------------------------
    # Remove segments shorter than 16 frames
    # --------------------------------------------------------

    original_count = len(df)

    df = df[
        df["frames"] >= CLIP_LENGTH
    ].copy()

    removed_count = (
        original_count -
        len(df)
    )

    print(
        f"Usable segments: "
        f"{len(df)}"
    )

    print(
        f"Removed < {CLIP_LENGTH}-frame "
        f"segments: {removed_count}"
    )

    # --------------------------------------------------------
    # Validate labels
    # --------------------------------------------------------

    valid_labels = {
        "NORMAL",
        "FAULT"
    }

    actual_labels = set(
        df["label"].unique()
    )

    invalid_labels = (
        actual_labels -
        valid_labels
    )

    if invalid_labels:

        raise ValueError(
            f"Unexpected labels: "
            f"{invalid_labels}"
        )

    # --------------------------------------------------------
    # Validate video split
    # --------------------------------------------------------

    all_split_videos = (
        TRAIN_VIDEOS |
        VAL_VIDEOS |
        TEST_VIDEOS
    )

    dataset_videos = set(
        df["video"].unique()
    )

    unassigned = (
        dataset_videos -
        all_split_videos
    )

    if unassigned:

        raise ValueError(
            "Videos without split assignment:\n"
            +
            "\n".join(
                sorted(unassigned)
            )
        )

    # Check overlap
    if TRAIN_VIDEOS & VAL_VIDEOS:
        raise ValueError(
            "TRAIN/VAL video overlap detected."
        )

    if TRAIN_VIDEOS & TEST_VIDEOS:
        raise ValueError(
            "TRAIN/TEST video overlap detected."
        )

    if VAL_VIDEOS & TEST_VIDEOS:
        raise ValueError(
            "VAL/TEST video overlap detected."
        )

    # --------------------------------------------------------
    # Print split
    # --------------------------------------------------------

    print()
    print("VIDEO SPLIT")
    print("-" * 90)

    print("TRAIN:")

    for video in sorted(
        TRAIN_VIDEOS
    ):
        print(
            f"  {video}"
        )

    print()
    print("VAL:")

    for video in sorted(
        VAL_VIDEOS
    ):
        print(
            f"  {video}"
        )

    print()
    print("TEST:")

    for video in sorted(
        TEST_VIDEOS
    ):
        print(
            f"  {video}"
        )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    manifest_rows = []

    global_clip_id = 0

    # --------------------------------------------------------
    # Process each video
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("GENERATING CLIPS")
    print("=" * 90)

    for video_name in sorted(
        df["video"].unique()
    ):

        split = determine_split(
            video_name
        )

        video_path = get_video_path(
            video_name
        )

        cap, actual_fps, recovered_frames = (
            open_video(video_path)
        )

        print()
        print(video_name)

        print(
            f"  split     : {split}"
        )

        print(
            f"  FPS       : "
            f"{actual_fps:.2f}"
        )

        print(
            f"  frames    : "
            f"{recovered_frames}"
        )

        video_segments = df[
            df["video"] == video_name
        ].copy()

        video_segments = (
            video_segments
            .sort_values(
                "start_frame_local"
            )
        )

        video_clip_count = 0

        # ----------------------------------------------------
        # Process annotated segments
        # ----------------------------------------------------

        for _, row in (
            video_segments.iterrows()
        ):

            label = str(
                row["label"]
            )

            start_frame = int(
                row[
                    "start_frame_local"
                ]
            )

            end_frame = int(
                row[
                    "end_frame_local"
                ]
            )

            segment_length = (
                end_frame -
                start_frame +
                1
            )

            if segment_length < CLIP_LENGTH:
                continue

            # ------------------------------------------------
            # Safety checks
            # ------------------------------------------------

            if start_frame < 0:

                print(
                    f"  WARNING: negative "
                    f"start frame "
                    f"{start_frame}"
                )

                continue

            if end_frame >= recovered_frames:

                print(
                    f"  WARNING: segment "
                    f"{start_frame}-{end_frame} "
                    f"exceeds video "
                    f"({recovered_frames})"
                )

                continue

            # ------------------------------------------------
            # Generate 16-frame windows
            # ------------------------------------------------

            clip_ranges = (
                generate_clip_ranges(
                    start_frame,
                    end_frame
                )
            )

            for (
                clip_start,
                clip_end
            ) in clip_ranges:

                # Final boundary check
                if clip_start < start_frame:
                    continue

                if clip_end > end_frame:
                    continue

                # ------------------------------------------------
                # Read and resize frames
                # ------------------------------------------------

                frames = read_frames(
                    cap,
                    clip_start,
                    clip_end
                )

                # ------------------------------------------------
                # Verify final array
                # ------------------------------------------------

                expected_shape = (
                    CLIP_LENGTH,
                    FRAME_SIZE[1],
                    FRAME_SIZE[0],
                    3,
                )

                if frames.shape != expected_shape:

                    raise RuntimeError(
                        f"Unexpected clip shape: "
                        f"{frames.shape}; "
                        f"expected "
                        f"{expected_shape}"
                    )

                # ------------------------------------------------
                # Filename
                # ------------------------------------------------

                filename = (
                    f"{global_clip_id:06d}_"
                    f"{label}_"
                    f"{Path(video_name).stem}.npy"
                )

                output_path = (
                    OUTPUT_DIR /
                    split /
                    label /
                    filename
                )

                # ------------------------------------------------
                # Save
                # ------------------------------------------------

                save_clip(
                    frames,
                    output_path
                )

                # ------------------------------------------------
                # Manifest
                # ------------------------------------------------

                manifest_rows.append({

                    "clip_id":
                        global_clip_id,

                    "video":
                        video_name,

                    "task_id":
                        int(row["task_id"]),

                    "split":
                        split,

                    "label":
                        label,

                    "start_frame_recovered":
                        clip_start,

                    "end_frame_recovered":
                        clip_end,

                    "fps_recovered":
                        float(actual_fps),

                    "frames":
                        CLIP_LENGTH,

                    "height":
                        FRAME_SIZE[1],

                    "width":
                        FRAME_SIZE[0],

                    "segment_start":
                        start_frame,

                    "segment_end":
                        end_frame,

                    "file":
                        str(
                            output_path
                            .relative_to(ROOT)
                        ),
                })

                global_clip_id += 1

                video_clip_count += 1

        cap.release()

        print(
            f"  clips     : "
            f"{video_clip_count}"
        )

    # --------------------------------------------------------
    # Save manifest
    # --------------------------------------------------------

    manifest = pd.DataFrame(
        manifest_rows
    )

    manifest.to_csv(
        MANIFEST_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("CLIP GENERATION COMPLETE")
    print("=" * 90)

    print(
        f"Total clips: "
        f"{len(manifest)}"
    )

    if len(manifest) > 0:

        print()
        print("BY SPLIT")
        print("-" * 90)

        print(
            manifest
            .groupby("split")
            .size()
        )

        print()
        print("BY LABEL")
        print("-" * 90)

        print(
            manifest
            .groupby("label")
            .size()
        )

        print()
        print("SPLIT × LABEL")
        print("-" * 90)

        print(
            manifest
            .groupby(
                ["split", "label"]
            )
            .size()
        )

        print()
        print("CLIPS PER VIDEO")
        print("-" * 90)

        print(
            manifest
            .groupby(
                ["video", "split", "label"]
            )
            .size()
            .to_string()
        )

        print()
        print("FRAME SIZE")
        print("-" * 90)

        print(
            f"{FRAME_SIZE[0]} x "
            f"{FRAME_SIZE[1]}"
        )

        print()
        print("CLIP LENGTH")
        print("-" * 90)

        print(
            manifest[
                "frames"
            ]
            .value_counts()
            .sort_index()
        )

        print()
        print("FPS")
        print("-" * 90)

        print(
            manifest[
                "fps_recovered"
            ]
            .value_counts()
            .sort_index()
        )

    print()
    print(
        f"Output  : "
        f"{OUTPUT_DIR}"
    )

    print(
        f"Manifest: "
        f"{MANIFEST_FILE}"
    )

    print()
    print("=" * 90)


if __name__ == "__main__":
    main()