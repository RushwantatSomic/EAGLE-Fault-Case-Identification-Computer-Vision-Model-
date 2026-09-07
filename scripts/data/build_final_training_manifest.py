from pathlib import Path
import pandas as pd
import numpy as np


# =============================================================================
# EAGLE-MARS
# FINAL TRAINING MANIFEST BUILDER - CORRECTED
#
# Combines:
#
#   Existing unified dataset       4814 clips
#   CVAT continual dataset          849 clips
#
# Expected final dataset:
#
#   5663 clips before exact-path deduplication
#
# Supports:
#
#   .npy
#   .npz
#
# Does NOT perform frame/window deduplication.
# =============================================================================


ROOT = Path(__file__).resolve().parents[1]


UNIFIED_MANIFEST = (
    ROOT
    / "outputs"
    / "unified_dataset"
    / "clips_manifest.csv"
)

CVAT_CONTINUAL_MANIFEST = (
    ROOT
    / "outputs"
    / "unified_dataset"
    / "cvat_continual"
    / "cvat_continual_manifest.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "unified_dataset"
    / "final_training"
)

FINAL_MANIFEST = (
    OUTPUT_DIR
    / "final_manifest.csv"
)


EXPECTED_SHAPE = (
    16,
    224,
    224,
    3,
)

EXPECTED_DTYPE = np.uint8

VALID_LABELS = {
    "NORMAL",
    "FAULT",
}

VALID_SPLITS = {
    "train",
    "val",
    "test",
}


# =============================================================================
# LOAD
# =============================================================================

def load_manifest(path, name):

    if not path.exists():
        raise FileNotFoundError(
            f"\nManifest not found:\n{path}"
        )

    print(f"Loading {name}:")
    print(f"  {path}")

    df = pd.read_csv(path)

    print(f"  Rows: {len(df)}")

    required = {
        "file",
        "video",
        "label",
        "split",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"{name} missing columns: "
            + ", ".join(sorted(missing))
        )

    return df


# =============================================================================
# NORMALIZE PATHS
# =============================================================================

def normalize_paths(df):

    df = df.copy()

    def normalize(value):

        path = Path(str(value))

        if not path.is_absolute():
            path = ROOT / path

        return str(path.resolve())

    df["file"] = df["file"].apply(normalize)

    return df


# =============================================================================
# SOURCE
# =============================================================================

def add_source(df, source):

    df = df.copy()

    df["dataset_source"] = source

    return df


# =============================================================================
# VALIDATION
# =============================================================================

def validate_labels(df):

    invalid = sorted(
        set(df["label"].dropna())
        - VALID_LABELS
    )

    if invalid:
        raise RuntimeError(
            "Invalid labels: "
            + ", ".join(invalid)
        )


def validate_splits(df):

    invalid = sorted(
        set(df["split"].dropna())
        - VALID_SPLITS
    )

    if invalid:
        raise RuntimeError(
            "Invalid splits: "
            + ", ".join(invalid)
        )


# =============================================================================
# EXACT FILE DEDUPLICATION ONLY
# =============================================================================

def remove_exact_duplicates(df):

    before = len(df)

    df = df.drop_duplicates(
        subset=["file"]
    ).copy()

    after = len(df)

    print()
    print(
        "EXACT FILE DEDUPLICATION"
    )
    print("-" * 90)

    print(
        f"Before : {before}"
    )

    print(
        f"After  : {after}"
    )

    print(
        f"Removed: {before - after}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "No video/frame/window deduplication "
        "was performed."
    )

    return df


# =============================================================================
# VERIFY FILES
# =============================================================================

def verify_files(df):

    print()
    print(
        "VERIFYING CLIP FILES"
    )

    print("-" * 90)

    missing = []

    for path in df["file"].unique():

        if not Path(path).exists():
            missing.append(path)

    if missing:

        print(
            f"Missing files: {len(missing)}"
        )

        for path in missing[:20]:
            print(
                f"  {path}"
            )

        raise FileNotFoundError(
            "Training clips are missing."
        )

    print(
        f"All {len(df)} clip references verified."
    )


# =============================================================================
# LOAD ONE CLIP
# =============================================================================

def load_clip(path):

    path = Path(path)

    suffix = path.suffix.lower()

    if suffix == ".npy":

        array = np.load(
            path,
            mmap_mode="r",
        )

        return array

    if suffix == ".npz":

        data = np.load(
            path
        )

        try:

            # Most existing clips use a single array.
            keys = list(data.keys())

            if not keys:
                raise RuntimeError(
                    "NPZ contains no arrays."
                )

            # Prefer common names if present.
            preferred = [
                "clip",
                "frames",
                "array",
                "arr_0",
            ]

            selected = None

            for key in preferred:

                if key in data:
                    selected = key
                    break

            if selected is None:
                selected = keys[0]

            array = data[selected]

            # Make a standalone array before closing NPZ.
            array = np.asarray(array)

            return array

        finally:

            data.close()

    raise RuntimeError(
        f"Unsupported clip format: {path}"
    )


# =============================================================================
# VALIDATE SAMPLES
# =============================================================================

def validate_sample_clips(
    df,
    samples_per_source=5,
):

    print()
    print(
        "VALIDATING SAMPLE CLIPS"
    )

    print("-" * 90)

    problems = []

    rng = 42

    for source in sorted(
        df["dataset_source"].unique()
    ):

        source_df = df[
            df["dataset_source"] == source
        ]

        sample = source_df.sample(
            n=min(
                samples_per_source,
                len(source_df),
            ),
            random_state=rng,
        )

        print()
        print(
            f"{source}: "
            f"checking {len(sample)} clips"
        )

        for _, row in sample.iterrows():

            path = row["file"]

            try:

                x = load_clip(path)

                if tuple(x.shape) != EXPECTED_SHAPE:

                    problems.append(
                        (
                            path,
                            f"shape={x.shape}",
                        )
                    )

                if x.dtype != EXPECTED_DTYPE:

                    problems.append(
                        (
                            path,
                            f"dtype={x.dtype}",
                        )
                    )

            except Exception as e:

                problems.append(
                    (
                        path,
                        str(e),
                    )
                )

    if problems:

        print()
        print(
            "PROBLEMS FOUND"
        )

        for path, error in problems:

            print(
                f"  {path}"
            )

            print(
                f"    {error}"
            )

        raise RuntimeError(
            "Clip validation failed."
        )

    print()
    print(
        "All sample clips validated successfully."
    )


# =============================================================================
# VIDEO SPLIT LEAKAGE
# =============================================================================

def check_video_split_leakage(df):

    print()
    print(
        "CHECKING VIDEO-LEVEL SPLIT LEAKAGE"
    )

    print("-" * 90)

    split_counts = (
        df.groupby("video")["split"]
        .nunique()
    )

    leaking = split_counts[
        split_counts > 1
    ]

    if len(leaking):

        print(
            "LEAKING VIDEOS:"
        )

        for video in leaking.index:

            splits = sorted(
                df.loc[
                    df["video"] == video,
                    "split",
                ].unique()
            )

            print(
                f"  {video}: {splits}"
            )

        raise RuntimeError(
            "Video-level split leakage detected."
        )

    print(
        "No video-level split leakage detected."
    )


# =============================================================================
# STATISTICS
# =============================================================================

def print_statistics(df):

    print()
    print("=" * 90)
    print(
        "FINAL DATASET STATISTICS"
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
                "dataset_source",
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
        "SOURCE × SPLIT × LABEL"
    )

    print("-" * 90)

    print(
        df.groupby(
            [
                "dataset_source",
                "split",
                "label",
            ]
        ).size()
    )

    print()
    print(
        "VIDEOS"
    )

    print("-" * 90)

    print(
        f"Total videos: "
        f"{df['video'].nunique()}"
    )

    for split in [
        "train",
        "val",
        "test",
    ]:

        subset = df[
            df["split"] == split
        ]

        print(
            f"{split.upper():5} "
            f"videos={subset['video'].nunique():3} "
            f"clips={len(subset):5}"
        )

    print()
    print(
        "TOTAL CLIPS"
    )

    print("-" * 90)

    print(
        len(df)
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 90)
    print(
        "EAGLE-MARS FINAL TRAINING "
        "MANIFEST BUILDER"
    )
    print("=" * 90)

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    unified = load_manifest(
        UNIFIED_MANIFEST,
        "Existing unified dataset",
    )

    cvat = load_manifest(
        CVAT_CONTINUAL_MANIFEST,
        "CVAT continual dataset",
    )

    # -------------------------------------------------------------------------
    # Sources
    # -------------------------------------------------------------------------

    unified = add_source(
        unified,
        "unified_base",
    )

    cvat = add_source(
        cvat,
        "cvat_continual",
    )

    # -------------------------------------------------------------------------
    # Normalize
    # -------------------------------------------------------------------------

    unified = normalize_paths(
        unified
    )

    cvat = normalize_paths(
        cvat
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

    print()
    print(
        f"Existing unified clips : {len(unified)}"
    )

    print(
        f"CVAT continual clips   : {len(cvat)}"
    )

    df = pd.concat(
        [
            unified,
            cvat,
        ],
        ignore_index=True,
    )

    print(
        f"Combined               : {len(df)}"
    )

    # -------------------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------------------

    validate_labels(
        df
    )

    validate_splits(
        df
    )

    # -------------------------------------------------------------------------
    # ONLY exact path deduplication
    # -------------------------------------------------------------------------

    df = remove_exact_duplicates(
        df
    )

    # -------------------------------------------------------------------------
    # Files
    # -------------------------------------------------------------------------

    verify_files(
        df
    )

    # -------------------------------------------------------------------------
    # Split leakage
    # -------------------------------------------------------------------------

    check_video_split_leakage(
        df
    )

    # -------------------------------------------------------------------------
    # Sample validation
    # -------------------------------------------------------------------------

    validate_sample_clips(
        df
    )

    # -------------------------------------------------------------------------
    # Sort
    # -------------------------------------------------------------------------

    sort_cols = [
        c
        for c in [
            "split",
            "label",
            "video",
            "start_frame_local",
            "file",
        ]
        if c in df.columns
    ]

    df = df.sort_values(
        sort_cols
    ).reset_index(
        drop=True
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        FINAL_MANIFEST,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------------

    print_statistics(
        df
    )

    print()
    print("=" * 90)
    print(
        "FINAL MANIFEST CREATED"
    )
    print("=" * 90)

    print()
    print(
        f"Saved:"
    )

    print(
        FINAL_MANIFEST
    )

    print()
    print(
        "Expected total:"
    )

    print(
        "4814 + 849 = 5663 clips"
    )

    print()
    print(
        "Ready for final 5-epoch training."
    )

    print("=" * 90)


if __name__ == "__main__":
    main()