import pandas as pd
from pathlib import Path

INPUT = Path("outputs/dataset/clips/clips_manifest.csv")
SEGMENTS = Path("outputs/dataset/training_segments.csv")
OUTPUT = Path("outputs/dataset/clips/clips_manifest_exp3.csv")


def main():

    print("=" * 100)
    print("EAGLE-MARS EXPERIMENT 3 MANIFEST")
    print("=" * 100)

    clips = pd.read_csv(INPUT)
    segments = pd.read_csv(SEGMENTS)

    boundary_ids = []

    for _, clip in clips.iterrows():

        video = clip["video"]
        start = int(clip["start_frame_recovered"])
        end = int(clip["end_frame_recovered"])

        video_segments = segments[
            segments["video"] == video
        ]

        overlaps = 0

        for _, seg in video_segments.iterrows():

            seg_start = int(
                seg["start_frame_local"]
            )

            seg_end = int(
                seg["end_frame_local"]
            )

            if (
                end >= seg_start
                and start <= seg_end
            ):
                overlaps += 1

        if overlaps > 1:
            boundary_ids.append(
                clip["clip_id"]
            )

    boundary_ids = set(boundary_ids)

    print()
    print(
        f"Original clips : {len(clips)}"
    )

    print(
        f"Boundary clips : {len(boundary_ids)}"
    )

    # --------------------------------------------------------
    # Remove boundary clips ONLY from train/validation.
    # Keep test unchanged.
    # --------------------------------------------------------

    keep = []

    for _, row in clips.iterrows():

        clip_id = row["clip_id"]
        split = row["split"]

        if (
            clip_id in boundary_ids
            and split != "test"
        ):
            keep.append(False)
        else:
            keep.append(True)

    result = clips[keep].copy()

    result.to_csv(
        OUTPUT,
        index=False
    )

    print()
    print("RESULT")
    print("-" * 100)

    print(
        "Total:",
        len(result)
    )

    print()

    print(
        result.groupby(
            ["split", "label"]
        ).size()
    )

    print()

    print(
        "Removed from TRAIN:",
        sum(
            (clips["clip_id"].isin(boundary_ids))
            & (clips["split"] == "train")
        )
    )

    print(
        "Removed from VAL:",
        sum(
            (clips["clip_id"].isin(boundary_ids))
            & (clips["split"] == "val")
        )
    )

    print(
        "Kept in TEST:",
        sum(
            (clips["clip_id"].isin(boundary_ids))
            & (clips["split"] == "test")
        )
    )

    print()
    print(
        f"Saved: {OUTPUT}"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()