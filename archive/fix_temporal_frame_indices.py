import pandas as pd
from pathlib import Path

INPUT = Path("outputs/dataset/temporal_segments.csv")
OUTPUT = Path("outputs/dataset/temporal_segments_local.csv")

df = pd.read_csv(INPUT)

# First CVAT frame for each recovered video/task.
# These are the global frame offsets in the CVAT export.
OFFSETS = {
    "Lamella Chain 2.mp4": 0,
    "Lamella Chain 3.mp4": 557,
    "Lamella Chain 4.mp4": 1114,
    "Lamella Chain.mp4": 1671,

    "Magazine.mp4": 2228,

    "Collecting Table.mp4": 2599,
    "Collecting Table 2.mp4": 3156,
    "Collecting Table 3.mp4": 3713,
    "Collecting Table 4.mp4": 4270,

    "Cartoning.mp4": 4827,
    "Cartoning 2.mp4": 5384,
    "Cartoning 3.mp4": 5941,
    "Cartoning 4.mp4": 6498,

    "Collecting Table 5.mp4": 7055,

    "Covering 2.mp4": 7612,
    "Covering 3.mp4": 8169,
    "Covering 4.mp4": 8726,

    "Magazine 2.mp4": 9283,
    "Magazine 3.mp4": 9654,
    "Magazine 4.mp4": 10025,

    "covering.mp4": 10437,
}

df["offset"] = df["video"].map(OFFSETS)

if df["offset"].isna().any():
    missing = df.loc[df["offset"].isna(), "video"].unique()
    raise RuntimeError(f"Missing offsets for: {missing}")

df["start_frame_local"] = (
    df["start_frame_original"] - df["offset"]
).astype(int)

df["end_frame_local"] = (
    df["end_frame_original"] - df["offset"]
).astype(int)

df.drop(columns=["offset"], inplace=True)

df.to_csv(OUTPUT, index=False)

print("=" * 100)
print("LOCAL CVAT FRAME INDICES")
print("=" * 100)

print(f"Rows: {len(df)}")
print(f"Videos: {df.video.nunique()}")
print()

for video in df.video.unique():
    x = df[df.video == video]

    print(f"{video}")
    print(
        f"  local range: "
        f"{x.start_frame_local.min()} -> "
        f"{x.end_frame_local.max()}"
    )

print()
print(f"Saved: {OUTPUT}")