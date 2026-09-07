import pandas as pd
from pathlib import Path

INPUT = Path("outputs/dataset/final_state_segments.csv")
OUTPUT = Path("outputs/dataset/training_segments.csv")

df = pd.read_csv(INPUT)

# Minimum useful temporal duration.
# At 30 FPS, 16 frames = ~0.53 seconds.
MIN_FAULT_FRAMES = 16

segments = []

for video, g in df.groupby("video"):

    g = g.sort_values("start_frame_local").reset_index(drop=True)

    # Convert very short fault segments into surrounding state.
    #
    # This prevents isolated single-frame Product-on-Floor
    # annotations from creating artificial state transitions.
    for i in range(len(g)):

        row = g.iloc[i].copy()

        duration = (
            int(row["end_frame_local"])
            - int(row["start_frame_local"])
            + 1
        )

        if row["label"] == "FAULT" and duration < MIN_FAULT_FRAMES:

            # If this is between two NORMAL segments,
            # absorb it into NORMAL.
            if (
                i > 0
                and i < len(g) - 1
                and g.iloc[i - 1]["label"] == "NORMAL"
                and g.iloc[i + 1]["label"] == "NORMAL"
            ):
                row["label"] = "NORMAL"

        segments.append(row)

result = pd.DataFrame(segments)

# Merge adjacent segments with the same label.
merged = []

for _, row in result.iterrows():

    row = row.to_dict()

    if not merged:
        merged.append(row)
        continue

    previous = merged[-1]

    if (
        previous["video"] == row["video"]
        and previous["label"] == row["label"]
        and int(row["start_frame_local"])
        <= int(previous["end_frame_local"]) + 1
    ):
        previous["end_frame_local"] = max(
            int(previous["end_frame_local"]),
            int(row["end_frame_local"])
        )
    else:
        merged.append(row)

result = pd.DataFrame(merged)

result = result[
    ["video", "label", "start_frame_local", "end_frame_local"]
]

result.to_csv(OUTPUT, index=False)

print("=" * 100)
print("EAGLE-MARS TRAINING SEGMENTS")
print("=" * 100)

print(f"Videos  : {result.video.nunique()}")
print(f"Segments: {len(result)}")
print()

print("LABEL COUNTS:")
print(result["label"].value_counts())
print()

for video, g in result.groupby("video"):
    print(f"\n{video}")
    print(g.to_string(index=False))

print()
print("=" * 100)
print(f"Saved: {OUTPUT}")
print("=" * 100)