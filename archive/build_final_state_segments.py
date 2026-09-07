import pandas as pd
from pathlib import Path

INPUT = Path("outputs/dataset/annotation_intervals.csv")
OUTPUT = Path("outputs/dataset/final_state_segments.csv")

df = pd.read_csv(INPUT)

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

FAULT_LABELS = {"Fault Start", "Product on Floor"}

segments = []

for video, g in df.groupby("video"):

    offset = OFFSETS[video]

    # Convert global CVAT frame numbers to local frames.
    g = g.copy()
    g["start"] = g["start_frame_original"].astype(int) - offset
    g["end"] = g["end_frame_original"].astype(int) - offset

    video_start = int(g["start"].min())
    video_end = int(g["end"].max())

    # Every Fault Start and Product on Floor annotation is an error interval.
    faults = g[g["source_label"].isin(FAULT_LABELS)]

    # Build fault intervals.
    fault_intervals = []

    for _, row in faults.iterrows():
        fault_intervals.append(
            (int(row["start"]), int(row["end"]))
        )

    # Merge overlapping/touching fault intervals.
    fault_intervals.sort()

    merged = []

    for start, end in fault_intervals:

        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    # Fault End explicitly closes a fault.
    # Extend the active fault to Fault End where applicable.
    fault_ends = g[g["source_label"] == "Fault End"]

    if not fault_ends.empty and merged:
        end_frame = int(fault_ends["end"].max())

        if end_frame > merged[-1][1]:
            merged[-1][1] = end_frame

    # Build NORMAL gaps around fault intervals.
    cursor = video_start

    for start, end in merged:

        if start > cursor:
            segments.append({
                "video": video,
                "label": "NORMAL",
                "start_frame_local": cursor,
                "end_frame_local": start - 1,
            })

        segments.append({
            "video": video,
            "label": "FAULT",
            "start_frame_local": start,
            "end_frame_local": end,
        })

        cursor = end + 1

    if cursor <= video_end:
        segments.append({
            "video": video,
            "label": "NORMAL",
            "start_frame_local": cursor,
            "end_frame_local": video_end,
        })


result = pd.DataFrame(segments)

result = result.sort_values(
    ["video", "start_frame_local"]
).reset_index(drop=True)

result.to_csv(OUTPUT, index=False)

print("=" * 100)
print("EAGLE-MARS FINAL STATE SEGMENTS")
print("=" * 100)

print(f"Videos  : {result.video.nunique()}")
print(f"Segments: {len(result)}")
print()

print(result["label"].value_counts())
print()

for video, g in result.groupby("video"):
    print(f"\n{video}")
    print(g[["label", "start_frame_local", "end_frame_local"]].to_string(index=False))

print()
print("=" * 100)
print(f"Saved: {OUTPUT}")
print("=" * 100)