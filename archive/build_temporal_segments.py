import pandas as pd
from pathlib import Path

INPUT = Path("outputs/dataset/annotation_intervals.csv")
OUTPUT = Path("outputs/dataset/temporal_segments.csv")

df = pd.read_csv(INPUT)

segments = []

for video, g in df.groupby("video"):

    g = g.sort_values("start_frame_original")

    # ---------------------------------------------------------
    # NORMAL
    # ---------------------------------------------------------
    normal = g[g["source_label"] == "Normal"]

    # ---------------------------------------------------------
    # FAULT START
    # The first Fault Start frame is the beginning of the fault.
    # It continues until Fault End or the end of the annotation.
    # ---------------------------------------------------------
    fault_start = g[g["source_label"] == "Fault Start"]

    fault_end = g[g["source_label"] == "Fault End"]

    if normal.empty and fault_start.empty:
        continue

    # Overall annotated range
    overall_start = int(g["start_frame_original"].min())
    overall_end = int(g["end_frame_original"].max())

    # Fault begins at first Fault Start
    if not fault_start.empty:

        fault_begin = int(fault_start["start_frame_original"].min())

        if not fault_end.empty:
            fault_finish = int(fault_end["end_frame_original"].max())
        else:
            fault_finish = overall_end

        # Normal is everything before fault begins
        if fault_begin > overall_start:
            segments.append({
                "video": video,
                "label": "NORMAL",
                "start_frame_original": overall_start,
                "end_frame_original": fault_begin - 1
            })

        segments.append({
            "video": video,
            "label": "FAULT",
            "start_frame_original": fault_begin,
            "end_frame_original": fault_finish
        })

    else:
        # Entire annotated video is normal
        segments.append({
            "video": video,
            "label": "NORMAL",
            "start_frame_original": overall_start,
            "end_frame_original": overall_end
        })


result = pd.DataFrame(segments)

result = result.sort_values(
    ["video", "start_frame_original"]
).reset_index(drop=True)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT, index=False)

print("=" * 100)
print("EAGLE-MARS TEMPORAL SEGMENTS")
print("=" * 100)

print(f"Videos: {result['video'].nunique()}")
print(f"Segments: {len(result)}")
print()

print("LABEL COUNTS:")
print(result["label"].value_counts())
print()

print("SEGMENTS:")
print(result.to_string(index=False))

print()
print("=" * 100)
print(f"Saved: {OUTPUT}")
print("=" * 100)