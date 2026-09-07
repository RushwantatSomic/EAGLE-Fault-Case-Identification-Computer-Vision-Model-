import pandas as pd

clips = pd.read_csv(
    "outputs/dataset/clips/clips_manifest.csv"
)

segments = pd.read_csv(
    "outputs/dataset/training_segments.csv"
)

print("=" * 100)
print("EAGLE-MARS CLIP / TEMPORAL BOUNDARY CHECK")
print("=" * 100)

crossing = []
clean = []

for _, clip in clips.iterrows():

    video = clip["video"]
    start = int(clip["start_frame_recovered"])
    end = int(clip["end_frame_recovered"])
    label = clip["label"]

    video_segments = segments[
        segments["video"] == video
    ]

    # Determine whether the clip contains
    # more than one training segment.
    overlaps = []

    for _, seg in video_segments.iterrows():

        seg_start = int(seg["start_frame_local"])
        seg_end = int(seg["end_frame_local"])

        if (
            end >= seg_start
            and start <= seg_end
        ):
            overlaps.append(
                (
                    seg_start,
                    seg_end,
                    seg["label"]
                )
            )

    if len(overlaps) > 1:

        crossing.append(
            {
                "clip_id": clip["clip_id"],
                "video": video,
                "start": start,
                "end": end,
                "clip_label": label,
                "segments": overlaps,
            }
        )

    else:

        clean.append(clip["clip_id"])


print()
print("TOTAL CLIPS :", len(clips))
print("CLEAN CLIPS :", len(clean))
print("CROSSING    :", len(crossing))

print()

if crossing:

    print("CLIPS CROSSING STATE BOUNDARIES")
    print("-" * 100)

    for item in crossing[:100]:

        print(
            f"{item['video']:<30} "
            f"{item['start']:>5}-{item['end']:<5} "
            f"label={item['clip_label']:<6} "
            f"segments={item['segments']}"
        )

else:

    print(
        "No clips cross a temporal state boundary."
    )

print()
print("=" * 100)