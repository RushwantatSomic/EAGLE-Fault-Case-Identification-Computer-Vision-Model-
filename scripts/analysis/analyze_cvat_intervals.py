import xml.etree.ElementTree as ET
from collections import defaultdict

XML_FILE = "annotations/cvat_export.xml"

root = ET.parse(XML_FILE).getroot()

tasks = defaultdict(lambda: defaultdict(list))

for track in root.findall(".//track"):
    task_id = track.attrib.get("task_id")
    label = track.attrib.get("label")
    source = track.attrib.get("source")

    for box in track.findall("./box"):
        frame = int(box.attrib["frame"])
        tasks[task_id][label].append((frame, source))


def analyze_frames(entries):
    """
    Remove duplicate frame numbers and return continuous frame segments.
    """
    frames = sorted(set(frame for frame, _ in entries))

    if not frames:
        return []

    segments = []
    start = previous = frames[0]

    for frame in frames[1:]:
        if frame == previous + 1:
            previous = frame
        else:
            segments.append((start, previous))
            start = previous = frame

    segments.append((start, previous))

    return segments


print("=" * 100)
print("EAGLE-MARS CVAT INTERVAL ANALYSIS")
print("=" * 100)

for task_id in sorted(tasks, key=lambda x: int(x)):
    print()
    print(f"TASK {task_id}")
    print("-" * 100)

    for label in sorted(tasks[task_id]):

        entries = tasks[task_id][label]

        segments = analyze_frames(entries)

        manual_frames = sorted(
            set(frame for frame, source in entries if source == "manual")
        )

        auto_frames = sorted(
            set(frame for frame, source in entries if source == "auto")
        )

        unique_frames = sorted(
            set(frame for frame, _ in entries)
        )

        print(f"\n{label}")
        print(f"  Unique frames : {len(unique_frames)}")

        if unique_frames:
            print(f"  Frame range   : {unique_frames[0]} -> {unique_frames[-1]}")

        print(f"  Manual frames : {len(manual_frames)}")
        print(f"  Auto frames   : {len(auto_frames)}")
        print(f"  Segments      : {len(segments)}")

        for start, end in segments:
            if start == end:
                print(f"    {start}")
            else:
                print(f"    {start} -> {end}")

print()
print("=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)