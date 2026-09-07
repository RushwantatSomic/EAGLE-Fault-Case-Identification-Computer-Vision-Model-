import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path


XML_PATH = Path("Annotations/cvat_export.xml")


def main():
    print("=" * 100)
    print("CVAT EXPORT INSPECTION")
    print("=" * 100)

    if not XML_PATH.exists():
        print(f"ERROR: XML file not found: {XML_PATH}")
        return

    print(f"XML: {XML_PATH}")
    print(f"Size: {XML_PATH.stat().st_size:,} bytes")
    print()

    tree = ET.parse(XML_PATH)
    root = tree.getroot()

    print(f"Root element: {root.tag}")
    print(f"Root attributes: {root.attrib}")
    print()

    # ---------------------------------------------------------
    # META / TASK INFORMATION
    # ---------------------------------------------------------

    tasks = root.findall(".//task")

    print("=" * 100)
    print(f"TASKS FOUND: {len(tasks)}")
    print("=" * 100)

    for i, task in enumerate(tasks, 1):
        print(f"\nTASK #{i}")

        name = task.findtext("name")
        task_id = task.findtext("id")
        size = task.findtext("size")
        mode = task.findtext("mode")
        overlap = task.findtext("overlap")
        start_frame = task.findtext("start_frame")
        stop_frame = task.findtext("stop_frame")
        source = task.findtext("source")

        print(f"  ID:          {task_id}")
        print(f"  Name:        {name}")
        print(f"  Size:        {size}")
        print(f"  Mode:        {mode}")
        print(f"  Overlap:     {overlap}")
        print(f"  Start frame: {start_frame}")
        print(f"  Stop frame:  {stop_frame}")
        print(f"  Source:      {source}")

        # Original data / video information
        original = task.find("original_size")
        if original is not None:
            print(
                f"  Original:    "
                f"{original.attrib.get('width')} x "
                f"{original.attrib.get('height')}"
            )

        labels = task.findall("./labels/label")

        print(f"  Labels:      {len(labels)}")

        for label in labels:
            label_name = label.findtext("name")
            print(f"      - {label_name}")

    # ---------------------------------------------------------
    # LABEL COUNTS
    # ---------------------------------------------------------

    print("\n")
    print("=" * 100)
    print("LABEL COUNTS")
    print("=" * 100)

    label_counter = Counter()

    for track in root.findall(".//track"):
        label = track.attrib.get("label", "UNKNOWN")
        label_counter[label] += 1

    for box in root.findall(".//box"):
        label = box.attrib.get("label", "UNKNOWN")
        label_counter[label] += 1

    for label, count in label_counter.most_common():
        print(f"{label:25} : {count}")

    # ---------------------------------------------------------
    # TRACK INFORMATION
    # ---------------------------------------------------------

    print("\n")
    print("=" * 100)
    print("TRACK SUMMARY")
    print("=" * 100)

    tracks = root.findall(".//track")

    print(f"Total tracks: {len(tracks)}")

    track_by_label = Counter()
    frames_by_label = defaultdict(int)

    for track in tracks:
        label = track.attrib.get("label", "UNKNOWN")
        track_by_label[label] += 1

        boxes = track.findall("./box")
        frames_by_label[label] += len(boxes)

    print()

    for label in sorted(track_by_label):
        print(
            f"{label:25} : "
            f"{track_by_label[label]:6} tracks | "
            f"{frames_by_label[label]:7} annotated boxes"
        )

    # ---------------------------------------------------------
    # FRAME RANGES
    # ---------------------------------------------------------

    print("\n")
    print("=" * 100)
    print("ANNOTATION FRAME RANGES BY TASK")
    print("=" * 100)

    for i, task in enumerate(tasks, 1):

        task_name = task.findtext("name") or f"TASK_{i}"

        # Find tracks belonging to this task.
        # Depending on CVAT export structure, tracks may be nested
        # under annotations rather than directly under task.
        task_tracks = task.findall(".//track")

        frame_ranges = []

        for track in task_tracks:
            boxes = track.findall("./box")

            if not boxes:
                continue

            frames = []

            for box in boxes:
                try:
                    frames.append(int(box.attrib["frame"]))
                except (KeyError, ValueError):
                    pass

            if frames:
                frame_ranges.append(
                    (
                        track.attrib.get("label", "UNKNOWN"),
                        min(frames),
                        max(frames),
                        len(frames),
                    )
                )

        print(f"\n{task_name}")
        print(f"  Tracks: {len(task_tracks)}")

        if frame_ranges:
            for label, first, last, count in frame_ranges[:20]:
                print(
                    f"    {label:20} "
                    f"frames {first:6} -> {last:6} "
                    f"({count} boxes)"
                )

            if len(frame_ranges) > 20:
                print(
                    f"    ... {len(frame_ranges) - 20} more tracks"
                )
        else:
            print("  No track frame information found.")

    # ---------------------------------------------------------
    # TOP-LEVEL STRUCTURE
    # ---------------------------------------------------------

    print("\n")
    print("=" * 100)
    print("TOP-LEVEL XML STRUCTURE")
    print("=" * 100)

    for child in root:
        print(f"  {child.tag}: {child.attrib}")

    # ---------------------------------------------------------
    # POTENTIAL VIDEO / FILE REFERENCES
    # ---------------------------------------------------------

    print("\n")
    print("=" * 100)
    print("POSSIBLE VIDEO / FILE REFERENCES")
    print("=" * 100)

    keywords = (
        "source",
        "file",
        "filename",
        "name",
        "path",
        "url",
        "video",
    )

    found = set()

    for element in root.iter():
        for key, value in element.attrib.items():
            key_lower = key.lower()

            if any(keyword in key_lower for keyword in keywords):
                item = f"{element.tag} @{key} = {value}"
                found.add(item)

        if element.text:
            text = element.text.strip()

            if text and len(text) < 500:
                parent = element.tag.lower()

                if any(keyword in parent for keyword in keywords):
                    found.add(f"{element.tag} = {text}")

    for item in sorted(found):
        print(f"  {item}")

    print("\n")
    print("=" * 100)
    print("INSPECTION COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()