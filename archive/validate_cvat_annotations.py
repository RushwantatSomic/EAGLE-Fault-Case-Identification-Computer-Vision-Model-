import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
from pathlib import Path


XML_PATH = Path("Annotations/cvat_export.xml")


def main():

    print("=" * 100)
    print("EAGLE-MARS CVAT ANNOTATION VALIDATION")
    print("=" * 100)

    root = ET.parse(XML_PATH).getroot()

    tasks = root.findall(".//task")
    tracks = root.findall(".//track")

    print(f"\nTasks found  : {len(tasks)}")
    print(f"Tracks found : {len(tracks)}")

    # ------------------------------------------------------------------
    # Task information
    # ------------------------------------------------------------------

    task_names = {}

    for task in tasks:
        task_id = task.findtext("id")
        name = task.findtext("name")

        task_names[task_id] = name

    # ------------------------------------------------------------------
    # Group tracks by task
    # ------------------------------------------------------------------

    tracks_by_task = defaultdict(list)

    for track in tracks:
        task_id = track.attrib.get("task_id")
        tracks_by_task[task_id].append(track)

    # ------------------------------------------------------------------
    # Environment inference
    # ------------------------------------------------------------------

    def environment(name):

        name_lower = name.lower()

        if "lamella chain" in name_lower:
            return "Lamella Chain"

        if "magazine" in name_lower:
            return "Magazine"

        if "collecting table" in name_lower:
            return "Collecting Table"

        if "cartoning" in name_lower:
            return "Cartoning"

        if "covering" in name_lower:
            return "Covering"

        return "UNKNOWN"

    # ------------------------------------------------------------------
    # Process each task
    # ------------------------------------------------------------------

    for task_id, name in task_names.items():

        task_tracks = tracks_by_task.get(task_id, [])

        env = environment(name)

        print("\n" + "=" * 100)
        print(f"TASK ID       : {task_id}")
        print(f"TASK NAME     : {name}")
        print(f"ENVIRONMENT   : {env}")
        print(f"TRACKS        : {len(task_tracks)}")
        print("-" * 100)

        label_counter = Counter()
        source_counter = Counter()

        label_frames = defaultdict(list)

        for track in task_tracks:

            label = track.attrib.get("label")
            source = track.attrib.get("source")

            label_counter[label] += 1
            source_counter[source] += 1

            for box in track.findall("./box"):

                frame = box.attrib.get("frame")

                if frame is not None:
                    label_frames[label].append(int(frame))

        print("\nLABEL COUNTS")
        for label, count in label_counter.most_common():
            print(f"  {label:20s}: {count}")

        print("\nANNOTATION SOURCES")
        for source, count in source_counter.most_common():
            print(f"  {source:20s}: {count}")

        print("\nFRAME RANGES")

        for label in label_counter:

            frames = label_frames[label]

            if frames:

                print(
                    f"  {label:20s}: "
                    f"{min(frames):5d} -> {max(frames):5d} "
                    f"({len(set(frames))} unique frames)"
                )

            else:
                print(
                    f"  {label:20s}: NO FRAME DATA"
                )

        # --------------------------------------------------------------
        # Fault interval
        # --------------------------------------------------------------

        fault_start = label_frames.get("Fault Start", [])
        fault_end = label_frames.get("Fault End", [])

        print("\nFAULT INTERVAL")

        if fault_start:

            start = min(fault_start)

            if fault_end:

                end = max(fault_end)

                print(f"  Fault start frame : {start}")
                print(f"  Fault end frame   : {end}")
                print(f"  Fault duration    : {end - start + 1} frames")

                if end < start:
                    print("  !!! ERROR: Fault End occurs before Fault Start !!!")

            else:

                print(f"  Fault start frame : {start}")
                print("  Fault end         : NOT ANNOTATED")

        else:

            print("  No Fault Start annotation")

        # --------------------------------------------------------------
        # Product on floor
        # --------------------------------------------------------------

        product = label_frames.get("Product on Floor", [])

        if product:

            print("\nPRODUCT ON FLOOR")

            print(
                f"  Frames: {min(product)} -> {max(product)}"
            )

        # --------------------------------------------------------------
        # Cause annotations
        # --------------------------------------------------------------

        cause_normal = label_frames.get("Cause Normal", [])
        cause_fault = label_frames.get("Cause Fault", [])

        if cause_normal or cause_fault:

            print("\nCAUSE ANNOTATIONS")

            if cause_normal:
                print(
                    f"  Cause Normal : "
                    f"{min(cause_normal)} -> {max(cause_normal)}"
                )

            if cause_fault:
                print(
                    f"  Cause Fault  : "
                    f"{min(cause_fault)} -> {max(cause_fault)}"
                )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print("\n")
    print("=" * 100)
    print("VALIDATION COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()