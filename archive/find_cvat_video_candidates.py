import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict


DB_PATH = Path("database/eagle_dataset.db")
XML_PATH = Path("Annotations/cvat_export.xml")


def environment(name):

    name = name.lower()

    if "lamella chain" in name:
        return "24320003"

    if "magazine" in name:
        return "24320015"

    if "collecting table" in name:
        return "24320021"

    # Based on the environment information established earlier.
    # These two should be verified before final manifest creation.
    if "cartoning" in name:
        return "24400019"

    if "covering" in name:
        return "24400017"

    return None


def main():

    print("=" * 110)
    print("EAGLE-MARS CVAT → ORIGINAL VIDEO CANDIDATES")
    print("=" * 110)

    # ------------------------------------------------------------
    # Load CVAT tasks
    # ------------------------------------------------------------

    root = ET.parse(XML_PATH).getroot()

    tasks = root.findall(".//task")

    # ------------------------------------------------------------
    # Load database
    # ------------------------------------------------------------

    conn = sqlite3.connect(DB_PATH)

    rows = conn.execute("""
        SELECT
            id,
            full_path,
            duration_seconds,
            fps,
            frame_count,
            width,
            height
        FROM videos
        WHERE readable = 1
        ORDER BY id
    """).fetchall()

    conn.close()

    # ------------------------------------------------------------
    # Build candidates
    # ------------------------------------------------------------

    for task in tasks:

        task_id = task.findtext("id")
        name = task.findtext("name")

        machine_id = environment(name)

        if machine_id is None:
            print(f"\nWARNING: Unknown environment for {name}")
            continue

        # Determine expected CVAT task frame count
        size = task.findtext("size")

        try:
            expected_frames = int(size)
        except (TypeError, ValueError):
            expected_frames = None

        candidates = []

        for row in rows:

            video_id, full_path, duration, fps, frame_count, width, height = row

            # Environment / machine check
            if machine_id not in full_path:
                continue

            # Resolution check
            if width != 1280 or height != 720:
                continue

            # Frame count is the strongest first filter
            if expected_frames is not None:

                if frame_count != expected_frames:
                    continue

            candidates.append(row)

        print("\n" + "=" * 110)
        print(f"TASK       : {task_id}")
        print(f"NAME       : {name}")
        print(f"MACHINE ID : {machine_id}")
        print(f"CVAT SIZE  : {expected_frames}")
        print(f"CANDIDATES : {len(candidates)}")
        print("-" * 110)

        # Show only first 20 candidates
        for row in candidates[:20]:

            video_id, full_path, duration, fps, frame_count, width, height = row

            folder = Path(full_path).parent.name

            print(
                f"DB ID={video_id:<5} "
                f"frames={frame_count:<5} "
                f"fps={fps:<8.3f} "
                f"folder={folder}"
            )

        if len(candidates) > 20:
            print(
                f"... {len(candidates) - 20} additional candidates"
            )

    print("\n")
    print("=" * 110)
    print("CANDIDATE SEARCH COMPLETE")
    print("=" * 110)


if __name__ == "__main__":
    main()