import sqlite3
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


DB_PATH = Path("database/eagle_dataset.db")
XML_PATH = Path("Annotations/cvat_export.xml")


def get_environment(name):

    name = name.lower()

    if "lamella chain" in name:
        return "Lamella Chain"

    if "magazine" in name:
        return "Magazine"

    if "collecting table" in name:
        return "Collecting Table"

    if "cartoning" in name:
        return "Cartoning"

    if "covering" in name:
        return "Covering"

    return "UNKNOWN"


def main():

    print("=" * 110)
    print("EAGLE-MARS CVAT ↔ ORIGINAL VIDEO MAPPING")
    print("=" * 110)

    # ------------------------------------------------------------
    # Load CVAT
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

    print("\nReadable videos:", len(rows))

    # ------------------------------------------------------------
    # Determine machine ID
    # ------------------------------------------------------------

    machine_videos = defaultdict(list)

    for row in rows:

        video_id, full_path, duration, fps, frame_count, width, height = row

        parts = full_path.replace("/", "\\").split("\\")

        machine_id = None

        for part in parts:

            if part.isdigit() and len(part) == 8:
                machine_id = part
                break

        if machine_id:

            machine_videos[machine_id].append(row)

    # ------------------------------------------------------------
    # Print CVAT tasks
    # ------------------------------------------------------------

    print("\nCVAT TASKS")
    print("-" * 110)

    for task in tasks:

        task_id = task.findtext("id")
        name = task.findtext("name")

        env = get_environment(name)

        print(
            f"TASK {task_id:<8} "
            f"{name:<22} "
            f"ENV={env}"
        )

    # ------------------------------------------------------------
    # Print database machine information
    # ------------------------------------------------------------

    print("\n\nDATABASE MACHINE COUNTS")
    print("-" * 110)

    for machine_id, videos in sorted(machine_videos.items()):

        total_frames = sum(v[4] for v in videos)

        print(
            f"{machine_id}: "
            f"{len(videos):4d} videos | "
            f"{total_frames:8d} frames"
        )

    # ------------------------------------------------------------
    # Print first videos from each machine
    # ------------------------------------------------------------

    print("\n\nFIRST VIDEOS PER MACHINE")
    print("-" * 110)

    for machine_id in sorted(machine_videos):

        print(f"\nMACHINE {machine_id}")

        for row in machine_videos[machine_id][:10]:

            video_id, full_path, duration, fps, frame_count, width, height = row

            folder = Path(full_path).parent.name

            print(
                f"  DB ID={video_id:<5} "
                f"frames={frame_count:<5} "
                f"folder={folder}"
            )

    print("\n")
    print("=" * 110)
    print("MAPPING DIAGNOSTIC COMPLETE")
    print("=" * 110)


if __name__ == "__main__":
    main()