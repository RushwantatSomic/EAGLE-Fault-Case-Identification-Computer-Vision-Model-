import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import csv


# ============================================================
# EAGLE-MARS
# Find Original Videos Used for CVAT Annotation
# ============================================================

DB_PATH = Path("database/eagle_dataset.db")
XML_PATH = Path("Annotations/cvat_export.xml")
OUTPUT_DIR = Path("outputs/cvat_mapping")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Environment mapping
# ------------------------------------------------------------

ENVIRONMENT_TO_MACHINE = {
    "Lamella Chain": "24320003",
    "Magazine": "24320015",
    "Collecting Table": "24320021",

    # Based on the mapping established in this project.
    "Cartoning": "24400019",
    "Covering": "24400017",
}


# ------------------------------------------------------------
# CVAT task name -> environment
# ------------------------------------------------------------

def get_environment(task_name):

    # Important: test longer names first.
    if "Lamella Chain" in task_name:
        return "Lamella Chain"

    if "Collecting Table" in task_name:
        return "Collecting Table"

    if "Magazine" in task_name:
        return "Magazine"

    if "Cartoning" in task_name:
        return "Cartoning"

    if "Covering" in task_name:
        return "Covering"

    return None


# ------------------------------------------------------------
# Extract frame numbers from a CVAT task
# ------------------------------------------------------------

def get_task_frames(root, task_id):

    frames = []

    for track in root.findall(".//track"):

        if track.attrib.get("task_id") != task_id:
            continue

        for box in track.findall("./box"):

            try:
                frame = int(box.attrib["frame"])
                frames.append(frame)
            except (KeyError, ValueError):
                pass

    return sorted(set(frames))


# ------------------------------------------------------------
# Determine maximum annotated frame
# ------------------------------------------------------------

def get_max_annotated_frame(root, task_id):

    frames = get_task_frames(root, task_id)

    if not frames:
        return None

    return max(frames)


# ------------------------------------------------------------
# Calculate a simple visual fingerprint
# ------------------------------------------------------------

def frame_fingerprint(frame):

    # Resize for fast comparison
    small = cv2.resize(frame, (64, 36))

    # Convert to grayscale
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # Normalize brightness
    gray = cv2.normalize(
        gray,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return gray.astype(np.float32)


# ------------------------------------------------------------
# Compare two fingerprints
# ------------------------------------------------------------

def fingerprint_similarity(a, b):

    if a.shape != b.shape:
        return 0.0

    # Mean absolute difference
    diff = np.mean(np.abs(a - b))

    # Convert difference to similarity
    similarity = 1.0 - (diff / 255.0)

    return float(similarity)


# ------------------------------------------------------------
# Get representative frames from CVAT annotations
#
# NOTE:
# The XML does NOT contain the original image.
# We therefore use annotated frame numbers to identify
# candidate frames in the original videos.
# ------------------------------------------------------------

def representative_frames(root, task_id):

    frames = get_task_frames(root, task_id)

    if not frames:
        return []

    # Select beginning / middle / end of annotated range.
    selected = []

    positions = [
        0.05,
        0.25,
        0.50,
        0.75,
        0.95,
    ]

    for p in positions:

        index = int(p * (len(frames) - 1))
        frame = frames[index]

        if frame not in selected:
            selected.append(frame)

    return selected


# ------------------------------------------------------------
# Open video and extract selected frames
# ------------------------------------------------------------

def extract_video_fingerprints(video_path, frame_numbers):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None

    fingerprints = {}

    for frame_number in frame_numbers:

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ok, frame = cap.read()

        if not ok:
            continue

        fingerprints[frame_number] = frame_fingerprint(frame)

    cap.release()

    return fingerprints


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    print("=" * 110)
    print("EAGLE-MARS CVAT → ORIGINAL VIDEO SEARCH")
    print("=" * 110)

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}"
        )

    if not XML_PATH.exists():
        raise FileNotFoundError(
            f"CVAT XML not found: {XML_PATH}"
        )

    # --------------------------------------------------------
    # Load CVAT XML
    # --------------------------------------------------------

    root = ET.parse(XML_PATH).getroot()

    tasks = root.findall(".//task")

    print(f"\nCVAT tasks found: {len(tasks)}")

    # --------------------------------------------------------
    # Load database
    # --------------------------------------------------------

    conn = sqlite3.connect(DB_PATH)

    videos = conn.execute("""
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

    print(f"Readable original videos: {len(videos)}")

    # --------------------------------------------------------
    # Prepare CSV
    # --------------------------------------------------------

    csv_path = OUTPUT_DIR / "cvat_video_candidates.csv"

    csv_file = open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    )

    writer = csv.writer(csv_file)

    writer.writerow([
        "task_id",
        "task_name",
        "environment",
        "machine_id",
        "annotated_first_frame",
        "annotated_last_frame",
        "candidate_db_id",
        "candidate_path",
        "candidate_frame_count",
        "candidate_fps",
        "status",
    ])

    # --------------------------------------------------------
    # Process tasks
    # --------------------------------------------------------

    for task in tasks:

        task_id = task.findtext("id")
        task_name = task.findtext("name") or "UNKNOWN"

        environment = get_environment(task_name)

        print("\n" + "=" * 110)
        print(f"TASK {task_id}")
        print(f"NAME       : {task_name}")
        print(f"ENVIRONMENT: {environment}")

        if environment is None:

            print("WARNING: Environment could not be determined.")

            continue

        machine_id = ENVIRONMENT_TO_MACHINE[environment]

        frames = get_task_frames(root, task_id)

        if not frames:

            print("WARNING: No annotated frames found.")
            continue

        first_frame = min(frames)
        last_frame = max(frames)

        print(
            f"ANNOTATED RANGE: {first_frame} → {last_frame}"
        )

        # ----------------------------------------------------
        # Find database candidates
        # ----------------------------------------------------

        candidates = []

        for row in videos:

            (
                db_id,
                full_path,
                duration,
                fps,
                frame_count,
                width,
                height,
            ) = row

            # Machine/environment filter
            if machine_id not in full_path:
                continue

            # Resolution
            if width != 1280 or height != 720:
                continue

            # The candidate must contain the annotated frames.
            if frame_count <= last_frame:
                continue

            # CVAT annotations usually start at frame 0.
            if first_frame < 0:
                continue

            candidates.append(row)

        print(f"MACHINE ID : {machine_id}")
        print(f"CANDIDATES : {len(candidates)}")

        # ----------------------------------------------------
        # Print candidates
        # ----------------------------------------------------

        for row in candidates:

            (
                db_id,
                full_path,
                duration,
                fps,
                frame_count,
                width,
                height,
            ) = row

            folder = Path(full_path).parent.name

            print(
                f"  DB {db_id:<5} "
                f"{folder:<20} "
                f"{frame_count:4d} frames "
                f"{fps:7.2f} FPS"
            )

            writer.writerow([
                task_id,
                task_name,
                environment,
                machine_id,
                first_frame,
                last_frame,
                db_id,
                full_path,
                frame_count,
                fps,
                "candidate",
            ])

        # ----------------------------------------------------
        # If exactly one candidate exists
        # ----------------------------------------------------

        if len(candidates) == 1:

            row = candidates[0]

            print("\n*** UNIQUE CANDIDATE ***")

            print(
                "DB ID:",
                row[0]
            )

            print(
                "VIDEO:",
                row[1]
            )

    csv_file.close()

    print("\n")
    print("=" * 110)
    print("SEARCH COMPLETE")
    print("=" * 110)

    print(
        f"\nCandidate report:\n{csv_path}"
    )


if __name__ == "__main__":
    main()