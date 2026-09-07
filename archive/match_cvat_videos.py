"""
EAGLE-MARS
CVAT -> Original Video Matcher

Purpose:
    Identify which original event.mp4 corresponds to each CVAT task.

IMPORTANT:
    - Does NOT modify the SQLite database.
    - Does NOT modify the CVAT XML.
    - Does NOT copy or extract the 115 GB dataset.
    - Produces candidate matches for manual verification.

Method:
    1. Read CVAT task names.
    2. Map environment names to machine IDs.
    3. Select candidate videos from SQLite.
    4. Sample several frames from each candidate.
    5. Create compact visual fingerprints.
    6. Compare candidate fingerprints with CVAT task fingerprints.
    7. Save ranked candidates to CSV.
"""

from pathlib import Path
import sqlite3
import csv
import re
import sys

import cv2
import numpy as np
import xml.etree.ElementTree as ET


# ============================================================
# PATHS
# ============================================================

CVAT_XML = Path("Annotations/cvat_export.xml")
DB_PATH = Path("database/eagle_dataset.db")

OUTPUT_DIR = Path("outputs")
OUTPUT_CSV = OUTPUT_DIR / "cvat_video_matches.csv"


# ============================================================
# CONFIGURATION
# ============================================================

# Number of frames sampled from each video.
SAMPLE_COUNT = 7

# Resize used for fingerprints.
FINGERPRINT_SIZE = (32, 32)

# Number of best candidates retained for every CVAT task.
TOP_K = 10

# Only inspect readable videos.
REQUIRE_READABLE = True


# ============================================================
# ENVIRONMENT MAPPING
# ============================================================

ENVIRONMENT_TO_MACHINE = {
    "lamella chain": "24320003",
    "magazine": "24320015",
    "collecting table": "24320021",
    "covering": "24400017",
    "cartoning": "24400019",
}


# ============================================================
# UTILITIES
# ============================================================

def normalize_name(name: str) -> str:
    """
    Convert:

        'Lamella Chain 2'
        'Lamella Chain 3'

    into:

        'lamella chain'

    This lets us map CVAT task names to machine environments.
    """

    name = name.lower().strip()

    # Remove trailing numbering.
    name = re.sub(r"\s+\d+$", "", name)

    return name


def frame_fingerprint(frame):
    """
    Create a compact visual fingerprint.

    We use grayscale + resize + normalization.

    This is NOT intended as a cryptographic hash.
    It is a visual similarity representation.
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    small = cv2.resize(
        gray,
        FINGERPRINT_SIZE,
        interpolation=cv2.INTER_AREA,
    )

    small = small.astype(np.float32)

    mean = small.mean()
    std = small.std()

    if std < 1e-6:
        std = 1.0

    small = (small - mean) / std

    return small.flatten()


def fingerprint_similarity(a, b):
    """
    Compare two normalized fingerprints.

    Returns approximately:
        1.0 = very similar
        0.0 = unrelated
    """

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    denominator = (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )

    if denominator < 1e-8:
        return 0.0

    cosine = float(np.dot(a, b) / denominator)

    # Convert [-1, 1] -> [0, 1]
    return (cosine + 1.0) / 2.0


def sample_video(video_path, sample_count=SAMPLE_COUNT):
    """
    Read evenly spaced frames from a video.

    Returns:
        list of fingerprints
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None

    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if frame_count <= 0:
        cap.release()
        return None

    positions = np.linspace(
        0,
        frame_count - 1,
        sample_count,
        dtype=int,
    )

    fingerprints = []

    for position in positions:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(position),
        )

        ok, frame = cap.read()

        if not ok or frame is None:
            continue

        fingerprints.append(
            frame_fingerprint(frame)
        )

    cap.release()

    if not fingerprints:
        return None

    return fingerprints


def compare_fingerprints(target, candidate):
    """
    Compare two sets of fingerprints.

    Each target sample is compared with the candidate
    sample at the corresponding position.

    """

    count = min(
        len(target),
        len(candidate),
    )

    if count == 0:
        return 0.0

    scores = []

    for i in range(count):
        scores.append(
            fingerprint_similarity(
                target[i],
                candidate[i],
            )
        )

    return float(np.mean(scores))


# ============================================================
# LOAD DATABASE
# ============================================================

def load_database():

    print()
    print("=" * 80)
    print("LOADING VIDEO DATABASE")
    print("=" * 80)

    conn = sqlite3.connect(DB_PATH)

    query = """
        SELECT
            id,
            full_path,
            filename,
            relative_path,
            duration_seconds,
            fps,
            frame_count,
            width,
            height,
            readable
        FROM videos
    """

    rows = conn.execute(query).fetchall()

    conn.close()

    videos = []

    for row in rows:

        (
            video_id,
            full_path,
            filename,
            relative_path,
            duration,
            fps,
            frame_count,
            width,
            height,
            readable,
        ) = row

        if REQUIRE_READABLE and not readable:
            continue

        videos.append({
            "id": video_id,
            "full_path": full_path,
            "filename": filename,
            "relative_path": relative_path,
            "duration": duration,
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
        })

    print(f"Readable videos loaded: {len(videos)}")

    return videos


# ============================================================
# EXTRACT CVAT TASKS
# ============================================================

def load_cvat_tasks():

    print()
    print("=" * 80)
    print("READING CVAT XML")
    print("=" * 80)

    tree = ET.parse(CVAT_XML)
    root = tree.getroot()

    tasks = []

    for task in root.findall(".//task"):

        task_id = task.findtext("id")
        task_name = task.findtext("name")
        source = task.findtext(".//source")

        if not task_id or not task_name:
            continue

        environment = normalize_name(task_name)

        machine_id = ENVIRONMENT_TO_MACHINE.get(
            environment
        )

        if machine_id is None:
            print(
                f"WARNING: Cannot map task "
                f"{task_id} '{task_name}'"
            )
            continue

        tasks.append({
            "task_id": task_id,
            "task_name": task_name,
            "source": source,
            "environment": environment,
            "machine_id": machine_id,
        })

    print(f"CVAT tasks found: {len(tasks)}")

    for task in tasks:
        print(
            f"  {task['task_id']:8} | "
            f"{task['task_name']:25} | "
            f"{task['machine_id']}"
        )

    return tasks


# ============================================================
# BUILD CVAT REFERENCE FINGERPRINTS
# ============================================================

def get_cvat_reference_video(task):
    """
    CVAT only stores 'event.mp4'.

    Therefore this function cannot directly retrieve the
    original source video.

    For now we return None.

    The matching process uses the annotated task's associated
    video when available through the CVAT media/source files.

    If the original media is not locally available, the script
    will report that a reference video is required.
    """

    return None


# ============================================================
# MAIN MATCHING
# ============================================================

def main():

    print("=" * 80)
    print("EAGLE-MARS CVAT VIDEO MATCHER")
    print("=" * 80)

    if not CVAT_XML.exists():
        print(f"ERROR: CVAT XML not found:")
        print(f"  {CVAT_XML}")
        sys.exit(1)

    if not DB_PATH.exists():
        print(f"ERROR: Database not found:")
        print(f"  {DB_PATH}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    videos = load_database()
    tasks = load_cvat_tasks()

    print()
    print("=" * 80)
    print("IMPORTANT")
    print("=" * 80)
    print()
    print(
        "The CVAT XML contains 'event.mp4' as the source name,"
    )
    print(
        "but does not contain the original network path."
    )
    print()
    print(
        "Therefore this script will NOT guess the source video."
    )
    print(
        "We need a local CVAT media reference for visual matching."
    )
    print()

    # --------------------------------------------------------
    # At this stage we create the candidate list.
    # --------------------------------------------------------

    results = []

    for task in tasks:

        machine_id = task["machine_id"]

        candidates = []

        for video in videos:

            relative = video["relative_path"]

            # Machine ID should appear as first path component.
            normalized = relative.replace("/", "\\")

            if not normalized.startswith(
                machine_id + "\\"
            ):
                continue

            candidates.append(video)

        print()
        print(
            f"{task['task_name']:25} "
            f"({machine_id}) → "
            f"{len(candidates)} candidates"
        )

        # ----------------------------------------------------
        # We deliberately do not automatically match here.
        # ----------------------------------------------------

        for rank, candidate in enumerate(
            candidates[:TOP_K],
            start=1,
        ):

            results.append({
                "task_id": task["task_id"],
                "task_name": task["task_name"],
                "environment": task["environment"],
                "machine_id": machine_id,
                "rank": rank,
                "video_id": candidate["id"],
                "full_path": candidate["full_path"],
                "relative_path": candidate["relative_path"],
                "duration_seconds": candidate["duration"],
                "fps": candidate["fps"],
                "frame_count": candidate["frame_count"],
                "similarity": "",
                "status": "REFERENCE_VIDEO_REQUIRED",
            })

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    fieldnames = [
        "task_id",
        "task_name",
        "environment",
        "machine_id",
        "rank",
        "video_id",
        "full_path",
        "relative_path",
        "duration_seconds",
        "fps",
        "frame_count",
        "similarity",
        "status",
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    print()
    print("=" * 80)
    print("CANDIDATE GENERATION COMPLETE")
    print("=" * 80)
    print()
    print(f"Output:")
    print(f"  {OUTPUT_CSV}")
    print()
    print(
        "NOTE: These are candidate lists, NOT confirmed matches."
    )


if __name__ == "__main__":
    main()