import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path
import csv

# ============================================================
# CONFIG
# ============================================================

CVAT_XML = Path("Annotations/cvat_export.xml")
DB_PATH = Path("database/eagle_dataset.db")
OUTPUT_CSV = Path("outputs/cvat_frame_video_mapping.csv")

# ============================================================
# READ DATABASE VIDEOS
# ============================================================

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("SELECT * FROM videos")
rows = cur.fetchall()

columns = [desc[0] for desc in cur.description]

print(f"Database videos found: {len(rows)}")
print("Columns:", columns)

# Find useful columns dynamically
def find_column(possible_names):
    for name in possible_names:
        for col in columns:
            if col.lower() == name.lower():
                return col
    return None

id_col = find_column(["id", "video_id"])
filename_col = find_column(["filename", "file_name", "name"])
frame_count_col = find_column(["frame_count", "frames", "num_frames"])

if not id_col or not filename_col or not frame_count_col:
    raise RuntimeError(
        f"Could not identify required columns.\n"
        f"id={id_col}, filename={filename_col}, frame_count={frame_count_col}"
    )

id_idx = columns.index(id_col)
filename_idx = columns.index(filename_col)
frame_count_idx = columns.index(frame_count_col)

# ============================================================
# BUILD CUMULATIVE VIDEO FRAME RANGES
# ============================================================

video_ranges = []

global_start = 0

for row in rows:
    video_id = row[id_idx]
    filename = row[filename_idx]
    frame_count = row[frame_count_idx]

    if frame_count is None:
        continue

    frame_count = int(frame_count)

    global_end = global_start + frame_count - 1

    video_ranges.append({
        "video_id": video_id,
        "filename": filename,
        "frame_count": frame_count,
        "global_start": global_start,
        "global_end": global_end,
    })

    global_start = global_end + 1

print(f"Total mapped database frames: {global_start}")

# ============================================================
# READ CVAT XML
# ============================================================

tree = ET.parse(CVAT_XML)
root = tree.getroot()

mapping = []

# CVAT annotations are stored inside <task> elements
for task in root.findall(".//task"):

    task_id = task.findtext("id")
    task_name = task.findtext("name")

    labels = set()

    # Find the corresponding annotations section.
    # We search by task ID in the XML structure.
    task_id_int = int(task_id) if task_id else None

    # --------------------------------------------------------
    # Find annotations belonging to this task
    # --------------------------------------------------------

    # CVAT XML exports may contain image tags with task/global
    # frame references. We therefore inspect all image tags.
    for image in root.findall(".//image"):

        frame_text = image.get("id")

        if frame_text is None:
            continue

        cvat_frame = int(frame_text)

        for box in image.findall(".//box"):

            label = box.get("label", "")

            # Find database video containing this global frame
            matched_video = None

            for video in video_ranges:
                if (
                    video["global_start"]
                    <= cvat_frame
                    <= video["global_end"]
                ):
                    matched_video = video
                    break

            if matched_video is None:
                mapping.append({
                    "cvat_task_id": task_id,
                    "cvat_task_name": task_name,
                    "cvat_frame": cvat_frame,
                    "label": label,
                    "video_id": "",
                    "filename": "",
                    "video_frame": "",
                    "status": "NO_VIDEO_MATCH",
                })
                continue

            video_frame = (
                cvat_frame
                - matched_video["global_start"]
            )

            mapping.append({
                "cvat_task_id": task_id,
                "cvat_task_name": task_name,
                "cvat_frame": cvat_frame,
                "label": label,
                "video_id": matched_video["video_id"],
                "filename": matched_video["filename"],
                "video_frame": video_frame,
                "status": "MATCHED",
            })

# ============================================================
# WRITE CSV
# ============================================================

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    fieldnames = [
        "cvat_task_id",
        "cvat_task_name",
        "cvat_frame",
        "label",
        "video_id",
        "filename",
        "video_frame",
        "status",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(mapping)

print()
print("======================================")
print("Mapping complete")
print("======================================")
print(f"CVAT annotation rows: {len(mapping)}")
print(f"Output: {OUTPUT_CSV}")

matched = sum(
    1 for x in mapping
    if x["status"] == "MATCHED"
)

unmatched = sum(
    1 for x in mapping
    if x["status"] != "MATCHED"
)

print(f"Matched:   {matched}")
print(f"Unmatched: {unmatched}")

conn.close()