import sqlite3
from pathlib import Path
from collections import Counter

DB_PATH = Path("database/eagle_dataset.db")

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

print("=" * 80)
print("EAGLE-MARS ENVIRONMENT CHECK")
print("=" * 80)

# ---------------------------------------------------------
# LOAD ALL VIDEO PATHS
# ---------------------------------------------------------

cursor.execute("""
    SELECT full_path, duration_seconds, fps
    FROM videos
    ORDER BY full_path
""")

rows = cursor.fetchall()

# ---------------------------------------------------------
# EXTRACT MACHINE ID FROM WINDOWS PATH
# ---------------------------------------------------------

machine_videos = Counter()
machine_durations = Counter()

machine_first_date = {}
machine_last_date = {}

video_records = []

for full_path, duration, fps in rows:

    # Normalize Windows / UNC path
    normalized = full_path.replace("/", "\\")

    # Find the dataset folder "unsorted"
    parts = normalized.split("\\")

    try:
        unsorted_index = [
            p.lower() for p in parts
        ].index("unsorted")

        machine_id = parts[unsorted_index + 1]
        recording_folder = parts[unsorted_index + 2]

    except (ValueError, IndexError):
        machine_id = "UNKNOWN"
        recording_folder = ""

    machine_videos[machine_id] += 1

    if duration:
        machine_durations[machine_id] += duration

    # Recording folder format:
    # YYYYMMDD_HH-MM-SS
    if recording_folder:

        if machine_id not in machine_first_date:
            machine_first_date[machine_id] = recording_folder

        if machine_id not in machine_last_date:
            machine_last_date[machine_id] = recording_folder

        if recording_folder < machine_first_date[machine_id]:
            machine_first_date[machine_id] = recording_folder

        if recording_folder > machine_last_date[machine_id]:
            machine_last_date[machine_id] = recording_folder

    video_records.append(
        (
            machine_id,
            recording_folder,
            full_path,
            duration,
            fps,
        )
    )


# ---------------------------------------------------------
# MACHINE IDS
# ---------------------------------------------------------

print()
print("MACHINE / ENVIRONMENT IDS")
print("-" * 80)

for machine_id, count in sorted(
    machine_videos.items(),
    key=lambda x: x[0]
):
    hours = machine_durations[machine_id] / 3600

    print(
        f"{machine_id:15} : "
        f"{count:5} videos | "
        f"{hours:7.2f} hours"
    )


# ---------------------------------------------------------
# KNOWN ENVIRONMENT MAPPING
# ---------------------------------------------------------

known_mapping = {
    "24320003": "Lamella Chain",
    "24320015": "Magazine",
    "24320017": "Covering",
    "24320019": "Cartoning",
    "24320021": "Collecting Table",
}

print()
print("KNOWN ENVIRONMENT MAPPING")
print("-" * 80)

for machine_id in sorted(machine_videos):

    if machine_id in known_mapping:
        environment = known_mapping[machine_id]
        status = "CONFIRMED FROM USER"
    else:
        environment = "UNKNOWN"
        status = "REQUIRES VERIFICATION"

    print(
        f"{machine_id:15} : "
        f"{environment:20} | "
        f"{status}"
    )


# ---------------------------------------------------------
# SAMPLE PATHS FOR UNKNOWN IDS
# ---------------------------------------------------------

for machine_id in sorted(machine_videos):

    if machine_id not in known_mapping:

        print()
        print(f"SAMPLE PATHS FOR {machine_id}")
        print("-" * 80)

        count = 0

        for record in video_records:

            record_machine = record[0]

            if record_machine == machine_id:

                print(record[2])

                count += 1

                if count >= 10:
                    break


# ---------------------------------------------------------
# DATE RANGES
# ---------------------------------------------------------

print()
print("RECORDING DATE RANGES")
print("-" * 80)

for machine_id in sorted(machine_videos):

    first_date = machine_first_date.get(
        machine_id,
        "UNKNOWN"
    )

    last_date = machine_last_date.get(
        machine_id,
        "UNKNOWN"
    )

    print(
        f"{machine_id:15} : "
        f"{first_date} → {last_date}"
    )


# ---------------------------------------------------------
# CHECK FOR 24320017 / 24320019
# ---------------------------------------------------------

print()
print("EXPECTED ENVIRONMENT IDS NOT PRESENT")
print("-" * 80)

for machine_id in [
    "24320017",
    "24320019",
]:

    if machine_id not in machine_videos:

        print(
            f"{machine_id} : NOT FOUND in scanned dataset"
        )


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"Total videos analysed : {len(rows)}")
print(f"Machine IDs found     : {len(machine_videos)}")

print()
print("Environment mapping is NOT automatically assigned")
print("until the 24400017 / 24400019 IDs are verified.")

print()
print("=" * 80)
print("CHECK COMPLETE")
print("=" * 80)

conn.close()