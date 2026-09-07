import sqlite3
from pathlib import Path

DB_PATH = Path("database/eagle_dataset.db")


def print_section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


connection = sqlite3.connect(DB_PATH)
cursor = connection.cursor()


# ---------------------------------------------------------
# BASIC DATABASE STATISTICS
# ---------------------------------------------------------

print_section("BASIC DATASET STATISTICS")

cursor.execute("SELECT COUNT(*) FROM videos")
print("Total videos:", cursor.fetchone()[0])

cursor.execute("""
    SELECT COUNT(*)
    FROM videos
    WHERE readable = 1
""")
print("Readable:", cursor.fetchone()[0])

cursor.execute("""
    SELECT COUNT(*)
    FROM videos
    WHERE readable = 0
""")
print("Unreadable:", cursor.fetchone()[0])


# ---------------------------------------------------------
# TOP-LEVEL FOLDERS
# ---------------------------------------------------------

print_section("TOP-LEVEL FOLDERS")

cursor.execute("""
    SELECT
        CASE
            WHEN instr(relative_path, '\\') > 0
            THEN substr(relative_path, 1, instr(relative_path, '\\') - 1)
            ELSE relative_path
        END AS top_folder,
        COUNT(*) AS video_count
    FROM videos
    GROUP BY top_folder
    ORDER BY video_count DESC
""")

for folder, count in cursor.fetchall():
    print(f"{count:6}  {folder}")


# ---------------------------------------------------------
# SOURCE FOLDERS
# ---------------------------------------------------------

print_section("MOST COMMON SOURCE FOLDERS")

cursor.execute("""
    SELECT source_folder, COUNT(*) AS count
    FROM videos
    GROUP BY source_folder
    ORDER BY count DESC
    LIMIT 50
""")

for folder, count in cursor.fetchall():
    print(f"{count:6}  {folder}")


# ---------------------------------------------------------
# DURATIONS
# ---------------------------------------------------------

print_section("DURATION STATISTICS")

cursor.execute("""
    SELECT
        MIN(duration_seconds),
        MAX(duration_seconds),
        AVG(duration_seconds)
    FROM videos
    WHERE readable = 1
""")

minimum, maximum, average = cursor.fetchone()

print(f"Minimum duration : {minimum:.3f} sec")
print(f"Maximum duration : {maximum:.3f} sec")
print(f"Average duration : {average:.3f} sec")


# ---------------------------------------------------------
# DURATION BUCKETS
# ---------------------------------------------------------

print_section("DURATION DISTRIBUTION")

cursor.execute("""
    SELECT
        CASE
            WHEN duration_seconds < 5 THEN '<5 sec'
            WHEN duration_seconds < 10 THEN '5-10 sec'
            WHEN duration_seconds < 15 THEN '10-15 sec'
            WHEN duration_seconds < 30 THEN '15-30 sec'
            WHEN duration_seconds < 60 THEN '30-60 sec'
            ELSE '>60 sec'
        END AS duration_group,
        COUNT(*)
    FROM videos
    WHERE readable = 1
    GROUP BY duration_group
    ORDER BY
        CASE duration_group
            WHEN '<5 sec' THEN 1
            WHEN '5-10 sec' THEN 2
            WHEN '10-15 sec' THEN 3
            WHEN '15-30 sec' THEN 4
            WHEN '30-60 sec' THEN 5
            ELSE 6
        END
""")

for group, count in cursor.fetchall():
    print(f"{count:6}  {group}")


# ---------------------------------------------------------
# FPS
# ---------------------------------------------------------

print_section("FRAME RATE DISTRIBUTION")

cursor.execute("""
    SELECT ROUND(fps, 2), COUNT(*)
    FROM videos
    WHERE readable = 1
    GROUP BY ROUND(fps, 2)
    ORDER BY COUNT(*) DESC
""")

for fps, count in cursor.fetchall():
    print(f"{count:6}  {fps} FPS")


# ---------------------------------------------------------
# RESOLUTION
# ---------------------------------------------------------

print_section("RESOLUTION DISTRIBUTION")

cursor.execute("""
    SELECT width, height, COUNT(*)
    FROM videos
    WHERE readable = 1
    GROUP BY width, height
    ORDER BY COUNT(*) DESC
""")

for width, height, count in cursor.fetchall():
    print(f"{count:6}  {width}x{height}")


# ---------------------------------------------------------
# CODECS
# ---------------------------------------------------------

print_section("CODEC DISTRIBUTION")

cursor.execute("""
    SELECT codec, COUNT(*)
    FROM videos
    WHERE readable = 1
    GROUP BY codec
    ORDER BY COUNT(*) DESC
""")

for codec, count in cursor.fetchall():
    print(f"{count:6}  {codec}")


# ---------------------------------------------------------
# UNREADABLE FILES
# ---------------------------------------------------------

print_section("UNREADABLE VIDEOS")

cursor.execute("""
    SELECT full_path, scan_error
    FROM videos
    WHERE readable = 0
""")

rows = cursor.fetchall()

if not rows:
    print("None")

else:
    for path, error in rows:
        print()
        print("FILE:")
        print(path)
        print("ERROR:")
        print(error)


connection.close()

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)