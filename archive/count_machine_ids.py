import sqlite3

DB = "database/eagle_dataset.db"

conn = sqlite3.connect(DB)
cursor = conn.cursor()

rows = cursor.execute("""
    SELECT
        substr(relative_path, 1, instr(relative_path, '\\') - 1) AS machine_id,
        COUNT(*) AS video_count
    FROM videos
    WHERE readable = 1
    GROUP BY machine_id
    ORDER BY machine_id
""").fetchall()

print("=" * 80)
print("MACHINE ID → VIDEO COUNT")
print("=" * 80)

for machine_id, count in rows:
    print(f"{machine_id:15} : {count:5} videos")

conn.close()