import sqlite3

DB = "database/eagle_dataset.db"

conn = sqlite3.connect(DB)
cursor = conn.cursor()

print("=" * 80)
print("VIDEOS TABLE")
print("=" * 80)

columns = cursor.execute("PRAGMA table_info(videos)").fetchall()

for column in columns:
    print(column)

conn.close()