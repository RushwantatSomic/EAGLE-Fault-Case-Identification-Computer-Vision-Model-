import sqlite3

DB = "database/eagle_dataset.db"

conn = sqlite3.connect(DB)
cursor = conn.cursor()

print("=" * 80)
print("EAGLE-MARS DATABASE SCHEMA")
print("=" * 80)

tables = cursor.execute("""
    SELECT name
    FROM sqlite_master
    WHERE type = 'table'
    ORDER BY name
""").fetchall()

for (table_name,) in tables:
    print(f"\nTABLE: {table_name}")
    print("-" * 80)

    columns = cursor.execute(
        f"PRAGMA table_info({table_name})"
    ).fetchall()

    for col in columns:
        print(f"  {col[1]:25} {col[2]}")

conn.close()