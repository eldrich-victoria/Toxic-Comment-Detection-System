import sqlite3

conn = sqlite3.connect("database/toxic_comments_benchmark.db")

with open("database/schema.sql", "r", encoding="utf-8") as f:
    conn.executescript(f.read())

conn.commit()
conn.close()

print("Database initialized successfully")