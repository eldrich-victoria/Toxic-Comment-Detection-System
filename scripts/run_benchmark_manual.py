import json
import sqlite3
from app.api.main import app
from fastapi.testclient import TestClient

def main():
    client = TestClient(app)

    print("Running benchmark via TestClient...")
    with open("Jigsaw_benchmark_1000_comments_targets.csv", "rb") as f:
        response = client.post("/benchmark/run", files={"file": ("Jigsaw_benchmark_1000_comments_targets.csv", f, "text/csv")})

    print("Status Code:", response.status_code)
    try:
        print("Response:", json.dumps(response.json(), indent=2))
    except Exception as e:
        print("Raw response:", response.text)

    print("\nVerifying database...")
    conn = sqlite3.connect("database/toxic_comments_benchmark.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, run_name, start_time, status FROM benchmark_runs ORDER BY start_time DESC LIMIT 1")
    row = cursor.fetchone()
    print("Latest benchmark run:", row)

    if row and row[3] == 'COMPLETED':
        print("SUCCESS: Found completed benchmark record in the database.")
        
        # Verify 6000 predictions were saved
        cursor.execute("SELECT COUNT(*) FROM model_predictions WHERE run_id = ?", (row[0],))
        pred_count = cursor.fetchone()[0]
        print(f"Predictions saved for this run: {pred_count}")
        if pred_count == 6000:
            print("SUCCESS: Exactly 6000 predictions were saved (1000 rows x 6 models).")
        else:
            print(f"WARNING: Expected 6000 predictions, but found {pred_count}.")
    else:
        print("FAILED: Did not find the expected completed benchmark record.")

    conn.close()

if __name__ == "__main__":
    main()
