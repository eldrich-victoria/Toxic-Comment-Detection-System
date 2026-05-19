"""Storage manager for handling database operations, CSV, and JSON exports."""
import aiosqlite
import csv
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from app.core.path_manager import PathManager
from app.core.logger import logger

class StorageManager:
    """Handles transaction-safe database writes and data exports."""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or (PathManager.DATABASE_DIR / "toxic_comments_benchmark.db")

    async def init_db(self):
        """Initialize database with schema."""
        schema_path = PathManager.DATABASE_DIR / "schema.sql"
        if not schema_path.exists():
            logger.error("Schema file not found.")
            return
            
        async with aiosqlite.connect(self.db_path) as db:
            with open(schema_path, "rt", encoding="utf-8") as f:
                schema_script = f.read()
            await db.executescript(schema_script)
            await db.commit()
            logger.info("Database initialized successfully.")

    async def insert_benchmark_run(self, run_name: str, config_snapshot: str) -> int:
        """Insert a new benchmark run and return its ID."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "INSERT INTO benchmark_runs (run_name, config_snapshot, status) VALUES (?, ?, ?)",
                (run_name, config_snapshot, 'STARTED')
            )
            await db.commit()
            return cursor.lastrowid

    async def complete_benchmark_run(self, run_id: int, status: str = 'COMPLETED'):
        """Mark a benchmark run as completed or failed."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE benchmark_runs SET status = ?, end_time = CURRENT_TIMESTAMP WHERE id = ?",
                (status, run_id)
            )
            await db.commit()

    async def save_predictions_batch(self, predictions: List[Dict[str, Any]]):
        """
        Save a batch of predictions to the SQLite database safely.
        """
        if not predictions:
            return
            
        query = """
            INSERT INTO model_predictions (
                run_id, model_name, model_version, sample_id, text_input, 
                normalized_text, ground_truth, prediction, confidence, latency_ms, xai_placeholder
            ) VALUES (
                :run_id, :model_name, :model_version, :sample_id, :text_input,
                :normalized_text, :ground_truth, :prediction, :confidence, :latency_ms, :xai_placeholder
            )
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.executemany(query, predictions)
            await db.commit()
            logger.info(f"Saved batch of {len(predictions)} predictions to database.")

    def export_to_csv(self, predictions: List[Dict[str, Any]], filename: str):
        """Export predictions to a CSV file."""
        if not predictions:
            return
            
        export_path = PathManager.BENCHMARK_RESULTS_DIR / filename
        fieldnames = predictions[0].keys()
        
        try:
            with open(export_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(predictions)
            logger.info(f"Exported {len(predictions)} rows to CSV: {export_path}")
        except Exception as e:
            logger.error(f"Failed to export CSV to {export_path}: {e}")

    def export_to_json(self, data: Any, filename: str):
        """Export data to a JSON file."""
        export_path = PathManager.BENCHMARK_RESULTS_DIR / filename
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            logger.info(f"Exported data to JSON: {export_path}")
        except Exception as e:
            logger.error(f"Failed to export JSON to {export_path}: {e}")
