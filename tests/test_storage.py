import pytest
from pathlib import Path
from database.storage_manager import StorageManager
from app.core.path_manager import PathManager

@pytest.mark.asyncio
async def test_init_db(tmp_path):
    db_path = tmp_path / "test.db"
    manager = StorageManager(db_path=db_path)
    assert str(db_path) == str(manager.db_path)
