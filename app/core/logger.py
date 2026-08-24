"""Logger setup and configuration."""
import logging
import logging.config
import yaml
from pathlib import Path
from typing import Optional

from .path_manager import PathManager

def setup_logger(config_filename: str = "logging_config.yaml") -> logging.Logger:
    """
    Configure and return the application logger.
    Uses RotatingFileHandler for log rotation as specified in the yaml config.
    """
    PathManager.LOGS_DIR.mkdir(exist_ok=True)
    config_path = PathManager.CONFIG_DIR / config_filename
    
    if config_path.exists():
        with open(config_path, "rt", encoding="utf-8") as f:
            config = yaml.safe_load(f.read())
            # dynamically update log file path to ensure it uses absolute path
            if 'handlers' in config and 'file' in config['handlers']:
                config['handlers']['file']['filename'] = str(PathManager.LOGS_DIR / "system.log")
            logging.config.dictConfig(config)
    else:
        # Fallback basic configuration
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.handlers.RotatingFileHandler(
                    filename=str(PathManager.LOGS_DIR / "system.log"),
                    maxBytes=10485760,
                    backupCount=5
                )
            ]
        )
    return logging.getLogger("app")

logger = setup_logger()
