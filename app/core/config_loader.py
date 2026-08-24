"""Configuration loading utility."""
import yaml
from typing import Dict, Any
from pathlib import Path

from .path_manager import PathManager
from .exceptions import ConfigurationError
from .logger import logger

class ConfigLoader:
    """Utility for loading YAML configurations safely."""

    @staticmethod
    def load_yaml(filename: str) -> Dict[str, Any]:
        """
        Load a YAML configuration file from the configs directory.
        
        Args:
            filename: The name of the config file.
            
        Returns:
            A dictionary containing the parsed YAML data.
            
        Raises:
            ConfigurationError: If the file is not found or invalid YAML.
        """
        config_path = PathManager.CONFIG_DIR / filename
        if not config_path.exists():
            error_msg = f"Configuration file not found: {config_path}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg)
            
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return data if data is not None else {}
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse YAML file {filename}: {e}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg) from e
