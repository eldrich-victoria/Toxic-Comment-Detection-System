"""Data validation utilities."""
from typing import Any, Dict
from pydantic import BaseModel, ValidationError
from .exceptions import DataValidationError

def validate_config(config_data: Dict[str, Any], schema_class: type[BaseModel]) -> BaseModel:
    """
    Validate dictionary data against a Pydantic schema.
    
    Args:
        config_data: Dictionary containing configuration data.
        schema_class: Pydantic BaseModel class for validation.
        
    Returns:
        Validated schema instance.
        
    Raises:
        DataValidationError: If validation fails.
    """
    try:
        return schema_class(**config_data)
    except ValidationError as e:
        raise DataValidationError(f"Configuration validation failed: {e}") from e
