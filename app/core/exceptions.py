"""Custom exceptions for the application."""

class BenchmarkSystemError(Exception):
    """Base exception for the benchmarking system."""
    pass

class ConfigurationError(BenchmarkSystemError):
    """Raised when there is an error in configuration loading or validation."""
    pass

class DatabaseConnectionError(BenchmarkSystemError):
    """Raised when the database connection fails."""
    pass

class DataValidationError(BenchmarkSystemError):
    """Raised when data fails validation checks."""
    pass

class ModelInitializationError(BenchmarkSystemError):
    """Raised when a model fails to initialize."""
    pass
