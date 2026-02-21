"""
📝 CENTRALIZED LOGGING SYSTEM
==============================
Standardized logging for all Python services

Features:
- Colored console output
- File rotation
- JSON structured logging
- Performance tracking
"""

import logging
import sys
from datetime import datetime
from typing import Optional
import json


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }

    def format(self, record):
        # Add color
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"

        # Format timestamp
        record.asctime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        return super().format(record)


class JSONFormatter(logging.Formatter):
    """Formatter for JSON structured logging"""

    def format(self, record):
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'level': record.levelname,
            'service': getattr(record, 'service', 'unknown'),
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # Add exception info if available
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def get_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False
) -> logging.Logger:
    """
    Get or create a logger instance

    Args:
        name: Logger name (usually service name)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for file logging
        json_format: Use JSON format instead of colored format

    Returns:
        logging.Logger instance
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper()))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))

    if json_format:
        console_formatter = JSONFormatter()
    else:
        console_formatter = ColoredFormatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
        )

    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        from logging.handlers import RotatingFileHandler

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)  # Log everything to file

        file_formatter = JSONFormatter()
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


class PerformanceLogger:
    """Context manager for performance logging"""

    def __init__(self, logger: logging.Logger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.debug(f"⏱️  Starting: {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds() * 1000
        if exc_type:
            self.logger.error(f"❌ Failed: {self.operation} ({duration:.2f}ms) - {exc_val}")
        else:
            self.logger.debug(f"✅ Completed: {self.operation} ({duration:.2f}ms)")


# Example usage
if __name__ == "__main__":
    # Create logger
    logger = get_logger("test-service", level="DEBUG")

    # Test logging levels
    logger.debug("🔍 Debug message")
    logger.info("✅ Info message")
    logger.warning("⚠️  Warning message")
    logger.error("❌ Error message")
    logger.critical("🚨 Critical message")

    # Test performance logging
    with PerformanceLogger(logger, "Test operation"):
        import time
        time.sleep(0.1)

    print("\n✅ Logger test completed")
