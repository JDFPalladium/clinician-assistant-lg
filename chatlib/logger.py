import logging
import sys
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler


LOG_DIR = Path(os.getenv("LOG_DIR", Path(__file__).resolve().parent.parent / "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "app.log"

LOG_CONFIG = {
    "console_format": "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
    "file_format": "%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)s:%(lineno)d | %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
    "max_bytes": 5 * 1024 * 1024,  # 5MB
    "backup_count": 3,
}

_logger_cache = {}


def get_logger(name: str = "clinical-assistant") -> logging.Logger:
    """Get a configured logger instance with console and file handlers

    Args:
        name: Logger name (usually __name__ of calling module)

    Returns:
        Configured Logger instance
    """
    if name in _logger_cache:
        return _logger_cache[name]

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        _logger_cache[name] = logger
        return logger

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        LOG_CONFIG["console_format"], LOG_CONFIG["date_format"]
    )
    console_handler.setFormatter(console_formatter)

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=LOG_CONFIG["max_bytes"],
        backupCount=LOG_CONFIG["backup_count"],
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        LOG_CONFIG["file_format"], LOG_CONFIG["date_format"]
    )
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    _logger_cache[name] = logger
    return logger
