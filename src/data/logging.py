import logging
from typing import Optional


def configure_logging(level: int = logging.INFO, name: Optional[str] = None) -> logging.Logger:
    logger_name = name or "fraud_data"
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
