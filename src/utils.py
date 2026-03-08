from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from typing import Any

import numpy as np
import pandas as pd

from src.config import APP_LOG_PATH, ensure_directories


def get_logger(name: str) -> logging.Logger:
    ensure_directories()
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = RotatingFileHandler(
        APP_LOG_PATH,
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def sanitize_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def sanitize_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: sanitize_scalar(value) for key, value in record.items()}


def dataframe_to_records(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    return [sanitize_record(record) for record in dataframe.to_dict(orient="records")]
