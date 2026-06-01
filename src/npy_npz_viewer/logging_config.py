"""Application logging setup."""

import logging
import os


def configure_logging() -> None:
    """Configure a small, predictable logging format once."""

    level_name = os.environ.get("NPY_NPZ_VIEWER_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
