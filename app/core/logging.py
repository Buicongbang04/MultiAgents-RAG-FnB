import logging
import sys

from app.core.config import get_settings


def setup_logging() -> None:
    """Configure simple structured-ish console logging for MVP."""

    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    # Giảm noise từ thư viện ngoài.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """Return logger by module name."""

    return logging.getLogger(name)
