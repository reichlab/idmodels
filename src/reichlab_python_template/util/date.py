import datetime

import structlog

logger = structlog.get_logger()


def get_current_date() -> str:
    """Return current date in human-readable format."""

    logger.info("getting the current date")
    current_date = datetime.datetime.now()
    formatted_date = current_date.strftime("%B %d, %Y")

    return formatted_date
