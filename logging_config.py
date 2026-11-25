"""Central logging setup — used by app.py, stt_service.py, database.py."""

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"

def setup_logging(level: int = logging.DEBUG) -> None:
    root = logging.getLogger()
    if root.handlers:
        return  # already configured
    root.setLevel(level)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
    root.addHandler(stream)

    # Silence noisy third-party loggers if you want cleaner output
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.WARNING)