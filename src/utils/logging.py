"""Logging utility stub for Nepal News Bot."""
import logging
import sys
from pathlib import Path


class CorrelatedFormatter(logging.Formatter):
    """Custom logging formatter that injects run_date and stage correlation fields."""

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "run_date"):
            record.run_date = "-"
        if not hasattr(record, "stage"):
            record.stage = "-"
        if record.run_date != "-" or record.stage != "-":
            prefix = f"[{self.formatTime(record, self.datefmt)}] [{record.levelname}] [{record.name}] [{record.run_date}:{record.stage}]"
        else:
            prefix = f"[{self.formatTime(record, self.datefmt)}] [{record.levelname}] [{record.name}]"

        message = record.getMessage()
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            if message[-1:] != "\n":
                message = message + "\n"
            message = message + record.exc_text
        if record.stack_info:
            if message[-1:] != "\n":
                message = message + "\n"
            message = message + self.formatStack(record.stack_info)
        return f"{prefix}: {message}"


class ContextFilter(logging.Filter):
    """Filter that sets default run_date and stage on records if not already provided."""

    def __init__(self, run_date: str | None = None, stage: str | None = None) -> None:
        super().__init__()
        self.run_date = run_date or "-"
        self.stage = stage or "-"

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "run_date") or record.run_date == "-":
            record.run_date = self.run_date
        if not hasattr(record, "stage") or record.stage == "-":
            record.stage = self.stage
        return True


def setup_logger(
    name: str = "nepal-news-bot",
    log_file: Path | None = None,
    run_date: str | None = None,
    stage: str | None = None,
) -> logging.Logger:
    """Configures and returns a structured logger with console and optional file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = CorrelatedFormatter(datefmt="%Y-%m-%d %H:%M:%S")

    # Clear existing context filters to avoid duplicate or stale filters
    for f in list(logger.filters):
        if isinstance(f, ContextFilter):
            logger.removeFilter(f)
    if run_date or stage:
        logger.addFilter(ContextFilter(run_date=run_date, stage=stage))

    has_console = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )
    if not has_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    else:
        for h in logger.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                h.setFormatter(formatter)

    if log_file:
        resolved_path = str(Path(log_file).resolve())
        matching_fh = False
        to_remove: list[logging.Handler] = []
        for h in logger.handlers:
            if isinstance(h, logging.FileHandler):
                if str(Path(h.baseFilename).resolve()) == resolved_path:
                    matching_fh = True
                    h.setFormatter(formatter)
                else:
                    to_remove.append(h)
        for h in to_remove:
            h.close()
            logger.removeHandler(h)

        if not matching_fh:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
