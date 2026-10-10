"""
utils/logger.py
===============
Structured JSON logger for FACE agents.
Every log line is valid JSON. Rotates daily, keeps 7 days.
Logs to state/logs/YYYY-MM-DD.log AND stdout (for GitHub Actions).
CRITICAL events are also mirrored to state/errors.json.

Usage:
    from utils.logger import get_logger
    log = get_logger("trend_hunter")
    log.info("Found %d trends", 10, extra={"trend_count": 10})
"""

import json
import logging
import logging.handlers
import os
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
_LOGS_DIR = _ROOT / "state" / "logs"
_ERRORS_FILE = _ROOT / "state" / "errors.json"

# Session ID is the same for all loggers within one process execution
_SESSION_ID = str(uuid.uuid4())[:8]

# Global registry so we never create duplicate loggers
_LOGGERS: Dict[str, logging.Logger] = {}


# ---------------------------------------------------------------------------
# Custom JSON formatter
# ---------------------------------------------------------------------------

class _JSONFormatter(logging.Formatter):
    """
    Format every log record as a single-line JSON object.

    Output fields:
        timestamp   ISO 8601 UTC string
        session_id  short UUID for this pipeline run
        agent_name  the logger name
        level       DEBUG / INFO / WARNING / ERROR / CRITICAL
        message     the formatted log message
        extra       any extra key=value pairs passed via the 'extra' kwarg
        exc_info    traceback string (only present when an exception is logged)
    """

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        """Format a log record as a single-line JSON string."""
        # Build the base record
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        entry: Dict[str, Any] = {
            "timestamp": ts,
            "session_id": _SESSION_ID,
            "agent_name": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
            "extra": {},
        }

        # Pull out any 'extra' dict injected by the caller
        for key, val in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "levelname", "levelno",
                "pathname", "filename", "module", "exc_info", "exc_text",
                "stack_info", "lineno", "funcName", "created", "msecs",
                "relativeCreated", "thread", "threadName", "processName",
                "process", "message", "taskName",
            ) and not key.startswith("_"):
                entry["extra"][key] = val

        # Traceback
        if record.exc_info:
            entry["exc_info"] = self.formatException(record.exc_info)

        try:
            return json.dumps(entry, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            # Fallback: stringify non-serialisable extras
            entry["extra"] = {k: str(v) for k, v in entry["extra"].items()}
            return json.dumps(entry, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Critical error mirror
# ---------------------------------------------------------------------------

class _CriticalMirrorHandler(logging.Handler):
    """
    On CRITICAL level: append the entry to state/errors.json for easy scanning.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Append CRITICAL log entries to state/errors.json."""
        if record.levelno < logging.CRITICAL:
            return
        try:
            _LOGS_DIR.parent.mkdir(parents=True, exist_ok=True)

            # Load existing errors
            if _ERRORS_FILE.exists():
                with open(_ERRORS_FILE, "r", encoding="utf-8") as fh:
                    try:
                        errors = json.load(fh)
                        if not isinstance(errors, list):
                            errors = []
                    except json.JSONDecodeError:
                        errors = []
            else:
                errors = []

            ts = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
            errors.append({
                "timestamp": ts,
                "session_id": _SESSION_ID,
                "agent_name": record.name,
                "message": record.getMessage(),
                "exc_info": self.formatException(record.exc_info) if record.exc_info else None,
            })

            # Keep only last 500 critical errors
            errors = errors[-500:]

            with open(_ERRORS_FILE, "w", encoding="utf-8") as fh:
                json.dump(errors, fh, indent=2, ensure_ascii=False, default=str)

        except Exception:  # noqa: BLE001
            # Never let the logger break the pipeline
            pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_logger(
    agent_name: str,
    level: Optional[str] = None,
) -> logging.Logger:
    """
    Return a configured logger for the given agent.

    Logs to:
      1. state/logs/YYYY-MM-DD.log (rotating daily, keep 7 days)
      2. stdout (so GitHub Actions shows live output)
      3. state/errors.json (CRITICAL only)

    The log level defaults to:
      - DEBUG if the LOG_LEVEL env var is 'DEBUG' or CI is not set
      - INFO in GitHub Actions (CI=true env var)

    Args:
        agent_name: Short name, e.g. 'trend_hunter', 'publisher'.
        level: Optional explicit level string ('DEBUG', 'INFO', etc.).

    Returns:
        Configured logging.Logger instance.
    """
    global _LOGGERS

    if agent_name in _LOGGERS:
        return _LOGGERS[agent_name]

    logger = logging.getLogger(agent_name)

    # Determine log level
    if level:
        log_level = getattr(logging, level.upper(), logging.INFO)
    elif os.environ.get("LOG_LEVEL", "").upper() == "DEBUG":
        log_level = logging.DEBUG
    elif os.environ.get("CI") == "true":
        log_level = logging.INFO
    else:
        log_level = logging.DEBUG

    logger.setLevel(log_level)

    # Prevent duplicate handlers if get_logger() called twice
    if logger.handlers:
        _LOGGERS[agent_name] = logger
        return logger

    formatter = _JSONFormatter()

    # ── File handler: daily rotation, keep 7 days ──────────────────────────
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_file = _LOGS_DIR / f"{today_str}.log"

    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=str(log_file),
        when="midnight",
        interval=1,
        backupCount=7,
        utc=True,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # ── Stdout handler ──────────────────────────────────────────────────────
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    # ── Critical mirror handler ─────────────────────────────────────────────
    critical_handler = _CriticalMirrorHandler()
    critical_handler.setLevel(logging.CRITICAL)
    critical_handler.setFormatter(formatter)
    logger.addHandler(critical_handler)

    # Don't propagate to root to avoid duplicate output
    logger.propagate = False

    _LOGGERS[agent_name] = logger
    return logger


def get_session_id() -> str:
    """Return the current pipeline session ID (short UUID)."""
    return _SESSION_ID


def log_exception(logger: logging.Logger, message: str, exc: Exception) -> None:
    """
    Convenience wrapper for logging an exception with full traceback at ERROR level.

    Args:
        logger: Logger instance from get_logger().
        message: Human-readable message prefix.
        exc: The caught exception.
    """
    logger.error(
        "%s: %s",
        message,
        exc,
        exc_info=sys.exc_info(),
        extra={
            "exception_type": type(exc).__name__,
            "traceback": traceback.format_exc(),
        },
    )
