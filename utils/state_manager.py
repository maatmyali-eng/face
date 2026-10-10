"""
utils/state_manager.py
======================
Thread-safe read/write for all state/*.json files using filelock.
Every write is atomic (write to .tmp, then rename).
Validates against JSON Schema on every write.
Auto-prunes post_history.json to keep last 30 days.

Usage:
    from utils.state_manager import read_state, write_state, update_state, append_to_state
"""

import json
import os
import shutil
import time
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import filelock

# Optional JSON schema validation — graceful degradation if jsonschema not installed
try:
    import jsonschema
    _JSONSCHEMA_AVAILABLE = True
except ImportError:
    _JSONSCHEMA_AVAILABLE = False
    logging.warning(
        "jsonschema not installed — schema validation on state writes is disabled."
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
STATE_DIR = _ROOT / "state"
LOCK_DIR = STATE_DIR / ".locks"
SCHEMA_DIR = STATE_DIR / "schema"

# Lock acquisition timeout in seconds
_LOCK_TIMEOUT = 30

# How many days of post_history.json to retain
_POST_HISTORY_DAYS = 30


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def _ensure_dirs() -> None:
    """Create state/ and lock dirs if they do not exist."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LOCK_DIR.mkdir(parents=True, exist_ok=True)


_ensure_dirs()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _state_path(filename: str) -> Path:
    """
    Resolve a state file path.

    Args:
        filename: bare name like 'trends.json' or full path.

    Returns:
        Absolute Path inside state/.
    """
    p = Path(filename)
    if p.is_absolute():
        return p
    if not p.suffix:
        p = p.with_suffix(".json")
    # If no directory given, put it in STATE_DIR
    if p.parent == Path("."):
        return STATE_DIR / p
    return p


def _lock_path(state_path: Path) -> Path:
    """Return the .lock file path corresponding to a state file."""
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    return LOCK_DIR / (state_path.name + ".lock")


def _load_schema(filename: str) -> Optional[Dict]:
    """
    Load the JSON schema for a given state file if one exists.

    Args:
        filename: e.g. 'trends.json'

    Returns:
        Schema dict, or None if no schema found.
    """
    if not _JSONSCHEMA_AVAILABLE:
        return None
    schema_file = SCHEMA_DIR / (Path(filename).stem + ".schema.json")
    if schema_file.exists():
        try:
            with open(schema_file, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            logger.warning("Failed to load schema %s: %s", schema_file, exc)
    return None


def _validate(data: Any, filename: str) -> None:
    """
    Validate data against schema if available. Raises jsonschema.ValidationError
    on failure; logs a warning if schema not available.

    Args:
        data: the Python object to validate.
        filename: state file name used to locate the schema.
    """
    if not _JSONSCHEMA_AVAILABLE:
        return
    schema = _load_schema(filename)
    if schema is None:
        return
    try:
        jsonschema.validate(instance=data, schema=schema)
    except jsonschema.ValidationError as exc:
        raise ValueError(
            f"State validation failed for {filename}: {exc.message}"
        ) from exc


def _atomic_write(path: Path, data: Any) -> None:
    """
    Write data as JSON to path atomically (write to .tmp, then os.replace).

    Args:
        path: destination file path.
        data: JSON-serialisable Python object.
    """
    tmp_path = path.with_suffix(".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except Exception:
        # Clean up tmp if something went wrong
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_state(filename: str) -> Any:
    """
    Read and return the contents of a state JSON file (thread-safe).

    Args:
        filename: e.g. 'trends.json' or 'state/trends.json'.

    Returns:
        Parsed Python object (dict, list, etc.), or an empty dict if file
        does not exist yet.
    """
    path = _state_path(filename)
    lock = filelock.FileLock(str(_lock_path(path)), timeout=_LOCK_TIMEOUT)

    with lock:
        if not path.exists():
            logger.debug("State file %s does not exist yet — returning empty dict.", path.name)
            return {}
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError as exc:
            logger.error(
                "Corrupt JSON in %s: %s — returning empty dict.", path.name, exc
            )
            _backup_corrupt(path)
            return {}

    return data


def write_state(filename: str, data: Any) -> None:
    """
    Validate and atomically write data to a state JSON file (thread-safe).

    Args:
        filename: e.g. 'trends.json'.
        data: JSON-serialisable Python object.

    Raises:
        ValueError: if schema validation fails.
    """
    _validate(data, filename)

    # Apply special retention rules
    if Path(filename).stem == "post_history":
        data = _prune_post_history(data)

    path = _state_path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = filelock.FileLock(str(_lock_path(path)), timeout=_LOCK_TIMEOUT)

    with lock:
        _atomic_write(path, data)

    logger.debug("Wrote state file: %s", path.name)


def update_state(filename: str, key: str, value: Any) -> None:
    """
    Read-modify-write: set state[key] = value atomically.

    Args:
        filename: state file name.
        key: top-level key to update.
        value: new value (must be JSON-serialisable).
    """
    path = _state_path(filename)
    lock = filelock.FileLock(str(_lock_path(path)), timeout=_LOCK_TIMEOUT)

    with lock:
        # Read existing
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    existing = json.load(fh)
            except (json.JSONDecodeError, OSError):
                existing = {}
        else:
            existing = {}

        if not isinstance(existing, dict):
            existing = {}

        existing[key] = value
        _validate(existing, filename)

        if Path(filename).stem == "post_history":
            existing = _prune_post_history(existing)

        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(path, existing)

    logger.debug("Updated key '%s' in %s", key, Path(filename).name)


def append_to_state(filename: str, key: str, item: Any) -> None:
    """
    Read-modify-write: append item to state[key] list atomically.
    Creates the list if the key doesn't exist yet.

    Args:
        filename: state file name.
        key: key whose value is (or should become) a list.
        item: item to append.
    """
    path = _state_path(filename)
    lock = filelock.FileLock(str(_lock_path(path)), timeout=_LOCK_TIMEOUT)

    with lock:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    existing = json.load(fh)
            except (json.JSONDecodeError, OSError):
                existing = {}
        else:
            existing = {}

        if not isinstance(existing, dict):
            existing = {}

        if key not in existing:
            existing[key] = []
        if not isinstance(existing[key], list):
            existing[key] = [existing[key]]

        existing[key].append(item)
        _validate(existing, filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(path, existing)

    logger.debug("Appended to key '%s' in %s", key, Path(filename).name)


def state_exists(filename: str) -> bool:
    """Return True if a state file exists and is non-empty."""
    path = _state_path(filename)
    return path.exists() and path.stat().st_size > 0


def delete_state(filename: str) -> None:
    """Delete a state file (e.g. after processing). Thread-safe."""
    path = _state_path(filename)
    lock = filelock.FileLock(str(_lock_path(path)), timeout=_LOCK_TIMEOUT)
    with lock:
        if path.exists():
            path.unlink()
            logger.info("Deleted state file: %s", path.name)


def list_state_files() -> List[str]:
    """Return a list of all .json files currently in the state directory."""
    return [f.name for f in STATE_DIR.glob("*.json") if f.is_file()]


# ---------------------------------------------------------------------------
# Retention / pruning helpers
# ---------------------------------------------------------------------------

def _prune_post_history(data: Any) -> Any:
    """
    Keep only the last _POST_HISTORY_DAYS days of entries in post_history.json.

    Expects data to be a dict with a 'posts' list where each post has a
    'published_at' ISO timestamp string.

    Args:
        data: the post_history data dict.

    Returns:
        Pruned data dict.
    """
    if not isinstance(data, dict):
        return data

    posts = data.get("posts", [])
    if not isinstance(posts, list):
        return data

    cutoff = datetime.now(timezone.utc) - timedelta(days=_POST_HISTORY_DAYS)
    retained = []
    pruned_count = 0

    for post in posts:
        ts_str = post.get("published_at", "")
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if ts >= cutoff:
                retained.append(post)
            else:
                pruned_count += 1
        except (ValueError, AttributeError):
            # Keep entries with unparseable timestamps
            retained.append(post)

    if pruned_count:
        logger.info(
            "post_history pruning: removed %d posts older than %d days.",
            pruned_count,
            _POST_HISTORY_DAYS,
        )

    data["posts"] = retained
    return data


def _backup_corrupt(path: Path) -> None:
    """Move a corrupt JSON file to .corrupt so it doesn't keep failing."""
    backup = path.with_suffix(".corrupt")
    try:
        shutil.move(str(path), str(backup))
        logger.warning("Moved corrupt state file to %s", backup.name)
    except OSError as exc:
        logger.error("Could not back up corrupt file %s: %s", path, exc)
