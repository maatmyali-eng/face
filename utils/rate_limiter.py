"""
utils/rate_limiter.py
=====================
Generic token-bucket rate limiter, thread-safe and persisted across
agent runs in the same pipeline execution.

Usage:
    from utils.rate_limiter import GEMINI_LIMITER, FACEBOOK_LIMITER
    GEMINI_LIMITER.acquire()   # blocks until a slot is available
    # ... make API call ...
"""

import json
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from utils.logger import get_logger

logger = get_logger("rate_limiter")

# ---------------------------------------------------------------------------
# Persistence path
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
_RATE_LIMITS_FILE = _ROOT / "state" / "rate_limits.json"
_FILE_LOCK = threading.Lock()


def _load_persisted_state() -> Dict[str, list]:
    """Load per-limiter request timestamps from disk."""
    with _FILE_LOCK:
        if not _RATE_LIMITS_FILE.exists():
            return {}
        try:
            with open(_RATE_LIMITS_FILE, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            return {}


def _save_persisted_state(state: Dict[str, list]) -> None:
    """Persist per-limiter timestamps to disk (best-effort)."""
    with _FILE_LOCK:
        try:
            _RATE_LIMITS_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = _RATE_LIMITS_FILE.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(state, fh)
            os.replace(tmp, _RATE_LIMITS_FILE)
        except OSError as exc:
            logger.warning("Failed to persist rate limit state: %s", exc)


# ---------------------------------------------------------------------------
# RateLimiter class
# ---------------------------------------------------------------------------

class RateLimiter:
    """
    Thread-safe token-bucket rate limiter using a sliding 60-second window.

    Args:
        name: Unique name for this limiter (used as persistence key).
        calls_per_minute: Maximum calls allowed per 60-second window.
    """

    def __init__(self, name: str, calls_per_minute: int) -> None:
        self.name = name
        self.calls_per_minute = calls_per_minute
        self._lock = threading.Lock()

        # Load any timestamps from a previous agent run in this pipeline
        persisted = _load_persisted_state()
        raw_timestamps = persisted.get(name, [])

        # Keep only timestamps from the last 60 seconds
        now = time.monotonic()
        # We can't directly compare monotonic times across processes,
        # so we store wall-clock (time.time()) and convert on load.
        # On load, convert wall-clock timestamps to approximate monotonic equivalents.
        wall_now = time.time()
        mono_offset = now - wall_now  # approximate

        self._requests: deque = deque()
        for wall_ts in raw_timestamps:
            # Convert stored wall-clock time to a monotonic-equivalent
            mono_ts = wall_ts + mono_offset
            if now - mono_ts < 60.0:
                self._requests.append(mono_ts)

        logger.debug(
            "RateLimiter '%s' initialised (%d CPM). Loaded %d recent timestamps.",
            name,
            calls_per_minute,
            len(self._requests),
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def acquire(self) -> None:
        """
        Block until a request slot is available within the rate limit window.
        Thread-safe — multiple threads can call this concurrently.
        """
        with self._lock:
            while True:
                now = time.monotonic()
                # Evict timestamps older than 60 seconds
                while self._requests and now - self._requests[0] > 60.0:
                    self._requests.popleft()

                if len(self._requests) < self.calls_per_minute:
                    # Slot available
                    self._requests.append(now)
                    self._persist_async()
                    return
                else:
                    # Must wait until the oldest request falls out of the window
                    wait_until = self._requests[0] + 60.0
                    sleep_for = wait_until - now + 0.05  # 50ms safety margin
                    logger.info(
                        "Rate limit: '%s' at %d/%d CPM. Sleeping %.1fs.",
                        self.name,
                        len(self._requests),
                        self.calls_per_minute,
                        sleep_for,
                        extra={
                            "limiter": self.name,
                            "current_count": len(self._requests),
                            "limit": self.calls_per_minute,
                            "sleep_seconds": round(sleep_for, 2),
                        },
                    )
                    # Release lock while sleeping so other threads can check
                    self._lock.release()
                    time.sleep(max(sleep_for, 0.1))
                    self._lock.acquire()

    def remaining_capacity(self) -> int:
        """
        Return how many more requests can be made in the current window
        without blocking.

        Returns:
            Number of available slots (0 = fully saturated).
        """
        with self._lock:
            now = time.monotonic()
            while self._requests and now - self._requests[0] > 60.0:
                self._requests.popleft()
            return max(0, self.calls_per_minute - len(self._requests))

    def reset(self) -> None:
        """Clear all recorded requests (use for testing or manual reset)."""
        with self._lock:
            self._requests.clear()
            logger.info("RateLimiter '%s' reset.", self.name)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _persist_async(self) -> None:
        """Persist current window to disk in a best-effort, non-blocking way."""
        # We store wall-clock times (time.time()) so they're meaningful across processes
        mono_offset = time.time() - time.monotonic()
        wall_timestamps = [t + mono_offset for t in list(self._requests)]

        def _write() -> None:
            try:
                state = _load_persisted_state()
                state[self.name] = wall_timestamps
                _save_persisted_state(state)
            except Exception as exc:
                logger.debug("Rate limit persist failed (non-fatal): %s", exc)

        # Run in background thread to not block the caller
        t = threading.Thread(target=_write, daemon=True)
        t.start()


# ---------------------------------------------------------------------------
# Pre-configured limiters (imported directly by other modules)
# ---------------------------------------------------------------------------

# Gemini: 14 RPM (buffer under 15 RPM free tier)
GEMINI_LIMITER = RateLimiter("gemini", calls_per_minute=14)

# Facebook Graph API: 180 calls/hr expressed as 3 CPM (180/60=3)
# Note: Facebook limit is per hour, so we approximate as ~3 per minute
# For more accurate control, the facebook_client tracks hourly windows too.
FACEBOOK_LIMITER = RateLimiter("facebook", calls_per_minute=3)

# Pexels: ~190/hr ≈ 3 CPM (conservative)
PEXELS_LIMITER = RateLimiter("pexels", calls_per_minute=3)

# HuggingFace: conservative 50 RPM
HF_LIMITER = RateLimiter("huggingface", calls_per_minute=50)

# Pixabay: conservative limit for low-volume use.
PIXABAY_LIMITER = RateLimiter("pixabay", calls_per_minute=3)
