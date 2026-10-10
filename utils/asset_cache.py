"""
utils/asset_cache.py
====================
Prevents re-downloading any asset that already exists locally.
Manages a JSON manifest at state/asset_cache.json.
Auto-cleanup removes video clips older than 7 days.

Usage:
    from utils.asset_cache import AssetCache
    cache = AssetCache()
    path = cache.download_if_missing(url, "noto-sans-bold", "font", "assets/fonts/")
"""

import hashlib
import json
import os
import time
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from utils.logger import get_logger
from utils.rate_limiter import PEXELS_LIMITER

logger = get_logger("asset_cache")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
_CACHE_MANIFEST = _ROOT / "state" / "asset_cache.json"
_VIDEO_CLIP_MAX_AGE_DAYS = 7

# Pexels API endpoint
_PEXELS_PHOTOS_URL = "https://api.pexels.com/v1/search"
_PEXELS_VIDEOS_URL = "https://api.pexels.com/videos/search"

# Asset types that are retained forever
_PERMANENT_TYPES = {"font", "audio"}
# Asset types subject to age-based cleanup
_EPHEMERAL_TYPES = {"video_clip", "stock_photo"}


# ---------------------------------------------------------------------------
# AssetCache class
# ---------------------------------------------------------------------------

class AssetCache:
    """
    Manages a local asset cache with a JSON manifest.

    The manifest maps asset IDs to:
        {
            "asset_type": str,
            "local_path": str,
            "cached_at": ISO timestamp string,
            "source_url": str (optional),
        }

    Thread safety: the manifest is loaded fresh on every operation to handle
    cases where multiple agents run concurrently (GitHub Actions matrix jobs).
    """

    def __init__(self) -> None:
        _CACHE_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        self._run_cleanup()

    # -----------------------------------------------------------------------
    # Core API
    # -----------------------------------------------------------------------

    def is_cached(self, asset_id: str, asset_type: str) -> bool:
        """
        Return True if the asset exists in the manifest AND the file is on disk.

        Args:
            asset_id: Unique string identifier for the asset.
            asset_type: 'font' | 'audio' | 'video_clip' | 'stock_photo'

        Returns:
            True if the asset is present locally.
        """
        manifest = self._load_manifest()
        key = self._make_key(asset_id, asset_type)
        entry = manifest.get(key)
        if entry is None:
            return False
        local_path = Path(entry.get("local_path", ""))
        return local_path.exists()

    def get_path(self, asset_id: str, asset_type: str) -> Optional[str]:
        """
        Return the local file path for a cached asset, or None if not cached.

        Args:
            asset_id: Unique asset identifier.
            asset_type: Asset category string.

        Returns:
            Absolute path string, or None.
        """
        manifest = self._load_manifest()
        key = self._make_key(asset_id, asset_type)
        entry = manifest.get(key)
        if entry is None:
            return None
        local_path = Path(entry.get("local_path", ""))
        if not local_path.exists():
            logger.debug("Asset '%s' in manifest but file missing — treating as uncached.", asset_id)
            return None
        return str(local_path)

    def register(
        self,
        asset_id: str,
        asset_type: str,
        local_path: str,
        source_url: Optional[str] = None,
    ) -> None:
        """
        Register a locally-existing asset in the manifest.

        Args:
            asset_id: Unique identifier.
            asset_type: Asset category.
            local_path: Absolute path to the local file.
            source_url: Original download URL (optional, for auditing).
        """
        manifest = self._load_manifest()
        key = self._make_key(asset_id, asset_type)
        manifest[key] = {
            "asset_id": asset_id,
            "asset_type": asset_type,
            "local_path": str(Path(local_path).resolve()),
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "source_url": source_url or "",
        }
        self._save_manifest(manifest)
        logger.debug("Registered asset '%s' (%s) → %s", asset_id, asset_type, local_path)

    def download_if_missing(
        self,
        url: str,
        asset_id: str,
        asset_type: str,
        dest_dir: str,
        filename: Optional[str] = None,
    ) -> str:
        """
        Return the cached path if available; otherwise download from URL and cache.

        Args:
            url: Source URL to download from.
            asset_id: Unique identifier for this asset.
            asset_type: Asset category ('font', 'audio', 'video_clip', 'stock_photo').
            dest_dir: Directory to download into.
            filename: Filename to save as. Defaults to last URL path segment.

        Returns:
            Local path string to the (possibly freshly downloaded) file.
        """
        cached = self.get_path(asset_id, asset_type)
        if cached:
            logger.debug("Cache hit for '%s' (%s).", asset_id, asset_type)
            return cached

        dest_dir_path = Path(dest_dir)
        dest_dir_path.mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename = url.split("?")[0].split("/")[-1] or f"{asset_id}.bin"

        dest_path = dest_dir_path / filename
        logger.info("Downloading asset '%s' from %s", asset_id, url)

        for attempt in range(3):
            try:
                urllib.request.urlretrieve(url, str(dest_path))
                if dest_path.exists() and dest_path.stat().st_size > 0:
                    self.register(asset_id, asset_type, str(dest_path), source_url=url)
                    logger.info("Asset cached → %s", dest_path)
                    return str(dest_path)
                raise IOError("Downloaded file is empty.")
            except Exception as exc:
                wait = 2 ** attempt
                logger.warning(
                    "Download attempt %d/3 failed for '%s': %s — retrying in %ds.",
                    attempt + 1, asset_id, exc, wait,
                )
                if attempt < 2:
                    time.sleep(wait)

        raise RuntimeError(
            f"Failed to download asset '{asset_id}' from {url} after 3 attempts."
        )

    # -----------------------------------------------------------------------
    # Pexels helpers
    # -----------------------------------------------------------------------

    def download_pexels_photo(
        self,
        query: str,
        orientation: str = "portrait",
        cache_dir: Optional[str] = None,
    ) -> str:
        """
        Search Pexels for a photo matching query, download, and cache it.
        Uses the Pexels API (requires PEXELS_API_KEY env var).

        Args:
            query: Search query string, e.g. 'fitness gym workout'.
            orientation: 'portrait' | 'landscape' | 'square'
            cache_dir: Override cache directory. Defaults to state/temp/.

        Returns:
            Local path to the downloaded photo.
        """
        api_key = os.environ.get("PEXELS_API_KEY", "")
        if not api_key:
            raise EnvironmentError("PEXELS_API_KEY environment variable is not set.")

        # Cache key based on query + orientation
        cache_key = f"pexels_photo_{hashlib.md5((query + orientation).encode()).hexdigest()[:12]}"

        if cache_dir is None:
            cache_dir = str(_ROOT / "state" / "temp")

        cached = self.get_path(cache_key, "stock_photo")
        if cached:
            return cached

        PEXELS_LIMITER.acquire()

        logger.info("Fetching Pexels photo: query=%r, orientation=%s", query, orientation)
        try:
            resp = requests.get(
                _PEXELS_PHOTOS_URL,
                headers={"Authorization": api_key},
                params={"query": query, "orientation": orientation, "per_page": 5},
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            raise RuntimeError(f"Pexels photo search failed: {exc}") from exc

        photos = data.get("photos", [])
        if not photos:
            raise RuntimeError(
                f"Pexels returned no results for query={query!r}. "
                "Try a different search term."
            )

        # Pick highest-resolution result
        best_photo = photos[0]
        photo_url = best_photo.get("src", {}).get("original", "")
        photo_id = str(best_photo.get("id", cache_key))
        filename = f"pexels_photo_{photo_id}.jpg"

        return self.download_if_missing(
            photo_url, cache_key, "stock_photo", cache_dir, filename
        )

    def download_pexels_video(
        self,
        query: str,
        orientation: str = "portrait",
        cache_dir: Optional[str] = None,
    ) -> str:
        """
        Search Pexels for a video clip matching query, download, and cache it.

        Args:
            query: Search query string, e.g. 'city traffic night'.
            orientation: 'portrait' | 'landscape'
            cache_dir: Override cache directory. Defaults to state/temp/.

        Returns:
            Local path to the downloaded video clip.
        """
        api_key = os.environ.get("PEXELS_API_KEY", "")
        if not api_key:
            raise EnvironmentError("PEXELS_API_KEY environment variable is not set.")

        cache_key = f"pexels_video_{hashlib.md5((query + orientation).encode()).hexdigest()[:12]}"

        if cache_dir is None:
            cache_dir = str(_ROOT / "state" / "temp")

        cached = self.get_path(cache_key, "video_clip")
        if cached:
            return cached

        PEXELS_LIMITER.acquire()

        logger.info("Fetching Pexels video: query=%r, orientation=%s", query, orientation)
        try:
            resp = requests.get(
                _PEXELS_VIDEOS_URL,
                headers={"Authorization": api_key},
                params={"query": query, "orientation": orientation, "per_page": 5, "size": "medium"},
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            raise RuntimeError(f"Pexels video search failed: {exc}") from exc

        videos = data.get("videos", [])
        if not videos:
            raise RuntimeError(
                f"Pexels returned no video results for query={query!r}."
            )

        # Pick the best HD file for portrait orientation
        best_video = videos[0]
        video_files = best_video.get("video_files", [])

        # Prefer HD portrait (1080×1920) or highest width available
        chosen_file = _pick_best_video_file(video_files, orientation)
        if not chosen_file:
            raise RuntimeError("No suitable video file found in Pexels response.")

        video_url = chosen_file.get("link", "")
        video_id = str(best_video.get("id", cache_key))
        filename = f"pexels_video_{video_id}.mp4"

        return self.download_if_missing(
            video_url, cache_key, "video_clip", cache_dir, filename
        )

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------

    def _run_cleanup(self) -> None:
        """
        Remove video_clip entries (and their files) older than 7 days.
        Audio and font assets are kept forever.
        """
        manifest = self._load_manifest()
        cutoff = datetime.now(timezone.utc) - timedelta(days=_VIDEO_CLIP_MAX_AGE_DAYS)
        keys_to_remove: List[str] = []
        files_removed = 0

        for key, entry in manifest.items():
            asset_type = entry.get("asset_type", "")
            if asset_type not in _EPHEMERAL_TYPES:
                continue

            try:
                cached_at = datetime.fromisoformat(
                    entry.get("cached_at", "1970-01-01T00:00:00+00:00")
                )
                if cached_at < cutoff:
                    local_path = Path(entry.get("local_path", ""))
                    if local_path.exists():
                        local_path.unlink()
                        files_removed += 1
                    keys_to_remove.append(key)
            except (ValueError, OSError) as exc:
                logger.warning("Cleanup error for key '%s': %s", key, exc)

        if keys_to_remove:
            for key in keys_to_remove:
                del manifest[key]
            self._save_manifest(manifest)
            logger.info(
                "Asset cleanup: removed %d ephemeral asset(s) older than %d days.",
                files_removed,
                _VIDEO_CLIP_MAX_AGE_DAYS,
            )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _load_manifest(self) -> Dict[str, Any]:
        """Load the manifest JSON from disk (fresh read each time)."""
        if not _CACHE_MANIFEST.exists():
            return {}
        try:
            with open(_CACHE_MANIFEST, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load asset manifest: %s — starting fresh.", exc)
            return {}

    def _save_manifest(self, manifest: Dict[str, Any]) -> None:
        """Atomically save the manifest JSON to disk."""
        tmp = _CACHE_MANIFEST.with_suffix(".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(manifest, fh, indent=2, ensure_ascii=False)
            os.replace(tmp, _CACHE_MANIFEST)
        except OSError as exc:
            logger.error("Failed to save asset manifest: %s", exc)

    @staticmethod
    def _make_key(asset_id: str, asset_type: str) -> str:
        """Create the manifest dict key from asset_id and type."""
        return f"{asset_type}::{asset_id}"


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _pick_best_video_file(
    video_files: List[Dict[str, Any]],
    orientation: str,
) -> Optional[Dict[str, Any]]:
    """
    Choose the best video file from a Pexels video_files list.

    For portrait: prefer 1080-wide portrait files.
    For landscape: prefer 1920-wide landscape files.

    Args:
        video_files: List of Pexels video file dicts.
        orientation: 'portrait' | 'landscape'

    Returns:
        Best-matching file dict, or None if list is empty.
    """
    if not video_files:
        return None

    # Score each file
    scored: List[tuple] = []
    for vf in video_files:
        w = vf.get("width", 0)
        h = vf.get("height", 0)
        quality = vf.get("quality", "")

        score = 0
        if orientation == "portrait":
            # Want w < h
            if h > w:
                score += 10
            if w == 1080:
                score += 5
        else:
            # Want w > h
            if w > h:
                score += 10
            if w == 1920:
                score += 5

        if quality in ("hd", "fhd"):
            score += 3

        scored.append((score, vf))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_CACHE_INSTANCE: Optional[AssetCache] = None


def get_asset_cache() -> AssetCache:
    """Return the shared AssetCache singleton."""
    global _CACHE_INSTANCE
    if _CACHE_INSTANCE is None:
        _CACHE_INSTANCE = AssetCache()
    return _CACHE_INSTANCE
