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
from utils.rate_limiter import PEXELS_LIMITER, PIXABAY_LIMITER

logger = get_logger("asset_cache")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
_CACHE_MANIFEST = _ROOT / "state" / "asset_cache.json"
_VIDEO_CLIP_MAX_AGE_DAYS = 7

_PIXABAY_IMAGES_URL = "https://pixabay.com/api/"
_PIXABAY_VIDEOS_URL = "https://pixabay.com/api/videos/"

# Required API-response cache: 24 hours.
_PIXABAY_SEARCH_CACHE_DIR = _ROOT / "state" / "pixabay_search_cache"
_PIXABAY_SEARCH_CACHE_TTL = 24 * 60 * 60

# Pexels API endpoint
# Official Pexels REST API endpoints
_PEXELS_PHOTOS_URL = "https://api.pexels.com/v1/search"
_PEXELS_VIDEOS_URL = "https://api.pexels.com/v1/videos/search"

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


        temp_path = dest_path.with_name(dest_path.name + ".part")
        headers = {
            "User-Agent": (
                "FACE/1.0 "
                "(https://github.com/maatmyali-eng/face)"
            ),
            "Accept": "*/*",
        }
        last_error = None

        for attempt in range(3):
            try:
                with requests.get(
                    url,
                    headers=headers,
                    stream=True,
                    allow_redirects=True,
                    timeout=(15, 90),
                ) as response:
                    if response.status_code == 403:
                        detail = response.text[:180].replace(
                            "\n", " "
                        )
                        raise requests.HTTPError(
                            f"HTTP 403 Forbidden at {response.url}. "
                            f"Response: {detail}",
                            response=response,
                        )

                    response.raise_for_status()

                    content_type = response.headers.get(
                        "Content-Type", ""
                    ).lower()

                    if "text/html" in content_type:
                        raise RuntimeError(
                            "The server returned HTML instead of "
                            "the requested media file. The download "
                            "may have been blocked."
                        )

                    with temp_path.open("wb") as output:
                        for chunk in response.iter_content(
                            chunk_size=256 * 1024
                        ):
                            if chunk:
                                output.write(chunk)

                if (
                    not temp_path.exists()
                    or temp_path.stat().st_size == 0
                ):
                    raise IOError("Downloaded file is empty.")

                os.replace(temp_path, dest_path)
                self.register(
                    asset_id,
                    asset_type,
                    str(dest_path),
                    source_url=url,
                )
                logger.info("Asset cached → %s", dest_path)
                return str(dest_path)

            except Exception as exc:
                last_error = exc

                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass

                status_code = getattr(
                    getattr(exc, "response", None),
                    "status_code",
                    None,
                )

                logger.warning(
                    "Download attempt %d/3 failed for '%s': %s",
                    attempt + 1,
                    asset_id,
                    exc,
                )

                # A 403 is an access denial, not a transient failure.
                # Avoid making two more identical requests.
                if status_code == 403:
                    break

                if attempt < 2:
                    wait = 2 ** attempt
                    time.sleep(wait)

        raise RuntimeError(
            f"Failed to download asset '{asset_id}'. "
            f"Last error: {last_error}"
        ) from last_error


    def _pixabay_search(
        self,
        endpoint: str,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Search Pixabay and cache successful API responses for 24 hours."""
        api_key = os.environ.get("PIXABAY_API_KEY", "").strip()
        if not api_key:
            raise EnvironmentError(
                "PIXABAY_API_KEY is not configured."
            )

        query = " ".join(query.split())[:100]
        if not query:
            raise ValueError("Pixabay search query cannot be empty.")

        search_filters = dict(filters or {})
        params = {
            "q": query,
            "safesearch": "true",
            "order": "popular",
            "per_page": 5,
            **search_filters,
        }

        # Do not include the secret key in the cache filename or saved data.
        cache_basis = json.dumps(
            {
                "endpoint": endpoint,
                "params": params,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        cache_id = hashlib.sha256(
            cache_basis.encode("utf-8")
        ).hexdigest()
        cache_path = _PIXABAY_SEARCH_CACHE_DIR / f"{cache_id}.json"

        # Reuse the API response for 24 hours.
        try:
            if cache_path.is_file():
                cached = json.loads(
                    cache_path.read_text(encoding="utf-8")
                )
                age = time.time() - float(cached["fetched_at"])
                data = cached["data"]
                if (
                    0 <= age < _PIXABAY_SEARCH_CACHE_TTL
                    and isinstance(data, dict)
                ):
                    logger.debug("Pixabay search-cache hit.")
                    return data
        except (OSError, ValueError, KeyError, TypeError):
            logger.warning("Ignoring invalid Pixabay search cache.")

        PIXABAY_LIMITER.acquire()

        try:
            response = requests.get(
                endpoint,
                params={**params, "key": api_key},
                timeout=25,
            )
        except requests.RequestException:
            raise RuntimeError(
                "Pixabay API network request failed."
            ) from None

        if response.status_code == 429:
            raise RuntimeError(
                "Pixabay rate limit reached (HTTP 429)."
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"Pixabay API returned HTTP {response.status_code}. "
                "Check the API key and request parameters."
            )

        try:
            data = response.json()
        except ValueError:
            raise RuntimeError(
                "Pixabay returned an invalid JSON response."
            ) from None

        if not isinstance(data, dict):
            raise RuntimeError("Unexpected Pixabay API response.")

        # Persist the response so subsequent runs can reuse it.
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = cache_path.with_name(
                f"{cache_path.stem}.{time.time_ns()}.tmp"
            )
            temporary_path.write_text(
                json.dumps(
                    {
                        "fetched_at": time.time(),
                        "data": data,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            os.replace(temporary_path, cache_path)
        except OSError as exc:
            # Don't lose usable results if local cache writing fails.
            logger.warning(
                "Could not persist Pixabay search cache: %s",
                type(exc).__name__,
            )

        return data


    def _record_pixabay_source(
        self,
        asset_id: str,
        asset_type: str,
        item: Dict[str, Any],
    ) -> None:
        """Keep source and contributor details in the asset manifest."""
        manifest = self._load_manifest()
        entry = manifest.get(self._make_key(asset_id, asset_type))
        if entry is None:
            return

        entry.update({
            "source_provider": "Pixabay",
            "source_page_url": str(item.get("pageURL", "")),
            "source_id": str(item.get("id", "")),
            "contributor": str(item.get("user", "")),
            "contributor_id": str(item.get("user_id", "")),
        })
        self._save_manifest(manifest)

    def download_pixabay_photo(
        self,
        query: str,
        orientation: str = "portrait",
        cache_dir: Optional[str] = None,
        image_type: str = "photo",
        category: Optional[str] = None,
        editors_choice: bool = False,
        order: str = "popular",
    ) -> str:
        """Find and download a Pixabay image through its official API."""
        orientation_map = {
            "portrait": "vertical",
            "landscape": "horizontal",
        }
        if orientation not in (
            "portrait", "landscape", "square", "all"
        ):
            raise ValueError(
                "orientation must be portrait, landscape, square, or all"
            )

        filters: Dict[str, Any] = {
            "image_type": image_type,
            "order": order,
        }
        if orientation in orientation_map:
            filters["orientation"] = orientation_map[orientation]
        if category:
            filters["category"] = category
        if editors_choice:
            filters["editors_choice"] = "true"

        data = self._pixabay_search(
            _PIXABAY_IMAGES_URL, query, filters
        )
        hits = data.get("hits", [])
        if not hits:
            raise RuntimeError(
                f"Pixabay returned no images for query={query!r}."
            )

        # For square content, prefer images closest to a 1:1 aspect ratio.
        if orientation == "square":
            hits = sorted(
                hits,
                key=lambda hit: abs(
                    (
                        float(hit.get("imageWidth") or 1)
                        / float(hit.get("imageHeight") or 1)
                    ) - 1.0
                ),
            )

        photo = hits[0]
        image_url = (
            photo.get("largeImageURL")
            or photo.get("webformatURL")
            or ""
        )
        if not image_url:
            raise RuntimeError(
                "Pixabay result did not contain a usable image URL."
            )

        photo_id = str(photo.get("id", ""))
        if not photo_id:
            raise RuntimeError("Pixabay image result has no ID.")

        asset_id = f"pixabay_photo_{photo_id}"
        if cache_dir is None:
            cache_dir = str(_ROOT / "state" / "temp")

        path = self.download_if_missing(
            image_url,
            asset_id,
            "stock_photo",
            cache_dir,
            f"{asset_id}.jpg",
        )
        self._record_pixabay_source(
            asset_id, "stock_photo", photo
        )
        return path

    def download_pixabay_video(
        self,
        query: str,
        orientation: str = "portrait",
        cache_dir: Optional[str] = None,
        category: Optional[str] = None,
        video_type: str = "all",
        editors_choice: bool = False,
        order: str = "popular",
    ) -> str:
        """Find, select, and download a Pixabay video clip."""
        if orientation not in ("portrait", "landscape"):
            raise ValueError(
                "Video orientation must be portrait or landscape."
            )

        filters: Dict[str, Any] = {
            "video_type": video_type,
            "order": order,
        }
        if category:
            filters["category"] = category
        if editors_choice:
            filters["editors_choice"] = "true"

        data = self._pixabay_search(
            _PIXABAY_VIDEOS_URL, query, filters
        )
        hits = data.get("hits", [])
        if not hits:
            raise RuntimeError(
                f"Pixabay returned no videos for query={query!r}."
            )

        # Prefer a rendition matching the requested orientation.
        # Prefer medium/small files to limit download size.
        quality_order = ("medium", "small", "large", "tiny")
        fallback = None
        selected = None
        max_bytes = 100 * 1024 * 1024

        for video in hits:
            renditions = video.get("videos") or {}
            candidates = []

            for quality in quality_order:
                rendition = renditions.get(quality) or {}
                url = rendition.get("url")
                if not url:
                    continue

                width = int(rendition.get("width") or 0)
                height = int(rendition.get("height") or 0)
                size = int(rendition.get("size") or 0)
                if size > max_bytes:
                    continue

                # Copy the dictionary; don't mutate the API response.
                candidate = dict(rendition)
                candidate["_quality"] = quality
                candidates.append(candidate)

            if not candidates:
                continue

            if fallback is None:
                fallback = (video, candidates[0])

            if orientation == "portrait":
                matching = [
                    item for item in candidates
                    if int(item.get("height") or 0)
                    > int(item.get("width") or 0)
                ]
            else:
                matching = [
                    item for item in candidates
                    if int(item.get("width") or 0)
                    >= int(item.get("height") or 0)
                ]

            if matching:
                selected = (video, matching[0])
                break

        if selected is None:
            selected = fallback
        if selected is None:
            raise RuntimeError(
                "Pixabay returned no usable video rendition."
            )

        video, rendition = selected
        video_url = rendition["url"]

        # Pixabay's documented download parameter for video URLs.
        separator = "&" if "?" in video_url else "?"
        if "download=1" not in video_url:
            video_url = f"{video_url}{separator}download=1"
        video_id = str(video.get("id", ""))
        if not video_id:
            raise RuntimeError("Pixabay video result has no ID.")

        asset_id = f"pixabay_video_{video_id}"
        if cache_dir is None:
            cache_dir = str(_ROOT / "state" / "temp")

        path = self.download_if_missing(
            video_url,
            asset_id,
            "video_clip",
            cache_dir,
            f"{asset_id}.mp4",
        )
        self._record_pixabay_source(
            asset_id, "video_clip", video
        )
        return path


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
