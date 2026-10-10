"""
utils/facebook_client.py
========================
Wraps Facebook Graph API v21.0 for FACE agents.
Handles: photo, video, story, text posts; insights; token expiry detection
         with automatic GitHub Issue creation.

Usage:
    from utils.facebook_client import FacebookClient
    fb = FacebookClient()
    post_id = fb.post_photo("image.jpg", "Caption here")
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from utils.logger import get_logger
from utils.state_manager import write_state, read_state
from utils.rate_limiter import FACEBOOK_LIMITER

logger = get_logger("facebook_client")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_BASE_URL = "https://graph.facebook.com/v21.0"
_TOKEN_EXPIRED_FILE = "token_expired.json"
_HOURLY_CALL_LIMIT = 200
_CALL_LIMIT_BUFFER = 190  # we stop at 190 and wait


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class FacebookTokenExpiredError(Exception):
    """Raised when the page access token has expired (error code 190)."""


class FacebookAPIError(Exception):
    """Generic Facebook Graph API error with code and message."""

    def __init__(self, message: str, code: Optional[int] = None):
        super().__init__(message)
        self.code = code


class FacebookRateLimitError(Exception):
    """Raised on 429 / rate limit responses."""


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------

class FacebookClient:
    """
    Wrapper around the Facebook Graph API v21.0.

    Reads credentials from environment variables:
        FB_PAGE_ACCESS_TOKEN  — long-lived page access token
        FB_PAGE_ID            — numeric page ID
        GITHUB_TOKEN          — used only for creating expiry alert issues
        GITHUB_REPO           — 'owner/repo' for the GitHub issue (optional,
                                defaults to GITHUB_REPOSITORY env var)
    """

    def __init__(self) -> None:
        self.access_token: str = os.environ.get("FB_PAGE_ACCESS_TOKEN", "")
        self.page_id: str = os.environ.get("FB_PAGE_ID", "")

        if not self.access_token:
            raise EnvironmentError("FB_PAGE_ACCESS_TOKEN environment variable is not set.")
        if not self.page_id:
            raise EnvironmentError("FB_PAGE_ID environment variable is not set.")

        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "FACE/1.0 (+GitHub Actions)"})

        # Hourly call tracking
        self._call_timestamps: list = []

        # Token validity cache (only check once per run unless forced)
        self._token_valid: Optional[bool] = None

        logger.info("FacebookClient initialised.", extra={"page_id": self.page_id})

    # -----------------------------------------------------------------------
    # Post methods
    # -----------------------------------------------------------------------

    def post_text(self, message: str) -> str:
        """
        Publish a plain-text post to the page feed.

        Args:
            message: The text content of the post.

        Returns:
            The post ID string (e.g. '123456789_987654321').
        """
        self._check_token_before_post()
        FACEBOOK_LIMITER.acquire()

        data = {
            "message": message,
            "access_token": self.access_token,
        }
        response = self._post(f"/{self.page_id}/feed", data=data)
        post_id: str = response["id"]

        logger.info("Text post published.", extra={"post_id": post_id})
        return post_id

    def post_photo(self, image_path: str, caption: str) -> str:
        """
        Upload a photo and publish it to the page feed.

        Args:
            image_path: Local filesystem path to the image (JPEG/PNG).
            caption: Caption text for the post.

        Returns:
            The post ID string.
        """
        self._check_token_before_post()
        FACEBOOK_LIMITER.acquire()

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with open(image_path, "rb") as img_file:
            files = {"source": (image_path.name, img_file, "image/jpeg")}
            data = {
                "caption": caption,
                "access_token": self.access_token,
            }
            response = self._post(
                f"/{self.page_id}/photos",
                data=data,
                files=files,
            )

        post_id: str = response.get("post_id", response.get("id", ""))
        logger.info("Photo post published.", extra={"post_id": post_id})
        return post_id

    def post_video(self, video_path: str, title: str, description: str) -> str:
        """
        Upload a video (Reel) to the page.
        Uses chunked/resumable upload for large files via the video endpoint.

        Args:
            video_path: Local path to the MP4 video file.
            title: Title of the reel.
            description: Description / caption.

        Returns:
            The video/post ID string.
        """
        self._check_token_before_post()
        FACEBOOK_LIMITER.acquire()

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        file_size = video_path.stat().st_size
        logger.info(
            "Uploading video.",
            extra={"path": str(video_path), "size_mb": round(file_size / 1_048_576, 1)},
        )

        with open(video_path, "rb") as vf:
            files = {"source": (video_path.name, vf, "video/mp4")}
            data = {
                "title": title,
                "description": description,
                "access_token": self.access_token,
            }
            # Use the video upload endpoint (supports up to 1 GB)
            response = self._post(
                f"/{self.page_id}/videos",
                data=data,
                files=files,
                timeout=300,  # video upload can take minutes
            )

        post_id: str = response.get("id", "")
        logger.info("Video post published.", extra={"post_id": post_id})
        return post_id

    def post_story_photo(self, image_path: str) -> str:
        """
        Publish a photo to the page's Story.

        Args:
            image_path: Local path to the story image (9:16 ratio recommended).

        Returns:
            The story ID string.
        """
        self._check_token_before_post()
        FACEBOOK_LIMITER.acquire()

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Story image not found: {image_path}")

        # Step 1: Upload photo without publishing (published=false)
        with open(image_path, "rb") as img_file:
            files = {"source": (image_path.name, img_file, "image/jpeg")}
            data = {
                "published": "false",
                "access_token": self.access_token,
            }
            upload_response = self._post(
                f"/{self.page_id}/photos",
                data=data,
                files=files,
            )

        photo_id = upload_response.get("id", "")

        # Step 2: Publish as story
        FACEBOOK_LIMITER.acquire()
        story_data = {
            "photo_id": photo_id,
            "access_token": self.access_token,
        }
        story_response = self._post(f"/{self.page_id}/photo_stories", data=story_data)
        story_id: str = story_response.get("id", photo_id)

        logger.info("Story photo published.", extra={"story_id": story_id})
        return story_id

    # -----------------------------------------------------------------------
    # Insights methods
    # -----------------------------------------------------------------------

    def get_post_insights(self, post_id: str) -> Dict[str, Any]:
        """
        Retrieve engagement metrics for a specific post.

        Args:
            post_id: The Facebook post ID.

        Returns:
            Dict with keys: impressions, reach, reactions, engaged_users,
            shares, comments, clicks. Values are integers (0 if unavailable).
        """
        FACEBOOK_LIMITER.acquire()
        metrics = [
            "post_impressions",
            "post_impressions_unique",
            "post_engaged_users",
            "post_reactions_by_type_total",
            "post_clicks",
        ]
        params = {
            "metric": ",".join(metrics),
            "access_token": self.access_token,
        }
        try:
            response = self._get(f"/{post_id}/insights", params=params)
            data_items = response.get("data", [])
        except FacebookAPIError as exc:
            logger.warning("Could not fetch insights for %s: %s", post_id, exc)
            data_items = []

        # Also fetch basic engagement counts
        FACEBOOK_LIMITER.acquire()
        try:
            basic = self._get(
                f"/{post_id}",
                params={
                    "fields": "reactions.summary(true),comments.summary(true),shares",
                    "access_token": self.access_token,
                },
            )
        except FacebookAPIError:
            basic = {}

        result: Dict[str, Any] = {
            "impressions": 0,
            "reach": 0,
            "engaged_users": 0,
            "reactions": 0,
            "comments": 0,
            "shares": 0,
            "clicks": 0,
        }

        for item in data_items:
            name = item.get("name", "")
            values = item.get("values", [{}])
            val = values[0].get("value", 0) if values else 0
            if name == "post_impressions":
                result["impressions"] = val
            elif name == "post_impressions_unique":
                result["reach"] = val
            elif name == "post_engaged_users":
                result["engaged_users"] = val
            elif name == "post_clicks":
                result["clicks"] = val

        if basic:
            result["reactions"] = (
                basic.get("reactions", {}).get("summary", {}).get("total_count", 0)
            )
            result["comments"] = (
                basic.get("comments", {}).get("summary", {}).get("total_count", 0)
            )
            result["shares"] = basic.get("shares", {}).get("count", 0)

        return result

    def get_page_insights(
        self,
        metric_list: List[str],
        since: str,
        until: str,
    ) -> Dict[str, Any]:
        """
        Retrieve page-level insights for a date range.

        Args:
            metric_list: List of metric names, e.g. ['page_impressions', 'page_fans'].
            since: Start date as 'YYYY-MM-DD'.
            until: End date as 'YYYY-MM-DD'.

        Returns:
            Dict mapping metric name → list of {end_time, value} dicts.
        """
        FACEBOOK_LIMITER.acquire()
        params = {
            "metric": ",".join(metric_list),
            "period": "day",
            "since": since,
            "until": until,
            "access_token": self.access_token,
        }
        response = self._get(f"/{self.page_id}/insights", params=params)
        data_items = response.get("data", [])

        result: Dict[str, Any] = {}
        for item in data_items:
            name = item.get("name", "")
            result[name] = item.get("values", [])

        return result

    # -----------------------------------------------------------------------
    # Token validity
    # -----------------------------------------------------------------------

    def check_token_validity(self) -> bool:
        """
        Check if the page access token is still valid.
        Calls GET /me?fields=id — error code 190 means token is expired.

        Returns:
            True if the token is valid, False if expired.
        """
        if self._token_valid is not None:
            return self._token_valid

        FACEBOOK_LIMITER.acquire()
        try:
            self._get("/me", params={"fields": "id", "access_token": self.access_token})
            self._token_valid = True
            logger.debug("Facebook token is valid.")
            return True
        except FacebookTokenExpiredError:
            self._token_valid = False
            self._handle_token_expiry()
            return False
        except FacebookAPIError as exc:
            logger.error("Token validity check failed: %s", exc)
            # Conservative — treat unknown errors as valid to not block the pipeline
            return True

    # -----------------------------------------------------------------------
    # Internal HTTP helpers
    # -----------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(FacebookRateLimitError),
        wait=wait_exponential(multiplier=2, min=5, max=120),
        stop=stop_after_attempt(4),
        before_sleep=before_sleep_log(logger, 20),
    )
    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make a GET request to the Graph API."""
        self._track_hourly_call()
        url = _BASE_URL + endpoint
        try:
            response = self._session.get(url, params=params, timeout=30)
        except requests.RequestException as exc:
            raise FacebookAPIError(f"Network error on GET {endpoint}: {exc}") from exc

        return self._handle_response(response, endpoint)

    @retry(
        retry=retry_if_exception_type(FacebookRateLimitError),
        wait=wait_exponential(multiplier=2, min=5, max=120),
        stop=stop_after_attempt(4),
        before_sleep=before_sleep_log(logger, 20),
    )
    def _post(
        self,
        endpoint: str,
        data: Optional[Dict] = None,
        files: Optional[Dict] = None,
        timeout: int = 60,
    ) -> Dict[str, Any]:
        """Make a POST request to the Graph API."""
        self._track_hourly_call()
        url = _BASE_URL + endpoint
        try:
            if files:
                response = self._session.post(
                    url, data=data, files=files, timeout=timeout
                )
            else:
                response = self._session.post(url, data=data, timeout=timeout)
        except requests.RequestException as exc:
            raise FacebookAPIError(f"Network error on POST {endpoint}: {exc}") from exc

        return self._handle_response(response, endpoint)

    def _handle_response(
        self, response: requests.Response, endpoint: str
    ) -> Dict[str, Any]:
        """Parse response JSON and raise appropriate exceptions on errors."""
        if response.status_code == 429:
            logger.warning("Facebook rate limit hit on %s.", endpoint)
            raise FacebookRateLimitError(f"Rate limit on {endpoint}")

        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise FacebookAPIError(
                f"Non-JSON response from {endpoint}: {response.text[:200]}"
            ) from exc

        error = body.get("error", {})
        if error:
            code = error.get("code", 0)
            sub_code = error.get("error_subcode", 0)
            msg = error.get("message", "Unknown Facebook API error")

            if code == 190 or sub_code == 190:
                logger.error("Facebook token EXPIRED. Code 190 on %s.", endpoint)
                raise FacebookTokenExpiredError(msg)

            if response.status_code in (429, 4):
                raise FacebookRateLimitError(msg)

            raise FacebookAPIError(msg, code=code)

        return body

    # -----------------------------------------------------------------------
    # Hourly call tracking
    # -----------------------------------------------------------------------

    def _track_hourly_call(self) -> None:
        """Track calls per hour; sleep if approaching 200/hr limit."""
        now = time.monotonic()
        # Evict timestamps older than 3600 seconds
        self._call_timestamps = [
            t for t in self._call_timestamps if now - t < 3600
        ]

        if len(self._call_timestamps) >= _CALL_LIMIT_BUFFER:
            # Wait until the oldest call falls out of the window
            oldest = self._call_timestamps[0]
            sleep_for = 3600 - (now - oldest) + 1
            logger.warning(
                "Facebook hourly limit approaching (%d calls). Sleeping %ds.",
                len(self._call_timestamps),
                int(sleep_for),
            )
            time.sleep(max(sleep_for, 1))

        self._call_timestamps.append(time.monotonic())

    # -----------------------------------------------------------------------
    # Token expiry handling
    # -----------------------------------------------------------------------

    def _check_token_before_post(self) -> None:
        """Raise if the token is known to be expired."""
        # Check if a previous run already recorded expiry
        expired_data = read_state(_TOKEN_EXPIRED_FILE)
        if expired_data.get("expired"):
            raise FacebookTokenExpiredError(
                "Facebook token is expired (recorded at "
                + expired_data.get("expired_at", "unknown")
                + "). Renew before posting."
            )

    def _handle_token_expiry(self) -> None:
        """
        Record token expiry to state file and create a GitHub Issue.
        Called automatically when error code 190 is detected.
        """
        now = datetime.now(timezone.utc).isoformat()

        # Write state marker so PublisherAgent can check before each post
        try:
            write_state(
                _TOKEN_EXPIRED_FILE,
                {
                    "expired": True,
                    "expired_at": now,
                    "page_id": self.page_id,
                },
            )
        except Exception as exc:
            logger.error("Failed to write token_expired.json: %s", exc)

        # Create GitHub Issue
        self._create_github_issue()

    def _create_github_issue(self) -> None:
        """
        Create a GitHub Issue titled '⚠️ FACE: Facebook token expired' with
        step-by-step renewal instructions.

        Uses GITHUB_TOKEN (auto-provided in GitHub Actions) and
        GITHUB_REPOSITORY (e.g. 'owner/repo') env vars.
        """
        github_token = os.environ.get("GITHUB_TOKEN", "")
        repo = os.environ.get("GITHUB_REPOSITORY", os.environ.get("GITHUB_REPO", ""))

        if not github_token or not repo:
            logger.warning(
                "Cannot create GitHub Issue: GITHUB_TOKEN or GITHUB_REPOSITORY not set."
            )
            return

        title = "⚠️ FACE: Facebook token expired — action required"
        body = (
            "## Facebook Page Access Token Has Expired\n\n"
            "The FACE pipeline cannot publish posts until the token is renewed.\n\n"
            "### Steps to Fix\n\n"
            "1. Go to [Meta for Developers](https://developers.facebook.com/tools/explorer/).\n"
            "2. Select your app and click **Get User Access Token**.\n"
            "3. Grant `pages_manage_posts`, `pages_read_engagement`, `publish_pages` permissions.\n"
            "4. Click **Generate Access Token** and copy the token.\n"
            "5. Go to the [Token Debugger](https://developers.facebook.com/tools/debug/accesstoken/) "
            "and exchange for a **Long-Lived Token** (60 days).\n"
            "6. Exchange for a **Page Access Token** using:\n"
            "   `GET /me/accounts?access_token=<long_lived_user_token>`\n"
            "7. Copy the `access_token` value for your page.\n"
            "8. In your GitHub repository, go to **Settings → Secrets and variables → Actions**.\n"
            "9. Update the `FB_PAGE_ACCESS_TOKEN` secret with the new token.\n"
            "10. Re-run the failed workflow, or wait for the next scheduled run.\n\n"
            "### Delete the Expiry Marker\n\n"
            "After updating the secret, delete `state/token_expired.json` from the repo "
            "to allow the pipeline to resume.\n\n"
            "---\n"
            f"*Auto-created by FACE at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*"
        )

        url = f"https://api.github.com/repos/{repo}/issues"
        headers = {
            "Authorization": f"Bearer {github_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        payload = {
            "title": title,
            "body": body,
            "labels": ["automation", "urgent"],
        }

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=15)
            if resp.status_code == 201:
                issue_url = resp.json().get("html_url", "")
                logger.info(
                    "GitHub Issue created for token expiry.",
                    extra={"issue_url": issue_url},
                )
            else:
                logger.error(
                    "Failed to create GitHub Issue. Status %d: %s",
                    resp.status_code,
                    resp.text[:200],
                )
        except requests.RequestException as exc:
            logger.error("GitHub Issue creation failed with network error: %s", exc)
