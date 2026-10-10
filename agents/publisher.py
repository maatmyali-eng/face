"""
agents/publisher.py
────────────────────────────────────────────────────────────────────────────
PublisherAgent — FACE System (Session 5)

Posts all ready content to the Facebook Page at their scheduled times.
Runs as the final step of the daily pipeline.

Cron trigger: every 15 minutes during posting windows (see GitHub Actions YAML)
Entry point: main()
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import sys
import time
import json
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

# ── Utils ────────────────────────────────────────────────────────────────────
from utils.config_loader import get_config
from utils.state_manager import read_state, write_state, update_state, append_to_state, state_exists
from utils.logger import get_logger, log_exception
from utils.facebook_client import FacebookClient, FacebookTokenExpiredError
from utils.rate_limiter import FACEBOOK_LIMITER

log = get_logger("publisher")

# ── Constants ────────────────────────────────────────────────────────────────
SCHEDULE_WINDOW_MINUTES = 10          # publish if within ±10 min of scheduled time
RETRY_MAX_ATTEMPTS = 3
RETRY_BASE_DELAY = 5                  # seconds
MAX_REEL_SIZE_MB = 100
STATE_GENERATED = "state/generated_content.json"
STATE_CONTENT_PLAN = "state/content_plan.json"
STATE_POST_HISTORY = "state/post_history.json"
STATE_PUBLISH_LOG = "state/publish_log.json"
STATE_SKIPPED = "state/skipped.json"
STATE_ERRORS = "state/errors.json"
STATE_TOKEN_EXPIRED = "state/token_expired.json"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _is_within_window(scheduled_iso: str, window_minutes: int = SCHEDULE_WINDOW_MINUTES) -> bool:
    """Return True if the current UTC time is within ±window_minutes of scheduled_iso."""
    try:
        scheduled = datetime.fromisoformat(scheduled_iso)
        if scheduled.tzinfo is None:
            scheduled = scheduled.replace(tzinfo=timezone.utc)
        delta = abs((_now_utc() - scheduled).total_seconds())
        return delta <= window_minutes * 60
    except Exception as exc:
        log.warning(f"Could not parse scheduled_time '{scheduled_iso}': {exc}")
        return False


def _build_full_caption(item: dict) -> str:
    """Concatenate caption and hashtags into the Facebook-ready string."""
    caption = item.get("caption", "").strip()
    hashtags = item.get("hashtags", [])
    tag_str = " ".join(h if h.startswith("#") else f"#{h}" for h in hashtags)
    if tag_str:
        return f"{caption}\n\n{tag_str}"
    return caption


def _get_reel_size_mb(reel_path: str) -> float:
    try:
        return Path(reel_path).stat().st_size / (1024 * 1024)
    except Exception:
        return 0.0


def _schedule_map(plan: dict) -> dict[str, dict]:
    """Build post_id → {scheduled_time_utc, content_pillar, hook_type, ...} from content_plan."""
    mapping: dict[str, dict] = {}
    for key in ("posts", "stories", "reels"):
        for entry in plan.get(key, []):
            pid = entry.get("post_id")
            if pid:
                mapping[pid] = entry
    return mapping


def _write_skipped(post_id: str, format_: str, topic: str, reason: str) -> None:
    record = {
        "post_id": post_id,
        "skipped_at": _now_utc().isoformat(),
        "reason": reason,
        "format": format_,
        "topic": topic,
    }
    try:
        if state_exists(STATE_SKIPPED):
            append_to_state(STATE_SKIPPED, "skipped", record)
        else:
            write_state(STATE_SKIPPED, {"skipped": [record]})
    except Exception as exc:
        log.error(f"Could not write skipped record for {post_id}: {exc}")


def _write_error(post_id: str, message: str) -> None:
    record = {
        "post_id": post_id,
        "error": message,
        "timestamp": _now_utc().isoformat(),
        "agent": "publisher",
    }
    try:
        if state_exists(STATE_ERRORS):
            append_to_state(STATE_ERRORS, "errors", record)
        else:
            write_state(STATE_ERRORS, {"errors": [record]})
    except Exception as exc:
        log.error(f"Could not write error record for {post_id}: {exc}")


def _append_post_history(fb_post_id: str, item: dict, plan_entry: dict) -> None:
    record = {
        "post_id": fb_post_id,
        "internal_id": item.get("post_id", ""),
        "published_at": _now_utc().isoformat(),
        "format": item.get("format", ""),
        "trend_topic": item.get("trend_topic", plan_entry.get("trend_topic", "")),
        "content_pillar": plan_entry.get("content_pillar", ""),
        "hook_type": plan_entry.get("hook_type", ""),
        "caption_length": len(item.get("caption", "")),
        "hashtag_count": len(item.get("hashtags", [])),
        "image_provider": item.get("image_provider_used", ""),
        "visual_style": item.get("visual_style_used", ""),
        "has_text_overlay": bool(item.get("image_path")),
        "is_reel": item.get("format") == "reel",
        "scheduled_time": plan_entry.get("scheduled_time_utc", ""),
        "actual_publish_time": _now_utc().isoformat(),
        "insights": None,
    }
    try:
        if state_exists(STATE_POST_HISTORY):
            append_to_state(STATE_POST_HISTORY, "posts", record)
        else:
            write_state(STATE_POST_HISTORY, {"posts": [record]})
    except Exception as exc:
        log.error(f"Could not write post_history record for {fb_post_id}: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Format-specific publish functions
# ─────────────────────────────────────────────────────────────────────────────

def _publish_image(fb: FacebookClient, item: dict) -> str:
    """Publish single_image or carousel post. Returns Facebook post_id."""
    image_path = item.get("image_path", "")
    if not item.get("image_ready") or not image_path or not Path(image_path).exists():
        raise ValueError(f"image_ready=False or missing image_path for {item['post_id']}")
    caption = _build_full_caption(item)
    FACEBOOK_LIMITER.acquire()
    return fb.post_photo(image_path=image_path, caption=caption)


def _publish_text(fb: FacebookClient, item: dict) -> str:
    """Publish text-only post. Returns Facebook post_id."""
    message = _build_full_caption(item)
    FACEBOOK_LIMITER.acquire()
    return fb.post_text(message=message)


def _publish_story(fb: FacebookClient, item: dict) -> str:
    """Publish story slides. Returns last story Facebook post_id (or synthetic id)."""
    slides = item.get("story_slides") or []
    if not slides:
        raise ValueError(f"No story_slides for {item['post_id']}")
    last_id = ""
    for slide in slides:
        slide_path = slide.get("image_path", "")
        if not slide_path or not Path(slide_path).exists():
            log.warning(f"Story slide missing image_path, skipping: {slide}")
            continue
        FACEBOOK_LIMITER.acquire()
        last_id = fb.post_story_photo(image_path=slide_path)
    if not last_id:
        raise ValueError(f"No story slides successfully published for {item['post_id']}")
    return last_id


def _publish_reel(fb: FacebookClient, item: dict) -> str:
    """Publish reel video. Returns Facebook post_id."""
    reel_path = item.get("reel_path", "")
    if not item.get("reel_ready") or not reel_path or not Path(reel_path).exists():
        raise ValueError(f"reel_ready=False or missing reel_path for {item['post_id']}")
    size_mb = _get_reel_size_mb(reel_path)
    if size_mb > MAX_REEL_SIZE_MB:
        raise ValueError(
            f"Reel too large ({size_mb:.1f} MB > {MAX_REEL_SIZE_MB} MB) for {item['post_id']}"
        )
    caption = _build_full_caption(item)
    title = item.get("reel_title", item.get("trend_topic", ""))
    FACEBOOK_LIMITER.acquire()
    return fb.post_video(video_path=reel_path, title=title, description=caption)


def _dispatch_publish(fb: FacebookClient, item: dict) -> str:
    """Route to the correct publish function by format. Returns Facebook post_id."""
    fmt = item.get("format", "")
    if fmt in ("single_image", "carousel"):
        return _publish_image(fb, item)
    elif fmt == "text":
        return _publish_text(fb, item)
    elif fmt == "story":
        return _publish_story(fb, item)
    elif fmt == "reel":
        return _publish_reel(fb, item)
    else:
        raise ValueError(f"Unknown format '{fmt}' for post {item.get('post_id')}")


# ─────────────────────────────────────────────────────────────────────────────
# Core publish loop
# ─────────────────────────────────────────────────────────────────────────────

def _publish_with_retry(fb: FacebookClient, item: dict) -> str:
    """Attempt to publish item up to RETRY_MAX_ATTEMPTS times. Returns FB post_id."""
    last_exc: Exception = RuntimeError("No attempts made")
    for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
        try:
            fb_post_id = _dispatch_publish(fb, item)
            log.info(
                f"[{item['post_id']}] Published (attempt {attempt}/{RETRY_MAX_ATTEMPTS}) "
                f"→ fb_post_id={fb_post_id}"
            )
            return fb_post_id
        except FacebookTokenExpiredError:
            raise  # bubble up immediately — no retry
        except Exception as exc:
            last_exc = exc
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            log.warning(
                f"[{item['post_id']}] Publish attempt {attempt} failed: {exc}. "
                f"Retrying in {delay}s…"
            )
            time.sleep(delay)
    raise last_exc


def run_publishing_loop(
    fb: FacebookClient,
    content: list[dict],
    schedule_map: dict[str, dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Iterate over ready content items and publish those in the schedule window.

    Returns: (published_items, failed_items, skipped_items)
    """
    published, failed, skipped = [], [], []
    now = _now_utc()

    for item in content:
        pid = item.get("post_id", "")
        fmt = item.get("format", "unknown")
        topic = item.get("trend_topic", "")
        status = item.get("status", "")

        # Only process "ready" items
        if status != "ready":
            log.debug(f"[{pid}] status='{status}' — skipping")
            continue

        plan_entry = schedule_map.get(pid, {})
        scheduled_time = plan_entry.get("scheduled_time_utc", "")

        if not scheduled_time:
            log.warning(f"[{pid}] No scheduled_time_utc in content_plan — skipping")
            _write_skipped(pid, fmt, topic, "no_scheduled_time")
            skipped.append({"post_id": pid, "format": fmt, "status": "skipped", "reason": "no_scheduled_time"})
            continue

        if not _is_within_window(scheduled_time):
            log.debug(f"[{pid}] Not in publish window (scheduled={scheduled_time}) — skipping this run")
            continue

        log.info(f"[{pid}] In publish window — attempting {fmt} post…")

        try:
            fb_post_id = _publish_with_retry(fb, item)
        except FacebookTokenExpiredError:
            log.critical("Facebook token expired mid-run — halting all publishing")
            # Mark current item as failed and stop loop
            _write_error(pid, "token_expired")
            failed.append({"post_id": pid, "format": fmt, "status": "failed", "error": "token_expired"})
            break
        except Exception as exc:
            log_exception(log, f"[{pid}] All retries exhausted", exc)
            _write_error(pid, str(exc))
            # Update status in generated_content to "failed"
            item["status"] = "failed"
            failed.append({"post_id": pid, "format": fmt, "status": "failed", "error": str(exc)})
            continue

        # ── Success ────────────────────────────────────────────────────
        item["status"] = "published"
        item["fb_post_id"] = fb_post_id
        _append_post_history(fb_post_id, item, plan_entry)
        published.append({"post_id": fb_post_id, "format": fmt, "status": "published"})

    return published, failed, skipped


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """PublisherAgent entry point."""
    log.info("══ PublisherAgent starting ══")
    run_at = _now_utc()

    # ── Pre-flight: token_expired halt flag ───────────────────────────────
    if state_exists(STATE_TOKEN_EXPIRED):
        log.warning("state/token_expired.json exists — Facebook token expired. Skipping all publishing.")
        sys.exit(0)

    # ── Pre-flight: token validity check ─────────────────────────────────
    try:
        fb = FacebookClient()
        if not fb.check_token_validity():
            log.critical("Facebook token is invalid or expired — halting publisher.")
            # FacebookClient.check_token_validity() already creates the GitHub Issue
            sys.exit(0)
    except FacebookTokenExpiredError:
        log.critical("FacebookTokenExpiredError on startup — halting publisher.")
        sys.exit(0)
    except Exception as exc:
        log_exception(log, "Could not initialize FacebookClient", exc)
        sys.exit(1)

    # ── Load state ────────────────────────────────────────────────────────
    try:
        generated = read_state(STATE_GENERATED)
        content_items: list[dict] = generated.get("content", [])
    except Exception as exc:
        log_exception(log, f"Cannot read {STATE_GENERATED}", exc)
        sys.exit(1)

    try:
        plan = read_state(STATE_CONTENT_PLAN)
        schedule_map = _schedule_map(plan)
    except Exception as exc:
        log_exception(log, f"Cannot read {STATE_CONTENT_PLAN}", exc)
        schedule_map = {}

    log.info(f"Loaded {len(content_items)} content items; {len(schedule_map)} plan entries")

    # ── Publishing loop ───────────────────────────────────────────────────
    published, failed, skipped = run_publishing_loop(fb, content_items, schedule_map)

    # ── Persist updated generated_content (status changes) ───────────────
    try:
        generated["content"] = content_items
        write_state(STATE_GENERATED, generated)
    except Exception as exc:
        log_exception(log, "Could not persist updated generated_content.json", exc)

    # ── Write publish log ─────────────────────────────────────────────────
    publish_log = {
        "date": run_at.strftime("%Y-%m-%d"),
        "run_at": run_at.isoformat(),
        "published_count": len(published),
        "failed_count": len(failed),
        "skipped_count": len(skipped),
        "items": published + failed + skipped,
    }
    try:
        write_state(STATE_PUBLISH_LOG, publish_log)
    except Exception as exc:
        log_exception(log, "Could not write publish_log.json", exc)

    log.info(
        f"══ PublisherAgent done — "
        f"published={len(published)} failed={len(failed)} skipped={len(skipped)} ══"
    )


if __name__ == "__main__":
    main()
