"""
agents/content_strategy.py
===========================
ContentStrategyAgent — Session 3, Deliverable 2

Reads today's top trends and algorithm rules, then produces a structured
daily content calendar written to state/content_plan.json.

Run:
    python agents/content_strategy.py
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any

from utils.config_loader import get_config, build_prompt_context, get_posting_times_utc
from utils.gemini_client import get_gemini_client, GeminiQuotaExhaustedError
from utils.logger import get_logger, log_exception
from utils.rate_limiter import GEMINI_LIMITER
from utils.state_manager import read_state, write_state, state_exists

# ── Logger ────────────────────────────────────────────────────────────────────
log = get_logger("content_strategy")

# ── Constants ─────────────────────────────────────────────────────────────────

# Content pillar pool — rotated to avoid back-to-back repeats
CONTENT_PILLARS = [
    "educational",
    "entertainment",
    "inspirational",
    "promotional",
    "community",
    "behind_the_scenes",
    "user_generated",
    "trending_commentary",
]

# Time-of-day → preferred content character
TIME_SLOT_PROFILE: dict[str, dict] = {
    "morning":   {   # 05:00–11:59 UTC
        "preferred_formats": ["single_image", "carousel", "text"],
        "hook_types": ["question", "statistic", "bold_claim"],
        "dominant_emotions": ["curiosity", "aspiration", "trust"],
        "pillars": ["educational", "inspirational"],
    },
    "afternoon": {   # 12:00–17:59 UTC
        "preferred_formats": ["carousel", "reel", "single_image"],
        "hook_types": ["listicle", "question", "bold_claim"],
        "dominant_emotions": ["curiosity", "urgency", "joy"],
        "pillars": ["entertainment", "trending_commentary", "community"],
    },
    "evening":   {   # 18:00–23:59 UTC
        "preferred_formats": ["reel", "story", "single_image"],
        "hook_types": ["story", "bold_claim", "question"],
        "dominant_emotions": ["nostalgia", "joy", "aspiration"],
        "pillars": ["entertainment", "behind_the_scenes", "community"],
    },
    "night":     {   # 00:00–04:59 UTC
        "preferred_formats": ["text", "single_image"],
        "hook_types": ["question", "statistic"],
        "dominant_emotions": ["trust", "curiosity"],
        "pillars": ["educational", "inspirational"],
    },
}

CTA_TYPES = ["comment", "share", "follow", "link", "save"]

# Day-of-week index (Mon=0) → whether a reel is expected this day
# Default schedule: reels on Wednesday (2) and Saturday (5)
DEFAULT_REEL_DAYS = {2, 5}

# Default algorithm rules used when state/algorithm_rules.json is absent
DEFAULT_ALGORITHM_RULES: dict = {
    "optimal_posting_times": [],
    "recommended_formats": ["single_image", "carousel", "reel"],
    "max_hashtags": 20,
    "engagement_triggers": ["question", "poll", "challenge"],
    "avoid": [],
    "notes": "Default rules (algorithm_rules.json not yet generated).",
}

# Default performance report fallback
DEFAULT_PERF_REPORT: dict = {
    "top_performing_formats": ["carousel", "reel"],
    "top_performing_pillars": ["educational", "entertainment"],
    "avg_engagement_rate": 0.0,
    "recommendations": [],
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _time_of_day(hour_utc: int) -> str:
    """Return time-slot label for a UTC hour."""
    if 5 <= hour_utc < 12:
        return "morning"
    if 12 <= hour_utc < 18:
        return "afternoon"
    if 18 <= hour_utc < 24:
        return "evening"
    return "night"


def _pick_pillar(
    slot_idx: int,
    prev_pillar: str | None,
    slot_profile: dict,
    preferred_pillars: list[str],
) -> str:
    """
    Choose a content pillar for this slot.
    Avoid repeating the previous pillar. Prefer pillars suggested by
    the slot profile and the performance report.
    """
    candidates = list(slot_profile.get("pillars", CONTENT_PILLARS))
    # Add performance-preferred pillars to front of candidate list
    for p in reversed(preferred_pillars):
        if p not in candidates:
            candidates.insert(0, p)
    # Remove previous pillar to avoid back-to-back repeat
    if prev_pillar and len(candidates) > 1:
        candidates = [p for p in candidates if p != prev_pillar]
    return candidates[slot_idx % len(candidates)] if candidates else "educational"


def _assign_format(
    slot_profile: dict,
    algo_formats: list[str],
    is_reel_slot: bool,
    is_story_slot: bool,
) -> str:
    """
    Decide the content format for a given slot.
    Priority: explicit reel/story slot > algo-recommended > time-profile preferred.
    """
    if is_reel_slot:
        return "reel"
    if is_story_slot:
        return "story"
    # Intersect algo recommendations with time-profile preferences
    preferred = slot_profile.get("preferred_formats", ["single_image"])
    if algo_formats:
        cross = [f for f in preferred if f in algo_formats]
        if cross:
            return cross[0]
    return preferred[0] if preferred else "single_image"


def _is_reel_day(cfg: dict) -> bool:
    """Return True if today (UTC) is a scheduled reel day."""
    reel_days_raw = cfg.get("posting_schedule", {}).get("reel_days", list(DEFAULT_REEL_DAYS))
    try:
        reel_days = set(reel_days_raw)
    except TypeError:
        reel_days = DEFAULT_REEL_DAYS
    today_weekday = datetime.now(timezone.utc).weekday()   # Mon=0
    return today_weekday in reel_days


def _trend_for_slot(
    slot_idx: int,
    trends: list[dict],
    used_topics: set[str],
) -> dict:
    """
    Select the best unused trend for this slot.
    Falls back to reuse if all trends are exhausted.
    """
    for trend in trends:
        if trend["topic"] not in used_topics:
            return trend
    # All trends used — cycle back to top
    return trends[slot_idx % len(trends)]

# ─────────────────────────────────────────────────────────────────────────────
# Gemini strategy call
# ─────────────────────────────────────────────────────────────────────────────

def _strategy_for_slot(
    client,
    ctx: dict,
    trend: dict,
    slot_info: dict,
    algo_rules: dict,
) -> dict:
    """
    Call Gemini strategy_decision.txt for one post slot.
    Returns a dict with format, pillar, hook_type, cta_type, dominant_emotion,
    optimal_post_time. Falls back to slot defaults on any failure.
    """
    default_profile = TIME_SLOT_PROFILE[slot_info["time_of_day"]]
    fallback = {
        "format":          slot_info["format"],
        "content_pillar":  slot_info["pillar"],
        "hook_type":       default_profile["hook_types"][0],
        "cta_type":        "comment",
        "dominant_emotion": default_profile["dominant_emotions"][0],
        "optimal_post_time": slot_info["scheduled_time_utc"],
    }
    try:
        GEMINI_LIMITER.acquire()
        result = client.generate_json(
            "strategy_decision.txt",
            **ctx,
            trend_topic=trend["topic"],
            trend_source=trend["source"],
            virality_score=str(trend.get("virality_score", 7)),
            niche_fit_score=str(trend.get("niche_fit_score", 7)),
            recommended_format=trend.get("recommended_format", "single_image"),
            hook_angle=trend.get("hook_angle", ""),
            time_of_day=slot_info["time_of_day"],
            slot_number=str(slot_info["slot"]),
            scheduled_time_utc=slot_info["scheduled_time_utc"],
            algorithm_rules=json.dumps(algo_rules),
        )
        # Merge: Gemini output wins, fallback fills missing keys
        merged = {**fallback, **result}
        return merged
    except GeminiQuotaExhaustedError:
        raise   # propagate — caller halts loop
    except Exception as exc:           # noqa: BLE001
        log_exception(log, exc, f"Gemini strategy failed for slot {slot_info['slot']}")
        return fallback

# ─────────────────────────────────────────────────────────────────────────────
# Build content plan
# ─────────────────────────────────────────────────────────────────────────────

def _build_content_plan(
    cfg: dict,
    trends: list[dict],
    algo_rules: dict,
    perf_report: dict,
    posting_times_utc: list[str],
) -> dict:
    """
    Build the full daily content calendar dict.
    """
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    client = get_gemini_client()
    ctx    = build_prompt_context()

    reel_day   = _is_reel_day(cfg)
    algo_formats = algo_rules.get("recommended_formats", [])
    preferred_pillars = perf_report.get("top_performing_pillars", [])

    posts:   list[dict] = []
    stories: list[dict] = []
    reels:   list[dict] = []

    prev_pillar: str | None = None
    used_topics: set[str]   = set()
    reel_assigned            = False
    story_count              = 0
    max_stories              = 2   # derive story slots from top 2 posts

    # Each posting_time_utc string is "HH:MM" or full ISO; parse hour
    for slot_idx, raw_time in enumerate(posting_times_utc):
        slot_num = slot_idx + 1

        # Parse hour from "HH:MM" or ISO string
        try:
            if "T" in raw_time:
                hour_utc = int(raw_time.split("T")[1][:2])
            else:
                hour_utc = int(raw_time.split(":")[0])
        except (ValueError, IndexError):
            hour_utc = 12   # noon fallback

        time_label = _time_of_day(hour_utc)
        slot_profile = TIME_SLOT_PROFILE[time_label]

        # Build scheduled datetime string for today
        try:
            if "T" in raw_time:
                scheduled_time_utc = raw_time
            else:
                scheduled_time_utc = f"{today_str}T{raw_time}:00+00:00"
        except Exception:
            scheduled_time_utc = f"{today_str}T12:00:00+00:00"

        # Format: assign reel to first afternoon slot on reel days
        is_reel_slot  = reel_day and not reel_assigned and time_label == "afternoon"
        is_story_slot = False   # stories are derived after posts

        assigned_format = _assign_format(
            slot_profile, algo_formats, is_reel_slot, is_story_slot
        )
        if assigned_format == "reel":
            reel_assigned = True

        # Pillar
        pillar = _pick_pillar(slot_idx, prev_pillar, slot_profile, preferred_pillars)
        prev_pillar = pillar

        # Trend selection
        trend = _trend_for_slot(slot_idx, trends, used_topics)
        used_topics.add(trend["topic"])

        slot_info = {
            "slot": slot_num,
            "time_of_day": time_label,
            "scheduled_time_utc": scheduled_time_utc,
            "format": assigned_format,
            "pillar": pillar,
        }

        # Gemini strategy
        try:
            strategy = _strategy_for_slot(client, ctx, trend, slot_info, algo_rules)
        except GeminiQuotaExhaustedError:
            log.critical("Gemini quota exhausted during strategy planning — halting slot loop")
            break

        post_id = str(uuid.uuid4())
        post_entry = {
            "post_id":            post_id,
            "slot":               slot_num,
            "scheduled_time_utc": strategy.get("optimal_post_time", scheduled_time_utc),
            "format":             strategy.get("format", assigned_format),
            "trend_topic":        trend["topic"],
            "content_pillar":     strategy.get("content_pillar", pillar),
            "hook_type":          strategy.get("hook_type", slot_profile["hook_types"][0]),
            "cta_type":           strategy.get("cta_type", "comment"),
            "dominant_emotion":   strategy.get("dominant_emotion",
                                               slot_profile["dominant_emotions"][0]),
            "status":             "planned",
        }
        posts.append(post_entry)

        # Promote top 2 posts to stories
        if story_count < max_stories and post_entry["format"] != "story":
            story_id = str(uuid.uuid4())
            stories.append({
                "post_id":            story_id,
                "slot":               slot_num,
                "scheduled_time_utc": scheduled_time_utc,
                "format":             "story",
                "trend_topic":        trend["topic"],
                "content_pillar":     pillar,
                "hook_type":          "bold_claim",
                "cta_type":           "follow",
                "dominant_emotion":   "curiosity",
                "parent_post_id":     post_id,
                "status":             "planned",
            })
            story_count += 1

        # Track reel slots separately
        if post_entry["format"] == "reel":
            reels.append(post_entry)

    total_pieces = len(posts) + len(stories)
    return {
        "date":          today_str,
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "posts":         posts,
        "stories":       stories,
        "reels":         reels,
        "total_pieces":  total_pieces,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Main agent
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point for ContentStrategyAgent."""
    log.info("=== ContentStrategyAgent starting ===")
    start_time = time.time()

    # ── Load config ───────────────────────────────────────────────────────────
    try:
        cfg = get_config()
    except Exception as exc:
        log_exception(log, exc, "Failed to load config — aborting")
        raise SystemExit(1)

    # ── Load trends (required) ────────────────────────────────────────────────
    if not state_exists("trends.json"):
        log.error("state/trends.json missing — run TrendHunterAgent first")
        raise SystemExit(1)

    try:
        trends_state = read_state("trends.json")
        trends = trends_state.get("trends", [])
    except Exception as exc:
        log_exception(log, exc, "Failed to read state/trends.json")
        raise SystemExit(1)

    if not trends:
        log.error("No trends found in state/trends.json — aborting")
        raise SystemExit(1)

    log.info("Loaded %d trends", len(trends))

    # ── Load algorithm rules (optional — first-run default) ───────────────────
    if state_exists("algorithm_rules.json"):
        try:
            algo_rules = read_state("algorithm_rules.json")
            log.info("Loaded algorithm_rules.json")
        except Exception as exc:
            log_exception(log, exc, "Failed to read algorithm_rules.json — using defaults")
            algo_rules = DEFAULT_ALGORITHM_RULES
    else:
        log.info("algorithm_rules.json not found — using defaults (first run)")
        algo_rules = DEFAULT_ALGORITHM_RULES

    # ── Load performance report (optional) ────────────────────────────────────
    if state_exists("performance_report.json"):
        try:
            perf_report = read_state("performance_report.json")
            log.info("Loaded performance_report.json")
        except Exception as exc:
            log_exception(log, exc, "Failed to read performance_report.json — using defaults")
            perf_report = DEFAULT_PERF_REPORT
    else:
        log.info("performance_report.json not found — using defaults (first run)")
        perf_report = DEFAULT_PERF_REPORT

    # ── Get posting times from config ─────────────────────────────────────────
    try:
        posting_times = get_posting_times_utc()
    except Exception as exc:
        log_exception(log, exc, "Could not get posting times — falling back to [09:00, 15:00, 19:00]")
        posting_times = ["09:00", "15:00", "19:00"]

    if not posting_times:
        log.warning("No posting times configured — using defaults")
        posting_times = ["09:00", "15:00", "19:00"]

    log.info("Planning %d post slots: %s", len(posting_times), posting_times)

    # ── Build content plan ────────────────────────────────────────────────────
    log.info("Building daily content plan with Gemini strategy calls")
    try:
        content_plan = _build_content_plan(
            cfg, trends, algo_rules, perf_report, posting_times
        )
    except GeminiQuotaExhaustedError as exc:
        log.critical("Gemini quota exhausted — writing partial plan: %s", exc)
        raise SystemExit(1)
    except Exception as exc:
        log_exception(log, exc, "Unexpected error building content plan")
        raise SystemExit(1)

    # ── Write state ───────────────────────────────────────────────────────────
    try:
        write_state("content_plan.json", content_plan)
        log.info(
            "Wrote content_plan.json: %d posts, %d stories, %d reels, %d total",
            len(content_plan["posts"]),
            len(content_plan["stories"]),
            len(content_plan["reels"]),
            content_plan["total_pieces"],
        )
    except Exception as exc:
        log_exception(log, exc, "Failed to write state/content_plan.json")
        raise SystemExit(1)

    elapsed = round(time.time() - start_time, 1)
    log.info("=== ContentStrategyAgent complete in %.1fs ===", elapsed)


if __name__ == "__main__":
    main()
