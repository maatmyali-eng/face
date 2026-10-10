"""
agents/algorithm_watcher.py
────────────────────────────────────────────────────────────────────────────
AlgorithmWatcherAgent — FACE System (Session 5)

Monthly research agent. Calls Gemini to update three state files:
  • state/algorithm_rules.json  — FB algorithm signals, format rankings, CTAs
  • state/visual_style.json     — Visual trend guidance for the month
  • state/best_practices.json   — Human-readable merged summary

Cron trigger: 1st of every month (00:30 UTC)
Entry point: main()
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

# ── Utils ────────────────────────────────────────────────────────────────────
from utils.config_loader import get_config, build_prompt_context
from utils.state_manager import read_state, write_state, state_exists
from utils.logger import get_logger, log_exception
from utils.gemini_client import GeminiClient, get_gemini_client, GeminiQuotaExhaustedError
from utils.rate_limiter import GEMINI_LIMITER

log = get_logger("algorithm_watcher")

# ── Prompt paths ──────────────────────────────────────────────────────────────
PROMPT_ALGORITHM_RESEARCH = "prompts/algorithm_research.txt"
PROMPT_VISUAL_TREND = "prompts/visual_trend_researcher.txt"

# ── State paths ───────────────────────────────────────────────────────────────
STATE_ALGORITHM_RULES = "state/algorithm_rules.json"
STATE_VISUAL_STYLE = "state/visual_style.json"
STATE_BEST_PRACTICES = "state/best_practices.json"


# ─────────────────────────────────────────────────────────────────────────────
# Default fallbacks (used when Gemini fails to parse)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_ALGORITHM_RULES: dict[str, Any] = {
    "top_ranking_signals": [
        "video content (Reels) gets highest organic reach",
        "posts with high save rates are boosted",
        "comments and replies signal quality conversations",
        "posting consistency matters for algorithm trust",
        "early engagement in first 60 minutes is heavily weighted",
    ],
    "content_formats_ranked": ["reel", "carousel", "single_image", "text", "story"],
    "optimal_post_times_utc": ["06:00", "12:00", "18:00"],
    "engagement_bait_to_avoid": [
        "like and share if you agree",
        "tag 3 friends to win",
        "comment YES or NO",
    ],
    "current_policy_highlights": [
        "no misleading health claims",
        "no sensational political content",
        "branded content requires disclosure",
    ],
    "hashtag_strategy": {
        "recommended_count": 15,
        "mix": "5 broad + 5 niche + 5 community",
        "avoid": "banned or overused hashtags",
    },
    "caption_length_sweet_spot": {
        "image": "80-150 characters",
        "reel": "50-100 characters",
        "carousel": "100-200 characters",
    },
    "best_cta_types_this_month": ["save", "comment", "share"],
    "region_specific_tips": [],
}

DEFAULT_VISUAL_STYLE: dict[str, Any] = {
    "optimal_image_style": "clean minimal flat illustration",
    "optimal_reel_style": "fast-cut with text overlays, trending audio",
    "trending_color_palettes": [
        {"name": "earthy neutrals", "hex_examples": ["#C4A882", "#7D6B5D", "#E8D5B7"]},
        {"name": "bold primaries", "hex_examples": ["#FF3B30", "#007AFF", "#34C759"]},
    ],
    "trending_video_editing_styles": ["jump cuts", "B-roll transitions", "text pop animations"],
    "content_formats_gaining": ["short-form vertical video", "interactive carousels"],
    "content_formats_losing": ["horizontal video", "text-only posts"],
    "cultural_moments_this_month": [],
    "visual_taboos": [
        "heavily filtered faces",
        "stock photo clichés",
        "low-resolution images",
    ],
    "recommended_visual_updates": [],
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _first_of_next_month(dt: datetime) -> datetime:
    """Return the first day of the month following dt."""
    if dt.month == 12:
        return dt.replace(year=dt.year + 1, month=1, day=1,
                          hour=0, minute=0, second=0, microsecond=0)
    return dt.replace(month=dt.month + 1, day=1,
                      hour=0, minute=0, second=0, microsecond=0)


def _safe_parse(raw: Any, default: dict, label: str) -> dict:
    """Return parsed dict or default if raw is not a usable dict."""
    if isinstance(raw, dict) and raw:
        return raw
    log.warning(f"{label}: Gemini returned non-dict result — using default fallback")
    return default


def _build_context(cfg: dict) -> dict:
    """Build the prompt injection context dict for algorithm research."""
    ctx = build_prompt_context()
    now = _now_utc()
    ctx["current_month"] = now.strftime("%B")
    ctx["current_year"] = now.strftime("%Y")
    ctx["current_month_year"] = now.strftime("%B %Y")
    return ctx


# ─────────────────────────────────────────────────────────────────────────────
# Part A — Algorithm + Policy Research
# ─────────────────────────────────────────────────────────────────────────────

def research_algorithm_rules(client: GeminiClient, ctx: dict) -> dict:
    """
    Call Gemini with algorithm_research.txt prompt.
    Returns a validated algorithm_rules dict.
    """
    log.info("Part A — Researching Facebook algorithm rules…")
    try:
        GEMINI_LIMITER.acquire()
        result = client.generate_json(PROMPT_ALGORITHM_RESEARCH, **ctx)
        rules = _safe_parse(result, DEFAULT_ALGORITHM_RULES, "algorithm_rules")
        log.info("Algorithm rules received from Gemini")
        return rules
    except GeminiQuotaExhaustedError:
        log.critical("Gemini quota exhausted during algorithm research — using defaults")
        return DEFAULT_ALGORITHM_RULES
    except Exception as exc:
        log_exception(log, "Algorithm research Gemini call failed", exc)
        return DEFAULT_ALGORITHM_RULES


# ─────────────────────────────────────────────────────────────────────────────
# Part B — Visual Trend Research
# ─────────────────────────────────────────────────────────────────────────────

def research_visual_style(client: GeminiClient, ctx: dict) -> dict:
    """
    Call Gemini with visual_trend_researcher.txt prompt.
    Returns a validated visual_style dict.
    """
    log.info("Part B — Researching visual trends…")
    try:
        GEMINI_LIMITER.acquire()
        result = client.generate_json(PROMPT_VISUAL_TREND, **ctx)
        style = _safe_parse(result, DEFAULT_VISUAL_STYLE, "visual_style")
        log.info("Visual style data received from Gemini")
        return style
    except GeminiQuotaExhaustedError:
        log.critical("Gemini quota exhausted during visual research — using defaults")
        return DEFAULT_VISUAL_STYLE
    except Exception as exc:
        log_exception(log, "Visual trend research Gemini call failed", exc)
        return DEFAULT_VISUAL_STYLE


# ─────────────────────────────────────────────────────────────────────────────
# Part C — Best Practices Merge
# ─────────────────────────────────────────────────────────────────────────────

def build_best_practices(rules: dict, style: dict, ctx: dict) -> dict:
    """
    Synthesize algorithm_rules + visual_style into a human-readable
    best_practices summary that all daily agents can reference.
    """
    log.info("Part C — Building best_practices.json…")
    now = _now_utc()
    return {
        "updated_at": now.isoformat(),
        "valid_until": _first_of_next_month(now).isoformat(),
        "month": now.strftime("%B %Y"),
        "niche": ctx.get("niche", ""),
        "region": ctx.get("target_region", ""),
        # ── Algorithm insights ────────────────────────────────────────────
        "top_formats_to_use": rules.get("content_formats_ranked", [])[:3],
        "best_ctas": rules.get("best_cta_types_this_month", []),
        "best_posting_times_utc": rules.get("optimal_post_times_utc", []),
        "top_ranking_signals": rules.get("top_ranking_signals", []),
        "hashtag_strategy": rules.get("hashtag_strategy", {}),
        "caption_length_guide": rules.get("caption_length_sweet_spot", {}),
        "content_to_avoid": rules.get("engagement_bait_to_avoid", []),
        "policy_reminders": rules.get("current_policy_highlights", []),
        "region_tips": rules.get("region_specific_tips", []),
        # ── Visual insights ───────────────────────────────────────────────
        "recommended_image_style": style.get("optimal_image_style", ""),
        "recommended_reel_style": style.get("optimal_reel_style", ""),
        "trending_palettes": style.get("trending_color_palettes", []),
        "trending_editing_styles": style.get("trending_video_editing_styles", []),
        "formats_gaining_traction": style.get("content_formats_gaining", []),
        "formats_losing_traction": style.get("content_formats_losing", []),
        "cultural_moments": style.get("cultural_moments_this_month", []),
        "visual_things_to_avoid": style.get("visual_taboos", []),
        "visual_updates_recommended": style.get("recommended_visual_updates", []),
    }


# ─────────────────────────────────────────────────────────────────────────────
# State writers — safe: keep previous file if parse failed
# ─────────────────────────────────────────────────────────────────────────────

def _write_algorithm_rules(rules: dict, now: datetime) -> None:
    payload = {
        "updated_at": now.isoformat(),
        "valid_until": _first_of_next_month(now).isoformat(),
        **rules,
    }
    # Ensure required top-level keys exist
    for key in (
        "top_ranking_signals", "content_formats_ranked", "optimal_post_times_utc",
        "engagement_bait_to_avoid", "current_policy_highlights", "hashtag_strategy",
        "caption_length_sweet_spot", "best_cta_types_this_month", "region_specific_tips",
    ):
        payload.setdefault(key, DEFAULT_ALGORITHM_RULES.get(key, []))
    try:
        write_state(STATE_ALGORITHM_RULES, payload)
        log.info(f"Written {STATE_ALGORITHM_RULES}")
    except Exception as exc:
        log_exception(log, f"Could not write {STATE_ALGORITHM_RULES}", exc)


def _write_visual_style(style: dict, now: datetime) -> None:
    payload = {
        "updated_at": now.isoformat(),
        "valid_until": _first_of_next_month(now).isoformat(),
        **style,
    }
    for key in (
        "optimal_image_style", "optimal_reel_style", "trending_color_palettes",
        "trending_video_editing_styles", "content_formats_gaining",
        "content_formats_losing", "cultural_moments_this_month",
        "visual_taboos", "recommended_visual_updates",
    ):
        payload.setdefault(key, DEFAULT_VISUAL_STYLE.get(key, []))
    try:
        write_state(STATE_VISUAL_STYLE, payload)
        log.info(f"Written {STATE_VISUAL_STYLE}")
    except Exception as exc:
        log_exception(log, f"Could not write {STATE_VISUAL_STYLE}", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """AlgorithmWatcherAgent entry point. Runs monthly."""
    log.info("══ AlgorithmWatcherAgent starting ══")
    now = _now_utc()

    cfg = get_config()
    ctx = _build_context(cfg)
    client = get_gemini_client()

    log.info(f"Researching for niche='{ctx.get('niche')}' region='{ctx.get('target_region')}' month='{ctx.get('current_month_year')}'")

    # ── Part A ────────────────────────────────────────────────────────────
    algorithm_rules = research_algorithm_rules(client, ctx)
    _write_algorithm_rules(algorithm_rules, now)

    # ── Part B ────────────────────────────────────────────────────────────
    visual_style = research_visual_style(client, ctx)
    _write_visual_style(visual_style, now)

    # ── Part C ────────────────────────────────────────────────────────────
    best_practices = build_best_practices(algorithm_rules, visual_style, ctx)
    try:
        write_state(STATE_BEST_PRACTICES, best_practices)
        log.info(f"Written {STATE_BEST_PRACTICES}")
    except Exception as exc:
        log_exception(log, f"Could not write {STATE_BEST_PRACTICES}", exc)

    log.info("══ AlgorithmWatcherAgent complete ══")


if __name__ == "__main__":
    main()
