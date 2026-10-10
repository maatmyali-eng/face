"""Configuration loader compatible with the current FACE config.yml schema.

Replace utils/config_loader.py with this file. Uses only the Python standard
library plus PyYAML; in particular, it does not import the undeclared `pytz`.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml

_CONFIG: dict[str, Any] | None = None
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yml"
_TIME_RE = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")
_DEFAULT_POSTING_TIMES_UTC = ["03:00", "07:00", "14:00"]


def get_config() -> dict[str, Any]:
    """Return the cached config after validating the repository's real schema."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = _load_and_validate()
    return _CONFIG


def reload_config() -> dict[str, Any]:
    """Clear the config cache and reload config.yml."""
    global _CONFIG
    _CONFIG = None
    return get_config()


def _load_and_validate() -> dict[str, Any]:
    if not _CONFIG_PATH.is_file():
        raise FileNotFoundError(f"config.yml not found: {_CONFIG_PATH}")
    try:
        with _CONFIG_PATH.open("r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"config.yml is invalid YAML: {exc}") from exc

    if not isinstance(cfg, dict):
        raise ValueError("config.yml must contain a YAML mapping at the top level")

    required_top_level = (
        "page_name", "niche", "target_region", "target_language",
        "audience_age_range", "posting_schedule", "tone", "content_pillars",
        "visual", "agents", "api", "quality_gate", "schedule",
        "self_improvement",
    )
    missing = [key for key in required_top_level if key not in cfg]
    if missing:
        raise ValueError(
            "config.yml is missing required top-level keys matching the current "
            f"schema: {', '.join(missing)}"
        )

    for section in ("posting_schedule", "visual", "agents", "api", "quality_gate", "schedule", "self_improvement"):
        if not isinstance(cfg.get(section), dict):
            raise ValueError(f"config.yml [{section}] must be a mapping")

    if not str(cfg.get("api", {}).get("gemini_model", "")).strip():
        raise ValueError("config.yml [api].gemini_model must be a non-empty model ID")
    if not isinstance(cfg.get("content_pillars"), list) or not cfg["content_pillars"]:
        raise ValueError("config.yml content_pillars must be a non-empty list")

    # Validate posting run-times without assuming these are local times.
    times = cfg.get("schedule", {}).get("daily_run_times_utc", [])
    if times:
        if not isinstance(times, list) or any(not isinstance(x, str) or not _TIME_RE.fullmatch(x) for x in times):
            raise ValueError("config.yml [schedule].daily_run_times_utc must be a list of HH:MM UTC strings")
    logging.info("config.yml loaded and validated")
    return cfg


def get_posting_times_utc() -> list[str]:
    """Return the UTC cron slots; convert legacy local times only as a fallback."""
    cfg = get_config()
    schedule = cfg.get("schedule", {})
    configured_utc = schedule.get("daily_run_times_utc")
    if isinstance(configured_utc, list) and configured_utc:
        if any(not isinstance(x, str) or not _TIME_RE.fullmatch(x) for x in configured_utc):
            raise ValueError("schedule.daily_run_times_utc entries must use HH:MM")
        return list(configured_utc)

    # Backward-compatible fallback if the config is switched to local posting times.
    posting = cfg.get("posting_schedule", {})
    local_times = posting.get("posting_times") or posting.get("daily_post_times")
    if isinstance(local_times, list) and local_times:
        zone_name = posting.get("timezone") or schedule.get("timezone") or "UTC"
        try:
            local_zone = ZoneInfo(zone_name)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Unknown timezone in config: {zone_name!r}") from exc
        today = datetime.now(local_zone).date()
        converted: list[str] = []
        for item in local_times:
            if not isinstance(item, str) or not _TIME_RE.fullmatch(item):
                raise ValueError(f"Invalid local posting time {item!r}; expected HH:MM")
            hour, minute = map(int, item.split(":"))
            local_dt = datetime.combine(today, time(hour, minute), tzinfo=local_zone)
            converted.append(local_dt.astimezone(timezone.utc).strftime("%H:%M"))
        return converted

    return list(_DEFAULT_POSTING_TIMES_UTC)


def get_active_agents() -> list[str]:
    """Return enabled feature/agent toggle names from either supported section."""
    cfg = get_config()
    toggles = cfg.get("agent_toggles") or cfg.get("agents", {})
    return [str(name) for name, enabled in toggles.items() if enabled is True]


def build_prompt_context() -> dict[str, str]:
    """Flatten config for prompt templates and add aliases used by existing prompts."""
    cfg = get_config()
    flat: dict[str, str] = {}

    def flatten(value: Any, prefix: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                flatten(child, f"{prefix}_{key}" if prefix else str(key))
        elif isinstance(value, list):
            flat[prefix] = ", ".join(str(item) for item in value)
        elif isinstance(value, (dict, list)):
            flat[prefix] = json.dumps(value, ensure_ascii=False)
        elif isinstance(value, bool):
            flat[prefix] = str(value).lower()
        elif value is None:
            flat[prefix] = ""
        else:
            flat[prefix] = str(value)

    flatten(cfg)

    # Canonical aliases expected by the current agents and prompts.
    posting = cfg.get("posting_schedule", {})
    visual = cfg.get("visual", {})
    agents = cfg.get("agents", {})
    schedule = cfg.get("schedule", {})
    flat.update({
        "page_name": str(cfg.get("page_name", "")),
        "niche": str(cfg.get("niche", "general")),
        "sub_niches": ", ".join(map(str, cfg.get("sub_niches", []))),
        "target_region": str(cfg.get("target_region", "Global")),
        "target_language": str(cfg.get("target_language", "English")),
        "audience_age_range": str(cfg.get("audience_age_range", "18+")),
        "tone": str(cfg.get("tone", "conversational")),
        "content_pillars": ", ".join(map(str, cfg.get("content_pillars", []))),
        "posts_per_day": str(posting.get("posts_per_day", 3)),
        "stories_per_day": str(posting.get("stories_per_day", 0)),
        "reels_per_week": str(posting.get("reels_per_week", 0)),
        "enable_reels": str(bool(agents.get("enable_reels", False))).lower(),
        "enable_stories": str(bool(agents.get("enable_stories", False))).lower(),
        "enable_safety_check": str(bool(agents.get("enable_safety_check", True))).lower(),
        "enable_image_qa": str(bool(agents.get("enable_image_qa", True))).lower(),
        "timezone": str(schedule.get("timezone", "UTC")),
        "run_slots": json.dumps(schedule.get("run_slots", []), ensure_ascii=False),
        "posting_times_utc": ", ".join(get_posting_times_utc()),
        "active_agents": ", ".join(get_active_agents()),
        "best_practices": "Use accurate, current, audience-relevant content; avoid misleading engagement bait.",
        "recent_post_history": "No recent post history was supplied to this prompt.",
    })
    return flat


def get_gemini_model() -> str:
    """Read the model from the actual config.yml [api] section."""
    return str(get_config().get("api", {}).get("gemini_model", "gemini-3.5-flash-lite"))


def get_min_image_score() -> float:
    return float(get_config().get("quality_gate", {}).get("min_image_score", 7.0))


def get_image_providers_order() -> list[str]:
    visual = get_config().get("visual", {})
    configured = visual.get("image_providers")
    if isinstance(configured, list) and configured:
        return [str(item) for item in configured]
    return ["pollinations", "hf_flux", "hf_sdxl", "pexels"]
