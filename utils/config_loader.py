"""
utils/config_loader.py
======================
Loads and validates config.yml at startup.
Provides a singleton get_config() and helpers for agents.

Usage:
    from utils.config_loader import get_config, get_posting_times_utc, get_active_agents
    cfg = get_config()
"""

import os
import re
import yaml
import pytz
import logging
from datetime import datetime, time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_CONFIG: Optional[Dict[str, Any]] = None
_CONFIG_PATH = Path(__file__).parent.parent / "config.yml"

# Required top-level sections and their required keys
_REQUIRED_FIELDS: Dict[str, List[str]] = {
    "page_identity": ["page_name", "niche", "target_language", "brand_voice"],
    "audience_definition": ["target_region", "audience_age_range", "interests"],
    "posting_schedule": ["timezone", "daily_post_count", "posting_times"],
    "content_style": ["tone", "hashtag_count", "cta_style"],
    "visual_settings": ["primary_color", "font_style", "image_style"],
    "agent_toggles": [],
    "api_settings": [],
    "quality_gate_settings": ["min_image_score"],
    "schedule_settings": [],
    "self_improvement": [],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_config() -> Dict[str, Any]:
    """
    Return the loaded config dict (singleton — loads once per process).

    Raises:
        FileNotFoundError: if config.yml does not exist.
        ValueError: if required fields are missing.
    """
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = _load_and_validate()
    return _CONFIG


def reload_config() -> Dict[str, Any]:
    """Force a re-read of config.yml (useful after self-improvement edits)."""
    global _CONFIG
    _CONFIG = None
    return get_config()


def get_posting_times_utc() -> List[str]:
    """
    Convert config posting times from the configured timezone to UTC.

    Returns:
        List of time strings in 'HH:MM' UTC format, e.g. ["06:00", "12:00", "18:00"]
    """
    cfg = get_config()
    tz_name: str = cfg["posting_schedule"]["timezone"]
    raw_times: List[str] = cfg["posting_schedule"]["posting_times"]

    try:
        local_tz = pytz.timezone(tz_name)
    except pytz.UnknownTimeZoneError:
        raise ValueError(f"Unknown timezone in config: {tz_name!r}")

    utc_tz = pytz.utc
    utc_times: List[str] = []
    today = datetime.utcnow().date()

    for t_str in raw_times:
        try:
            h, m = map(int, t_str.split(":"))
        except (ValueError, AttributeError):
            raise ValueError(
                f"Invalid posting_time format {t_str!r} — expected 'HH:MM'"
            )
        local_dt = local_tz.localize(
            datetime.combine(today, time(h, m))
        )
        utc_dt = local_dt.astimezone(utc_tz)
        utc_times.append(utc_dt.strftime("%H:%M"))

    return utc_times


def get_active_agents() -> List[str]:
    """
    Return a list of agent names that are enabled in config.yml.

    Returns:
        e.g. ["trend_hunter", "content_strategy", "content_generator", ...]
    """
    cfg = get_config()
    toggles: Dict[str, bool] = cfg.get("agent_toggles", {})
    return [name for name, enabled in toggles.items() if enabled]


def build_prompt_context() -> Dict[str, str]:
    """
    Extract all {placeholder} values from config into a flat dict for
    use with str.format(**context) when injecting into prompt templates.

    Returns:
        Flat dict of all injectable values, e.g.:
        {
            "page_name": "My Page",
            "niche": "fitness",
            "tone": "inspirational",
            ...
        }
    """
    cfg = get_config()

    def _flatten(obj: Any, prefix: str = "") -> Dict[str, str]:
        """Recursively flatten nested dicts, joining keys with underscores."""
        result: Dict[str, str] = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{prefix}{k}" if not prefix else f"{prefix}_{k}"
                result.update(_flatten(v, new_key))
        elif isinstance(obj, list):
            # Join lists as comma-separated strings for prompt injection
            result[prefix] = ", ".join(str(x) for x in obj)
        else:
            result[prefix] = str(obj) if obj is not None else ""
        return result

    flat = _flatten(cfg)

    # Add derived helpers
    try:
        flat["posting_times_utc"] = ", ".join(get_posting_times_utc())
    except Exception:
        flat["posting_times_utc"] = ""

    flat["active_agents"] = ", ".join(get_active_agents())

    return flat


def get_gemini_model() -> str:
    """Return the Gemini model name from config, defaulting to gemini-1.5-flash."""
    cfg = get_config()
    return cfg.get("api_settings", {}).get("gemini_model", "gemini-1.5-flash")


def get_min_image_score() -> float:
    """Return the quality gate threshold (default 7.0)."""
    cfg = get_config()
    return float(cfg.get("quality_gate_settings", {}).get("min_image_score", 7.0))


def get_image_providers_order() -> List[str]:
    """
    Return the 4-provider waterfall order from config.
    Defaults to Pollinations → HuggingFace FLUX → HuggingFace SDXL → Pexels.
    """
    cfg = get_config()
    return cfg.get("visual_settings", {}).get(
        "image_providers",
        ["pollinations", "hf_flux", "hf_sdxl", "pexels"],
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_and_validate() -> Dict[str, Any]:
    """Load config.yml from disk and validate required fields."""
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"config.yml not found at {_CONFIG_PATH}. "
            "Copy config.yml.example and fill in your values."
        )

    with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
        try:
            cfg = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise ValueError(f"config.yml is not valid YAML: {exc}") from exc

    if not isinstance(cfg, dict):
        raise ValueError("config.yml must be a YAML mapping (dict) at the top level.")

    errors: List[str] = []

    for section, required_keys in _REQUIRED_FIELDS.items():
        if section not in cfg:
            errors.append(f"Missing required section: [{section}]")
            continue
        for key in required_keys:
            if key not in cfg[section]:
                errors.append(f"Missing required key: [{section}].{key}")

    # Validate posting_times is a list
    try:
        pt = cfg["posting_schedule"]["posting_times"]
        if not isinstance(pt, list) or len(pt) == 0:
            errors.append(
                "[posting_schedule].posting_times must be a non-empty list of 'HH:MM' strings"
            )
    except (KeyError, TypeError):
        pass  # Already caught by required fields check

    # Validate color format for primary_color
    try:
        pc = cfg["visual_settings"]["primary_color"]
        if not re.match(r"^#[0-9A-Fa-f]{6}$", str(pc)):
            errors.append(
                f"[visual_settings].primary_color must be a 6-digit hex color like #1a2b3c, got: {pc!r}"
            )
    except (KeyError, TypeError):
        pass

    # Validate min_image_score is numeric in 0–10
    try:
        score = float(cfg["quality_gate_settings"]["min_image_score"])
        if not (0 <= score <= 10):
            errors.append(
                f"[quality_gate_settings].min_image_score must be between 0 and 10, got: {score}"
            )
    except (KeyError, TypeError, ValueError):
        pass

    if errors:
        msg = "config.yml validation failed:\n  " + "\n  ".join(errors)
        raise ValueError(msg)

    logging.info("config.yml loaded and validated successfully.")
    return cfg
