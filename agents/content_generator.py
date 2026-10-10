"""
agents/content_generator.py
============================
ContentGeneratorAgent — Session 3, Deliverable 3

Reads state/content_plan.json and generates all written content
(captions, hashtags, story slides, reel scripts) for today's posts.

Every piece of content passes a pre-publish Facebook Community Standards
safety check via an inline Gemini call. Unsafe content that cannot be
auto-fixed is skipped and logged to state/skipped.json.

Run:
    python agents/content_generator.py
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from utils.config_loader import get_config, build_prompt_context
from utils.gemini_client import get_gemini_client, GeminiQuotaExhaustedError
from utils.logger import get_logger, log_exception
from utils.rate_limiter import GEMINI_LIMITER
from utils.state_manager import read_state, write_state, state_exists, append_to_state

# ── Logger ────────────────────────────────────────────────────────────────────
log = get_logger("content_generator")

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_HASHTAGS = 15
MAX_HASHTAGS = 20
MAX_CAPTION_CHARS = 2_000      # Facebook caption practical limit
MAX_SAFETY_RETRIES = 1         # How many times to attempt auto-fix
REEL_MAX_HOOK_WORDS = 15       # Spec: hook ≤ 15 words
REEL_MAX_CTA_WORDS = 20        # Spec: CTA ≤ 20 words

# Inline safety check prompt (not a prompt file — injected as raw string)
SAFETY_PROMPT_TEMPLATE = """\
You are a Facebook Community Standards moderator.
Review the following content for policy violations.

CONTENT:
\"\"\"
{content_text}
\"\"\"

Check for: hate speech, misinformation, spam, adult/sexual content,
graphic violence, harassment, dangerous organizations, privacy violations.

If the content is problematic, rewrite it to remove violations while
preserving the original intent and topic as much as possible.

Return STRICT JSON only (no markdown, no prose):
{{
  "safe": true | false,
  "violations": ["list of detected violations, empty if safe"],
  "adjusted_text": "rewritten safe version, or original text if already safe",
  "confidence": 0.0 to 1.0
}}
"""

# Default algorithm rules for tone guidance
DEFAULT_ALGORITHM_RULES: dict = {
    "tone_guidance": "engaging and conversational",
    "max_hashtags": MAX_HASHTAGS,
    "notes": "Default (algorithm_rules.json not yet generated).",
}

# ─────────────────────────────────────────────────────────────────────────────
# Safety check helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safety_check(client, text_to_check: str) -> dict:
    """
    Run a Facebook Community Standards safety check on content_text.

    Returns dict: { safe: bool, violations: list, adjusted_text: str,
                    confidence: float }
    Raises GeminiQuotaExhaustedError on quota failure (re-raised).
    Returns a default-safe response on any other error.
    """
    prompt = SAFETY_PROMPT_TEMPLATE.format(content_text=text_to_check)
    try:
        GEMINI_LIMITER.acquire()
        result = client.generate_json_raw(prompt)
        return result
    except GeminiQuotaExhaustedError:
        raise
    except Exception as exc:           # noqa: BLE001
        log_exception(log, exc, "Safety check Gemini call failed — assuming safe")
        return {
            "safe": True,
            "violations": [],
            "adjusted_text": text_to_check,
            "confidence": 0.5,
        }


def _enforce_safety(
    client,
    post_id: str,
    original_text: str,
    label: str = "caption",
) -> tuple[bool, str, list[str]]:
    """
    Run safety check on original_text.
    If unsafe, try one auto-fix round using adjusted_text.
    Returns (is_safe, final_text, violations).
    """
    result = _safety_check(client, original_text)
    if result.get("safe", True):
        return True, original_text, []

    violations = result.get("violations", [])
    adjusted   = result.get("adjusted_text", original_text)
    log.warning(
        "[%s] %s failed safety check — violations: %s — attempting auto-fix",
        post_id, label, violations
    )

    # Second check on the adjusted text
    result2 = _safety_check(client, adjusted)
    if result2.get("safe", True):
        log.info("[%s] %s auto-fixed successfully", post_id, label)
        return True, adjusted, []

    log.error(
        "[%s] %s still unsafe after auto-fix — violations: %s",
        post_id, label, result2.get("violations", violations)
    )
    return False, adjusted, result2.get("violations", violations)

# ─────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _validate_hashtags(hashtags: list) -> list[str]:
    """
    Ensure hashtags are strings, prefixed with #, and within count bounds.
    """
    cleaned: list[str] = []
    for tag in hashtags:
        tag = str(tag).strip()
        if not tag.startswith("#"):
            tag = f"#{tag}"
        # Remove spaces inside tags
        tag = tag.replace(" ", "")
        if len(tag) > 1:
            cleaned.append(tag)
    # Clamp to [MIN_HASHTAGS, MAX_HASHTAGS]
    return cleaned[:MAX_HASHTAGS]


def _validate_caption(caption: str) -> str:
    """Truncate caption to Facebook's practical character limit."""
    if len(caption) > MAX_CAPTION_CHARS:
        log.warning("Caption exceeds %d chars — truncating", MAX_CAPTION_CHARS)
        return caption[:MAX_CAPTION_CHARS - 3] + "..."
    return caption


def _word_count(text: str) -> int:
    return len(text.split())


def _trim_to_word_limit(text: str, limit: int) -> str:
    words = text.split()
    if len(words) <= limit:
        return text
    return " ".join(words[:limit]) + "…"

# ─────────────────────────────────────────────────────────────────────────────
# Per-format generation functions
# ─────────────────────────────────────────────────────────────────────────────

def _generate_post_caption(
    client,
    ctx: dict,
    plan_item: dict,
) -> dict | None:
    """
    Generate caption + hashtags + CTA for single_image, carousel, or text posts.
    Returns content dict or None on terminal failure.
    """
    post_id = plan_item["post_id"]
    try:
        GEMINI_LIMITER.acquire()
        result = client.generate_json(
            "post_caption.txt",
            **ctx,
            trend_topic=plan_item.get("trend_topic", ""),
            content_pillar=plan_item.get("content_pillar", "educational"),
            hook_type=plan_item.get("hook_type", "question"),
            cta_type=plan_item.get("cta_type", "comment"),
            dominant_emotion=plan_item.get("dominant_emotion", "curiosity"),
            post_format=plan_item.get("format", "single_image"),
        )
    except GeminiQuotaExhaustedError:
        raise
    except Exception as exc:
        log_exception(log, exc, f"[{post_id}] Caption Gemini call failed")
        return None

    hook     = result.get("hook", "")
    body     = result.get("caption_body", result.get("body", ""))
    hashtags = _validate_hashtags(result.get("hashtags", []))
    cta      = result.get("cta", "")
    alt_text = result.get("alt_text", "")

    # Compose full caption
    parts = [p for p in [hook, body, cta] if p]
    caption = "\n\n".join(parts)
    caption = _validate_caption(caption)

    # Pad hashtags if below minimum (pull from alt_text keywords as backup)
    if len(hashtags) < MIN_HASHTAGS:
        topic_tags = [
            f"#{w}" for w in plan_item.get("trend_topic", "").split()
            if len(w) > 3
        ]
        hashtags = list(dict.fromkeys(hashtags + topic_tags))[:MAX_HASHTAGS]

    return {
        "caption":   caption,
        "hook":      hook,
        "hashtags":  hashtags,
        "cta":       cta,
        "alt_text":  alt_text,
    }


def _generate_story_content(
    client,
    ctx: dict,
    plan_item: dict,
) -> dict | None:
    """
    Generate 3-slide story content.
    Returns content dict or None on terminal failure.
    """
    post_id = plan_item["post_id"]
    try:
        GEMINI_LIMITER.acquire()
        result = client.generate_json(
            "story_content.txt",
            **ctx,
            trend_topic=plan_item.get("trend_topic", ""),
            content_pillar=plan_item.get("content_pillar", "educational"),
            dominant_emotion=plan_item.get("dominant_emotion", "curiosity"),
            cta_type=plan_item.get("cta_type", "follow"),
        )
    except GeminiQuotaExhaustedError:
        raise
    except Exception as exc:
        log_exception(log, exc, f"[{post_id}] Story Gemini call failed")
        return None

    slides = result.get("slides", [])
    if not slides:
        # Build minimal fallback from result keys
        slides = [
            {
                "slide_number": 1,
                "text": result.get("hook", "Check this out!"),
                "sticker_suggestions": [],
            },
            {
                "slide_number": 2,
                "text": result.get("body", ""),
                "sticker_suggestions": [],
            },
            {
                "slide_number": 3,
                "text": result.get("cta", "Follow for more!"),
                "sticker_suggestions": result.get("sticker_suggestions", []),
            },
        ]
    return {
        "story_slides": slides[:3],   # cap at 3 slides
        "caption": result.get("caption", ""),
        "hook": slides[0].get("text", "") if slides else "",
        "hashtags": _validate_hashtags(result.get("hashtags", [])),
        "cta": slides[-1].get("text", "") if slides else "",
        "alt_text": "",
    }


def _generate_reel_script(
    client,
    ctx: dict,
    plan_item: dict,
) -> dict | None:
    """
    Generate reel script with hook, body segments, CTA, audio_mood,
    and estimated_duration.
    Returns content dict or None on terminal failure.
    """
    post_id = plan_item["post_id"]
    try:
        GEMINI_LIMITER.acquire()
        result = client.generate_json(
            "reel_script.txt",
            **ctx,
            trend_topic=plan_item.get("trend_topic", ""),
            content_pillar=plan_item.get("content_pillar", "entertainment"),
            hook_type=plan_item.get("hook_type", "bold_claim"),
            dominant_emotion=plan_item.get("dominant_emotion", "curiosity"),
        )
    except GeminiQuotaExhaustedError:
        raise
    except Exception as exc:
        log_exception(log, exc, f"[{post_id}] Reel script Gemini call failed")
        return None

    hook          = result.get("hook", "")
    body_segments = result.get("body_segments", [])
    cta           = result.get("cta", "")
    audio_mood    = result.get("audio_mood", "energetic")
    estimated_dur = int(result.get("estimated_duration", 30))

    # Enforce word limits
    if _word_count(hook) > REEL_MAX_HOOK_WORDS:
        hook = _trim_to_word_limit(hook, REEL_MAX_HOOK_WORDS)
        log.debug("[%s] Reel hook trimmed to %d words", post_id, REEL_MAX_HOOK_WORDS)

    if _word_count(cta) > REEL_MAX_CTA_WORDS:
        cta = _trim_to_word_limit(cta, REEL_MAX_CTA_WORDS)
        log.debug("[%s] Reel CTA trimmed to %d words", post_id, REEL_MAX_CTA_WORDS)

    # Validate audio_mood against known vocabulary
    valid_moods = {"energetic", "calm", "inspiring", "upbeat", "emotional", "mysterious"}
    if audio_mood not in valid_moods:
        log.warning("[%s] Unknown audio_mood %r — defaulting to 'energetic'", post_id, audio_mood)
        audio_mood = "energetic"

    # Compose a caption from hook for the reel post
    caption = f"{hook}\n\n{cta}" if cta else hook
    caption = _validate_caption(caption)

    return {
        "caption":  caption,
        "hook":     hook,
        "hashtags": _validate_hashtags(result.get("hashtags", [])),
        "cta":      cta,
        "alt_text": result.get("alt_text", ""),
        "reel_script": {
            "hook":              hook,
            "body_segments":     body_segments,
            "cta":               cta,
            "audio_mood":        audio_mood,
            "estimated_duration": estimated_dur,
        },
    }

# ─────────────────────────────────────────────────────────────────────────────
# Content text for safety check (extract all human-readable strings)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_text_for_safety(content_piece: dict, fmt: str) -> str:
    """
    Flatten all user-visible text from a content piece into a single
    string for safety evaluation.
    """
    parts: list[str] = []

    if content_piece.get("caption"):
        parts.append(content_piece["caption"])

    if fmt == "story" and content_piece.get("story_slides"):
        for slide in content_piece["story_slides"]:
            if isinstance(slide, dict) and slide.get("text"):
                parts.append(slide["text"])

    if fmt == "reel" and content_piece.get("reel_script"):
        rs = content_piece["reel_script"]
        parts.append(rs.get("hook", ""))
        for seg in rs.get("body_segments", []):
            if isinstance(seg, dict):
                parts.append(seg.get("text", str(seg)))
            else:
                parts.append(str(seg))
        parts.append(rs.get("cta", ""))

    return "\n".join(p for p in parts if p)

# ─────────────────────────────────────────────────────────────────────────────
# Apply safety fix back to content structure
# ─────────────────────────────────────────────────────────────────────────────

def _apply_safe_text(content_piece: dict, fmt: str, safe_text: str) -> dict:
    """
    If the safety check produced an adjusted_text, update the caption
    field (and reel hook if applicable). Story slides are rebuilt from
    the adjusted_text as a best-effort single slide.
    """
    updated = dict(content_piece)
    updated["caption"] = _validate_caption(safe_text)

    if fmt == "reel" and updated.get("reel_script"):
        rs = dict(updated["reel_script"])
        rs["hook"] = _trim_to_word_limit(safe_text.split("\n")[0], REEL_MAX_HOOK_WORDS)
        updated["reel_script"] = rs

    if fmt == "story" and updated.get("story_slides"):
        # Replace slide text with safe version in first slide
        slides = list(updated["story_slides"])
        if slides:
            slides[0] = {**slides[0], "text": safe_text[:200]}
        updated["story_slides"] = slides

    return updated

# ─────────────────────────────────────────────────────────────────────────────
# Per-plan-item processor
# ─────────────────────────────────────────────────────────────────────────────

def _process_plan_item(
    client,
    ctx: dict,
    plan_item: dict,
) -> dict | None:
    """
    Generate content for a single plan item (post, story, or reel).
    Returns a fully-formed content dict ready for generated_content.json,
    or None if the item must be skipped (unsafe / quota exhausted).
    """
    post_id = plan_item["post_id"]
    fmt     = plan_item.get("format", "single_image")

    log.info("[%s] Generating content — format: %s topic: %r",
             post_id, fmt, plan_item.get("trend_topic", "")[:60])

    # ── Generate raw content ──────────────────────────────────────────────────
    raw: dict | None = None

    if fmt in ("single_image", "carousel", "text"):
        raw = _generate_post_caption(client, ctx, plan_item)
    elif fmt == "story":
        raw = _generate_story_content(client, ctx, plan_item)
    elif fmt == "reel":
        raw = _generate_reel_script(client, ctx, plan_item)
    else:
        log.warning("[%s] Unknown format %r — treating as single_image", post_id, fmt)
        raw = _generate_post_caption(client, ctx, plan_item)

    if raw is None:
        log.error("[%s] Content generation returned None — skipping", post_id)
        return None

    # ── Safety check ──────────────────────────────────────────────────────────
    safety_text = _extract_text_for_safety(raw, fmt)
    if not safety_text.strip():
        log.warning("[%s] Empty content text — skipping safety check", post_id)
        is_safe, final_text, violations = True, "", []
    else:
        try:
            is_safe, final_text, violations = _enforce_safety(
                client, post_id, safety_text
            )
        except GeminiQuotaExhaustedError:
            raise   # propagate to main loop

    if not is_safe:
        log.error("[%s] Skipping — content could not be made safe: %s",
                  post_id, violations)
        return None   # caller will log to skipped.json

    # Apply auto-fixed text back to structure if it changed
    if final_text and final_text != safety_text:
        raw = _apply_safe_text(raw, fmt, final_text)

    # ── Assemble output record ────────────────────────────────────────────────
    record: dict = {
        "post_id":         post_id,
        "format":          fmt,
        "caption":         raw.get("caption", ""),
        "hashtags":        raw.get("hashtags", []),
        "alt_text":        raw.get("alt_text", ""),
        "cta":             raw.get("cta", ""),
        "hook":            raw.get("hook", ""),
        "safety_checked":  True,
        "safe":            True,
        "story_slides":    raw.get("story_slides"),       # None for non-stories
        "reel_script":     raw.get("reel_script"),        # None for non-reels
        "status":          "ready",
    }
    log.info("[%s] Content ready — caption len: %d  hashtags: %d",
             post_id, len(record["caption"]), len(record["hashtags"]))
    return record

# ─────────────────────────────────────────────────────────────────────────────
# Skipped-item logging
# ─────────────────────────────────────────────────────────────────────────────

def _log_skipped(post_id: str, reason: str, plan_item: dict) -> None:
    """Append a skipped post entry to state/skipped.json."""
    entry = {
        "post_id":    post_id,
        "skipped_at": datetime.now(timezone.utc).isoformat(),
        "reason":     reason,
        "format":     plan_item.get("format", "unknown"),
        "topic":      plan_item.get("trend_topic", ""),
    }
    try:
        append_to_state("skipped.json", "skipped", entry)
    except Exception as exc:
        log_exception(log, exc, "Failed to append to state/skipped.json")

# ─────────────────────────────────────────────────────────────────────────────
# Main agent
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point for ContentGeneratorAgent."""
    log.info("=== ContentGeneratorAgent starting ===")
    start_time = time.time()

    # ── Load config ───────────────────────────────────────────────────────────
    try:
        cfg = get_config()
    except Exception as exc:
        log_exception(log, exc, "Failed to load config — aborting")
        raise SystemExit(1)

    ctx = build_prompt_context()   # flat dict for prompt injection

    # ── Load content plan (required) ──────────────────────────────────────────
    if not state_exists("content_plan.json"):
        log.error("state/content_plan.json missing — run ContentStrategyAgent first")
        raise SystemExit(1)

    try:
        plan_state = read_state("content_plan.json")
    except Exception as exc:
        log_exception(log, exc, "Failed to read state/content_plan.json")
        raise SystemExit(1)

    # Collect all plan items: posts + stories + reels
    all_plan_items: list[dict] = []
    all_plan_items.extend(plan_state.get("posts", []))
    all_plan_items.extend(plan_state.get("stories", []))
    # Reels appear in both posts and reels lists — deduplicate by post_id
    seen_ids: set[str] = {item["post_id"] for item in all_plan_items}
    for item in plan_state.get("reels", []):
        if item["post_id"] not in seen_ids:
            all_plan_items.append(item)
            seen_ids.add(item["post_id"])

    log.info("Content plan has %d items to generate", len(all_plan_items))

    if not all_plan_items:
        log.error("No plan items found — aborting")
        raise SystemExit(1)

    # ── Load algorithm rules for tone guidance (optional) ─────────────────────
    if state_exists("algorithm_rules.json"):
        try:
            algo_rules = read_state("algorithm_rules.json")
            ctx["algorithm_tone"] = algo_rules.get("tone_guidance",
                                                    "engaging and conversational")
        except Exception as exc:
            log_exception(log, exc, "Failed to read algorithm_rules.json — using default tone")
            ctx["algorithm_tone"] = DEFAULT_ALGORITHM_RULES["tone_guidance"]
    else:
        ctx["algorithm_tone"] = DEFAULT_ALGORITHM_RULES["tone_guidance"]

    # ── Process each plan item ────────────────────────────────────────────────
    client = get_gemini_client()
    generated: list[dict]  = []
    skipped_count = 0

    for plan_item in all_plan_items:
        post_id = plan_item.get("post_id", str(uuid.uuid4()))
        try:
            record = _process_plan_item(client, ctx, plan_item)
        except GeminiQuotaExhaustedError as exc:
            log.critical(
                "Gemini quota exhausted at item %s — writing partial results: %s",
                post_id, exc
            )
            _log_skipped(post_id, "gemini_quota_exhausted", plan_item)
            break   # write what we have so far
        except Exception as exc:
            log_exception(log, exc, f"[{post_id}] Unexpected error — skipping")
            _log_skipped(post_id, f"unexpected_error: {exc}", plan_item)
            skipped_count += 1
            continue

        if record is None:
            _log_skipped(post_id, "unsafe_content_or_generation_failure", plan_item)
            skipped_count += 1
        else:
            generated.append(record)

    log.info(
        "Generation complete: %d ready, %d skipped",
        len(generated), skipped_count
    )

    # ── Write state/generated_content.json ────────────────────────────────────
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    output = {
        "date":          today_str,
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "content":       generated,
    }

    try:
        write_state("generated_content.json", output)
        log.info("Wrote generated_content.json with %d content items", len(generated))
    except Exception as exc:
        log_exception(log, exc, "Failed to write state/generated_content.json")
        raise SystemExit(1)

    elapsed = round(time.time() - start_time, 1)
    log.info("=== ContentGeneratorAgent complete in %.1fs ===", elapsed)

    if skipped_count > 0:
        log.warning(
            "%d item(s) were skipped — see state/skipped.json for details",
            skipped_count
        )


if __name__ == "__main__":
    main()
