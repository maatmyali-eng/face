"""
agents/visual_director.py
─────────────────────────
Agent 4 — VisualDirectorAgent
FACE — Facebook Autonomous Content Engine

Responsibilities:
  1. Generate visual briefs for every post (via Gemini)
  2. Produce images via 4-provider waterfall (Pollinations → HF FLUX → HF SDXL → Pexels)
  3. Quality-gate each image with Gemini Vision (min score 7/10, up to 3 attempts)
  4. Apply text overlays (hook text, bilingual if configured)
  5. Write state/visual_briefs.json and state/qa_log.json
  6. Patch state/generated_content.json with image_path, image_provider_used,
     qa_score, visual_style_used, image_ready fields
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from utils.asset_cache import get_asset_cache
from utils.config_loader import build_prompt_context, get_config
from utils.gemini_client import GeminiQuotaExhaustedError, get_gemini_client
from utils.image_utils import add_text_overlay, resize_for_format
from utils.logger import get_logger, log_exception
from utils.rate_limiter import GEMINI_LIMITER, HF_LIMITER, PEXELS_LIMITER
from utils.state_manager import (
    append_to_state,
    read_state,
    state_exists,
    update_state,
    write_state,
)

log = get_logger("visual_director")

# ── Directory constants ────────────────────────────────────────────────────────
ASSETS_RAW = Path("assets/images/raw")
ASSETS_APPROVED = Path("assets/images/approved")
ASSETS_FINAL = Path("assets/images/final")

for _d in (ASSETS_RAW, ASSETS_APPROVED, ASSETS_FINAL):
    _d.mkdir(parents=True, exist_ok=True)

# ── HuggingFace model endpoints ────────────────────────────────────────────────
HF_FLUX_URL = (
    "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
)
HF_SDXL_URL = (
    "https://api-inference.huggingface.co/models/"
    "stabilityai/stable-diffusion-xl-base-1.0"
)

# ── Format → (width, height) map ──────────────────────────────────────────────
FORMAT_DIMENSIONS: dict[str, tuple[int, int]] = {
    "single_image": (1200, 628),
    "carousel": (1080, 1080),
    "story": (1080, 1920),
    "reel": (1080, 1920),
    "text": (1200, 628),
}


def get_dimensions_for_format(fmt: str) -> tuple[int, int]:
    """Return (width, height) for the given post format."""
    return FORMAT_DIMENSIONS.get(fmt, (1200, 628))


# ── Niche-specific visual style defaults ──────────────────────────────────────

NICHE_STYLE_DEFAULTS: dict[str, dict[str, Any]] = {
    "travel": {
        "style": "cinematic",
        "color_palette": ["warm orange", "teal", "golden yellow"],
        "lighting": "golden hour",
        "composition": "drone-style wide shot",
        "dominant_emotion": "wanderlust",
        "mood": "inspiring",
    },
    "tech": {
        "style": "futuristic minimalist",
        "color_palette": ["neon cyan", "electric blue", "dark charcoal"],
        "lighting": "dark mode ambient",
        "composition": "hexagonal patterns",
        "dominant_emotion": "curiosity",
        "mood": "energetic",
    },
    "food": {
        "style": "photorealistic macro",
        "color_palette": ["warm red", "burnt orange", "rich brown"],
        "lighting": "soft diffused with steam details",
        "composition": "close-up hero shot",
        "dominant_emotion": "joy",
        "mood": "upbeat",
    },
    "motivation": {
        "style": "high contrast cinematic",
        "color_palette": ["black", "gold", "white"],
        "lighting": "sunrise silhouette",
        "composition": "epic wide establishing",
        "dominant_emotion": "aspiration",
        "mood": "inspiring",
    },
    "education": {
        "style": "flat illustration",
        "color_palette": ["clean blue", "white", "light grey"],
        "lighting": "even flat lighting",
        "composition": "open diagram-friendly layout",
        "dominant_emotion": "trust",
        "mood": "calm",
    },
    "lifestyle": {
        "style": "soft pastel aesthetic",
        "color_palette": ["soft pink", "peach", "cream"],
        "lighting": "golden hour diffused",
        "composition": "negative space heavy",
        "dominant_emotion": "joy",
        "mood": "upbeat",
    },
}


def get_niche_style_defaults(niche: str) -> dict[str, Any]:
    """
    Return default visual style values for the given niche.
    Used when state/visual_style.json hasn't been written yet (first run).
    Falls back to 'motivation' defaults if niche is unrecognised.
    """
    niche_key = niche.lower().strip()
    return NICHE_STYLE_DEFAULTS.get(niche_key, NICHE_STYLE_DEFAULTS["motivation"])


# ── Provider helpers ───────────────────────────────────────────────────────────


def _save_raw(data: bytes, post_id: str, attempt: int) -> Path:
    """Write raw image bytes to assets/images/raw/ and return the path."""
    path = ASSETS_RAW / f"{post_id}_attempt{attempt}.jpg"
    path.write_bytes(data)
    log.debug("Saved raw image: %s (%d bytes)", path, len(data))
    return path


def _provider_pollinations(
    prompt: str, width: int, height: int, post_id: str, attempt: int
) -> Path | None:
    """
    Provider 1 — Pollinations.ai FLUX (primary, no API key required).
    Returns local path on success, None on failure.
    """
    seed = random.randint(1, 999_999)
    encoded = urllib.parse.quote(prompt)
    url = (
        f"https://image.pollinations.ai/prompt/{encoded}"
        f"?model=flux&width={width}&height={height}&seed={seed}&nologo=true"
    )
    log.info("[P1-Pollinations] Requesting image for post %s (seed=%d)", post_id, seed)
    try:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        if not resp.content or len(resp.content) < 1000:
            log.warning("[P1-Pollinations] Response too small — likely an error page")
            return None
        return _save_raw(resp.content, post_id, attempt)
    except Exception as exc:
        log.warning("[P1-Pollinations] Failed: %s", exc)
        return None


def _provider_hf(
    api_url: str,
    prompt: str,
    width: int,
    height: int,
    post_id: str,
    attempt: int,
    provider_tag: str,
) -> Path | None:
    """
    Generic HuggingFace Inference API image provider.
    Handles 503 model-loading with up to 3 retries (30 s apart).
    Returns local path on success, None on failure.
    """
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        log.warning("[%s] HF_TOKEN not set — skipping", provider_tag)
        return None

    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {"inputs": prompt, "parameters": {"width": width, "height": height}}

    for retry in range(1, 4):
        HF_LIMITER.acquire()
        log.info(
            "[%s] Requesting image for post %s (retry %d/3)", provider_tag, post_id, retry
        )
        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=120)
            if resp.status_code == 503:
                log.warning(
                    "[%s] Model loading (503), waiting 30 s...", provider_tag
                )
                time.sleep(30)
                continue
            resp.raise_for_status()
            content = resp.content
            if not content or len(content) < 1000:
                log.warning("[%s] Response too small", provider_tag)
                return None
            return _save_raw(content, post_id, attempt)
        except Exception as exc:
            log.warning("[%s] Attempt %d failed: %s", provider_tag, retry, exc)
            if retry < 3:
                time.sleep(10)

    return None


def _provider_pexels(
    query: str, orientation: str, post_id: str, attempt: int
) -> Path | None:
    """
    Provider 4 — Pexels stock photo (final fallback).
    Uses AssetCache to avoid redundant downloads.
    Returns local path on success, None on failure.
    """
    PEXELS_LIMITER.acquire()
    log.info(
        "[P4-Pexels] Stock photo fallback for post %s | query: %s", post_id, query
    )
    try:
        cache = get_asset_cache()
        path = cache.download_pexels_photo(query=query, orientation=orientation)
        if path:
            return Path(path)
        log.warning("[P4-Pexels] No photo found for query: %s", query)
        return None
    except Exception as exc:
        log.warning("[P4-Pexels] Failed: %s", exc)
        return None


def _provider_pixabay(
    query: str,
    orientation: str,
    post_id: str,
    attempt: int,
) -> Path | None:
    """Pixabay stock-photo fallback."""
    if not os.environ.get("PIXABAY_API_KEY", "").strip():
        log.info("[Pixabay] Skipping: PIXABAY_API_KEY is not configured.")
        return None

    try:
        cache = get_asset_cache()
        path = cache.download_pixabay_photo(
            query=query,
            orientation=orientation,
        )
        log.info("[Pixabay] Photo ready for post %s", post_id)
        return Path(path)
    except Exception as exc:
        log.warning("[Pixabay] Failed for post %s: %s", post_id, exc)
        return None

# ── Quality gate ──────────────────────────────────────────────────────────────


def _run_quality_gate(
    gemini,
    image_path: Path,
    config: dict,
) -> dict:
    """
    Score an image using Gemini Vision + image_quality_evaluator.txt prompt.
    Returns parsed JSON from Gemini (expected keys: overall_score, improvement_instructions, …).
    Falls back to a neutral score dict on any error.
    """
    GEMINI_LIMITER.acquire()
    prompt_path = Path("prompts/image_quality_evaluator.txt")
    try:
        prompt_text = prompt_path.read_text(encoding="utf-8").format(
            niche=config.get("niche", ""),
            target_region=config.get("target_region", ""),
            audience_age_range=config.get("audience_age_range", ""),
        )
        result = gemini.generate_with_image(prompt_text, str(image_path))
        if isinstance(result, str):
            result = json.loads(result)
        return result
    except Exception as exc:
        log_exception(log, "QA gate failed", exc)
        return {
            "overall_score": 5.0,
            "policy_safe": True,
            "improvement_instructions": "QA evaluation failed; using fallback score.",
        }


# ── Visual brief generation ────────────────────────────────────────────────────


def _generate_visual_brief(
    gemini,
    post: dict,
    content: dict,
    visual_style: dict,
    config: dict,
    context: dict,
) -> dict:
    """
    Call Gemini with visual_brief_generator.txt to produce a visual brief
    for a single post. Returns a dict; falls back to a minimal stub on error.
    """
    GEMINI_LIMITER.acquire()
    runtime_ctx = {
        **context,
        "trend_topic": post.get("trend_topic", ""),
        "dominant_emotion": post.get("dominant_emotion", ""),
        "content_pillar": post.get("content_pillar", ""),
        "format": post.get("format", "single_image"),
        "caption": content.get("caption", ""),
        "hook": content.get("hook", ""),
        "visual_style_json": json.dumps(visual_style, ensure_ascii=False),
    }
    try:
        brief = gemini.generate_json("visual_brief_generator.txt", **runtime_ctx)
        brief["post_id"] = post["post_id"]
        return brief
    except GeminiQuotaExhaustedError:
        raise
    except Exception as exc:
        log_exception(log, f"Brief generation failed for post {post['post_id']}", exc)
        niche_defaults = get_niche_style_defaults(config.get("niche", "motivation"))
        return {
            "post_id": post["post_id"],
            "primary_subject": post.get("trend_topic", ""),
            "generation_prompt_pollinations": (
                f"{post.get('trend_topic', '')} {niche_defaults['style']} photography"
            ),
            "generation_prompt_huggingface": (
                f"{post.get('trend_topic', '')} {niche_defaults['style']}"
            ),
            "text_overlay_space": "bottom",
            "background_is_busy": False,
            "fallback_pexels_query": post.get("trend_topic", "inspiration"),
        }


# ── Image generation waterfall ────────────────────────────────────────────────


def _generate_image(
    brief: dict,
    post_id: str,
    fmt: str,
    config: dict,
    gemini,
    qa_records: list[dict],
    min_score: float,
) -> tuple[Path | None, str, float, int]:
    """
    Run the 4-provider waterfall + quality gate loop.

    Returns:
        (approved_path, provider_used, final_score, attempt_count)
    """
    w, h = get_dimensions_for_format(fmt)
    orientation = "portrait" if fmt in ("story", "reel") else "landscape"

    prompt_flux = brief.get("generation_prompt_pollinations", "")
    prompt_hf = brief.get("generation_prompt_huggingface", prompt_flux)
    pexels_query = (
        f"{brief.get('primary_subject', '')} "
        f"{config.get('niche', '')} {config.get('target_region', '')}"
    ).strip()

    # Provider order for generation attempts
    providers = [
        ("Pollinations", lambda att: _provider_pollinations(prompt_flux, w, h, post_id, att)),
        ("HF-FLUX", lambda att: _provider_hf(HF_FLUX_URL, prompt_hf, w, h, post_id, att, "P2-FLUX")),
        ("HF-SDXL", lambda att: _provider_hf(HF_SDXL_URL, prompt_hf, w, h, post_id, att, "P3-SDXL")),
    ]

    attempt = 0
    improved_prompt = prompt_flux  # evolves with QA feedback
    final_score = 0.0
    provider_used = "pexels"

    for provider_name, provider_fn in providers:
        attempt += 1
        raw_path = provider_fn(attempt)
        if raw_path is None:
            log.warning("Provider %s returned no image for %s", provider_name, post_id)
            continue

        qa = _run_quality_gate(gemini, raw_path, config)
        score = float(qa.get("overall_score", 0.0))
        policy_safe = qa.get("policy_safe", True)

        if not policy_safe:
            score = 0.0
            log.warning(
                "Policy violation for post %s (provider %s) — zeroing score",
                post_id,
                provider_name,
            )

        qa_records.append({
            "post_id": post_id,
            "attempt": attempt,
            "provider": provider_name,
            "score": score,
            "policy_safe": policy_safe,
            "improvement_instructions": qa.get("improvement_instructions", ""),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        log.info(
            "QA: post=%s provider=%s attempt=%d score=%.1f",
            post_id, provider_name, attempt, score,
        )

        if score >= min_score and policy_safe:
            approved_path = ASSETS_APPROVED / f"{post_id}.jpg"
            raw_path.replace(approved_path)
            log.info(
                "Image approved — post=%s score=%.1f provider=%s",
                post_id, score, provider_name,
            )
            return approved_path, provider_name, score, attempt

        # Improve prompt with QA feedback for next provider
        improvement = qa.get("improvement_instructions", "")
        if improvement:
            improved_prompt = f"{prompt_hf}. {improvement}"
            log.debug("Improved prompt: %s", improved_prompt[:120])

        # Try Pixabay before Pexels when all image-generation providers fail.
    pixabay_path = _provider_pixabay(
        pexels_query, orientation, post_id, attempt + 1
    )

    if pixabay_path:
        qa = _run_quality_gate(gemini, pixabay_path, config)
        score = float(qa.get("overall_score", 0.0))
        policy_safe = bool(qa.get("policy_safe", False))

        qa_records.append({
            "post_id": post_id,
            "attempt": attempt + 1,
            "provider": "Pixabay",
            "score": score,
            "policy_safe": policy_safe,
            "improvement_instructions": qa.get(
                "improvement_instructions", ""
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fallback": True,
        })

        # Do not approve a stock image that fails the quality gate.
        if policy_safe and score >= min_score:
            approved_path = ASSETS_APPROVED / f"{post_id}.jpg"
            pixabay_path.replace(approved_path)
            return (
                approved_path,
                "pixabay",
                score,
                attempt + 1,
            )

        log.warning(
            "Pixabay image failed the quality gate for post %s.",
            post_id,
        )

    # ── All generative providers failed — force Pexels ──────────────────────
    log.warning(
        "All 3 generation providers failed quality gate for post %s — using Pexels",
        post_id,
    )
    pexels_path = _provider_pexels(pexels_query, orientation, post_id, attempt + 1)
    if pexels_path:
        qa = _run_quality_gate(gemini, pexels_path, config)
        final_score = float(qa.get("overall_score", 5.0))
        qa_records.append({
            "post_id": post_id,
            "attempt": attempt + 1,
            "provider": "Pexels",
            "score": final_score,
            "policy_safe": qa.get("policy_safe", True),
            "improvement_instructions": qa.get("improvement_instructions", ""),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fallback": True,
        })
        approved_path = ASSETS_APPROVED / f"{post_id}.jpg"
        pexels_path.replace(approved_path)
        return approved_path, "pexels", final_score, attempt + 1

    log.error("All providers (including Pexels) failed for post %s", post_id)
    return None, "none", 0.0, attempt + 1


# ── Text overlay ──────────────────────────────────────────────────────────────


def _apply_text_overlay(
    approved_path: Path,
    post_id: str,
    content: dict,
    brief: dict,
    config: dict,
) -> Path:
    """
    Apply hook text overlay to the approved image.
    Handles bilingual (English + Bangla) if configured.
    Returns the path to the final image in assets/images/final/.
    """
    hook_text = content.get("hook", "")
    position = brief.get("text_overlay_space", "bottom")
    bg_busy = brief.get("background_is_busy", False)
    target_lang = config.get("target_language", "English")

    if "bilingual" in target_lang.lower():
        # story_slides may carry a Bangla version; otherwise keep English only
        bangla_subtext = ""
        slides = content.get("story_slides") or []
        if slides and isinstance(slides[0], dict):
            bangla_subtext = slides[0].get("text_bn", "")
        text: Any = {"en": hook_text, "bn": bangla_subtext} if bangla_subtext else hook_text
    else:
        text = hook_text

    final_path = ASSETS_FINAL / f"{post_id}.jpg"
    try:
        result = add_text_overlay(
            image_path=str(approved_path),
            text=text,
            position=position,
            font_size=84,
            color="auto",
            shadow=True,
            backdrop=bg_busy,
        )
        Path(result).replace(final_path)
    except Exception as exc:
        log_exception(log, f"Text overlay failed for {post_id}", exc)
        # Fall back: use approved image as-is
        import shutil
        shutil.copy2(approved_path, final_path)

    log.info("Final image written: %s", final_path)
    return final_path


# ── Main agent ────────────────────────────────────────────────────────────────


def main() -> None:
    log.info("=== VisualDirectorAgent starting ===")
    config = get_config()
    context = build_prompt_context()
    gemini = get_gemini_client()

    min_score: float = float(config.get("quality_gate", {}).get("min_image_score", 7))

    # ── Load inputs ──────────────────────────────────────────────────────────
    if not state_exists("content_plan.json"):
        log.critical("state/content_plan.json missing — cannot run VisualDirectorAgent")
        return

    if not state_exists("generated_content.json"):
        log.critical("state/generated_content.json missing — cannot run VisualDirectorAgent")
        return

    content_plan = read_state("content_plan.json")
    generated_content_list: list[dict] = read_state("generated_content.json").get(
        "content", []
    )

    # Build lookup: post_id → content
    content_by_id: dict[str, dict] = {
        item["post_id"]: item for item in generated_content_list
    }

    # Load visual style (may not exist on first run)
    if state_exists("visual_style.json"):
        visual_style = read_state("visual_style.json")
        log.info("Loaded state/visual_style.json")
    else:
        niche = config.get("niche", "motivation")
        visual_style = get_niche_style_defaults(niche)
        log.warning(
            "state/visual_style.json not found — using niche defaults for '%s'", niche
        )

    # ── Collect posts that need images ───────────────────────────────────────
    all_posts: list[dict] = content_plan.get("posts", [])
    image_formats = {"single_image", "carousel", "text", "reel", "story"}
    posts_needing_image = [
        p
        for p in all_posts
        if p.get("format", "") in image_formats
        and p["post_id"] in content_by_id
    ]

    log.info(
        "Posts requiring image: %d / %d total",
        len(posts_needing_image),
        len(all_posts),
    )

    # ── Step 1: Generate visual briefs ───────────────────────────────────────
    visual_briefs: dict[str, dict] = {}
    log.info("Step 1: Generating visual briefs...")

    for post in posts_needing_image:
        pid = post["post_id"]
        content = content_by_id[pid]
        try:
            brief = _generate_visual_brief(
                gemini, post, content, visual_style, config, context
            )
            visual_briefs[pid] = brief
            log.info("Brief generated for post %s", pid)
        except GeminiQuotaExhaustedError:
            log.critical("Gemini quota exhausted during brief generation — halting")
            return
        except Exception as exc:
            log_exception(log, f"Brief skipped for {pid}", exc)

    # Persist all briefs
    write_state(
        "visual_briefs.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "date": content_plan.get("date", ""),
            "briefs": visual_briefs,
        },
    )
    log.info("state/visual_briefs.json written (%d briefs)", len(visual_briefs))

    # ── Steps 2–4: Image generation + QA + text overlay ─────────────────────
    qa_records: list[dict] = []
    content_updates: dict[str, dict] = {}

    log.info("Step 2-4: Image generation → QA → text overlay...")

    for post in posts_needing_image:
        pid = post["post_id"]
        fmt = post.get("format", "single_image")
        content = content_by_id[pid]
        brief = visual_briefs.get(pid, {})

        log.info("Processing post %s (format=%s)", pid, fmt)

        # ── Image generation waterfall ────────────────────────────────────
        try:
            approved_path, provider_used, qa_score, attempt_count = _generate_image(
                brief=brief,
                post_id=pid,
                fmt=fmt,
                config=config,
                gemini=gemini,
                qa_records=qa_records,
                min_score=min_score,
            )
        except GeminiQuotaExhaustedError:
            log.critical("Gemini quota exhausted during QA for post %s — halting", pid)
            return
        except Exception as exc:
            log_exception(log, f"Image generation failed entirely for {pid}", exc)
            approved_path, provider_used, qa_score, attempt_count = None, "none", 0.0, 0

        if approved_path is None:
            log.error("No image produced for post %s — skipping overlay", pid)
            content_updates[pid] = {
                "image_path": None,
                "image_provider_used": "none",
                "qa_score": 0.0,
                "visual_style_used": visual_style.get("style", "unknown"),
                "image_ready": False,
            }
            continue

        # ── Text overlay ──────────────────────────────────────────────────
        try:
            final_path = _apply_text_overlay(approved_path, pid, content, brief, config)
        except Exception as exc:
            log_exception(log, f"Overlay failed for {pid}", exc)
            final_path = approved_path  # use approved image without overlay

        content_updates[pid] = {
            "image_path": str(final_path),
            "image_provider_used": provider_used,
            "qa_score": qa_score,
            "visual_style_used": visual_style.get("style", "unknown"),
            "image_ready": True,
        }
        log.info(
            "Post %s complete — provider=%s score=%.1f path=%s",
            pid, provider_used, qa_score, final_path,
        )

    # ── Step 5: Patch generated_content.json ─────────────────────────────────
    log.info("Step 5: Patching state/generated_content.json...")
    current_content = read_state("generated_content.json")
    for item in current_content.get("content", []):
        pid = item["post_id"]
        if pid in content_updates:
            item.update(content_updates[pid])
    write_state("generated_content.json", current_content)
    log.info("state/generated_content.json patched for %d posts", len(content_updates))

    # ── Write QA log ──────────────────────────────────────────────────────────
    write_state(
        "qa_log.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "date": content_plan.get("date", ""),
            "records": qa_records,
        },
    )
    log.info("state/qa_log.json written (%d records)", len(qa_records))

    log.info("=== VisualDirectorAgent complete ===")


if __name__ == "__main__":
    main()
