"""
agents/self_improver.py
────────────────────────────────────────────────────────────────────────────
SelfImprovementAgent — FACE System (Session 5)

Weekly self-optimization agent:
  Step 1 — Fetch Facebook Insights for posts older than 48 hours
  Step 2 — Analyze last 30 days of performance; write performance_report.json
  Step 3 — Rewrite underperforming prompts; version previous files
  Step 4 — Auto-rollback prompts that degraded engagement
  Step 5 — Apply same analysis to visual prompts

Cron trigger: every Sunday at 02:00 UTC
Entry point: main()
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import re
import shutil
import json
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from statistics import mean
from typing import Any

# ── Utils ────────────────────────────────────────────────────────────────────
from utils.config_loader import get_config, build_prompt_context
from utils.state_manager import read_state, write_state, update_state, state_exists
from utils.logger import get_logger, log_exception
from utils.gemini_client import GeminiClient, get_gemini_client, GeminiQuotaExhaustedError
from utils.facebook_client import FacebookClient, FacebookTokenExpiredError
from utils.rate_limiter import GEMINI_LIMITER, FACEBOOK_LIMITER

log = get_logger("self_improver")

# ── Paths ─────────────────────────────────────────────────────────────────────
PROMPTS_DIR = Path("prompts")
VERSIONS_DIR = Path("state/prompt_versions")
STATE_POST_HISTORY = "state/post_history.json"
STATE_PERFORMANCE_REPORT = "state/performance_report.json"
STATE_IMPROVEMENT_LOG = "state/improvement_log.json"
STATE_VISUAL_STYLE = "state/visual_style.json"
STATE_BEST_PRACTICES = "state/best_practices.json"
PROMPT_SELF_IMPROVEMENT = "prompts/self_improvement.txt"

# ── Config ────────────────────────────────────────────────────────────────────
INSIGHTS_MIN_AGE_HOURS = 48    # insights need time to populate
ANALYSIS_WINDOW_DAYS = 30
ROLLBACK_THRESHOLD_PCT = 5.0   # roll back if engagement dropped by this %
MAX_PROMPT_VERSIONS = 8        # keep latest N versions per prompt

# ── All prompt files that can be improved ────────────────────────────────────
IMPROVABLE_PROMPTS = [
    "post_caption.txt",
    "story_content.txt",
    "reel_script.txt",
    "trend_analysis.txt",
    "strategy_decision.txt",
    "algorithm_research.txt",
    "visual_brief_generator.txt",
    "image_quality_evaluator.txt",
    "reel_script_visual_director.txt",
    "visual_trend_researcher.txt",
]

VISUAL_PROMPTS = {
    "visual_brief_generator.txt",
    "image_quality_evaluator.txt",
    "reel_script_visual_director.txt",
    "visual_trend_researcher.txt",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(s: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _engagement_rate(post: dict) -> float:
    insights = post.get("insights") or {}
    return float(insights.get("engagement_rate", 0.0))


def _caption_length_bucket(length: int) -> str:
    if length < 100:
        return "short"
    if length <= 300:
        return "medium"
    return "long"


def _ensure_versions_dir(prompt_name: str) -> Path:
    version_dir = VERSIONS_DIR / prompt_name.replace(".txt", "")
    version_dir.mkdir(parents=True, exist_ok=True)
    return version_dir


def _next_version_number(version_dir: Path) -> int:
    existing = sorted(version_dir.glob("*.txt"))
    if not existing:
        return 1
    # filenames: YYYY-MM-DD_vN.txt
    max_n = 0
    for f in existing:
        m = re.search(r"_v(\d+)\.txt$", f.name)
        if m:
            max_n = max(max_n, int(m.group(1)))
    return max_n + 1


def _prune_old_versions(version_dir: Path) -> None:
    """Keep only the latest MAX_PROMPT_VERSIONS versions."""
    versions = sorted(version_dir.glob("*.txt"))
    while len(versions) > MAX_PROMPT_VERSIONS:
        oldest = versions.pop(0)
        oldest.unlink(missing_ok=True)
        log.debug(f"Pruned old prompt version: {oldest}")


def _read_prompt(prompt_name: str) -> str:
    path = PROMPTS_DIR / prompt_name
    if not path.exists():
        log.warning(f"Prompt file not found: {path}")
        return ""
    return path.read_text(encoding="utf-8")


def _write_prompt(prompt_name: str, content: str) -> None:
    path = PROMPTS_DIR / prompt_name
    path.write_text(content, encoding="utf-8")
    log.info(f"Wrote updated prompt: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Fetch Insights
# ─────────────────────────────────────────────────────────────────────────────

def step1_fetch_insights(fb: FacebookClient, post_history: dict) -> dict:
    """
    Fetch Facebook Insights for posts older than 48 hours where insights==null.
    Mutates post_history in place and returns it.
    """
    log.info("Step 1 — Fetching Facebook Insights for unpopulated posts…")
    posts: list[dict] = post_history.get("posts", [])
    cutoff = _now_utc() - timedelta(hours=INSIGHTS_MIN_AGE_HOURS)
    updated = 0

    for post in posts:
        if post.get("insights") is not None:
            continue
        published_at = _parse_iso(post.get("published_at", ""))
        if published_at is None or published_at > cutoff:
            continue  # too recent

        fb_post_id = post.get("post_id", "")
        if not fb_post_id:
            continue

        try:
            FACEBOOK_LIMITER.acquire()
            data = fb.get_post_insights(fb_post_id)
            if data:
                reach = data.get("reach", 1) or 1
                engaged = data.get("engaged_users", 0)
                post["insights"] = {
                    "impressions": data.get("impressions", 0),
                    "reach": reach,
                    "engaged_users": engaged,
                    "reactions": data.get("reactions", 0),
                    "shares": data.get("shares", 0),
                    "comments": data.get("comments", 0),
                    "engagement_rate": round(engaged / reach * 100, 4),
                }
                updated += 1
                log.debug(f"Insights fetched for post {fb_post_id}: ER={post['insights']['engagement_rate']:.2f}%")
        except FacebookTokenExpiredError:
            log.critical("Facebook token expired during insights fetch — aborting Step 1")
            break
        except Exception as exc:
            log_exception(log, f"Could not fetch insights for {fb_post_id}", exc)

    log.info(f"Step 1 complete — updated insights for {updated} posts")
    return post_history


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Performance Analysis
# ─────────────────────────────────────────────────────────────────────────────

def _group_avg(posts: list[dict], key_fn) -> dict[str, float]:
    """Group posts by key_fn(post) and compute average engagement_rate per group."""
    groups: dict[str, list[float]] = defaultdict(list)
    for p in posts:
        if p.get("insights") is None:
            continue
        k = key_fn(p)
        if k:
            groups[k].append(_engagement_rate(p))
    return {k: round(mean(v), 4) for k, v in groups.items() if v}


def step2_analyze_performance(post_history: dict) -> dict:
    """Build performance_report.json from last 30 days of posts with insights."""
    log.info("Step 2 — Analyzing performance data…")
    now = _now_utc()
    cutoff = now - timedelta(days=ANALYSIS_WINDOW_DAYS)

    recent = [
        p for p in post_history.get("posts", [])
        if p.get("insights") is not None
        and _parse_iso(p.get("published_at", "")) is not None
        and _parse_iso(p["published_at"]) >= cutoff  # type: ignore[operator]
    ]

    if not recent:
        log.warning("No posts with insights in last 30 days — skipping analysis")
        return {}

    all_er = [_engagement_rate(p) for p in recent]
    avg_er = round(mean(all_er), 4) if all_er else 0.0

    # Previous week avg for delta
    prev_cutoff = cutoff - timedelta(days=ANALYSIS_WINDOW_DAYS)
    prev_posts = [
        p for p in post_history.get("posts", [])
        if p.get("insights") is not None
        and _parse_iso(p.get("published_at", "")) is not None
        and prev_cutoff <= _parse_iso(p["published_at"]) < cutoff  # type: ignore[operator]
    ]
    prev_er = round(mean([_engagement_rate(p) for p in prev_posts]), 4) if prev_posts else avg_er
    vs_last = round(avg_er - prev_er, 4)

    # Sort by engagement_rate
    scored = sorted(recent, key=_engagement_rate, reverse=True)
    top_3 = [_post_summary(p) for p in scored[:3]]
    bottom_3 = [_post_summary(p) for p in scored[-3:] if _engagement_rate(p) < avg_er]

    # Group averages
    format_avg = _group_avg(recent, lambda p: p.get("format", ""))
    pillar_avg = _group_avg(recent, lambda p: p.get("content_pillar", ""))
    hook_avg = _group_avg(recent, lambda p: p.get("hook_type", ""))
    hour_avg = _group_avg(recent, lambda p: str(_parse_iso(p.get("published_at", "")).hour if _parse_iso(p.get("published_at", "")) else ""))
    cap_avg = _group_avg(recent, lambda p: _caption_length_bucket(p.get("caption_length", 0)))

    def _best(d: dict) -> str:
        return max(d, key=d.get) if d else ""

    def _worst(d: dict) -> str:
        return min(d, key=d.get) if d else ""

    recommendations = _build_recommendations(format_avg, pillar_avg, hook_avg, avg_er)

    report = {
        "week_ending": now.strftime("%Y-%m-%d"),
        "posts_analyzed": len(recent),
        "avg_engagement_rate": avg_er,
        "vs_last_week": vs_last,
        "top_3_posts": top_3,
        "bottom_3_posts": bottom_3,
        "format_avg_engagement": format_avg,
        "pillar_avg_engagement": pillar_avg,
        "hook_avg_engagement": hook_avg,
        "hour_avg_engagement": hour_avg,
        "caption_length_avg_engagement": cap_avg,
        "best_performing": {
            "format": _best(format_avg),
            "pillar": _best(pillar_avg),
            "hook_type": _best(hook_avg),
            "posting_hour": _best(hour_avg),
        },
        "worst_performing": {
            "format": _worst(format_avg),
            "pillar": _worst(pillar_avg),
            "hook_type": _worst(hook_avg),
        },
        "recommendations": recommendations,
    }

    try:
        write_state(STATE_PERFORMANCE_REPORT, report)
        log.info(f"Written {STATE_PERFORMANCE_REPORT} (analyzed {len(recent)} posts, avg ER={avg_er}%)")
    except Exception as exc:
        log_exception(log, f"Could not write {STATE_PERFORMANCE_REPORT}", exc)

    return report


def _post_summary(p: dict) -> dict:
    return {
        "post_id": p.get("post_id", ""),
        "internal_id": p.get("internal_id", ""),
        "format": p.get("format", ""),
        "content_pillar": p.get("content_pillar", ""),
        "hook_type": p.get("hook_type", ""),
        "engagement_rate": _engagement_rate(p),
        "published_at": p.get("published_at", ""),
    }


def _build_recommendations(
    format_avg: dict, pillar_avg: dict, hook_avg: dict, avg_er: float
) -> list[str]:
    recs = []
    if format_avg:
        best_fmt = max(format_avg, key=format_avg.get)
        worst_fmt = min(format_avg, key=format_avg.get)
        recs.append(f"Increase '{best_fmt}' posts — highest avg engagement ({format_avg[best_fmt]:.2f}%)")
        if format_avg[worst_fmt] < avg_er * 0.6:
            recs.append(f"Reduce '{worst_fmt}' posts — engagement below average ({format_avg[worst_fmt]:.2f}%)")
    if pillar_avg:
        best_pillar = max(pillar_avg, key=pillar_avg.get)
        recs.append(f"Lean into '{best_pillar}' content pillar this week")
    if hook_avg:
        best_hook = max(hook_avg, key=hook_avg.get)
        recs.append(f"Use '{best_hook}' hooks more frequently — best performing hook type")
    return recs


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Prompt Improvement
# ─────────────────────────────────────────────────────────────────────────────

def _read_all_prompts() -> dict[str, str]:
    return {name: _read_prompt(name) for name in IMPROVABLE_PROMPTS}


def _version_prompt(prompt_name: str, current_content: str, avg_er: float) -> str:
    """Archive current prompt file and return path to archived version."""
    version_dir = _ensure_versions_dir(prompt_name)
    n = _next_version_number(version_dir)
    date_str = _now_utc().strftime("%Y-%m-%d")
    archive_path = version_dir / f"{date_str}_v{n}.txt"

    # Write archived version
    archive_path.write_text(current_content, encoding="utf-8")

    # Write metadata sidecar
    meta_path = version_dir / f"{date_str}_v{n}.meta.json"
    meta = {
        "version": n,
        "created_at": _now_utc().isoformat(),
        "avg_engagement_at_time": avg_er,
        "prompt_name": prompt_name,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    _prune_old_versions(version_dir)
    log.info(f"Archived prompt '{prompt_name}' → {archive_path}")
    return str(archive_path)


def _append_improvement_log(entry: dict) -> None:
    try:
        if state_exists(STATE_IMPROVEMENT_LOG):
            existing = read_state(STATE_IMPROVEMENT_LOG)
            improvements = existing.get("improvements", [])
        else:
            improvements = []
        improvements.append(entry)
        write_state(STATE_IMPROVEMENT_LOG, {"improvements": improvements})
    except Exception as exc:
        log_exception(log, "Could not write improvement_log.json", exc)


def step3_improve_prompts(client: GeminiClient, report: dict) -> None:
    """
    Ask Gemini to suggest prompt rewrites based on performance report.
    Versions and overwrites each suggested prompt file.
    """
    if not report:
        log.warning("Step 3 — No performance report to work from; skipping prompt improvement")
        return

    log.info("Step 3 — Requesting prompt improvement suggestions from Gemini…")

    avg_er = report.get("avg_engagement_rate", 0.0)
    current_prompts = _read_all_prompts()

    # Build the self-improvement prompt context
    ctx = build_prompt_context()
    ctx["performance_report_json"] = json.dumps(report, indent=2)
    ctx["current_prompts_json"] = json.dumps(
        {k: v[:2000] for k, v in current_prompts.items()},  # truncate for token budget
        indent=2
    )

    try:
        GEMINI_LIMITER.acquire()
        result = client.generate_json(PROMPT_SELF_IMPROVEMENT, **ctx)
    except GeminiQuotaExhaustedError:
        log.critical("Gemini quota exhausted during self-improvement — skipping prompt rewrites")
        return
    except Exception as exc:
        log_exception(log, "Self-improvement Gemini call failed", exc)
        return

    if not isinstance(result, dict):
        log.warning("Step 3 — Gemini did not return a dict; skipping rewrites")
        return

    rewrites: list[dict] = result.get("prompt_rewrites", [])
    if not rewrites:
        log.info("Step 3 — Gemini suggested no prompt rewrites this week")
        return

    now = _now_utc()
    rollback_date = (now + timedelta(weeks=2)).strftime("%Y-%m-%d")

    for item in rewrites:
        prompt_name: str = item.get("prompt_name", "")
        new_content: str = item.get("new_content", "")
        changes_summary: str = item.get("changes_made", "")
        expected_improvement: str = item.get("expected_improvement", "")

        if not prompt_name or not new_content:
            log.warning(f"Step 3 — Skipping incomplete rewrite entry: {item}")
            continue
        if prompt_name not in IMPROVABLE_PROMPTS:
            log.warning(f"Step 3 — Prompt '{prompt_name}' not in improvable list; skipping")
            continue

        current_content = current_prompts.get(prompt_name, "")

        # Archive current version before overwriting
        archived_path = _version_prompt(prompt_name, current_content, avg_er)

        # Write improved prompt
        _write_prompt(prompt_name, new_content)

        # Log the change
        log_entry = {
            "date": now.strftime("%Y-%m-%d"),
            "prompt_name": prompt_name,
            "previous_version": archived_path,
            "changes_made": changes_summary,
            "expected_improvement": expected_improvement,
            "avg_engagement_at_change": avg_er,
            "rollback_if_no_improvement_by": rollback_date,
            "rollback_executed": False,
        }
        _append_improvement_log(log_entry)
        log.info(f"Step 3 — Rewrote '{prompt_name}'. Expected: {expected_improvement}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Auto-Rollback Check
# ─────────────────────────────────────────────────────────────────────────────

def step4_auto_rollback(current_avg_er: float) -> None:
    """
    Check improvement_log for entries past their rollback deadline.
    Restore previous prompt version if engagement dropped.
    """
    log.info("Step 4 — Checking for auto-rollback candidates…")

    if not state_exists(STATE_IMPROVEMENT_LOG):
        log.info("Step 4 — No improvement_log.json found; nothing to check")
        return

    try:
        log_data = read_state(STATE_IMPROVEMENT_LOG)
        improvements: list[dict] = log_data.get("improvements", [])
    except Exception as exc:
        log_exception(log, "Could not read improvement_log.json", exc)
        return

    today = _now_utc().date()
    changed = False

    for entry in improvements:
        if entry.get("rollback_executed"):
            continue

        deadline_str = entry.get("rollback_if_no_improvement_by", "")
        if not deadline_str:
            continue
        try:
            deadline = datetime.strptime(deadline_str, "%Y-%m-%d").date()
        except Exception:
            continue

        if today < deadline:
            continue  # still within trial window

        prompt_name = entry.get("prompt_name", "")
        er_at_change = float(entry.get("avg_engagement_at_change", current_avg_er))
        threshold = er_at_change * (1 - ROLLBACK_THRESHOLD_PCT / 100)

        if current_avg_er < threshold:
            # Performance degraded — roll back
            log.warning(
                f"Auto-rollback triggered for '{prompt_name}': "
                f"ER now={current_avg_er:.2f}% was={er_at_change:.2f}% threshold={threshold:.2f}%"
            )
            _restore_previous_version(prompt_name, entry.get("previous_version", ""))
            entry["rollback_executed"] = True
            entry["rollback_reason"] = "engagement_degraded"
            entry["rollback_at"] = _now_utc().isoformat()
        else:
            log.info(
                f"'{prompt_name}' improvement confirmed: ER now={current_avg_er:.2f}% >= threshold={threshold:.2f}%"
            )
            entry["rollback_executed"] = True
            entry["rollback_reason"] = "improvement_confirmed"

        changed = True

    if changed:
        try:
            write_state(STATE_IMPROVEMENT_LOG, {"improvements": improvements})
            log.info("Step 4 — improvement_log.json updated with rollback decisions")
        except Exception as exc:
            log_exception(log, "Could not update improvement_log.json after rollbacks", exc)


def _restore_previous_version(prompt_name: str, archived_path: str) -> None:
    """Restore a prompt from its archived version path."""
    src = Path(archived_path)
    dst = PROMPTS_DIR / prompt_name
    if not src.exists():
        log.error(f"Cannot rollback '{prompt_name}' — archived file missing: {src}")
        return
    shutil.copy2(src, dst)
    log.info(f"Rolled back '{prompt_name}' ← {src}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Visual Prompt Improvement
# ─────────────────────────────────────────────────────────────────────────────

def step5_visual_improvement(report: dict) -> None:
    """
    Correlate image_provider + visual_style with engagement_rate.
    Update visual_style.json with performance-driven recommendations.
    """
    log.info("Step 5 — Analyzing visual performance correlations…")

    if not report:
        log.warning("Step 5 — No performance report; skipping visual analysis")
        return

    # We need the raw post history for these correlations
    try:
        post_history = read_state(STATE_POST_HISTORY)
        posts: list[dict] = post_history.get("posts", [])
    except Exception as exc:
        log_exception(log, "Could not read post_history for visual analysis", exc)
        return

    now = _now_utc()
    cutoff = now - timedelta(days=ANALYSIS_WINDOW_DAYS)
    recent = [
        p for p in posts
        if p.get("insights") is not None
        and _parse_iso(p.get("published_at", "")) is not None
        and _parse_iso(p["published_at"]) >= cutoff  # type: ignore[operator]
    ]

    if not recent:
        log.info("Step 5 — No recent posts with insights for visual analysis")
        return

    # Provider correlation
    provider_avg = _group_avg(recent, lambda p: p.get("image_provider", ""))
    style_avg = _group_avg(recent, lambda p: p.get("visual_style", ""))

    log.info(f"Provider engagement averages: {provider_avg}")
    log.info(f"Visual style engagement averages: {style_avg}")

    # Build recommendations
    visual_updates: list[str] = []

    if provider_avg:
        best_provider = max(provider_avg, key=provider_avg.get)
        if best_provider:
            visual_updates.append(
                f"'{best_provider}' images have highest engagement — prioritize in visual pipeline"
            )

    if style_avg:
        best_style = max(style_avg, key=style_avg.get)
        worst_style = min(style_avg, key=style_avg.get)
        if best_style:
            visual_updates.append(
                f"Visual style '{best_style}' outperforms others — increase usage"
            )
        if worst_style and style_avg[worst_style] < report.get("avg_engagement_rate", 0) * 0.7:
            visual_updates.append(
                f"Visual style '{worst_style}' underperforming — consider replacing"
            )

    # Update visual_style.json with performance-driven notes
    if state_exists(STATE_VISUAL_STYLE):
        try:
            vs = read_state(STATE_VISUAL_STYLE)
            vs["performance_driven_updates"] = visual_updates
            vs["provider_engagement_avg"] = provider_avg
            vs["style_engagement_avg"] = style_avg
            vs["last_performance_analysis"] = now.isoformat()
            write_state(STATE_VISUAL_STYLE, vs)
            log.info(f"Updated {STATE_VISUAL_STYLE} with performance correlations")
        except Exception as exc:
            log_exception(log, f"Could not update {STATE_VISUAL_STYLE}", exc)

    # Update best_practices with visual notes
    if state_exists(STATE_BEST_PRACTICES):
        try:
            bp = read_state(STATE_BEST_PRACTICES)
            bp["visual_performance_notes"] = visual_updates
            write_state(STATE_BEST_PRACTICES, bp)
        except Exception as exc:
            log_exception(log, f"Could not update {STATE_BEST_PRACTICES}", exc)

    # Check if visual_brief_generator.txt needs improvement
    if visual_updates:
        avg_er = report.get("avg_engagement_rate", 0.0)
        vs_rate = mean(style_avg.values()) if style_avg else avg_er
        if vs_rate < avg_er * 0.85:
            log.info(
                "Step 5 — Visual style engagement below overall average; "
                "visual_brief_generator.txt is a candidate for next improvement cycle"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """SelfImprovementAgent entry point. Runs weekly (Sunday 02:00 UTC)."""
    log.info("══ SelfImprovementAgent starting ══")

    # Ensure required directories exist
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load post history ─────────────────────────────────────────────────
    if not state_exists(STATE_POST_HISTORY):
        log.warning(f"{STATE_POST_HISTORY} not found — nothing to analyze. Exiting.")
        return

    try:
        post_history = read_state(STATE_POST_HISTORY)
    except Exception as exc:
        log_exception(log, f"Cannot read {STATE_POST_HISTORY}", exc)
        return

    # ── Step 1: Fetch Insights ────────────────────────────────────────────
    try:
        fb = FacebookClient()
        post_history = step1_fetch_insights(fb, post_history)
        write_state(STATE_POST_HISTORY, post_history)
    except FacebookTokenExpiredError:
        log.critical("Facebook token expired — skipping insights fetch (will use existing data)")
    except Exception as exc:
        log_exception(log, "Step 1 failed", exc)

    # ── Step 2: Analyze Performance ───────────────────────────────────────
    try:
        report = step2_analyze_performance(post_history)
    except Exception as exc:
        log_exception(log, "Step 2 failed", exc)
        report = {}

    if not report:
        log.warning("No performance report generated — Steps 3-5 will be skipped or limited")

    # ── Step 3: Prompt Improvement ────────────────────────────────────────
    try:
        client = get_gemini_client()
        step3_improve_prompts(client, report)
    except Exception as exc:
        log_exception(log, "Step 3 failed", exc)

    # ── Step 4: Auto-Rollback ─────────────────────────────────────────────
    try:
        current_avg_er = report.get("avg_engagement_rate", 0.0) if report else 0.0
        step4_auto_rollback(current_avg_er)
    except Exception as exc:
        log_exception(log, "Step 4 failed", exc)

    # ── Step 5: Visual Prompt Improvement ────────────────────────────────
    try:
        step5_visual_improvement(report)
    except Exception as exc:
        log_exception(log, "Step 5 failed", exc)

    log.info("══ SelfImprovementAgent complete ══")


if __name__ == "__main__":
    main()
