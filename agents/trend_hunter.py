"""
agents/trend_hunter.py
======================
TrendHunterAgent — Session 3, Deliverable 1

Fetches trending topics from 4 free public sources concurrently,
deduplicates by keyword overlap, scores each trend via Gemini, and
writes the top 10 to state/trends.json.

Sources:
  1. Google Trends RSS  (no auth)
  2. Reddit JSON API    (no auth)
  3. YouTube Trending RSS (no auth)
  4. Nitter RSS         (no auth, multiple fallback instances)

Run:
    python agents/trend_hunter.py
"""

from __future__ import annotations

import asyncio
import html
import json
import os
import re
import time
import uuid
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote_plus

import aiohttp
import requests

from utils.config_loader import get_config, build_prompt_context
from utils.gemini_client import get_gemini_client, GeminiQuotaExhaustedError
from utils.logger import get_logger, log_exception
from utils.rate_limiter import GEMINI_LIMITER
from utils.state_manager import write_state

# ── Logger ────────────────────────────────────────────────────────────────────
log = get_logger("trend_hunter")

# ── Constants ─────────────────────────────────────────────────────────────────
REQUEST_TIMEOUT = 15          # seconds per HTTP request
MAX_REDDIT_POSTS = 25
KEYWORD_OVERLAP_THRESHOLD = 0.60   # 60 % keyword overlap → duplicate
TOP_N_TRENDS = 10

NITTER_INSTANCES = [
    "https://nitter.net",
    "https://nitter.privacydev.net",
    "https://nitter.poast.org",
    "https://nitter.kavin.rocks",
]

# Niche → subreddit mapping (extend freely)
NICHE_SUBREDDITS: dict[str, list[str]] = {
    "travel":      ["travel", "solotravel", "backpacking"],
    "tech":        ["technology", "programming", "artificial"],
    "food":        ["food", "recipes", "cooking"],
    "motivation":  ["GetMotivated", "selfimprovement", "DecidingToBeBetter"],
    "education":   ["learnprogramming", "todayilearned", "explainlikeimfive"],
    "fitness":     ["fitness", "bodyweightfitness", "loseit"],
    "business":    ["entrepreneur", "smallbusiness", "startups"],
    "finance":     ["personalfinance", "investing", "financialindependence"],
    "beauty":      ["SkincareAddiction", "MakeupAddiction", "Hair"],
    "gaming":      ["gaming", "Games", "pcgaming"],
    "parenting":   ["Parenting", "mommit", "daddit"],
    "fashion":     ["femalefashionadvice", "malefashionadvice", "streetwear"],
    "health":      ["Health", "HealthyFood", "mentalhealth"],
    "pets":        ["aww", "dogs", "cats"],
    "diy":         ["DIY", "crafts", "HomeImprovement"],
    "default":     ["popular", "all"],
}

# target_region (config) → ISO country code for Google Trends / YouTube
REGION_TO_COUNTRY: dict[str, str] = {
    "Bangladesh": "BD",
    "USA": "US",
    "United States": "US",
    "UK": "GB",
    "United Kingdom": "GB",
    "India": "IN",
    "Canada": "CA",
    "Australia": "AU",
    "Germany": "DE",
    "France": "FR",
    "Brazil": "BR",
    "Pakistan": "PK",
    "Indonesia": "ID",
    "Nigeria": "NG",
    "South Africa": "ZA",
    "Philippines": "PH",
    "global": "US",       # Google Trends / YouTube don't have a "global" code
    "Global": "US",
}

HEADERS = {
    "User-Agent": "FACE-Bot/1.0 (Facebook Autonomous Content Engine; research purposes)",
    "Accept": "application/rss+xml, application/xml, text/xml, application/json, */*",
}

# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Strip HTML tags and decode HTML entities from a string."""
    text = html.unescape(text or "")
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


def _tokenise(text: str) -> set[str]:
    """Lowercase words ≥ 3 chars, stripped of punctuation."""
    return {w for w in re.findall(r"[a-z0-9]{3,}", text.lower())}


def _keyword_overlap(a: str, b: str) -> float:
    """Return Jaccard-style overlap ratio between two topic strings."""
    ta, tb = _tokenise(a), _tokenise(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _deduplicate(raw_trends: list[dict]) -> list[dict]:
    """
    Remove duplicate trends using keyword overlap ≥ KEYWORD_OVERLAP_THRESHOLD.
    When two items are duplicates, keep the one with the higher raw_score.
    """
    unique: list[dict] = []
    for item in raw_trends:
        is_dup = False
        for existing in unique:
            if _keyword_overlap(item["topic"], existing["topic"]) >= KEYWORD_OVERLAP_THRESHOLD:
                # Keep whichever has the higher raw score
                if item.get("raw_score", 0) > existing.get("raw_score", 0):
                    existing.update(item)
                is_dup = True
                break
        if not is_dup:
            unique.append(item)
    return unique


def _country_code(cfg: dict) -> str:
    region = cfg.get("target_region", "global")
    return REGION_TO_COUNTRY.get(region, "US")


def _get_subreddits(cfg: dict) -> list[str]:
    niche = cfg.get("niche", "default").lower()
    return NICHE_SUBREDDITS.get(niche, NICHE_SUBREDDITS["default"])

# ─────────────────────────────────────────────────────────────────────────────
# Source 1 — Google Trends RSS
# ─────────────────────────────────────────────────────────────────────────────

def fetch_google_trends(country_code: str) -> list[dict]:
    """
    Fetch current Google Trends RSS topics for the configured country.

    Google retired the old /trends/trendingsearches/daily/rss endpoint. The
    current /trending/rss feed uses its own XML namespace and K/M traffic
    suffixes, which must be parsed explicitly.
    """
    url = f"https://trends.google.com/trending/rss?geo={country_code}"
    log.info("Google Trends: fetching %s", url)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log_exception(log, exc, "Google Trends fetch failed")
        return []

    def parse_traffic(value: str) -> int:
        match = re.search(r"([0-9][0-9,.]*)\s*([KMB]?)", str(value or "0"), re.IGNORECASE)
        if not match:
            return 0
        try:
            number = float(match.group(1).replace(",", ""))
        except ValueError:
            return 0
        multiplier = {"": 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
        return int(number * multiplier.get(match.group(2).upper(), 1))

    trends: list[dict] = []
    try:
        root = ET.fromstring(resp.content)
        ns = {"ht": "https://trends.google.com/trending/rss"}
        channel = root.find("channel")
        if channel is None:
            log.warning("Google Trends RSS response has no channel element")
            return []

        for item in channel.findall("item"):
            title_el = item.find("title")
            title = _clean_text(title_el.text if title_el is not None else "")
            if not title:
                continue

            traffic_str = item.findtext("ht:approx_traffic", default="0", namespaces=ns)
            traffic = parse_traffic(traffic_str)
            published = item.findtext("pubDate", default="", namespaces=ns)

            news_items: list[dict[str, str]] = []
            for news in item.findall("ht:news_item", ns):
                news_title = news.findtext("ht:news_item_title", default="", namespaces=ns).strip()
                news_url = news.findtext("ht:news_item_url", default="", namespaces=ns).strip()
                news_source = news.findtext("ht:news_item_source", default="", namespaces=ns).strip()
                news_snippet = news.findtext("ht:news_item_snippet", default="", namespaces=ns).strip()
                if news_title:
                    news_items.append({
                        "title": news_title[:240],
                        "url": news_url,
                        "source": news_source[:120],
                        "snippet": news_snippet[:400],
                    })

            trends.append({
                "topic": title,
                "source": "google_trends",
                "raw_score": min(traffic / 100_000, 10.0),
                "raw_data": {
                    "approx_traffic": traffic,
                    "approx_traffic_display": traffic_str,
                    "published_at": published,
                    "news_items": news_items[:3],
                },
            })
        log.info("Google Trends: collected %d current items", len(trends))
    except ET.ParseError as exc:
        log_exception(log, exc, "Google Trends XML parse error")
    return trends

# ─────────────────────────────────────────────────────────────────────────────
# Source 2 — Reddit JSON API
# ─────────────────────────────────────────────────────────────────────────────

def fetch_reddit(subreddits: list[str]) -> list[dict]:
    """
    Reddit retired unauthenticated public .json API access. Do not repeatedly
    call the known-blocked endpoint; add an official OAuth adapter before
    enabling this source again.
    """
    log.info(
        "Reddit source skipped: public .json access requires authenticated OAuth; "
        "Google Trends and fallback topics remain available."
    )
    return []

# ─────────────────────────────────────────────────────────────────────────────
# Source 3 — YouTube Trending RSS
# ─────────────────────────────────────────────────────────────────────────────

def fetch_youtube_trending(country_code: str) -> list[dict]:
    """
    Fetch YouTube's current mostPopular category-chart videos using Data API v3.

    YouTube retired its old trending feed/page. Data API's mostPopular chart
    now represents Music, Movies, and Gaming charts; an API key is required.
    """
    api_key = os.environ.get("YOUTUBE_API_KEY", "").strip()
    if not api_key:
        log.info("YouTube chart source skipped: YOUTUBE_API_KEY is not configured.")
        return []

    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        "part": "snippet,statistics",
        "chart": "mostPopular",
        "regionCode": country_code,
        "maxResults": min(MAX_REDDIT_POSTS, 50),
        "key": api_key,
    }
    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()
    except (requests.RequestException, json.JSONDecodeError, ValueError) as exc:
        log_exception(log, exc, "YouTube Data API request failed")
        return []

    trends: list[dict] = []
    for video in payload.get("items", []):
        snippet = video.get("snippet") or {}
        stats = video.get("statistics") or {}
        title = _clean_text(snippet.get("title", ""))
        video_id = str(video.get("id", "")).strip()
        if not title or not video_id:
            continue
        try:
            views = int(stats.get("viewCount", 0) or 0)
        except (TypeError, ValueError):
            views = 0
        trends.append({
            "topic": title,
            "source": "youtube_most_popular",
            "raw_score": min(views / 1_000_000, 10.0),
            "raw_data": {
                "view_count": views,
                "channel_title": str(snippet.get("channelTitle", ""))[:160],
                "published_at": str(snippet.get("publishedAt", "")),
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "chart_scope": "YouTube Music, Movies, and Gaming charts; not a general Trending page",
            },
        })

    log.info("YouTube Data API returned %d chart items", len(trends))
    return trends

# ─────────────────────────────────────────────────────────────────────────────
# Source 4 — Nitter RSS
# ─────────────────────────────────────────────────────────────────────────────

def fetch_nitter(niche: str) -> list[dict]:
    """
    Optionally fetch niche-specific posts through a public Nitter RSS instance.
    Nitter is unreliable; keep it disabled unless explicitly opted in.
    """
    if os.environ.get("ENABLE_NITTER_SOURCE", "").strip().lower() != "true":
        log.info("Nitter source skipped by default; set ENABLE_NITTER_SOURCE=true to opt in.")
        return []
    query = quote_plus(f"{niche} trending")
    trends: list[dict] = []
    for base in NITTER_INSTANCES:
        url = f"{base}/search/rss?q={query}&f=tweets"
        log.info("Nitter: trying %s", url)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                log.warning("Nitter: %s returned %d", base, resp.status_code)
                continue
            root = ET.fromstring(resp.content)
            channel = root.find("channel")
            if channel is None:
                continue
            for item in channel.findall("item"):
                title_el = item.find("title")
                pub_el = item.find("pubDate")
                title = _clean_text(title_el.text if title_el is not None else "")
                pub_date = (pub_el.text or "") if pub_el is not None else ""
                if not title or len(title) < 10:
                    continue
                trends.append({
                    "topic": title[:120],     # cap length
                    "source": "twitter",
                    "raw_score": 6.5,         # tweets are noisy — conservative score
                    "raw_data": {"pub_date": pub_date},
                })
            log.info("Nitter: collected %d items from %s", len(trends), base)
            if trends:
                return trends           # first working instance is enough
        except (requests.RequestException, ET.ParseError) as exc:
            log_exception(log, exc, f"Nitter instance {base} failed")
            continue

    log.warning("Nitter: all instances failed or returned no data")
    return trends

# ─────────────────────────────────────────────────────────────────────────────
# Async parallel fetch
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_all_sources_parallel(cfg: dict) -> list[dict]:
    """
    Fetch all 4 configured sources concurrently using a thread pool.
    Individual source failures are caught and logged — pipeline continues.
    """
    country_code = _country_code(cfg)
    subreddits   = _get_subreddits(cfg)
    niche        = cfg.get("niche", "general")

    tasks = {
        "google_trends": (fetch_google_trends, (country_code,)),
        "reddit":        (fetch_reddit,         (subreddits,)),
        "youtube":       (fetch_youtube_trending, (country_code,)),
        "nitter":        (fetch_nitter,          (niche,)),
    }

    all_trends: list[dict] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(fn, *args): name
            for name, (fn, args) in tasks.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                results = future.result()
                log.info("Source %s returned %d items", name, len(results))
                all_trends.extend(results)
            except Exception as exc:           # noqa: BLE001
                log_exception(log, exc, f"Source {name} raised unexpected error")

    return all_trends

def _build_evergreen_fallback(cfg: dict) -> list[dict]:
    """
    Return niche-specific evergreen ideas when every external trend source fails.

    These are explicitly marked as fallback topics, not live trends. This keeps
    the pipeline productive without fabricating current popularity or metrics.
    """
    niche = str(cfg.get("niche", "general")).strip() or "general"
    region = str(cfg.get("target_region", "global")).strip() or "global"
    sub_niches = cfg.get("sub_niches", [])
    sub_niche = str(sub_niches[0]).strip() if isinstance(sub_niches, list) and sub_niches else niche

    if niche.lower() == "travel":
        topics = [
            f"Budget-friendly travel ideas in {region}",
            f"Hidden-gem destinations and local experiences in {region}",
            f"Travel-planning mistakes to avoid when visiting {region}",
            f"Practical tips for first-time travelers in {region}",
            f"How to plan a memorable trip in {region} on a realistic budget",
        ]
    else:
        topics = [
            f"Beginner-friendly {niche} tips for people in {region}",
            f"Common {niche} mistakes and how to avoid them",
            f"Budget-friendly {niche} ideas for people in {region}",
            f"Practical questions people ask about {niche}",
            f"How to get started with {sub_niche} in {region}",
        ]

    now = datetime.now(timezone.utc).isoformat()
    return [
        {
            "topic": topic,
            "source": "evergreen_fallback",
            "raw_score": 0.0,
            "raw_data": {
                "fallback": True,
                "not_a_live_trend": True,
                "evidence": "No configured public trend source returned usable items in this run.",
                "generated_at": now,
            },
        }
        for topic in topics
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Gemini scoring
# ─────────────────────────────────────────────────────────────────────────────

def _score_trend_with_gemini(
    client,
    trend: dict,
    ctx: dict,
) -> dict | None:
    """
    Call Gemini trend_analysis.txt for a single trend topic.
    Returns enriched trend dict or None on failure.
    """
    try:
        GEMINI_LIMITER.acquire()
        result = client.generate_json(
            "trend_analysis.txt",
            **ctx,
            trend_topic=trend["topic"],
            trend_source=trend["source"],
            trend_raw_data=json.dumps(trend.get("raw_data", {})),
        )
    except GeminiQuotaExhaustedError as exc:
        log.critical("Gemini quota exhausted while scoring trends: %s", exc)
        raise   # re-raise so caller can halt loop
    except Exception as exc:           # noqa: BLE001
        log_exception(log, exc, f"Gemini scoring failed for topic: {trend['topic'][:60]}")
        return None

    # Gemini must return: virality_score, niche_fit_score, region_fit_score,
    # recommended_format, hook_angle
    required = {"virality_score", "niche_fit_score", "region_fit_score",
                "recommended_format", "hook_angle"}
    if not required.issubset(result.keys()):
        log.warning("Gemini response missing fields for topic %r: %s",
                    trend["topic"][:60], result)
        return None

    vs  = float(result["virality_score"])
    ns  = float(result["niche_fit_score"])
    rs  = float(result["region_fit_score"])
    combined = round((vs * 0.4 + ns * 0.35 + rs * 0.25), 2)

    return {
        **trend,
        "virality_score":    vs,
        "niche_fit_score":   ns,
        "region_fit_score":  rs,
        "combined_score":    combined,
        "recommended_format": result["recommended_format"],
        "hook_angle":         result.get("hook_angle", ""),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Main agent
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point for TrendHunterAgent."""
    log.info("=== TrendHunterAgent starting ===")
    start_time = time.time()

    # ── Load config ───────────────────────────────────────────────────────────
    try:
        cfg = get_config()
    except Exception as exc:
        log_exception(log, exc, "Failed to load config — aborting")
        raise SystemExit(1)

    niche  = cfg.get("niche", "general")
    region = cfg.get("target_region", "global")
    ctx    = build_prompt_context()   # flat dict of all config placeholders

    log.info("Niche: %s | Region: %s", niche, region)

    # ── Step 1: Fetch all sources concurrently ─────────────────────────────
    log.info("Step 1/4 — Fetching trend sources concurrently")
    raw_trends = _fetch_all_sources_parallel(cfg)
    log.info("Total raw items collected: %d", len(raw_trends))

    evergreen_backups = _build_evergreen_fallback(cfg)
    if not raw_trends:
        log.warning(
            "All live sources failed; using clearly labelled evergreen ideas "
            "instead of pretending we have current trend data."
        )
        raw_trends = evergreen_backups
    elif len(raw_trends) < 25:
        # Let niche-relevant evergreen ideas compete with general country trends;
        # raw_score=0 keeps live evidence ahead when enough live items exist.
        raw_trends.extend(evergreen_backups)
        log.info("Added %d low-priority evergreen backup candidates", len(evergreen_backups))
    log.info("Total candidates after fallback: %d", len(raw_trends))

    # ── Step 2: Deduplicate ─────────────────────────────────────────────────
    log.info("Step 2/4 — Deduplicating %d items", len(raw_trends))
    unique_trends = _deduplicate(raw_trends)
    log.info("After dedup: %d unique topics", len(unique_trends))

    # Sort by raw_score descending before Gemini scoring
    # (this limits API calls if quota is tight — we stop after TOP_N_TRENDS)
    unique_trends.sort(key=lambda t: t.get("raw_score", 0), reverse=True)

    # ── Step 3: Score with Gemini ───────────────────────────────────────────
    log.info("Step 3/4 — Scoring top candidates with Gemini")
    client = get_gemini_client()
    scored: list[dict] = []

    # Score up to 25 candidates (then slice to top 10)
    candidates = unique_trends[:25]
    for i, trend in enumerate(candidates, 1):
        log.debug("Scoring %d/%d: %s", i, len(candidates), trend["topic"][:60])
        try:
            scored_trend = _score_trend_with_gemini(client, trend, ctx)
        except GeminiQuotaExhaustedError:
            log.critical("Stopping Gemini scoring due to quota exhaustion")
            break
        if scored_trend:
            scored.append(scored_trend)

    if not scored:
        log.error("No trends survived Gemini scoring — writing raw top 10 as fallback")
        # Build fallback entries with placeholder scores
        fallback: list[dict] = []
        for i, t in enumerate(unique_trends[:TOP_N_TRENDS], 1):
            fallback.append({
                "rank": i,
                "topic": t["topic"],
                "source": t["source"],
                "virality_score":   t.get("raw_score", 5.0),
                "niche_fit_score":  5.0,
                "region_fit_score": 5.0,
                "combined_score":   t.get("raw_score", 5.0),
                "recommended_format": "single_image",
                "hook_angle": "General interest angle",
                "raw_data": t.get("raw_data", {}),
            })
        _write_trends(region, niche, fallback)
        return

    # ── Step 4: Rank and write top 10 ──────────────────────────────────────
    log.info("Step 4/4 — Ranking and writing state/trends.json")
    scored.sort(key=lambda t: t["combined_score"], reverse=True)
    top_10 = scored[:TOP_N_TRENDS]

    output_trends: list[dict] = []
    for rank, t in enumerate(top_10, 1):
        output_trends.append({
            "rank": rank,
            "topic": t["topic"],
            "source": t["source"],
            "virality_score":    round(t["virality_score"], 2),
            "niche_fit_score":   round(t["niche_fit_score"], 2),
            "region_fit_score":  round(t["region_fit_score"], 2),
            "combined_score":    round(t["combined_score"], 2),
            "recommended_format": t["recommended_format"],
            "hook_angle":         t["hook_angle"],
            "raw_data":           t.get("raw_data", {}),
        })

    _write_trends(region, niche, output_trends)
    elapsed = round(time.time() - start_time, 1)
    log.info("=== TrendHunterAgent complete in %.1fs — top topic: %r ===",
             elapsed, output_trends[0]["topic"] if output_trends else "none")


def _write_trends(region: str, niche: str, trends: list[dict]) -> None:
    """Write trends to state/trends.json."""
    state = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "region": region,
        "niche": niche,
        "trends": trends,
    }
    write_state("trends.json", state)
    log.info("Wrote %d trends to state/trends.json", len(trends))


if __name__ == "__main__":
    main()
