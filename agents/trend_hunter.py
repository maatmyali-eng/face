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
    Fetch daily trending searches from Google Trends RSS for the given country.

    Returns a list of raw trend dicts with keys: topic, source, raw_score.
    """
    url = (
        f"https://trends.google.com/trends/trendingsearches/daily/rss"
        f"?geo={country_code}"
    )
    log.info("Google Trends: fetching %s", url)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log_exception(log, exc, "Google Trends fetch failed")
        return []

    trends: list[dict] = []
    try:
        root = ET.fromstring(resp.content)
        ns = {"ht": "https://trends.google.com/trends/trendingsearches/daily"}
        channel = root.find("channel")
        if channel is None:
            return []
        for item in channel.findall("item"):
            title_el = item.find("title")
            traffic_el = item.find("ht:approx_traffic", ns)
            title = _clean_text(title_el.text if title_el is not None else "")
            if not title:
                continue
            # traffic looks like "200,000+" — strip non-digits
            traffic_str = (traffic_el.text or "0") if traffic_el is not None else "0"
            traffic = int(re.sub(r"[^\d]", "", traffic_str) or "0")
            # related queries
            related: list[str] = []
            for rq in item.findall("ht:related_queries/ht:related_query", ns):
                t = rq.find("ht:query", ns)
                if t is not None and t.text:
                    related.append(t.text.strip())
            trends.append({
                "topic": title,
                "source": "google_trends",
                "raw_score": min(traffic / 100_000, 10.0),   # normalise to 0-10
                "raw_data": {
                    "approx_traffic": traffic,
                    "related_queries": related[:5],
                },
            })
        log.info("Google Trends: collected %d items", len(trends))
    except ET.ParseError as exc:
        log_exception(log, exc, "Google Trends XML parse error")
    return trends

# ─────────────────────────────────────────────────────────────────────────────
# Source 2 — Reddit JSON API
# ─────────────────────────────────────────────────────────────────────────────

def fetch_reddit(subreddits: list[str]) -> list[dict]:
    """
    Fetch hot posts from a list of subreddits using the public Reddit JSON API.

    Returns a list of raw trend dicts.
    """
    trends: list[dict] = []
    for sub in subreddits:
        url = f"https://www.reddit.com/r/{sub}/hot.json?limit={MAX_REDDIT_POSTS}"
        log.info("Reddit: fetching r/%s", sub)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                log.warning("Reddit: rate-limited on r/%s, sleeping 2s", sub)
                time.sleep(2)
                resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, json.JSONDecodeError) as exc:
            log_exception(log, exc, f"Reddit fetch failed for r/{sub}")
            continue

        posts = data.get("data", {}).get("children", [])
        for post in posts:
            pd = post.get("data", {})
            title = _clean_text(pd.get("title", ""))
            score = pd.get("score", 0)
            comments = pd.get("num_comments", 0)
            if not title or score < 100:
                continue
            # Normalise Reddit score (upvotes) → 0-10
            raw_score = min(score / 5000, 10.0)
            trends.append({
                "topic": title,
                "source": "reddit",
                "raw_score": raw_score,
                "raw_data": {
                    "subreddit": sub,
                    "score": score,
                    "num_comments": comments,
                    "url": pd.get("url", ""),
                },
            })
        time.sleep(0.5)   # be polite to Reddit
    log.info("Reddit: collected %d items total", len(trends))
    return trends

# ─────────────────────────────────────────────────────────────────────────────
# Source 3 — YouTube Trending RSS
# ─────────────────────────────────────────────────────────────────────────────

def fetch_youtube_trending(country_code: str) -> list[dict]:
    """
    Fetch trending YouTube videos via the public Atom feed.

    Returns a list of raw trend dicts.
    """
    url = (
        f"https://www.youtube.com/feeds/videos.xml"
        f"?chart=trending&regionCode={country_code}&hl=en"
    )
    log.info("YouTube Trending: fetching %s", url)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log_exception(log, exc, "YouTube Trending fetch failed")
        return []

    trends: list[dict] = []
    try:
        ns = {
            "atom":  "http://www.w3.org/2005/Atom",
            "media": "http://search.yahoo.com/mrss/",
            "yt":    "http://www.youtube.com/xml/schemas/2015",
        }
        root = ET.fromstring(resp.content)
        for entry in root.findall("atom:entry", ns):
            title_el = entry.find("atom:title", ns)
            published_el = entry.find("atom:published", ns)
            summary_el = entry.find("media:group/media:description", ns)
            title = _clean_text(title_el.text if title_el is not None else "")
            published = (published_el.text or "") if published_el is not None else ""
            summary = _clean_text(summary_el.text if summary_el is not None else "")
            if not title:
                continue
            trends.append({
                "topic": title,
                "source": "youtube",
                "raw_score": 7.0,   # YouTube trending is already curated — flat score
                "raw_data": {
                    "published": published,
                    "summary": summary[:200],
                },
            })
        log.info("YouTube Trending: collected %d items", len(trends))
    except ET.ParseError as exc:
        log_exception(log, exc, "YouTube Trending XML parse error")
    return trends

# ─────────────────────────────────────────────────────────────────────────────
# Source 4 — Nitter RSS
# ─────────────────────────────────────────────────────────────────────────────

def fetch_nitter(niche: str) -> list[dict]:
    """
    Fetch trending tweets related to the niche via public Nitter RSS instances.
    Tries each instance in order and returns on first success.

    Returns a list of raw trend dicts.
    """
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
    Fetch all 4 trend sources concurrently using a thread pool.
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

    if not raw_trends:
        log.error("All trend sources returned zero items — writing empty trends file")
        write_state("trends.json", {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "region": region,
            "niche": niche,
            "trends": [],
        })
        raise SystemExit(1)

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
