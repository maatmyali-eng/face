# FACE — Facebook Autonomous Content Engine

> **Fully autonomous, $0-cost Facebook Page management system powered by Google Gemini AI.**

[![FACE Daily Pipeline](https://github.com/YOUR_USERNAME/face/actions/workflows/daily_pipeline.yml/badge.svg)](https://github.com/YOUR_USERNAME/face/actions/workflows/daily_pipeline.yml)

---

## What This Does

FACE is a complete, self-managing Facebook content system that runs entirely inside GitHub Actions — no server, no hosting, no monthly fees. Once configured, it wakes up three times a day, researches what is trending in your niche and region, writes captions and reel scripts tailored to your audience, generates or sources images and video, and posts everything to your Facebook Page automatically.

Every Sunday, the system analyzes the past week's engagement data from your page and rewrites its own AI prompts to improve performance. On the first of every month, a separate agent researches the current Facebook algorithm, updates its understanding of ranking signals, and adjusts content format weights accordingly. The only time you need to touch the system is every ~60 days when your Facebook token expires — the system creates a GitHub Issue to remind you.

The entire system is built on free tiers: Gemini AI (1,500 calls/day free), Hugging Face (image generation, free), Pexels (stock images, free), GitHub Actions (2,000 minutes/month free), and edge-tts (Microsoft's TTS, completely free). The only requirement is a GitHub account, a Google account, a Facebook Page, and about 30 minutes of initial setup.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GITHUB ACTIONS (Cron Scheduler)                  │
│                                                                     │
│   Daily 03:00/07:00/14:00 UTC          Monthly 1st    Weekly Sun   │
│          │                                  │              │        │
│          ▼                                  ▼              ▼        │
│  ┌───────────────────┐          ┌──────────────────┐  ┌──────────┐ │
│  │  DAILY PIPELINE   │          │ ALGORITHM WATCHER│  │  SELF-   │ │
│  │                   │          │                  │  │IMPROVER  │ │
│  │ 1. TrendHunter   ─┼─►trends  │ algorithm_rules  │  │          │ │
│  │ 2. ContentStrat  ─┼─►plan    │ visual_style     │  │perf anal │ │
│  │ 3. ContentGen    ─┼─►content │ best_practices   │  │prompt    │ │
│  │ 4. VisualDir     ─┼─►images  └──────────────────┘  │rewrite   │ │
│  │ 5. ReelProducer  ─┼─►videos                        └──────────┘ │
│  │ 6. Publisher     ─┼─►posts to Facebook                          │
│  └───────────────────┘                                             │
│          │                                                          │
│          ▼                                                          │
│   state/ branch (git)  ◄── All inter-agent state stored here       │
└─────────────────────────────────────────────────────────────────────┘

External APIs:
  ┌──────────────────────────────────────────────────────────────────┐
  │  Google Gemini API   — AI text/JSON generation (free 1500/day)   │
  │  Pollinations.ai     — Image generation (no key, free)           │
  │  Hugging Face API    — FLUX.1 / SDXL image gen (free 1000/day)   │
  │  Pexels API          — Stock image fallback (free 20k/month)     │
  │  edge-tts            — Microsoft TTS for reel voiceover (free)   │
  │  yt-dlp              — Background music download (free)          │
  │  Facebook Graph API  — Post publishing and insights              │
  └──────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

- **GitHub account** (free) — where the code lives and runs
- **Google account** (free) — for Gemini API access
- **Facebook account + Page** (free) — the page you want to automate
- **Hugging Face account** (free) — for AI image generation
- **Pexels account** (free) — for stock image fallback

No credit card required for any of the above.

---

## Setup Guide

### Step 1: Fork This Repository

1. Click the **Fork** button at the top-right of this page.
2. Select your GitHub account as the destination.
3. Leave the repository name as `face` (or rename it — it does not matter).
4. Click **Create fork**.
5. You now have your own copy at `https://github.com/YOUR_USERNAME/face`.

All subsequent steps are done inside your forked repository.

---

### Step 2: Configure `config.yml`

Open `config.yml` in your forked repo (click the file → pencil icon to edit).
Fill in every field. Below is a complete filled example for a
**Bangladesh travel page targeting 18–30 year olds**:

```yaml
# ── PAGE IDENTITY ─────────────────────────────────────────────
page:
  name: "Bhromon BD"
  niche: "travel"
  description: "Exploring the hidden beauty of Bangladesh — beaches, hills, rivers, culture."
  brand_voice: "adventurous, inspiring, locally rooted"
  target_language: "Bilingual (English + Bangla)"

# ── AUDIENCE DEFINITION ───────────────────────────────────────
audience:
  age_range: "18-30"
  gender: "all"
  target_region: "Bangladesh"
  interests:
    - "travel"
    - "photography"
    - "adventure"
    - "culture"
    - "food"
  pain_points:
    - "can't afford expensive international travel"
    - "don't know hidden local gems"
    - "want travel content in Bangla"

# ── POSTING SCHEDULE ──────────────────────────────────────────
posting_schedule:
  posts_per_day: 3
  daily_post_times:          # Your local time (Bangladesh = UTC+6)
    - "09:00"
    - "13:00"
    - "20:00"
  timezone: "Asia/Dhaka"
  reel_days:
    - 3   # Wednesday (0=Monday, 6=Sunday)
    - 6   # Sunday
  reels_per_week: 2
  story_days:
    - 1   # Tuesday
    - 4   # Friday
    - 6   # Sunday

# ── CONTENT STYLE ────────────────────────────────────────────
content:
  tone: "warm, adventurous, conversational"
  content_pillars:
    - "destination_spotlight"
    - "travel_tips"
    - "local_culture"
    - "community_stories"
    - "educational"
  caption_style: "hook + 3 short paragraphs + cta"
  hashtag_count: 18
  cta_preference: "comment"
  emojis: true

# ── VISUAL SETTINGS ──────────────────────────────────────────
visuals:
  image_style: "cinematic travel photography"
  color_mood: "warm golden hour"
  text_overlay: true
  text_overlay_font_size: "large"
  watermark: false
  reel_style: "fast cuts with ambient music"
  aspect_ratio_feed: "4:5"
  aspect_ratio_reel: "9:16"

# ── AGENT TOGGLES ────────────────────────────────────────────
agents:
  enable_trend_hunter: true
  enable_content_strategy: true
  enable_content_generator: true
  enable_visual_director: true
  enable_reel_producer: true
  enable_publisher: true
  enable_algorithm_watcher: true
  enable_self_improver: true

# ── API SETTINGS ─────────────────────────────────────────────
api:
  gemini_model: "gemini-1.5-flash"
  gemini_temperature: 0.85
  image_providers:
    - "pollinations"
    - "huggingface"
    - "sdxl"
    - "pexels"

# ── QUALITY GATE ─────────────────────────────────────────────
quality_gate:
  min_image_score: 7
  max_image_attempts: 3
  enable_safety_check: true

# ── SELF-IMPROVEMENT ─────────────────────────────────────────
self_improvement:
  enable: true
  max_prompt_versions: 8
  rollback_window_days: 14
  prompts_to_improve:
    - "post_caption.txt"
    - "reel_script.txt"
    - "story_content.txt"
    - "visual_brief_generator.txt"
```

**Key fields explained:**

| Field | What to put | Example |
|---|---|---|
| `page.niche` | Single word describing your page topic | `travel`, `food`, `fitness`, `tech` |
| `page.target_language` | Language for content | `English`, `Bangla`, `Bilingual (English + Bangla)` |
| `audience.target_region` | Country or region for trend research | `Bangladesh`, `India`, `USA` |
| `posting_schedule.daily_post_times` | Your local timezone times | `09:00`, `13:00`, `20:00` |
| `posting_schedule.timezone` | Your timezone string | `Asia/Dhaka`, `America/New_York` |
| `content.content_pillars` | 3–6 themes you post about | See example above |
| `quality_gate.min_image_score` | Minimum image quality (7 recommended) | `6`–`8` |

---

### Step 3: Get Your Facebook Page Access Token

**3a. Create a Facebook Developer App:**
1. Go to https://developers.facebook.com
2. Click **My Apps** → **Create App**
3. Select **Business** as the app type → click **Next**
4. Enter an app name (e.g. "FACE Automation") and your email → **Create App**
5. On the app dashboard, click **Set Up** next to **Facebook Login**
6. Choose **Web** → enter `https://localhost` → **Save**

**3b. Generate a short-lived token:**
1. Go to https://developers.facebook.com/tools/explorer
2. Select your app (top-right dropdown)
3. Select your Facebook **Page** (not personal account) in the User or Page dropdown
4. Add these permissions (search and tick each):
   - `pages_manage_posts`
   - `pages_read_engagement`
   - `pages_show_list`
   - `read_insights`
5. Click **Generate Access Token** → log in when prompted → copy the token

**3c. Extend to a 60-day long-lived token:**

Open this URL in your browser (replace placeholders with values from App Dashboard → Settings → Basic):

```
https://graph.facebook.com/v21.0/oauth/access_token
  ?grant_type=fb_exchange_token
  &client_id=YOUR_APP_ID
  &client_secret=YOUR_APP_SECRET
  &fb_exchange_token=YOUR_SHORT_LIVED_TOKEN
```

The response JSON contains `access_token` — this is your 60-day user token.

**3d. Get your Page Access Token:**

```
https://graph.facebook.com/v21.0/me/accounts?access_token=YOUR_60_DAY_USER_TOKEN
```

Find your page in the response array. Copy its `access_token` value.
This is what goes into the GitHub secret `FB_PAGE_ACCESS_TOKEN`.

> **Token expiry:** Tokens expire every ~60 days. FACE automatically creates a
> GitHub Issue titled "⚠️ Facebook Token Expired — Action Required" on expiry.
> Follow Q4 in Troubleshooting to refresh it.

---

### Step 4: Get Your Gemini API Key

1. Go to https://aistudio.google.com
2. Sign in with your Google account
3. Click **Get API key** in the left sidebar
4. Click **Create API key** → select or create a Google Cloud project
5. Copy the key — it starts with `AIza...`

**Free tier:** 15 requests/minute, 1,500 requests/day, 1 million tokens/day.
FACE uses approximately 150–300 requests/day, well within the free limit.

---

### Step 5: Get Your Hugging Face Token

1. Go to https://huggingface.co → **Sign Up** (free)
2. Click your profile picture → **Settings**
3. Click **Access Tokens** → **New token**
4. Name it `face-token`, select **Read** permission → **Generate token**
5. Copy the token — it starts with `hf_...`

**Free tier:** ~1,000 inference API requests/day. Used for FLUX.1-schnell and
Stable Diffusion XL image generation (second and third fallbacks).

---

### Step 6: Get Your Pexels API Key

1. Go to https://www.pexels.com/api/
2. Click **Get Started** → create a free account
3. Fill in the API request form (describe your use: "automated social media content")
4. Once approved (usually instant), copy your API key from the dashboard

**Free tier:** 200 requests/hour, 20,000 requests/month. Used only when all
3 AI image providers fail for a given post.

---

### Step 7: Add GitHub Secrets

In your forked repository:
1. Click **Settings** (top repo menu)
2. Left sidebar: **Secrets and variables** → **Actions**
3. Click **New repository secret** and add each of the following:

| Secret Name | Where to find it | Notes |
|---|---|---|
| `GEMINI_API_KEY` | Google AI Studio dashboard | Starts with `AIza...` |
| `FB_PAGE_ACCESS_TOKEN` | Facebook Graph API Explorer (Step 3d) | 60-day page token |
| `FB_PAGE_ID` | Facebook Page → About → Page ID | Numeric only (e.g. `123456789`) |
| `HF_TOKEN` | Hugging Face → Settings → Tokens | Starts with `hf_...` |
| `PEXELS_API_KEY` | Pexels API dashboard | Random alphanumeric string |

Secret names are **case-sensitive** — copy exactly as shown.
`GITHUB_TOKEN` is auto-provided by GitHub Actions — do not add it manually.

**How to find your Facebook Page ID:**
Go to your Page → **About** → scroll to bottom → **Page ID** (long number).

---

### Step 8: Initialize the State Branch

FACE stores all inter-agent state in a separate git branch called `state`.
Run these commands once from your terminal:

```bash
# Clone your forked repo
git clone https://github.com/YOUR_USERNAME/face.git
cd face

# Create an orphan branch (no parent commits — clean slate)
git checkout --orphan state
git rm -rf .

# Create initial state directory structure
mkdir -p state/logs state/.locks state/prompt_versions state/schema
touch state/.gitkeep

# Commit and push
git add state/
git commit -m "init: state branch"
git push origin state

# Return to main
git checkout main
```

**No terminal?** Create the branch in GitHub's web UI:
1. Click the branch dropdown (shows `main`)
2. Type `state` → click **Create branch: state from main**

---

### Step 9: Enable GitHub Actions

1. Click the **Actions** tab in your forked repo
2. If you see: "Workflows aren't being run on this forked repository"
3. Click **I understand my workflows, go ahead and enable them**
4. You should see three workflows:
   - FACE Daily Pipeline
   - FACE Monthly Algorithm Watcher
   - FACE Weekly Self-Improver

---

### Step 10: Run Your First Manual Test

1. Click **Actions** → **FACE Daily Pipeline**
2. Click **Run workflow** → **Run workflow**
3. Click the running workflow to watch live logs

**What to look for:**
- Each step shows a green checkmark ✅
- `Run PublisherAgent` step logs: `Published post_id: 123456789_987654321`
- Your Facebook Page has a new post within 1–2 minutes

---

### Step 11: Bootstrap the Algorithm Watcher (Recommended)

The Algorithm Watcher runs on the 1st of every month. For immediate region-specific knowledge:

1. Click **Actions** → **FACE Monthly Algorithm Watcher**
2. Click **Run workflow** → **Run workflow**
3. Wait ~3 minutes

This populates `state/algorithm_rules.json`, `state/visual_style.json`, and
`state/best_practices.json` so the daily pipeline uses region-specific rules from day one.

---

### Step 12: You're Done 🎉

| Schedule | Agent | What it does |
|---|---|---|
| 9 AM / 1 PM / 8 PM BDT (daily) | Daily Pipeline | Research trends → create content → generate images → post to Facebook |
| 1st of every month, 8 AM BDT | Algorithm Watcher | Update Facebook algorithm knowledge |
| Every Sunday, 9 AM BDT | Self-Improver | Analyze performance, rewrite underperforming prompts |

**Your only ongoing task:** Refresh the Facebook token every ~60 days
when FACE creates a GitHub Issue reminding you.

---

## Free Tier Limits Reference

| Service | Free Limit | FACE Daily Usage | Monthly Budget Used |
|---|---|---|---|
| Google Gemini | 1,500 req/day, 15 RPM | ~300 req/day | ~9,000/45,000 (20%) |
| Pollinations.ai | Unlimited (no key) | ~3–9 images/day | Unlimited |
| Hugging Face | ~1,000 req/day | ~3–9 images/day (fallback) | ~1–2% used |
| Pexels API | 20,000 req/month | 0–9 images/day (last resort) | <2% used |
| Facebook Graph API | 200 calls/hour | ~6–12 calls/day | No monthly limit |
| GitHub Actions | 2,000 min/month | ~11–33 min/run × 3/day | ~990–2,970 min ⚠️ |
| edge-tts | Unlimited | ~2–8 TTS/reel | Unlimited |
| yt-dlp | Free (cached) | ~2–3 downloads/week | Cached indefinitely |
| GitHub Storage | 1 GB | ~5 MB state/day | ~150 MB/month |

> ⚠️ **GitHub Actions minutes:** Running reels every day uses ~2,970 min/month
> against a 2,000 min free limit. Set `reels_per_week: 2` or `3` in config.yml.
> With 3 reels/week, usage drops to ~1,395 min/month (safely within limit).

---

## Troubleshooting

**Q1: The workflow fails with `ModuleNotFoundError`**
The pip install step failed or cached incorrectly.
Re-run the workflow. If it persists: Actions → Caches → delete the `pip-` cache → re-run.

**Q2: Images are not being generated — all posts use Pexels photos**
Open `state/qa_log.json` on the state branch. If all records show `provider: "Pexels"`,
all 3 AI providers failed. Verify `HF_TOKEN` is set correctly.
Pollinations.ai occasionally goes offline — auto-fallback is working as designed.
Check https://status.huggingface.co for outages.

**Q3: Posts are not appearing on Facebook**
Check if `state/token_expired.json` exists on the state branch (see Q4).
Open `state/publish_log.json` and look for `"status": "failed"` entries.
Verify `FB_PAGE_ACCESS_TOKEN` and `FB_PAGE_ID` secrets are correct.
Ensure your page token has `pages_manage_posts` permission.

**Q4: GitHub Issue says "⚠️ Facebook Token Expired" — how to fix**
1. Go to https://developers.facebook.com/tools/explorer
2. Select your app and Page → add the four permissions → Generate Access Token
3. Exchange for a 60-day token (same URL as Step 3c)
4. Repo → Settings → Secrets → `FB_PAGE_ACCESS_TOKEN` → Update
5. On the state branch, delete `state/token_expired.json`
6. Manually run the Daily Pipeline to verify the new token
7. Close the GitHub Issue

**Q5: Reels are failing to upload**
Facebook Reels requirements: H.264, AAC audio, max 100 MB, 1080×1920 px, 3–90 seconds.
Check `state/logs/` for `reel_producer` errors.
Verify `ffmpeg` is installed — check the "Install system dependencies" step in workflow logs.
If reel files exceed 100 MB, reduce `reels_per_week` or shorten reel scripts in config.yml.

**Q6: Content feels generic and not region-specific**
Manually trigger **FACE Monthly Algorithm Watcher** from the Actions tab.
This populates Bangladesh-specific tips in `state/algorithm_rules.json`.
The next daily pipeline run will use this knowledge.

**Q7: The self-improver rolled back a prompt — should I be worried?**
No. Rollback means the new prompt variant performed worse than the original.
The system detected the regression and restored the better version automatically.
Check `state/improvement_log.json` for `rollback_reason: "engagement_degraded"`.

**Q8: GitHub Actions minutes are running low**
Set `reels_per_week: 1` (or `0`) in config.yml — reels account for most runtime.
Set `posts_per_day: 2` instead of 3.
Set `enable_visual_director: false` to use Pexels only (fastest, no AI generation).

**Q9: Gemini quota exhausted mid-pipeline**
The system logs `GeminiQuotaExhaustedError` and skips remaining AI calls gracefully.
Affected posts show `"failed"` in `state/publish_log.json`.
Quota resets at midnight Pacific. Next scheduled run starts fresh. No data corrupted.

**Q10: The `state/` directory is growing very large**
`state/post_history.json` auto-prunes to 30 days.
Video clips in `assets/videos/` auto-delete after each successful run.
If still growing: delete `state/logs/` files older than 7 days manually.
Delete `state/pexels_cache.json` to clear the Pexels search cache (rebuilds automatically).

---

## Customization

**Change your niche:** Update `page.niche` and `page.description` in `config.yml` → commit → next run adapts automatically.

**Change your region:** Update `audience.target_region` and `page.target_language` → TrendHunter researches the new region immediately.

**Change posting times:** Update `posting_schedule.daily_post_times` in `config.yml` AND update the cron strings in `.github/workflows/daily_pipeline.yml`. Convert your local time to UTC (Bangladesh is UTC+6, subtract 6 hours).

**Disable reels entirely:** Set `agents.enable_reel_producer: false` and `posting_schedule.reels_per_week: 0`. Saves ~8 minutes of Actions time per reel.

**Add a new content pillar:** Add it to `content.content_pillars` in `config.yml`. ContentStrategy incorporates it from the next run.

**Post in a different language:** Change `page.target_language` to `"Hindi"`, `"Arabic"`, `"French"`, etc. All captions, stories, and reel scripts generate in that language.

---

## License

MIT — free to use, modify, and deploy for personal or commercial use.

---

*Built with ❤️ for zero-cost autonomous content creation.*
