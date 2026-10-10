# FACE static code audit — findings and repair priorities

Target repository: `maatmyali-eng/face`

Model target: `gemini-3.5-flash-lite`

Audit type: static source review of the public repository. No user secrets were supplied, and no live Gemini/Meta/Pexels/Hugging Face API run was performed. Findings below describe code paths inspected; they should be confirmed by running the included smoke tests and an end-to-end no-publish test on the actual clone.

## Blockers

| Area | Finding | Impact | Repair in this pack? |
|---|---|---|---|
| Dependencies / Gemini | `requirements.txt` installs `google-genai`, while `utils/gemini_client.py` imported the legacy `google.generativeai` package and called `configure` / `GenerativeModel`. | Clean install can fail on import before generation. | Yes: full replacement `replacements/utils/gemini_client.py`. |
| Configuration | `utils/config_loader.py` required config sections absent from the repository's `config.yml`. | Startup/config validation fails. | Yes: full replacement `replacements/utils/config_loader.py`. |
| Configuration | Loader imports `pytz`, but it is not present in pinned requirements. Model lookup reads `api_settings` although actual YAML uses `api`. | Missing dependency or selected model silently falling back. | Yes: replacement loader uses `zoneinfo` and reads `api.gemini_model`. |
| Prompt rendering | Shared client uses `.format_map()` on prompts containing literal JSON braces. Visual QA separately calls `.format()` on a prompt containing JSON braces. | Formatting exceptions break generation or trigger fallback QA scores. | Yes: shared client is brace-safe; replace the QA prompt with the supplied file whose JSON braces are escaped for `.format()`. |
| Trend schema | Trend prompt returns nested score fields but the trend agent validates flat keys. Prompt raw-data placeholder and caller keyword differ. | Good responses can fail schema checks or lack scores. | Yes: supplied `trend_analysis.txt` aligns with current caller and consumer. |
| Strategy schema | Strategy prompt describes a full calendar, while code calls it per scheduled slot and expects a flat slot decision. | Agent output may be unusable or rely on defaults. | Yes: supplied `strategy_decision.txt` matches the slot-based flow. |
| Schedule | Config UTC slots (`04:00`, `11:00`, `17:00`) differ from workflow cron (`03:00`, `07:00`, `14:00`). | Publisher's narrow schedule window can miss posts. | Manual config edit in `PATCHES.md`. |

## High-risk publishing / policy issues

| Area | Finding | Impact | Repair guidance |
|---|---|---|---|
| Content safety | `agents/content_generator.py` treats safety-check exceptions as safe. | A failed or blocked moderation call can still approve content. | Manual fail-closed `except` replacement in `PATCHES.md`. |
| Facebook rate limiting | `utils/facebook_client.py` tests `response.status_code in (429, 4)` as if Meta error code 4 were an HTTP status. | Graph API rate-limit responses may not go through intended retry handling. | `PATCHES.md` provides a separate HTTP-status / JSON error-code check. Integrate with existing exception classes. |
| Duplicate publishing | Publisher retry logic retries broad exceptions around a remote POST. A timeout may occur after Facebook has accepted a post. | Retrying an ambiguous outcome can create duplicates. | Only retry known-safe transient responses; mark unknown outcomes for reconciliation. |
| Reels | Page `/videos` upload is not by itself proof of a correct Reels publishing workflow. | Reel publishing may fail or create an ordinary video. | Disable until Meta's current Reels upload/publish flow is implemented and tested. |
| Carousel | Strategy/publisher path sends carousel to the single-image publisher. | Content claiming to be a carousel may publish as only one image. | Replacement strategy disallows carousel until true multi-image publishing is implemented. |
| Reels schedule | Code reads `posting_schedule.reel_days`, but the actual config uses `reels_per_week`. | Configured weekly frequency is not honored reliably. | Manual note in `PATCHES.md`; keep Reels disabled until scheduling is corrected. |
| State persistence | Workflow builds a state update on the main branch and pushes it to `state`, risking divergent/non-fast-forward history. | Persistent memory/history may stop saving after runs. | Replacement state-commit step in `PATCHES.md` uses a detached worktree based on `origin/state`. |

## Operational and quality improvements

1. Keep `google-genai==2.29.0`; do not install the legacy Gemini package merely to mask the import bug.
2. Make the selected model visible in logs, and fail startup if `GEMINI_API_KEY` or model configuration is missing.
3. Keep Gemini temperature at `1.0` for this Gemini 3-family model; the replacement wrapper deliberately ignores older per-agent low-temperature values.
4. JSON mode improves format reliability but does not alone guarantee business-level correctness. Keep schema validation in each agent and reject missing/extra/wrongly typed fields before downstream use.
5. Validate facts before posting trending/news-like claims. AI-generated trend summaries can invent causes, engagement metrics, or sources; preserve original source URLs and do not publish factual claims without evidence.
6. Test image generation providers separately from Gemini. Gemini Flash-Lite can analyze images but should not be treated as an image generator.
7. Check actual API quota in AI Studio rather than assuming a fixed free RPM/RPD limit; it varies by model/project/tier.
8. Review the hard-coded Facebook Graph API version against Meta's current lifecycle before production deployment. Test the API endpoints and permissions; do not only swap the version string.

## Validation steps

Run the included checks from the repository root after applying replacements:

```bash
python -m compileall -q .
python -m unittest discover -s tests -v
```

Then add/run targeted tests for `trend_hunter`, `content_strategy`, visual QA, safety-gate failure, rate-limit handling, schedule selection, and state persistence. Finally run the workflow with publishing disabled and inspect every generated output. Only then use a separate test Page for one controlled publishing test.
