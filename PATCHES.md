# Additional fixes to apply after copying replacement files

These are source-specific manual patches. Apply them on a branch, inspect the diff, and run tests before merging. Some published API behavior and Graph API versions can change; these patches do not replace a live integration test.

## 1. Fix trend-agent input name if you keep your original prompt

The replacement `trend_analysis.txt` uses `{trend_raw_data}`, matching the current call in `agents/trend_hunter.py`. Do not rename this placeholder to `{raw_trend_data}` unless you also change the keyword passed by the caller.

The replacement prompt returns a flat result with `virality_score`, `niche_fit_score`, `region_fit_score`, `recommended_format`, and `hook_angle`, which is what the current trend agent validates. This resolves the old nested-vs-flat JSON contract mismatch.

## 2. Make the safety gate fail closed

In `agents/content_generator.py`, find `_safety_check()` and its broad `except Exception as exc:` branch. The current exception path treats a failed safety check as safe. Replace the body of that `except` branch with:

```python
        except Exception as exc:
            logger.exception("Safety check failed; blocking this item")
            return {
                "safe": False,
                "reason": f"Safety check could not complete: {exc}",
                "confidence": 0.0,
            }
```

Keep indentation compatible with the existing method. If the method uses a different logger name, use its existing logger. If its return contract requires additional fields, preserve them but keep `safe=False` when the check cannot complete. A failed check must block publication rather than silently approve it.

## 3. Fix schedule mismatch (mandatory)

In `config.yml`, set the values in `schedule.daily_run_times_utc` to:

```yaml
schedule:
  timezone: "Asia/Dhaka"
  daily_run_times_utc:
    - "03:00"
    - "07:00"
    - "14:00"
```

This is only the relevant section; merge it into the existing `schedule:` mapping. Do not add a second top-level `schedule:` key. These values match the current workflow cron. If you change the cron, change this list at the same time.

## 4. Repair state persistence in `.github/workflows/daily_pipeline.yml`

The current workflow commits updated `state/` on the checked-out default branch and then pushes that commit to `state`. Those histories can diverge, causing non-fast-forward rejection after the first run. Replace the final “commit and push state” step with a worktree-based update, preserving the workflow's existing `permissions: contents: write`:

```yaml
      - name: Commit and push persistent state
        if: always()
        shell: bash
        run: |
          set -euo pipefail
          git config user.name "github-actions[bot]"
          git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

          git fetch origin state
          git worktree add --detach /tmp/face-state origin/state

          mkdir -p /tmp/face-state/state
          if [ -d state ]; then
            cp -a state/. /tmp/face-state/state/
          fi

          cd /tmp/face-state
          git add state/
          if git diff --cached --quiet; then
            echo "No persistent state changes to commit."
          else
            git commit -m "Update FACE persistent state [skip ci]"
            git push origin HEAD:state
          fi
```

Check the workflow's earlier restore step: it should fetch `origin/state` and restore the saved `state/` directory into the working tree before the agents run. If the current workflow uses a different state directory path, update both restore and save consistently. Ensure the state branch exists before the first run.

## 5. Correct the Facebook Graph API rate-limit test

In `utils/facebook_client.py`, the current handler checks HTTP status using `response.status_code in (429, 4)`. HTTP 4 is not a valid way to detect a Meta Graph API error code: code `4` is inside the JSON response body, often with an HTTP 400 status. In the response handler, extract the API error object and use its code separately:

```python
payload = response.json()
error = payload.get("error") if isinstance(payload, dict) else None
error = error if isinstance(error, dict) else {}
api_code = error.get("code")
is_rate_limited = response.status_code == 429 or api_code in {4, 17, 32, 613}
```

Use `is_rate_limited` in the handler's existing rate-limit exception/retry branch. Do not retry all failed POSTs blindly: a network timeout may occur after Facebook accepted a post, and retrying could publish a duplicate. Only retry known-safe transient errors; mark ambiguous publish outcomes for manual reconciliation.

Also review the hard-coded Graph API version in this file against Meta's current version lifecycle before deployment. Do not just change the version string without testing every used endpoint and permission.

## 6. Do not claim Reels or carousel support until verified

The current publishing client uses the Page `/videos` endpoint for video posting; this is not sufficient evidence that the Reels-specific publishing flow is implemented. The carousel path also routes to the single-image publisher. Until a tested Meta Reels upload/publish flow and a true multi-image carousel flow exist:

- Turn off Reels in `config.yml` (use the existing `agents.enable_reels: false` field).
- Do not let strategy output `carousel` as a supported format; the replacement strategy prompt avoids it.
- Use ordinary image/text posts for the initial test. Implement and test Reels/carousels as separate features with current Meta docs and Page permissions.

## 7. Honor the configured Reels toggle and frequency

The strategy code currently looks for `posting_schedule.reel_days`, while the repository config defines `posting_schedule.reels_per_week`. This means the setting is not honored consistently. Until the scheduling function has been rewritten to distribute exactly `reels_per_week` over a rolling week, keep `agents.enable_reels` set to `false`. Do not assume `reels_per_week` alone activates reliable Reel scheduling.

## 8. De-duplicate dependency/secret handling

- Keep `google-genai==2.29.0`; do not add `google-generativeai` just to satisfy the legacy import. The replacement Gemini client uses the new SDK.
- Confirm Pillow is installed (needed by `generate_with_image`). If it is missing in your `requirements.txt`, add `Pillow`.
- Do not write API keys or Facebook tokens into `config.yml`, source files, prompt files, logs, or commits. Continue using GitHub Actions secrets.

## 9. Validation order

1. Run `python -m compileall -q .`.
2. Run the unit tests included in this pack.
3. Run the content-generation workflow with the publisher disabled.
4. Inspect prompt outputs, schema validation, generated images, and logs.
5. Enable publishing only after a controlled test on a separate test Page.

This pack is a static-audit repair set, not a claim that all Meta endpoints or external API quotas have been live-tested.
