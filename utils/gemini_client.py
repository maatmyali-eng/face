"""Shared Gemini client migrated to Google's current `google-genai` SDK.

Drop-in replacement for the public methods used by FACE agents:
  - generate_text
  - generate_json
  - generate_json_raw
  - generate_with_image
  - load_prompt

The configured model remains `gemini-3.5-flash-lite`. Gemini provides text and
image understanding; image/video creation stays in FACE's separate providers.
"""
from __future__ import annotations

import json
import os
import re
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from utils.config_loader import get_config, get_gemini_model
from utils.logger import get_logger
from utils.state_manager import read_state, write_state

logger = get_logger("gemini_client")
_ROOT = Path(__file__).resolve().parent.parent
_PROMPTS_DIR = _ROOT / "prompts"
_QUOTA_FILE = "gemini_quota.json"
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


class GeminiQuotaExhaustedError(RuntimeError):
    """The Gemini API/project quota is exhausted or its configured daily cap hit."""


class GeminiRateLimitError(RuntimeError):
    """A transient API rate-limit response after retries are considered."""


class GeminiInvalidResponseError(RuntimeError):
    """Gemini returned empty, blocked, or invalid output for a requested task."""


class GeminiClient:
    """Reusable wrapper with brace-safe prompt rendering and bounded retries."""

    def __init__(self) -> None:
        cfg = get_config()
        self.model_name = get_gemini_model()
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY environment variable is not set")

        self.client = genai.Client(api_key=api_key)
        api_cfg = cfg.get("api", {})
        self.max_retries = max(1, int(api_cfg.get("max_retries", 3)))
        self.retry_delay_seconds = max(1.0, float(api_cfg.get("retry_delay_seconds", 2)))
        # Server-side limits vary by model, project, and usage tier. Only apply a
        # local daily cap if the owner explicitly configures one.
        self.daily_limit = int(api_cfg.get("gemini_daily_request_limit", 0) or 0)
        self.rpm_limit = max(1, int(api_cfg.get("gemini_rpm_limit", 12)))
        self._request_times: deque[float] = deque()

        quota = read_state(_QUOTA_FILE) or {}
        if not isinstance(quota, dict):
            quota = {}
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._daily_count = int(quota.get("count", 0)) if quota.get("date") == today else 0
        logger.info("Gemini client ready (model=%s, successful_calls_today=%d)", self.model_name, self._daily_count)

    def load_prompt(self, filename: str, **kwargs: Any) -> str:
        """Load a text prompt and replace only named `{identifier}` placeholders.

        This intentionally does NOT use str.format/format_map: JSON examples in
        the prompt files contain literal braces that Python would otherwise parse
        as formatting syntax.
        """
        prompt_path = (_PROMPTS_DIR / filename).resolve()
        if _PROMPTS_DIR.resolve() not in prompt_path.parents:
            raise ValueError(f"Prompt path must be inside {_PROMPTS_DIR}")
        if not prompt_path.is_file():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        template = prompt_path.read_text(encoding="utf-8")

        def replace(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in kwargs:
                # Keep unknown placeholders intact instead of corrupting literal
                # JSON examples or explanatory tokens such as `{X}`.
                return match.group(0)
            return self._stringify(kwargs[key])

        return _PLACEHOLDER_RE.sub(replace, template)

    @staticmethod
    def _stringify(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=False, default=str)
        if value is None:
            return ""
        return str(value)

    def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 2048,
    ) -> str:
        return self._generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            json_output=False,
        )

    def generate_json(
        self,
        prompt_file: str,
        system_prompt: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        **context: Any,
    ) -> dict[str, Any]:
        prompt = self.load_prompt(prompt_file, **context)
        raw = self._generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            json_output=True,
        )
        return self._parse_json(raw, prompt_file)

    def generate_json_raw(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        raw = self._generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            json_output=True,
        )
        return self._parse_json(raw, "<inline>")

    def generate_with_image(
        self,
        prompt: str,
        image_path: str | Path,
        temperature: float = 1.0,
    ) -> dict[str, Any]:
        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required for Gemini image understanding") from exc

        # Keep the image open until the request has completed. Gemini 3.5
        # Flash-Lite supports image input, but does not generate images.
        with Image.open(image_path) as image:
            image.load()
            raw = self._generate(
                prompt=prompt,
                system_prompt=None,
                temperature=temperature,
                max_tokens=1024,
                json_output=True,
                extra_contents=[image.copy()],
            )
        return self._parse_json(raw, "<vision>")

    def _generate(
        self,
        prompt: str,
        system_prompt: str | None,
        temperature: float,
        max_tokens: int,
        json_output: bool,
        extra_contents: list[Any] | None = None,
    ) -> str:
        if not prompt or not prompt.strip():
            raise ValueError("Gemini prompt must not be empty")
        self._pre_request_checks()

        config_args: dict[str, Any] = {
            "max_output_tokens": max(1, int(max_tokens)),
            # Gemini 3 models recommend temperature 1.0. Existing callers may
            # pass legacy low-temperature values; deliberately ignore them.
            "temperature": 1.0,
        }
        if system_prompt:
            config_args["system_instruction"] = system_prompt
        if json_output:
            config_args["response_mime_type"] = "application/json"
        generation_config = types.GenerateContentConfig(**config_args)
        contents: Any = [prompt, *(extra_contents or [])]
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=generation_config,
                )
                self._record_success()
                text = getattr(response, "text", None)
                if not isinstance(text, str) or not text.strip():
                    finish = "unknown"
                    try:
                        finish = str(response.candidates[0].finish_reason)
                    except Exception:
                        pass
                    raise GeminiInvalidResponseError(
                        f"Gemini returned no text (finish_reason={finish}); the response may have been blocked or truncated"
                    )
                return text.strip()
            except GeminiInvalidResponseError:
                raise
            except Exception as exc:  # SDK error types differ slightly by release.
                last_error = exc
                message = str(exc).lower()
                status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
                exhausted = (
                    "resource_exhausted" in message
                    or "quota exceeded" in message
                    or "requests per day" in message
                    or "daily quota" in message
                )
                if exhausted:
                    raise GeminiQuotaExhaustedError(str(exc)) from exc

                rate_limited = status == 429 or "429" in message or "rate limit" in message
                transient_server_error = status in (500, 502, 503, 504)
                if not (rate_limited or transient_server_error) or attempt >= self.max_retries - 1:
                    if rate_limited:
                        raise GeminiRateLimitError(str(exc)) from exc
                    raise

                delay = min(60.0, self.retry_delay_seconds * (2 ** attempt))
                logger.warning("Transient Gemini error (attempt %d/%d); retrying in %.1fs: %s", attempt + 1, self.max_retries, delay, exc)
                time.sleep(delay)

        if last_error is not None:
            raise last_error
        raise RuntimeError("Gemini request failed without an error object")

    def _pre_request_checks(self) -> None:
        if self.daily_limit and self._daily_count >= self.daily_limit:
            raise GeminiQuotaExhaustedError(
                f"Configured local Gemini request cap reached ({self.daily_limit}); increase api.gemini_daily_request_limit only after verifying your actual quota"
            )
        now = time.monotonic()
        while self._request_times and now - self._request_times[0] >= 60:
            self._request_times.popleft()
        if len(self._request_times) >= self.rpm_limit:
            delay = max(0.1, 60 - (now - self._request_times[0]) + 0.1)
            logger.info("Local Gemini RPM guard sleeping %.1fs", delay)
            time.sleep(delay)
            now = time.monotonic()
            while self._request_times and now - self._request_times[0] >= 60:
                self._request_times.popleft()
        self._request_times.append(time.monotonic())

    def _record_success(self) -> None:
        self._daily_count += 1
        try:
            write_state(_QUOTA_FILE, {
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "count": self._daily_count,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as exc:
            logger.warning("Could not persist Gemini usage counter: %s", exc)

    @staticmethod
    def _parse_json(raw_text: str, source: str) -> dict[str, Any]:
        if not raw_text or not raw_text.strip():
            raise GeminiInvalidResponseError(f"Empty JSON response from Gemini for {source}")
        text = raw_text.strip()
        # Be defensive if a model version returns a code fence despite JSON mode.
        fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
        if fence:
            text = fence.group(1).strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            # Last-resort extraction: start at the earliest JSON object/array.
            starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
            if not starts:
                raise GeminiInvalidResponseError(f"Gemini response for {source} contains no JSON")
            start = min(starts)
            try:
                value, _ = json.JSONDecoder().raw_decode(text[start:])
            except json.JSONDecodeError as exc:
                raise GeminiInvalidResponseError(f"Invalid JSON from Gemini for {source}: {exc}") from exc
        if not isinstance(value, dict):
            raise GeminiInvalidResponseError(
                f"Expected a JSON object from {source}; got {type(value).__name__}"
            )
        return value


_CLIENT: GeminiClient | None = None


def get_gemini_client() -> GeminiClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = GeminiClient()
    return _CLIENT
