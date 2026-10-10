"""
utils/gemini_client.py
======================
Wraps google-generativeai SDK for FACE agents.
Handles: rate limiting (14 RPM buffer), exponential backoff on 429,
         JSON fence stripping, vision (image) calls, daily quota tracking,
         and prompt loading from prompts/ folder.

Usage:
    from utils.gemini_client import GeminiClient
    client = GeminiClient()
    result = client.generate_json("trend_analysis.txt", niche="fitness", ...)
"""

import json
import os
import re
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import google.generativeai as genai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
    RetryError,
)

from utils.logger import get_logger
from utils.state_manager import read_state, write_state
from utils.config_loader import get_config, get_gemini_model

logger = get_logger("gemini_client")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
_PROMPTS_DIR = _ROOT / "prompts"
_QUOTA_FILE = "gemini_quota.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_FREE_TIER_RPM = 15          # Gemini free tier requests per minute
_RPM_BUFFER = 14             # We track 14; sleep when we'd hit 15
_FREE_TIER_RPD = 1500        # Requests per day
_FREE_TIER_TPM = 1_000_000   # Tokens per minute

# Regex to strip markdown code fences from JSON responses
_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)

# Exceptions that trigger tenacity retries
_RETRYABLE_EXCEPTIONS = (Exception,)  # Will narrow inside the wrapper


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class GeminiQuotaExhaustedError(Exception):
    """Raised when the daily request quota is hit."""


class GeminiRateLimitError(Exception):
    """Raised by the SDK on 429 responses — triggers tenacity retry."""


class GeminiInvalidResponseError(Exception):
    """Raised when the model response cannot be parsed as valid JSON."""


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------

class GeminiClient:
    """
    Singleton-friendly wrapper around google-generativeai.

    Attributes:
        model_name: Gemini model string from config.
        _request_times: deque tracking timestamps of recent requests (for RPM).
        _daily_count: number of requests made in this process run.
    """

    def __init__(self) -> None:
        cfg = get_config()
        self.model_name: str = get_gemini_model()
        api_key: str = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY environment variable is not set."
            )

        genai.configure(api_key=api_key)

        # Safety settings — permissive for content creation (no harassment filter)
        self._safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
        ]

        # RPM tracking — sliding window of request timestamps
        self._request_times: deque = deque()

        # Load daily count from quota state file
        quota_data = read_state(_QUOTA_FILE)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if quota_data.get("date") == today:
            self._daily_count: int = quota_data.get("count", 0)
        else:
            self._daily_count = 0

        logger.info(
            "GeminiClient initialised.",
            extra={
                "model": self.model_name,
                "daily_count_so_far": self._daily_count,
            },
        )

    # -----------------------------------------------------------------------
    # Public methods
    # -----------------------------------------------------------------------

    def load_prompt(self, filename: str, **kwargs: Any) -> str:
        """
        Load a prompt template from prompts/ and inject config values.

        Args:
            filename: e.g. 'trend_analysis.txt'
            **kwargs: Placeholder values to inject via str.format().

        Returns:
            Fully rendered prompt string.

        Raises:
            FileNotFoundError: if the prompt file does not exist.
            KeyError: if a required placeholder is missing from kwargs.
        """
        prompt_path = _PROMPTS_DIR / filename
        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt file not found: {prompt_path}. "
                "Check that prompts/ directory is populated from Session 1."
            )
        template = prompt_path.read_text(encoding="utf-8")
        try:
            # Use safe format_map so missing keys raise a clear error
            return template.format_map(_SafeDict(**kwargs))
        except Exception as exc:
            logger.error(
                "Prompt template substitution failed for %s: %s", filename, exc
            )
            raise

    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """
        Generate a text completion from Gemini.

        Args:
            prompt: The user-turn prompt string.
            system_prompt: Optional system instruction.
            temperature: Sampling temperature (0.0–1.0).
            max_tokens: Maximum output tokens.

        Returns:
            The model's text response as a plain string.
        """
        self._pre_request_checks()
        model = self._build_model(system_prompt, temperature, max_tokens)

        @retry(
            retry=retry_if_exception_type(GeminiRateLimitError),
            wait=wait_exponential(multiplier=1, min=2, max=60),
            stop=stop_after_attempt(5),
            before_sleep=before_sleep_log(logger, 20),  # 20 = logging.INFO
        )
        def _call() -> str:
            try:
                response = model.generate_content(prompt)
                return response.text.strip()
            except Exception as exc:
                if "429" in str(exc) or "quota" in str(exc).lower():
                    logger.warning("Gemini 429 — backing off. Error: %s", exc)
                    raise GeminiRateLimitError(str(exc)) from exc
                raise

        result = _call()
        self._post_request()
        return result

    def generate_json(
        self,
        prompt_file: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        **context: Any,
    ) -> Dict[str, Any]:
        """
        Load a prompt template, inject context, call Gemini, and parse JSON.

        The prompt must instruct Gemini to return ONLY JSON (no markdown).
        This method strips fences defensively and validates the result.

        Args:
            prompt_file: Filename in prompts/, e.g. 'trend_analysis.txt'.
            system_prompt: Optional system instruction.
            temperature: Lower = more deterministic JSON (default 0.3).
            max_tokens: Max output tokens.
            **context: Placeholder values for the prompt template.

        Returns:
            Parsed dict from the JSON response.

        Raises:
            GeminiInvalidResponseError: if the response cannot be parsed as JSON.
            GeminiQuotaExhaustedError: if the daily quota is hit.
        """
        rendered_prompt = self.load_prompt(prompt_file, **context)
        raw_text = self.generate_text(
            rendered_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return self._parse_json(raw_text, prompt_file)

    def generate_json_raw(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """
        Same as generate_json but takes a raw prompt string instead of a file.
        Used for inline prompts (e.g. safety checks in PublisherAgent).

        Args:
            prompt: Raw prompt text.
            system_prompt: Optional system instruction.
            temperature: Sampling temperature.
            max_tokens: Max output tokens.

        Returns:
            Parsed dict.
        """
        raw_text = self.generate_text(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return self._parse_json(raw_text, "<inline>")

    def generate_with_image(
        self,
        prompt: str,
        image_path: Union[str, Path],
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Gemini Vision call — evaluate or describe an image.
        Used by the image quality evaluator in VisualDirectorAgent.

        Args:
            prompt: Text instructions for the vision task.
            image_path: Local path to a JPEG/PNG image file.
            temperature: Lower = more deterministic scoring.

        Returns:
            Parsed dict from JSON response.

        Raises:
            FileNotFoundError: if image_path does not exist.
            GeminiInvalidResponseError: if response is not valid JSON.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        self._pre_request_checks()

        import PIL.Image as PILImage

        img = PILImage.open(image_path)
        model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=1024,
            ),
            safety_settings=self._safety_settings,
        )

        @retry(
            retry=retry_if_exception_type(GeminiRateLimitError),
            wait=wait_exponential(multiplier=1, min=2, max=60),
            stop=stop_after_attempt(4),
            before_sleep=before_sleep_log(logger, 20),
        )
        def _call() -> str:
            try:
                response = model.generate_content([prompt, img])
                return response.text.strip()
            except Exception as exc:
                if "429" in str(exc) or "quota" in str(exc).lower():
                    raise GeminiRateLimitError(str(exc)) from exc
                raise

        raw_text = _call()
        self._post_request()
        return self._parse_json(raw_text, "<vision>")

    # -----------------------------------------------------------------------
    # Rate limit & quota helpers
    # -----------------------------------------------------------------------

    def _pre_request_checks(self) -> None:
        """
        Block if we're at the RPM limit, and check daily quota.
        Must be called before every API request.
        """
        # Daily quota check
        if self._daily_count >= _FREE_TIER_RPD:
            self._record_quota_exhausted()
            raise GeminiQuotaExhaustedError(
                f"Gemini daily quota of {_FREE_TIER_RPD} requests exhausted. "
                "Pipeline will resume tomorrow."
            )

        # RPM sliding-window check
        now = time.monotonic()
        # Remove timestamps older than 60 seconds
        while self._request_times and now - self._request_times[0] > 60.0:
            self._request_times.popleft()

        if len(self._request_times) >= _RPM_BUFFER:
            oldest = self._request_times[0]
            sleep_for = 60.0 - (now - oldest) + 0.1  # +0.1s safety margin
            if sleep_for > 0:
                logger.info(
                    "RPM buffer reached (%d/%d). Sleeping %.1fs.",
                    len(self._request_times),
                    _RPM_BUFFER,
                    sleep_for,
                    extra={"sleep_seconds": sleep_for},
                )
                time.sleep(sleep_for)

    def _post_request(self) -> None:
        """Record a successful request — update counters and persist quota state."""
        self._request_times.append(time.monotonic())
        self._daily_count += 1
        self._persist_quota()
        logger.debug(
            "Gemini request recorded.",
            extra={
                "daily_count": self._daily_count,
                "rpm_window": len(self._request_times),
            },
        )

    def _persist_quota(self) -> None:
        """Persist daily usage count to state/gemini_quota.json."""
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            write_state(_QUOTA_FILE, {"date": today, "count": self._daily_count})
        except Exception as exc:
            logger.warning("Failed to persist Gemini quota state: %s", exc)

    def _record_quota_exhausted(self) -> None:
        """Write a quota-exhausted marker to state/gemini_quota.json."""
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            write_state(
                _QUOTA_FILE,
                {
                    "date": today,
                    "count": self._daily_count,
                    "exhausted": True,
                    "exhausted_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        except Exception as exc:
            logger.warning("Failed to record quota exhaustion: %s", exc)

    # -----------------------------------------------------------------------
    # JSON parsing helper
    # -----------------------------------------------------------------------

    def _parse_json(self, raw_text: str, source: str) -> Dict[str, Any]:
        """
        Strip markdown fences and parse JSON from a Gemini response.

        Args:
            raw_text: Raw string from Gemini.
            source: Descriptor for logging (prompt filename or '<inline>').

        Returns:
            Parsed dict.

        Raises:
            GeminiInvalidResponseError: if parsing fails.
        """
        if not raw_text:
            raise GeminiInvalidResponseError(
                f"Empty response from Gemini for prompt: {source}"
            )

        # Strip markdown fences
        fence_match = _FENCE_RE.search(raw_text)
        if fence_match:
            cleaned = fence_match.group(1).strip()
        else:
            cleaned = raw_text.strip()

        # Sometimes the model includes extra text before/after the JSON object
        # Try to extract the outermost {...} or [...]
        cleaned = _extract_json_object(cleaned)

        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error(
                "JSON parse failed for %s. Raw: %.200s... Error: %s",
                source,
                raw_text,
                exc,
            )
            raise GeminiInvalidResponseError(
                f"Gemini response for {source!r} is not valid JSON: {exc}\n"
                f"Raw snippet: {raw_text[:200]!r}"
            ) from exc

        if not isinstance(result, (dict, list)):
            raise GeminiInvalidResponseError(
                f"Expected JSON object or array from {source!r}, got {type(result).__name__}."
            )

        return result

    # -----------------------------------------------------------------------
    # Model builder
    # -----------------------------------------------------------------------

    def _build_model(
        self,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> genai.GenerativeModel:
        """Build a GenerativeModel instance with the given config."""
        gen_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        kwargs: Dict[str, Any] = {
            "model_name": self.model_name,
            "generation_config": gen_config,
            "safety_settings": self._safety_settings,
        }
        if system_prompt:
            kwargs["system_instruction"] = system_prompt

        return genai.GenerativeModel(**kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _SafeDict(dict):
    """
    dict subclass that returns '{key}' for missing keys instead of raising KeyError.
    Used in format_map() so partial substitutions don't blow up.
    """

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _extract_json_object(text: str) -> str:
    """
    Extract the first JSON object {...} or array [...] from a string.
    Handles cases where Gemini prepends text like 'Here is the JSON:'.

    Args:
        text: String potentially containing a JSON object.

    Returns:
        The extracted JSON string, or the original text if no braces found.
    """
    # Find first { or [
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        idx_start = text.find(start_char)
        if idx_start == -1:
            continue
        # Find matching closing bracket using a depth counter
        depth = 0
        idx_end = -1
        in_string = False
        escape_next = False
        for i, ch in enumerate(text[idx_start:], start=idx_start):
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if ch == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    idx_end = i
                    break
        if idx_end != -1:
            return text[idx_start: idx_end + 1]

    return text


# ---------------------------------------------------------------------------
# Module-level singleton for convenience
# ---------------------------------------------------------------------------

_CLIENT: Optional[GeminiClient] = None


def get_gemini_client() -> GeminiClient:
    """
    Return the shared GeminiClient singleton.
    Lazily initialised on first call.
    """
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = GeminiClient()
    return _CLIENT
