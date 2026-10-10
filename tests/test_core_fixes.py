"""Run after applying replacements: python -m unittest discover -s tests."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class PromptReplacementTests(unittest.TestCase):
    def test_brace_safe_prompt_rendering(self):
        # Test the replacement strategy without making a live Gemini request.
        from utils.gemini_client import GeminiClient
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompts = root / "prompts"
            prompts.mkdir()
            (prompts / "example.txt").write_text(
                'Topic: {topic}\nJSON: {"key": "literal", "n": 1}', encoding="utf-8"
            )
            with patch("utils.gemini_client._PROMPTS_DIR", prompts):
                obj = object.__new__(GeminiClient)
                rendered = obj.load_prompt("example.txt", topic="test")
        self.assertIn("Topic: test", rendered)
        self.assertIn('{"key": "literal", "n": 1}', rendered)

    def test_config_matches_current_schema_and_model(self):
        from utils.config_loader import get_config, get_gemini_model, get_posting_times_utc
        cfg = get_config()
        self.assertEqual(get_gemini_model(), cfg["api"]["gemini_model"])
        times = get_posting_times_utc()
        self.assertTrue(times)
        self.assertTrue(all(len(t) == 5 and t[2] == ":" for t in times))


if __name__ == "__main__":
    unittest.main()
