"""
utils/audio_utils.py
====================
Audio utilities for FACE: TTS via edge-tts, background music via yt-dlp,
subtitle timing generation, and audio duration measurement.

Usage:
    from utils.audio_utils import generate_tts, get_music_track, get_audio_duration
"""

import asyncio
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.logger import get_logger
from utils.config_loader import get_config

logger = get_logger("audio_utils")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
_AUDIO_DIR = _ROOT / "assets" / "audio"
_MANIFEST_FILE = _AUDIO_DIR / "manifest.json"
_TEMP_DIR = _ROOT / "state" / "temp"

# ---------------------------------------------------------------------------
# Voice mapping
# ---------------------------------------------------------------------------
VOICE_MAP: Dict[str, str] = {
    "english": "en-US-AriaNeural",
    "bangla": "bn-BD-NabanitaNeural",
    "english_bd": "en-IN-NeerjaNeural",
}

# Default voice for unknown languages
_DEFAULT_VOICE = "en-US-AriaNeural"

# ---------------------------------------------------------------------------
# Audio library: royalty-free YouTube Audio Library tracks per mood
# (4 real YouTube Audio Library URLs per mood — direct download via yt-dlp)
# ---------------------------------------------------------------------------
AUDIO_LIBRARY: Dict[str, List[str]] = {
    "energetic": [
        "https://www.youtube.com/watch?v=xGRjCa49C6U",  # Upbeat Party — YouTube Audio Library
        "https://www.youtube.com/watch?v=fHI8X4OXluQ",  # Rock It — YouTube Audio Library
        "https://www.youtube.com/watch?v=Q44Z9YRQfHo",  # Electro Cabello — YouTube Audio Library
        "https://www.youtube.com/watch?v=nkqoGLBzwsc",  # On My Way — YouTube Audio Library
    ],
    "calm": [
        "https://www.youtube.com/watch?v=aLqc5cCMVS8",  # Acoustic Breeze — YouTube Audio Library
        "https://www.youtube.com/watch?v=LMBF16tz7go",  # Morning Routine — YouTube Audio Library
        "https://www.youtube.com/watch?v=V1bNOwNN2dI",  # Happy Life — YouTube Audio Library
        "https://www.youtube.com/watch?v=mqPYB4mBCVo",  # Inspired — YouTube Audio Library
    ],
    "inspiring": [
        "https://www.youtube.com/watch?v=zvCBSSwgtg4",  # Almost Original — YouTube Audio Library
        "https://www.youtube.com/watch?v=qMUGRmCQBkM",  # Inspiring — YouTube Audio Library
        "https://www.youtube.com/watch?v=5z6GhZd2dP8",  # Gonna Start — YouTube Audio Library
        "https://www.youtube.com/watch?v=u6poTQc640c",  # Breaking Point — YouTube Audio Library
    ],
    "upbeat": [
        "https://www.youtube.com/watch?v=3aaZFEBTrkA",  # Ukulele — YouTube Audio Library
        "https://www.youtube.com/watch?v=WkF3OgkNkrs",  # Happy Clappy Ukulele — YouTube Audio Library
        "https://www.youtube.com/watch?v=hVV3xLHDuEw",  # Cool Vibes — YouTube Audio Library
        "https://www.youtube.com/watch?v=y_SzL5GVMlg",  # Sunflower — YouTube Audio Library
    ],
    "emotional": [
        "https://www.youtube.com/watch?v=mKmD3Oa5jFQ",  # Chill Abstract Intention — YouTube Audio Library
        "https://www.youtube.com/watch?v=4g79GFVsMg8",  # Emotional Piano — YouTube Audio Library
        "https://www.youtube.com/watch?v=3pHnEQxJmVY",  # Piano Moment — YouTube Audio Library
        "https://www.youtube.com/watch?v=tT5fEhMBWvI",  # Inner Light — YouTube Audio Library
    ],
    "mysterious": [
        "https://www.youtube.com/watch?v=jqWnCE1vNnE",  # Dark Mystery — YouTube Audio Library
        "https://www.youtube.com/watch?v=TlBct_ZZVTI",  # Sneaky Snitch — YouTube Audio Library
        "https://www.youtube.com/watch?v=UgHKb_7884o",  # Investigations — YouTube Audio Library
        "https://www.youtube.com/watch?v=8LCTd2I-8Co",  # Cipher — YouTube Audio Library
    ],
}


# ---------------------------------------------------------------------------
# TTS generation
# ---------------------------------------------------------------------------

def generate_tts(
    text: str,
    voice: Optional[str] = None,
    speed: str = "+0%",
    output_path: Optional[str] = None,
) -> str:
    """
    Generate a voiceover MP3 using edge-tts (Microsoft Neural TTS).
    Automatically selects voice from config target_language if not specified.

    Args:
        text: The text to convert to speech.
        voice: Voice name (e.g. 'en-US-AriaNeural'). If None, inferred from config.
        speed: Speed adjustment, e.g. '+5%', '-10%', '+0%'.
        output_path: Output file path. Defaults to state/temp/tts_{hash}.mp3.

    Returns:
        Path to the generated audio file.

    Raises:
        RuntimeError: if edge-tts fails after retries.
    """
    if voice is None:
        voice = _infer_voice_from_config()

    _TEMP_DIR.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        output_path = str(_TEMP_DIR / f"tts_{text_hash}.mp3")

    logger.info(
        "Generating TTS.",
        extra={"voice": voice, "speed": speed, "chars": len(text)},
    )

    # edge-tts is async — wrap with asyncio.run()
    for attempt in range(3):
        try:
            asyncio.run(_run_edge_tts(text, voice, speed, output_path))
            if Path(output_path).exists() and Path(output_path).stat().st_size > 0:
                logger.info("TTS saved → %s (%.1f KB)", output_path,
                            Path(output_path).stat().st_size / 1024)
                return output_path
            raise RuntimeError("TTS output file is empty.")
        except Exception as exc:
            wait = 2 ** attempt
            logger.warning("TTS attempt %d failed: %s — retrying in %ds.", attempt + 1, exc, wait)
            if attempt < 2:
                time.sleep(wait)

    raise RuntimeError(f"edge-tts failed after 3 attempts for voice={voice!r}.")


async def _run_edge_tts(text: str, voice: str, speed: str, output_path: str) -> None:
    """Async inner function: calls edge-tts communicate API."""
    import edge_tts  # Imported here to avoid startup overhead

    communicate = edge_tts.Communicate(text, voice, rate=speed)
    await communicate.save(output_path)


def _infer_voice_from_config() -> str:
    """
    Pick the appropriate TTS voice based on config target_language.

    Returns:
        Edge TTS voice name string.
    """
    try:
        cfg = get_config()
        lang = cfg.get("page_identity", {}).get("target_language", "English").lower()

        if "bangla" in lang or "bengali" in lang or "bn" in lang:
            if "english" in lang:
                # Bilingual: use BD-accented English as primary
                return VOICE_MAP["english_bd"]
            return VOICE_MAP["bangla"]
        return VOICE_MAP["english"]
    except Exception:
        return _DEFAULT_VOICE


# ---------------------------------------------------------------------------
# Audio duration
# ---------------------------------------------------------------------------

def get_audio_duration(audio_path: str) -> float:
    """
    Get the duration of an audio file in seconds using ffprobe.

    Args:
        audio_path: Path to the audio file.

    Returns:
        Duration in seconds (0.0 if unavailable).
    """
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return float(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError) as exc:
        logger.warning("ffprobe duration check failed for %s: %s", audio_path, exc)

    # Fallback: try mutagen if available
    try:
        import mutagen
        audio = mutagen.File(audio_path)
        if audio and audio.info:
            return audio.info.length
    except (ImportError, Exception):
        pass

    return 0.0


# ---------------------------------------------------------------------------
# Subtitle timing generation
# ---------------------------------------------------------------------------

def generate_subtitle_timings(
    script_segments: List[Dict[str, str]],
    audio_path: str,
) -> List[Dict[str, Any]]:
    """
    Generate per-segment subtitle timings by proportional distribution
    based on character count vs total audio duration.

    This is a heuristic approach (not forced-alignment). For a small
    number of segments it is accurate enough for social media reels.

    Args:
        script_segments: List of {'text': str} dicts (ordered as spoken).
        audio_path: Path to the generated TTS audio file.

    Returns:
        List of {'text': str, 'start_time': float, 'end_time': float} dicts.
    """
    total_duration = get_audio_duration(audio_path)
    if total_duration <= 0:
        logger.warning("Could not determine audio duration — using 3s per segment.")
        total_duration = len(script_segments) * 3.0

    # Weight time by character count (simple but effective for TTS)
    texts = [seg.get("text", "") for seg in script_segments]
    char_counts = [max(len(t), 1) for t in texts]
    total_chars = sum(char_counts)

    timings: List[Dict[str, Any]] = []
    cursor = 0.0

    for text, chars in zip(texts, char_counts):
        proportion = chars / total_chars
        seg_duration = total_duration * proportion
        timings.append({
            "text": text,
            "start_time": round(cursor, 2),
            "end_time": round(cursor + seg_duration, 2),
        })
        cursor += seg_duration

    return timings


# ---------------------------------------------------------------------------
# Background music cache
# ---------------------------------------------------------------------------

def get_music_track(mood: str, cache_dir: Optional[str] = None) -> str:
    """
    Return a path to a cached background music track for the given mood.
    Downloads via yt-dlp if not already cached.

    Audio tracks are royalty-free from YouTube Audio Library.
    The manifest (assets/audio/manifest.json) tracks what has been downloaded.

    Args:
        mood: 'energetic' | 'calm' | 'inspiring' | 'upbeat' | 'emotional' | 'mysterious'
        cache_dir: Override cache directory. Defaults to assets/audio/{mood}/.

    Returns:
        Local path to the .mp3 audio track.

    Raises:
        ValueError: if mood is unknown.
        RuntimeError: if all download attempts fail.
    """
    if mood not in AUDIO_LIBRARY:
        raise ValueError(
            f"Unknown mood {mood!r}. Valid options: {list(AUDIO_LIBRARY.keys())}"
        )

    if cache_dir is None:
        mood_dir = _AUDIO_DIR / mood
    else:
        mood_dir = Path(cache_dir)

    mood_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    manifest = _load_manifest()
    manifest_key = f"audio_{mood}"
    cached_tracks = manifest.get(manifest_key, [])

    # Return first valid cached track
    for track_info in cached_tracks:
        path = Path(track_info.get("path", ""))
        if path.exists():
            logger.debug("Returning cached track for mood '%s': %s", mood, path)
            return str(path)

    # Need to download — try each URL in order
    urls = AUDIO_LIBRARY[mood]
    for i, url in enumerate(urls):
        track_filename = f"{mood}_{i+1}.mp3"
        dest_path = mood_dir / track_filename

        if dest_path.exists():
            # Register in manifest in case it was lost
            _register_track(manifest, manifest_key, str(dest_path))
            return str(dest_path)

        logger.info("Downloading audio track: mood=%s, url=%s", mood, url)
        try:
            _download_with_ytdlp(url, dest_path)
            if dest_path.exists() and dest_path.stat().st_size > 0:
                _register_track(manifest, manifest_key, str(dest_path))
                logger.info("Music track cached → %s", dest_path)
                return str(dest_path)
        except Exception as exc:
            logger.warning("Track download failed (URL %d/%d): %s", i + 1, len(urls), exc)

    raise RuntimeError(
        f"All {len(urls)} download attempts failed for mood '{mood}'. "
        "Check yt-dlp installation and network access."
    )


def _download_with_ytdlp(url: str, dest_path: Path) -> None:
    """
    Download audio from a URL using yt-dlp and convert to MP3.

    Args:
        url: YouTube URL.
        dest_path: Destination .mp3 file path (without extension — yt-dlp adds it).
    """
    # yt-dlp output template uses stem (without .mp3 — yt-dlp adds format extension)
    output_template = str(dest_path.with_suffix("")) + ".%(ext)s"

    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", "192K",
        "--output", output_template,
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        url,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(
            f"yt-dlp exit code {result.returncode}: {result.stderr[:200]}"
        )


def _load_manifest() -> Dict[str, Any]:
    """Load the audio manifest JSON from disk."""
    if not _MANIFEST_FILE.exists():
        return {}
    try:
        with open(_MANIFEST_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_manifest(manifest: Dict[str, Any]) -> None:
    """Save the audio manifest to disk."""
    _AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    try:
        tmp = _MANIFEST_FILE.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        os.replace(tmp, _MANIFEST_FILE)
    except OSError as exc:
        logger.error("Failed to save audio manifest: %s", exc)


def _register_track(manifest: Dict[str, Any], key: str, path: str) -> None:
    """Register a downloaded track in the manifest and save."""
    if key not in manifest:
        manifest[key] = []
    existing_paths = [t.get("path") for t in manifest[key]]
    if path not in existing_paths:
        manifest[key].append({"path": path, "cached_at": time.strftime("%Y-%m-%d")})
    _save_manifest(manifest)
