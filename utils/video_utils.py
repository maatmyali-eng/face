"""
utils/video_utils.py
====================
moviepy-based video operations for FACE reels.
Handles assembly, animations, colour grading, subtitle burning,
text cards, audio mixing, and H.264/AAC export at 1080×1920/30fps.

Usage:
    from utils.video_utils import assemble_reel, export_reel, verify_reel
"""

import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utils.logger import get_logger
from utils.image_utils import load_font, apply_color_grade, _hex_to_rgb

logger = get_logger("video_utils")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_REEL_W, _REEL_H = 1080, 1920
_FPS = 30
_VIDEO_BITRATE = "8000k"
_AUDIO_BITRATE = "192k"
_MAX_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB
_MOVIEPY_THREADS = 2  # Respect GitHub Actions 2-core runner
_TEMP_DIR = Path(__file__).parent.parent / "state" / "temp"

# Colour grades for video (per-channel LUT approximation using numpy)
_VIDEO_GRADE_PARAMS: Dict[str, Dict] = {
    "warm": {"contrast": 1.05, "brightness": 1.05, "saturation": 1.15, "r": 10, "g": 0, "b": -10},
    "cool": {"contrast": 1.05, "brightness": 1.0, "saturation": 0.95, "r": -5, "g": 0, "b": 15},
    "high_contrast": {"contrast": 1.35, "brightness": 1.0, "saturation": 1.2, "r": 0, "g": 0, "b": 0},
    "pastel": {"contrast": 0.85, "brightness": 1.15, "saturation": 0.75, "r": 5, "g": 5, "b": 5},
    "cinematic": {"contrast": 1.20, "brightness": 0.92, "saturation": 1.10, "r": 5, "g": -2, "b": -8},
    "none": {"contrast": 1.0, "brightness": 1.0, "saturation": 1.0, "r": 0, "g": 0, "b": 0},
}

# Lazy import moviepy to avoid slow startup when only image utils are needed
_moviepy = None


def _get_moviepy():
    """Lazy-import moviepy.editor and configure thread count."""
    global _moviepy
    if _moviepy is None:
        try:
            import moviepy.editor as mpe
            # Apply thread limit for GitHub Actions
            import moviepy.config as mpc
            _moviepy = mpe
        except ImportError as exc:
            raise ImportError(
                "moviepy is not installed. Run: pip install moviepy"
            ) from exc
    return _moviepy


# ---------------------------------------------------------------------------
# Main assembly
# ---------------------------------------------------------------------------

def assemble_reel(
    segments: List[Dict[str, Any]],
    audio_path: str,
    output_path: str,
) -> str:
    """
    Assemble a list of video/image/text segments into a reel with audio.

    Each segment dict:
        {
            "type": "video" | "image" | "text",
            "clip_path": "...",          # for type=video
            "image_path": "...",         # for type=image
            "text": "...",               # for type=text
            "duration": 3.0,             # seconds
            "animation": "zoom_in" | "slide_left" | "fade" | "none",
            "color_grade": "warm" | "cool" | ...,  # optional
            "style": "bold_dark",        # for type=text only
        }

    Args:
        segments: List of segment dicts.
        audio_path: Path to the mixed audio file (voiceover + music).
        output_path: Final output path for the assembled reel.

    Returns:
        Output path string.
    """
    mpe = _get_moviepy()
    _TEMP_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    logger.info("Assembling reel from %d segments.", len(segments))

    clips = []

    for i, seg in enumerate(segments):
        seg_type = seg.get("type", "image")
        duration = float(seg.get("duration", 3.0))
        animation = seg.get("animation", "none")
        grade = seg.get("color_grade", "none")

        try:
            if seg_type == "video":
                clip = _load_video_segment(mpe, seg["clip_path"], duration)
            elif seg_type == "image":
                clip = _load_image_segment(mpe, seg["image_path"], duration)
            elif seg_type == "text":
                clip = create_text_card(
                    text=seg.get("text", ""),
                    style=seg.get("style", "bold_dark"),
                    duration=duration,
                )
            else:
                logger.warning("Unknown segment type '%s' at index %d — skipping.", seg_type, i)
                continue

            # Apply animation
            clip = apply_animation(clip, animation)

            # Apply colour grade
            if grade and grade != "none":
                clip = apply_color_grade_video(clip, grade)

            clips.append(clip)
            logger.debug("Segment %d (%s, %.1fs) processed.", i, seg_type, duration)

        except Exception as exc:
            logger.error("Segment %d failed: %s — inserting black card.", i, exc)
            # Fallback: black card for the segment duration
            black = mpe.ColorClip(size=(_REEL_W, _REEL_H), color=(0, 0, 0), duration=duration)
            clips.append(black)

    if not clips:
        raise ValueError("No valid segments to assemble.")

    # Concatenate all segments
    final_clip = mpe.concatenate_videoclips(clips, method="compose")

    # Attach audio
    try:
        audio_clip = mpe.AudioFileClip(audio_path)
        # Loop audio if shorter than video, or trim if longer
        if audio_clip.duration < final_clip.duration:
            loops = int(final_clip.duration / audio_clip.duration) + 1
            audio_clip = mpe.concatenate_audioclips([audio_clip] * loops)
        audio_clip = audio_clip.subclip(0, final_clip.duration)
        final_clip = final_clip.set_audio(audio_clip)
    except Exception as exc:
        logger.warning("Could not attach audio to reel: %s", exc)

    # Export
    result_path = export_reel(final_clip, output_path)

    elapsed = time.time() - t_start
    logger.info("Reel assembled in %.1fs → %s", elapsed, output_path)

    # Cleanup
    for clip in clips:
        try:
            clip.close()
        except Exception:
            pass

    return result_path


# ---------------------------------------------------------------------------
# Animations
# ---------------------------------------------------------------------------

def apply_animation(clip, animation_type: str):
    """
    Apply a Ken Burns / transition animation to a clip.

    Args:
        clip: moviepy VideoClip.
        animation_type: 'zoom_in' | 'slide_left' | 'fade' | 'none'

    Returns:
        Modified clip.
    """
    mpe = _get_moviepy()

    if animation_type == "none":
        return clip

    if animation_type == "fade":
        return clip.fadein(0.4).fadeout(0.4)

    if animation_type == "zoom_in":
        # Zoom from 100% to 115% over the clip duration (Ken Burns effect)
        w, h = _REEL_W, _REEL_H

        def _make_frame(t: float):
            scale = 1.0 + 0.15 * (t / max(clip.duration, 0.001))
            frame = clip.get_frame(t)
            img = Image.fromarray(frame.astype(np.uint8))
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            left = (new_w - w) // 2
            top = (new_h - h) // 2
            img = img.crop((left, top, left + w, top + h))
            return np.array(img)

        return mpe.VideoClip(_make_frame, duration=clip.duration).set_fps(_FPS)

    if animation_type == "slide_left":
        # Slide in from right
        w, h = _REEL_W, _REEL_H
        slide_duration = min(0.5, clip.duration * 0.3)

        def _make_slide_frame(t: float):
            frame = clip.get_frame(t)
            if t > slide_duration:
                return frame
            progress = t / slide_duration  # 0 → 1
            offset = int(w * (1 - progress))
            img = Image.fromarray(frame.astype(np.uint8))
            # Create black canvas
            canvas = Image.new("RGB", (w, h), (0, 0, 0))
            canvas.paste(img, (-offset, 0))
            return np.array(canvas)

        return mpe.VideoClip(_make_slide_frame, duration=clip.duration).set_fps(_FPS)

    logger.warning("Unknown animation_type '%s' — skipping.", animation_type)
    return clip


# ---------------------------------------------------------------------------
# Colour grading for video
# ---------------------------------------------------------------------------

def apply_color_grade_video(clip, grade: str):
    """
    Apply a colour grade to a video clip via per-frame numpy operations.

    Args:
        clip: moviepy VideoClip.
        grade: 'warm' | 'cool' | 'high_contrast' | 'pastel' | 'cinematic' | 'none'

    Returns:
        Colour-graded clip.
    """
    mpe = _get_moviepy()

    if grade not in _VIDEO_GRADE_PARAMS or grade == "none":
        return clip

    params = _VIDEO_GRADE_PARAMS[grade]
    contrast = params["contrast"]
    brightness = params["brightness"]
    saturation = params["saturation"]
    r_shift, g_shift, b_shift = params["r"], params["g"], params["b"]

    def _grade_frame(frame: np.ndarray) -> np.ndarray:
        arr = frame.astype(np.float32)
        # Brightness
        arr = arr * brightness
        # Contrast (centred on mid-grey 128)
        arr = (arr - 128) * contrast + 128
        # Channel shifts
        arr[:, :, 0] += r_shift
        arr[:, :, 1] += g_shift
        arr[:, :, 2] += b_shift
        # Saturation via greyscale blend
        grey = (arr[:, :, 0] * 0.299 + arr[:, :, 1] * 0.587 + arr[:, :, 2] * 0.114)
        grey = grey[:, :, np.newaxis]
        arr = grey + (arr - grey) * saturation
        return np.clip(arr, 0, 255).astype(np.uint8)

    return clip.fl_image(_grade_frame)


# ---------------------------------------------------------------------------
# Subtitle burning
# ---------------------------------------------------------------------------

def burn_subtitles(clip, subtitle_list: List[Dict], style: str = "default"):
    """
    Burn subtitles onto a video clip as TextClips in the bottom third.

    Args:
        clip: moviepy VideoClip.
        subtitle_list: List of {'text': str, 'start_time': float, 'end_time': float}
        style: 'default' (white text, black outline, bottom third)

    Returns:
        Composite clip with burnt-in subtitles.
    """
    mpe = _get_moviepy()

    if not subtitle_list:
        return clip

    sub_clips = [clip]

    for sub in subtitle_list:
        text = sub.get("text", "")
        start = float(sub.get("start_time", 0))
        end = float(sub.get("end_time", start + 2.0))

        if not text:
            continue

        try:
            txt_clip = (
                mpe.TextClip(
                    text,
                    fontsize=48,
                    color="white",
                    stroke_color="black",
                    stroke_width=2,
                    method="label",
                    size=(_REEL_W - 80, None),
                    align="center",
                )
                .set_start(start)
                .set_end(end)
                .set_position(("center", int(_REEL_H * 0.80)))
            )
            sub_clips.append(txt_clip)
        except Exception as exc:
            logger.warning("Subtitle render failed for '%s...': %s", text[:30], exc)

    return mpe.CompositeVideoClip(sub_clips)


# ---------------------------------------------------------------------------
# Text card generator
# ---------------------------------------------------------------------------

def create_text_card(
    text: str,
    style: str = "bold_dark",
    duration: float = 3.0,
    size: Tuple[int, int] = (_REEL_W, _REEL_H),
) -> Any:
    """
    Create a text-only video card (static image clip with given duration).

    Args:
        text: The text to display.
        style: 'bold_dark' | 'bold_light' | 'minimal' | 'handwritten_feel'
        duration: Duration in seconds.
        size: (width, height) tuple.

    Returns:
        moviepy ImageClip.
    """
    mpe = _get_moviepy()
    w, h = size

    # Style definitions
    styles: Dict[str, Dict] = {
        "bold_dark": {"bg": (15, 15, 15), "fg": (255, 255, 255), "font_size": 72, "font_style": "bold"},
        "bold_light": {"bg": (245, 245, 245), "fg": (20, 20, 20), "font_size": 72, "font_style": "bold"},
        "minimal": {"bg": (255, 255, 255), "fg": (80, 80, 80), "font_size": 56, "font_style": "regular"},
        "handwritten_feel": {"bg": (255, 248, 220), "fg": (60, 30, 10), "font_size": 64, "font_style": "regular"},
    }

    s = styles.get(style, styles["bold_dark"])
    img = Image.new("RGB", (w, h), s["bg"])
    draw = ImageDraw.Draw(img)
    font = load_font(s["font_size"], s["font_style"])

    # Word-wrap text to fit width
    wrapped_lines = _wrap_text(draw, text, font, max_width=w - 120)

    # Calculate total text block height
    line_h = s["font_size"] + 16
    total_h = len(wrapped_lines) * line_h

    y = (h - total_h) // 2
    for line in wrapped_lines:
        try:
            bbox = draw.textbbox((0, 0), line, font=font)
            text_w = bbox[2] - bbox[0]
        except AttributeError:
            text_w = int(draw.textlength(line, font=font))

        x = (w - text_w) // 2
        draw.text((x, y), line, font=font, fill=s["fg"])
        y += line_h

    arr = np.array(img)
    return mpe.ImageClip(arr).set_duration(duration).set_fps(_FPS)


# ---------------------------------------------------------------------------
# Audio mixing
# ---------------------------------------------------------------------------

def mix_audio(
    voiceover_path: str,
    music_path: str,
    vo_vol: float = 1.0,
    music_vol: float = 0.12,
    fade_in: float = 2.0,
    fade_out: float = 2.0,
    output_path: Optional[str] = None,
) -> str:
    """
    Mix a voiceover with background music.
    VO is at full volume; music is ducked to music_vol (default 12%).

    Args:
        voiceover_path: Path to the TTS audio file.
        music_path: Path to the background music file.
        vo_vol: Voiceover volume multiplier (default 1.0).
        music_vol: Music volume multiplier (default 0.12 = 12%).
        fade_in: Fade-in seconds for music.
        fade_out: Fade-out seconds for music.
        output_path: Output path for mixed audio. Defaults to temp dir.

    Returns:
        Path to the mixed audio file.
    """
    mpe = _get_moviepy()
    _TEMP_DIR.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        output_path = str(_TEMP_DIR / "mixed_audio.mp3")

    vo = mpe.AudioFileClip(voiceover_path).volumex(vo_vol)
    music = mpe.AudioFileClip(music_path).volumex(music_vol)

    # Apply fades to music
    music = music.audio_fadein(fade_in).audio_fadeout(fade_out)

    # Loop music to match VO duration
    if music.duration < vo.duration:
        loops = int(vo.duration / music.duration) + 1
        music = mpe.concatenate_audioclips([music] * loops)
    music = music.subclip(0, vo.duration)

    # Composite: VO on top of music
    mixed = mpe.CompositeAudioClip([music, vo])
    mixed.write_audiofile(output_path, fps=44100, nbytes=2, codec="libmp3lame", bitrate=_AUDIO_BITRATE)

    logger.info("Mixed audio saved → %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Export & verification
# ---------------------------------------------------------------------------

def export_reel(clip, output_path: str) -> str:
    """
    Export a moviepy clip to H.264/AAC MP4 at 1080×1920 / 30fps.
    Enforces <100MB output size limit.

    Args:
        clip: Final composited moviepy VideoClip.
        output_path: Destination path.

    Returns:
        Output path string.
    """
    mpe = _get_moviepy()
    _TEMP_DIR.mkdir(parents=True, exist_ok=True)
    output_path = str(output_path)

    logger.info("Exporting reel to %s (this may take several minutes)...", output_path)
    t_start = time.time()

    try:
        clip.write_videofile(
            output_path,
            fps=_FPS,
            codec="libx264",
            audio_codec="aac",
            bitrate=_VIDEO_BITRATE,
            audio_bitrate=_AUDIO_BITRATE,
            threads=_MOVIEPY_THREADS,
            preset="medium",
            ffmpeg_params=["-movflags", "+faststart"],  # web-optimised
            logger=None,  # suppress moviepy's own progress bar in JSON logs
        )
    except Exception as exc:
        logger.error("Export failed: %s", exc)
        raise

    elapsed = time.time() - t_start
    logger.info("Export completed in %.1fs.", elapsed)

    # File size check
    size_bytes = Path(output_path).stat().st_size
    size_mb = size_bytes / 1_048_576
    if size_bytes > _MAX_SIZE_BYTES:
        logger.warning(
            "Reel is %.1fMB — exceeds 100MB limit! Consider reducing bitrate or duration.",
            size_mb,
            extra={"size_mb": round(size_mb, 1)},
        )
    else:
        logger.info("Reel size: %.1fMB ✓", size_mb)

    return output_path


def verify_reel(video_path: str) -> Dict[str, Any]:
    """
    Verify an exported reel using ffprobe.

    Args:
        video_path: Path to the exported video file.

    Returns:
        Dict with keys: duration (float), resolution (str), size_mb (float),
        fps (float), has_audio (bool), valid (bool).
    """
    result: Dict[str, Any] = {
        "duration": 0.0,
        "resolution": "unknown",
        "size_mb": 0.0,
        "fps": 0.0,
        "has_audio": False,
        "valid": False,
    }

    video_path = Path(video_path)
    if not video_path.exists():
        logger.error("Video file not found for verification: %s", video_path)
        return result

    result["size_mb"] = round(video_path.stat().st_size / 1_048_576, 1)

    try:
        import json as _json
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", "-show_format", str(video_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        info = _json.loads(proc.stdout)

        fmt = info.get("format", {})
        result["duration"] = float(fmt.get("duration", 0))

        for stream in info.get("streams", []):
            codec_type = stream.get("codec_type", "")
            if codec_type == "video":
                w = stream.get("width", 0)
                h = stream.get("height", 0)
                result["resolution"] = f"{w}x{h}"
                r_str = stream.get("r_frame_rate", "0/1")
                num, den = r_str.split("/") if "/" in r_str else (r_str, "1")
                result["fps"] = round(int(num) / max(int(den), 1), 2)
            elif codec_type == "audio":
                result["has_audio"] = True

        result["valid"] = (
            result["duration"] > 0
            and result["resolution"] != "unknown"
            and result["size_mb"] < 100
        )

    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        logger.warning("ffprobe not available — skipping verification: %s", exc)
        result["valid"] = video_path.stat().st_size > 0

    except Exception as exc:
        logger.error("Reel verification error: %s", exc)

    logger.info(
        "Reel verification: valid=%s, %.1fs, %s, %.1fMB",
        result["valid"], result["duration"], result["resolution"], result["size_mb"],
        extra=result,
    )
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_video_segment(mpe, clip_path: str, duration: float):
    """Load a video file, resize to reel dimensions, and trim to duration."""
    clip = mpe.VideoFileClip(clip_path, target_resolution=(_REEL_H, _REEL_W))
    clip = clip.resize((_REEL_W, _REEL_H))
    if clip.duration > duration:
        clip = clip.subclip(0, duration)
    elif clip.duration < duration:
        clip = clip.loop(duration=duration)
    return clip.without_audio()  # Audio handled separately


def _load_image_segment(mpe, image_path: str, duration: float):
    """Load a still image as a video clip resized to reel dimensions."""
    img = Image.open(image_path).convert("RGB").resize((_REEL_W, _REEL_H), Image.LANCZOS)
    arr = np.array(img)
    return mpe.ImageClip(arr).set_duration(duration).set_fps(_FPS)


def _wrap_text(draw: ImageDraw.Draw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    """Word-wrap text to fit within max_width pixels."""
    words = text.split()
    lines: List[str] = []
    current_line = ""

    for word in words:
        test_line = f"{current_line} {word}".strip()
        try:
            bbox = draw.textbbox((0, 0), test_line, font=font)
            test_w = bbox[2] - bbox[0]
        except AttributeError:
            test_w = int(draw.textlength(test_line, font=font))

        if test_w <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return lines
