"""
utils/image_utils.py
====================
Pillow-based image helpers for FACE.
Handles text overlays (bilingual EN/BN), format resizing, dominant colours,
contrast checking, colour grading, and NotoSans font auto-download.

Usage:
    from utils.image_utils import add_text_overlay, resize_for_format, get_dominant_colors
"""

import io
import math
import os
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from utils.logger import get_logger

logger = get_logger("image_utils")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
_FONTS_DIR = _ROOT / "assets" / "fonts"
_TEMP_DIR = _ROOT / "state" / "temp"

# NotoSans download URL (supports both Latin and Bangla Unicode)
_NOTO_SANS_URLS = {
    "regular": "https://github.com/notofonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf",
    "bold": "https://github.com/notofonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Bold.ttf",
    "bengali": "https://github.com/notofonts/noto-fonts/raw/main/hinted/ttf/NotoSansBengali/NotoSansBengali-Regular.ttf",
}

# In-memory font cache: {(size, style) → ImageFont}
_FONT_CACHE: Dict[Tuple[int, str], ImageFont.FreeTypeFont] = {}

# Target sizes for each output format (width × height)
FORMAT_SIZES: Dict[str, Tuple[int, int]] = {
    "feed_square": (1080, 1080),
    "feed_portrait": (1080, 1350),
    "story": (1080, 1920),
    "reel_cover": (1080, 1920),
    "link": (1200, 630),
}

# Color grades: (contrast, brightness, saturation, colour_matrix_hint)
_COLOR_GRADES: Dict[str, Dict] = {
    "warm": {"contrast": 1.05, "brightness": 1.05, "saturation": 1.15, "color_shift": (10, 0, -10)},
    "cool": {"contrast": 1.05, "brightness": 1.0, "saturation": 0.95, "color_shift": (-5, 0, 15)},
    "high_contrast": {"contrast": 1.35, "brightness": 1.0, "saturation": 1.2, "color_shift": (0, 0, 0)},
    "pastel": {"contrast": 0.85, "brightness": 1.15, "saturation": 0.75, "color_shift": (5, 5, 5)},
    "cinematic": {"contrast": 1.20, "brightness": 0.92, "saturation": 1.10, "color_shift": (5, -2, -8)},
}


# ---------------------------------------------------------------------------
# Font management
# ---------------------------------------------------------------------------

def load_font(size: int, style: str = "regular") -> ImageFont.FreeTypeFont:
    """
    Return a NotoSans TrueType font at the requested size.
    Downloads to assets/fonts/ on first call; cached in memory after that.

    Args:
        size: Font size in points.
        style: 'regular' | 'bold' | 'bengali'

    Returns:
        Loaded ImageFont.FreeTypeFont instance.
    """
    cache_key = (size, style)
    if cache_key in _FONT_CACHE:
        return _FONT_CACHE[cache_key]

    _FONTS_DIR.mkdir(parents=True, exist_ok=True)
    font_filename = f"NotoSans-{style.capitalize()}.ttf"
    font_path = _FONTS_DIR / font_filename

    if not font_path.exists():
        _download_font(style, font_path)

    try:
        font = ImageFont.truetype(str(font_path), size=size)
    except OSError as exc:
        logger.warning("Could not load font %s, using default: %s", font_path, exc)
        font = ImageFont.load_default()  # type: ignore[assignment]

    _FONT_CACHE[cache_key] = font
    return font


def _download_font(style: str, dest: Path) -> None:
    """Download a NotoSans font file to assets/fonts/."""
    url = _NOTO_SANS_URLS.get(style, _NOTO_SANS_URLS["regular"])
    logger.info("Downloading NotoSans font (%s) from %s", style, url)
    try:
        urllib.request.urlretrieve(url, str(dest))
        logger.info("Font saved to %s", dest)
    except Exception as exc:
        logger.error("Font download failed for style=%s: %s", style, exc)
        raise


# ---------------------------------------------------------------------------
# Text overlay
# ---------------------------------------------------------------------------

def add_text_overlay(
    image_path: Union[str, Path],
    text: Union[str, Dict[str, str]],
    position: str = "bottom",
    font_size: int = 52,
    color: Optional[str] = None,
    shadow: bool = True,
    backdrop: bool = True,
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """
    Overlay text on an image. Handles bilingual text (EN + BN stacked).

    Args:
        image_path: Source image path.
        text: Either a plain string, or {'en': '...', 'bn': '...'} for bilingual.
        position: 'top' | 'bottom' | 'center' | 'left' | 'right'
        font_size: Font size in points.
        color: Hex colour string like '#FFFFFF'. If None, auto-chosen for contrast.
        shadow: If True, add a 3px drop shadow at 60% opacity.
        backdrop: If True, add a semi-transparent bar behind the text.
        output_path: Where to save. Defaults to <original>_overlay.<ext>.

    Returns:
        Path string of the output image.
    """
    image_path = Path(image_path)
    img = Image.open(image_path).convert("RGBA")
    width, height = img.size

    _TEMP_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = _TEMP_DIR / (image_path.stem + "_overlay.jpg")
    output_path = Path(output_path)

    # Normalise text input
    if isinstance(text, dict):
        lines = []
        if text.get("en"):
            lines.append(("en", text["en"]))
        if text.get("bn"):
            lines.append(("bn", text["bn"]))
    else:
        lines = [("en", str(text))]

    # ── Determine backdrop region ────────────────────────────────────────────
    # Estimate total text block height
    line_height = font_size + 12  # padding
    block_height = line_height * len(lines) + 24

    margin = 20
    if position in ("bottom", "right"):
        bar_y0 = height - block_height - margin
        bar_y1 = height - margin + 8
    elif position in ("top", "left"):
        bar_y0 = margin - 8
        bar_y1 = margin + block_height
    else:  # center
        bar_y0 = (height - block_height) // 2 - 12
        bar_y1 = (height + block_height) // 2 + 12

    bar_y0 = max(0, bar_y0)
    bar_y1 = min(height, bar_y1)

    # Auto-contrast: sample the backdrop region to pick text colour
    if color is None:
        crop_box = (0, bar_y0, width, bar_y1)
        region = img.crop(crop_box).convert("RGB")
        avg_lum = _region_luminance(region)
        color = "#FFFFFF" if avg_lum < 0.5 else "#1A1A1A"

    text_rgb = _hex_to_rgb(color)
    shadow_rgb = (0, 0, 0) if text_rgb[0] > 128 else (255, 255, 255)

    # ── Draw backdrop ────────────────────────────────────────────────────────
    draw_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(draw_layer)

    if backdrop:
        backdrop_alpha = 160  # ~63% opacity
        draw.rectangle(
            [(0, bar_y0), (width, bar_y1)],
            fill=(0, 0, 0, backdrop_alpha),
        )

    # ── Draw text lines ──────────────────────────────────────────────────────
    y_cursor = bar_y0 + 12

    for lang, line_text in lines:
        font_style = "bengali" if lang == "bn" else "regular"
        font = load_font(font_size, style=font_style)

        # Measure text width for centering
        try:
            bbox = draw.textbbox((0, 0), line_text, font=font)
            text_w = bbox[2] - bbox[0]
        except AttributeError:
            text_w = draw.textlength(line_text, font=font)

        x = (width - text_w) // 2  # horizontally centred

        if shadow:
            # Drop shadow: 3px offset, 60% opacity black
            shadow_color = shadow_rgb + (153,)  # 60% of 255
            draw.text((x + 3, y_cursor + 3), line_text, font=font, fill=shadow_color)

        draw.text((x, y_cursor), line_text, font=font, fill=text_rgb + (255,))
        y_cursor += line_height

    # ── Composite layers and save ────────────────────────────────────────────
    composited = Image.alpha_composite(img, draw_layer).convert("RGB")
    composited.save(str(output_path), "JPEG", quality=92)

    logger.debug("Text overlay saved to %s", output_path)
    return str(output_path)


# ---------------------------------------------------------------------------
# Dominant colours
# ---------------------------------------------------------------------------

def get_dominant_colors(image_path: Union[str, Path], n: int = 5) -> List[str]:
    """
    Return the top N dominant colours from an image as hex strings.
    Uses k-means clustering on downscaled pixel data for speed.

    Args:
        image_path: Path to the image file.
        n: Number of dominant colours to return.

    Returns:
        List of hex colour strings like ['#FF5733', '#C70039', ...].
    """
    img = Image.open(image_path).convert("RGB")
    # Downscale for speed
    img = img.resize((150, 150), Image.LANCZOS)
    pixels = np.array(img).reshape(-1, 3).astype(np.float32)

    # Simple k-means with numpy (avoids requiring sklearn)
    # Initialise centroids from random pixels
    rng = np.random.default_rng(seed=42)
    indices = rng.choice(len(pixels), size=n, replace=False)
    centroids = pixels[indices].copy()

    for _ in range(10):  # 10 iterations is enough for colour quantisation
        # Assign each pixel to nearest centroid
        distances = np.linalg.norm(pixels[:, None] - centroids[None, :], axis=2)
        labels = np.argmin(distances, axis=1)
        # Recompute centroids
        new_centroids = np.array([
            pixels[labels == k].mean(axis=0) if np.any(labels == k) else centroids[k]
            for k in range(n)
        ])
        if np.allclose(centroids, new_centroids, atol=1.0):
            break
        centroids = new_centroids

    # Sort by cluster size (most dominant first)
    counts = [np.sum(labels == k) for k in range(n)]
    sorted_pairs = sorted(zip(counts, centroids), reverse=True)

    return [_rgb_to_hex(tuple(int(c) for c in centroid)) for _, centroid in sorted_pairs]


# ---------------------------------------------------------------------------
# Format resizing
# ---------------------------------------------------------------------------

def resize_for_format(
    image_path: Union[str, Path],
    format_name: str,
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """
    Resize (and crop if necessary) an image to a standard Facebook format.

    Args:
        image_path: Source image.
        format_name: 'feed_square' | 'feed_portrait' | 'story' | 'reel_cover' | 'link'
        output_path: Destination path. Defaults to temp dir.

    Returns:
        Output path string.
    """
    if format_name not in FORMAT_SIZES:
        raise ValueError(
            f"Unknown format {format_name!r}. Choose from: {list(FORMAT_SIZES)}"
        )

    target_w, target_h = FORMAT_SIZES[format_name]
    _TEMP_DIR.mkdir(parents=True, exist_ok=True)

    image_path = Path(image_path)
    if output_path is None:
        output_path = _TEMP_DIR / f"{image_path.stem}_{format_name}.jpg"
    output_path = Path(output_path)

    img = Image.open(image_path).convert("RGB")
    src_w, src_h = img.size

    # Scale to fill (cover), then centre-crop
    scale = max(target_w / src_w, target_h / src_h)
    new_w = int(math.ceil(src_w * scale))
    new_h = int(math.ceil(src_h * scale))
    img = img.resize((new_w, new_h), Image.LANCZOS)

    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    img = img.crop((left, top, left + target_w, top + target_h))

    img.save(str(output_path), "JPEG", quality=92, optimize=True)
    logger.debug("Resized to %s → %s", format_name, output_path)
    return str(output_path)


# ---------------------------------------------------------------------------
# Contrast checking (WCAG)
# ---------------------------------------------------------------------------

def check_text_contrast(text_color_hex: str, bg_color_hex: str) -> float:
    """
    Calculate WCAG 2.1 contrast ratio between text and background colours.
    Minimum 4.5:1 is required for normal text accessibility.

    Args:
        text_color_hex: Hex colour of the text, e.g. '#FFFFFF'.
        bg_color_hex: Hex colour of the background.

    Returns:
        Contrast ratio as a float (e.g. 7.5). Higher = more contrast.
    """
    l1 = _relative_luminance(_hex_to_rgb(text_color_hex))
    l2 = _relative_luminance(_hex_to_rgb(bg_color_hex))
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


# ---------------------------------------------------------------------------
# Colour grading
# ---------------------------------------------------------------------------

def apply_color_grade(
    image_path: Union[str, Path],
    grade: str,
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """
    Apply a colour grade (warm / cool / high_contrast / pastel / cinematic)
    using PIL ImageEnhance operations and per-channel shifts.

    Args:
        image_path: Source image path.
        grade: One of the keys in _COLOR_GRADES.
        output_path: Destination path. Defaults to temp dir.

    Returns:
        Output path string.
    """
    if grade not in _COLOR_GRADES:
        raise ValueError(f"Unknown grade {grade!r}. Options: {list(_COLOR_GRADES)}")

    params = _COLOR_GRADES[grade]
    _TEMP_DIR.mkdir(parents=True, exist_ok=True)

    image_path = Path(image_path)
    if output_path is None:
        output_path = _TEMP_DIR / f"{image_path.stem}_{grade}.jpg"
    output_path = Path(output_path)

    img = Image.open(image_path).convert("RGB")

    # Apply contrast
    img = ImageEnhance.Contrast(img).enhance(params["contrast"])
    # Apply brightness
    img = ImageEnhance.Brightness(img).enhance(params["brightness"])
    # Apply saturation (colour)
    img = ImageEnhance.Color(img).enhance(params["saturation"])

    # Per-channel colour shift
    r_shift, g_shift, b_shift = params.get("color_shift", (0, 0, 0))
    if any((r_shift, g_shift, b_shift)):
        img = _apply_channel_shift(img, r_shift, g_shift, b_shift)

    img.save(str(output_path), "JPEG", quality=92)
    logger.debug("Applied %s colour grade → %s", grade, output_path)
    return str(output_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert '#RRGGBB' hex to (R, G, B) int tuple."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join(c * 2 for c in hex_color)
    r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return r, g, b


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """Convert (R, G, B) to '#RRGGBB'."""
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def _relative_luminance(rgb: Tuple[int, int, int]) -> float:
    """
    Compute WCAG relative luminance from an sRGB colour.
    Returns a value between 0 (black) and 1 (white).
    """
    def _linearise(c: int) -> float:
        v = c / 255.0
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4

    r, g, b = (_linearise(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _region_luminance(region: Image.Image) -> float:
    """Return average perceptual luminance of an image region (0=dark, 1=bright)."""
    arr = np.array(region).reshape(-1, 3)
    lum = 0.2126 * arr[:, 0] + 0.7152 * arr[:, 1] + 0.0722 * arr[:, 2]
    return float(lum.mean()) / 255.0


def _apply_channel_shift(
    img: Image.Image,
    r_shift: int,
    g_shift: int,
    b_shift: int,
) -> Image.Image:
    """Add integer offsets to each RGB channel using numpy (clamped 0–255)."""
    arr = np.array(img).astype(np.int16)
    arr[:, :, 0] = np.clip(arr[:, :, 0] + r_shift, 0, 255)
    arr[:, :, 1] = np.clip(arr[:, :, 1] + g_shift, 0, 255)
    arr[:, :, 2] = np.clip(arr[:, :, 2] + b_shift, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))
