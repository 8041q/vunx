"""Emissive Dreamy style processing pipeline.

Implements the 6-step workflow from emissive_dreamy_style.txt:
    1. Levels (contrast)
    2. Gradient map / Colorize (cool palette)
    3. Emissive glow (blur + additive blend)
    4. Dream softness (blur + soft light)
    5. Color bloom (optional accent)
    6. Grain / texture

Each step is a standalone function.  ``process_emissive()`` orchestrates them.
"""

from __future__ import annotations

from .blending import BLEND_MODES

np = None
Image = None
ImageFilter = None
_n = None
_Image = None
_ImageFilter = None


def _lazy_emissive_deps():
    global np, Image, ImageFilter, _n, _Image, _ImageFilter
    if _n is None or _Image is None or _ImageFilter is None:
        import numpy as local_np
        from PIL import Image as local_Image, ImageFilter as local_ImageFilter
        np = local_np
        Image = local_Image
        ImageFilter = local_ImageFilter
        _n, _Image, _ImageFilter = np, Image, ImageFilter
    return np, Image, ImageFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _img_to_float(img: Image.Image) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Convert PIL RGBA → (rgb float32 [0,1] shape (H,W,3), alpha uint8 (H,W), (W,H))."""
    np, Image, ImageFilter = _lazy_emissive_deps()
    rgba = np.asarray(img.convert("RGBA"))
    h, w = rgba.shape[:2]
    rgb = rgba[..., :3].astype(np.float32) / 255.0
    alpha = rgba[..., 3]
    return rgb, alpha, (w, h)


def _float_to_img(rgb: np.ndarray, alpha: np.ndarray, size: tuple[int, int]) -> Image.Image:
    """float32 (H,W,3) [0,1] + uint8 alpha (H,W) → PIL RGBA."""
    np, Image, ImageFilter = _lazy_emissive_deps()
    w, h = size
    out = np.empty((h, w, 4), dtype=np.uint8)
    out[..., :3] = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    out[..., 3] = alpha
    return Image.fromarray(out, mode="RGBA")


def _luminance(rgb: np.ndarray) -> np.ndarray:
    """Rec. 709 luminance from float32 (H,W,3) → (H,W) float32."""
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


# ---------------------------------------------------------------------------
# Step 1 – Levels / Contrast
# ---------------------------------------------------------------------------

def adjust_levels(
    img: Image.Image,
    shadow_point: int = 30,
    highlight_point: int = 220,
    gamma: float = 1.2,
) -> Image.Image:
    """Remap [shadow_point, highlight_point] → [0, 255] then apply gamma."""
    rgb, alpha, size = _img_to_float(img)

    lo = shadow_point / 255.0
    hi = highlight_point / 255.0
    span = max(hi - lo, 1e-6)
    rgb = (rgb - lo) / span
    rgb = np.clip(rgb, 0.0, 1.0)

    inv_gamma = 1.0 / max(gamma, 0.01)
    rgb = np.power(rgb, inv_gamma)

    return _float_to_img(rgb, alpha, size)


# ---------------------------------------------------------------------------
# Step 2a – Gradient Map
# ---------------------------------------------------------------------------

def apply_gradient_map(
    img: Image.Image,
    shadow_color: tuple[int, int, int] = (10, 15, 40),
    mid_color: tuple[int, int, int] = (0, 180, 220),
    highlight_color: tuple[int, int, int] = (230, 240, 255),
) -> Image.Image:
    """Map luminance through a 3-stop colour gradient."""
    rgb, alpha, size = _img_to_float(img)
    lum = _luminance(rgb)  # (H, W)

    # Build 256-entry LUT for each channel
    lut = np.zeros((256, 3), dtype=np.float32)
    s = np.array(shadow_color, dtype=np.float32) / 255.0
    m = np.array(mid_color, dtype=np.float32) / 255.0
    h = np.array(highlight_color, dtype=np.float32) / 255.0

    for i in range(256):
        t = i / 255.0
        if t < 0.5:
            frac = t / 0.5
            lut[i] = s + (m - s) * frac
        else:
            frac = (t - 0.5) / 0.5
            lut[i] = m + (h - m) * frac

    idx = np.clip((lum * 255.0).astype(np.int32), 0, 255)
    mapped = lut[idx]  # (H, W, 3)

    return _float_to_img(mapped, alpha, size)


# ---------------------------------------------------------------------------
# Step 2b – Colorize (single-hue)
# ---------------------------------------------------------------------------

def apply_colorize(
    img: Image.Image,
    hue: int = 210,
    saturation: int = 70,
) -> Image.Image:
    """Desaturate then tint to a single hue (HSL model, hue 0-360, sat 0-100)."""
    rgb, alpha, size = _img_to_float(img)
    lum = _luminance(rgb)  # (H, W)

    h_rad = (hue % 360) / 360.0
    s = saturation / 100.0

    # HSL→RGB for a given hue at each pixel's luminance
    def _hsl_channel(p, q, t):
        t = t % 1.0
        r = np.where(t < 1 / 6, p + (q - p) * 6.0 * t,
            np.where(t < 1 / 2, q,
            np.where(t < 2 / 3, p + (q - p) * (2 / 3 - t) * 6.0, p)))
        return r

    q = np.where(lum < 0.5, lum * (1.0 + s), lum + s - lum * s)
    p = 2.0 * lum - q

    r = _hsl_channel(p, q, h_rad + 1 / 3)
    g = _hsl_channel(p, q, h_rad)
    b = _hsl_channel(p, q, h_rad - 1 / 3)
    mapped = np.stack([r, g, b], axis=-1)
    mapped = np.clip(mapped, 0.0, 1.0)

    return _float_to_img(mapped, alpha, size)


# ---------------------------------------------------------------------------
# Step 3 – Emissive Glow
# ---------------------------------------------------------------------------

def create_glow(
    img: Image.Image,
    blur_radius: float = 20.0,
    blend_mode: str = "Screen",
    opacity: float = 0.6,
    threshold: int = 128,
) -> Image.Image:
    """Duplicate → blur → threshold-mask → blend for emissive glow."""
    rgb_base, alpha, size = _img_to_float(img)

    # Apply brightness threshold mask: only glow from bright areas
    lum = _luminance(rgb_base)
    mask = np.clip((lum - threshold / 255.0) / max(1.0 - threshold / 255.0, 1e-6), 0.0, 1.0)
    masked_rgb = rgb_base * mask[..., np.newaxis]

    # Build a masked image for blurring
    masked_uint8 = np.clip(masked_rgb * 255.0, 0, 255).astype(np.uint8)
    h, w = rgb_base.shape[:2]
    masked_img = Image.fromarray(masked_uint8, mode="RGB")

    # Gaussian blur
    blurred_img = masked_img.filter(ImageFilter.GaussianBlur(blur_radius))
    blurred = np.asarray(blurred_img).astype(np.float32) / 255.0

    # Blend
    blend_fn = BLEND_MODES.get(blend_mode, BLEND_MODES["Screen"])
    result = blend_fn(rgb_base, blurred, opacity)

    return _float_to_img(result, alpha, size)


# ---------------------------------------------------------------------------
# Step 4 – Dream Softness
# ---------------------------------------------------------------------------

def create_softness(
    img: Image.Image,
    blur_radius: float = 3.0,
    blend_mode: str = "Soft Light",
    opacity: float = 0.2,
) -> Image.Image:
    """Slight blur blended with Soft Light for floating, surreal softness."""
    rgb_base, alpha, size = _img_to_float(img)

    blurred_img = img.convert("RGB").filter(ImageFilter.GaussianBlur(blur_radius))
    blurred = np.asarray(blurred_img).astype(np.float32) / 255.0

    blend_fn = BLEND_MODES.get(blend_mode, BLEND_MODES["Soft Light"])
    result = blend_fn(rgb_base, blurred, opacity)

    return _float_to_img(result, alpha, size)


# ---------------------------------------------------------------------------
# Step 5 – Color Bloom
# ---------------------------------------------------------------------------

def apply_color_bloom(
    img: Image.Image,
    color: tuple[int, int, int] = (0, 200, 255),
    blend_mode: str = "Overlay",
    opacity: float = 0.15,
) -> Image.Image:
    """Solid colour layer blended over the image to amplify emissive tones."""
    rgb_base, alpha, size = _img_to_float(img)

    solid = np.full_like(rgb_base, np.array(color, dtype=np.float32) / 255.0)

    blend_fn = BLEND_MODES.get(blend_mode, BLEND_MODES["Overlay"])
    result = blend_fn(rgb_base, solid, opacity)

    return _float_to_img(result, alpha, size)


# ---------------------------------------------------------------------------
# Step 6 – Grain
# ---------------------------------------------------------------------------

def add_grain(
    img: Image.Image,
    intensity: float = 0.08,
    grain_type: str = "gaussian",
    seed: int = 42,
) -> Image.Image:
    """Add film-like grain to prevent a sterile digital look."""
    rgb, alpha, size = _img_to_float(img)

    rng = np.random.RandomState(seed)
    shape = rgb.shape[:2]  # (H, W)

    if grain_type == "uniform":
        noise = rng.uniform(-1.0, 1.0, shape).astype(np.float32)
    else:  # gaussian
        noise = rng.standard_normal(shape).astype(np.float32)
        noise = np.clip(noise, -3.0, 3.0) / 3.0  # normalise to roughly [-1, 1]

    # Apply monochrome noise across all channels
    rgb = rgb + intensity * noise[..., np.newaxis]
    rgb = np.clip(rgb, 0.0, 1.0)

    return _float_to_img(rgb, alpha, size)


# ---------------------------------------------------------------------------
# Draft-mode helper
# ---------------------------------------------------------------------------

_DRAFT_MAX_PIXELS = 1024 * 768  # ~0.78 MP for fast live preview


def _draft_downscale(img: Image.Image) -> tuple[Image.Image, float]:
    """Downscale for live preview if the image exceeds _DRAFT_MAX_PIXELS.

    Returns (possibly-resized image, scale_factor).  scale_factor is 1.0 if
    no resize was needed.
    """
    w, h = img.size
    n = w * h
    if n <= _DRAFT_MAX_PIXELS:
        return img, 1.0
    factor = ((_DRAFT_MAX_PIXELS) / n) ** 0.5
    new_w = max(1, int(w * factor))
    new_h = max(1, int(h * factor))
    return img.resize((new_w, new_h), Image.BILINEAR), factor


def _draft_upscale(img: Image.Image, original_size: tuple[int, int]) -> Image.Image:
    """Upscale a draft result back to the original dimensions."""
    if img.size == original_size:
        return img
    return img.resize(original_size, Image.BILINEAR)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

# Neutral / identity parameters — every effect is a pass-through or disabled.
# This is what ``process_emissive`` falls back to for any missing key, and
# what the GUI uses when the user picks "Default" (clean slate / reset).
#
# Rules for "neutral":
#   - Levels  : shadow=0, highlight=255, gamma=1.0  → identity remap
#   - Color   : "none"                               → no colour change
#   - Glow / Softness / Bloom / Grain : enabled=False → no effect applied
#
# Preset files (JSON) override whichever keys they need; they do NOT live here.
EMISSIVE_NEUTRAL: dict = {
    # Levels — identity pass-through
    "levels_shadow": 0,
    "levels_highlight": 255,
    "levels_gamma": 1.0,
    # Color grading — disabled
    "color_mode": "none",          # "none" | "gradient_map" | "colorize"
    "gm_shadow": (0, 0, 0),
    "gm_mid": (128, 128, 128),
    "gm_highlight": (255, 255, 255),
    "colorize_hue": 0,
    "colorize_sat": 0,
    # Glow — disabled
    "glow_enabled": False,
    "glow_blur": 10.0,
    "glow_blend": "Screen",
    "glow_opacity": 0.5,
    "glow_threshold": 128,
    # Softness — disabled
    "soft_enabled": False,
    "soft_blur": 2.0,
    "soft_blend": "Soft Light",
    "soft_opacity": 0.2,
    # Color bloom — disabled
    "bloom_enabled": False,
    "bloom_color": (255, 255, 255),
    "bloom_blend": "Overlay",
    "bloom_opacity": 0.2,
    # Grain — disabled
    "grain_enabled": False,
    "grain_intensity": 0.08,
    "grain_type": "gaussian",
}


def process_emissive(
    img: Image.Image,
    params: dict | None = None,
    *,
    draft: bool = False,
) -> Image.Image:
    """Run the full emissive dreamy pipeline.

    Args:
        img: Source PIL Image.
        params: Dict of parameters (keys from EMISSIVE_NEUTRAL).
                Missing keys fall back to defaults.
        draft: If True, process at reduced resolution for live preview.
    """
    p = {**EMISSIVE_NEUTRAL, **(params or {})}

    original_size = img.size
    if draft:
        img, _scale = _draft_downscale(img)

    # 1. Levels
    img = adjust_levels(img, p["levels_shadow"], p["levels_highlight"], p["levels_gamma"])

    # 2. Color grading
    if p["color_mode"] == "gradient_map":
        img = apply_gradient_map(img, p["gm_shadow"], p["gm_mid"], p["gm_highlight"])
    elif p["color_mode"] == "colorize":
        img = apply_colorize(img, p["colorize_hue"], p["colorize_sat"])
    # "none" → skip, pass image through unchanged

    # 3. Glow
    if p["glow_enabled"]:
        img = create_glow(img, p["glow_blur"], p["glow_blend"], p["glow_opacity"], p["glow_threshold"])

    # 4. Softness
    if p["soft_enabled"]:
        img = create_softness(img, p["soft_blur"], p["soft_blend"], p["soft_opacity"])

    # 5. Color bloom
    if p["bloom_enabled"]:
        img = apply_color_bloom(img, p["bloom_color"], p["bloom_blend"], p["bloom_opacity"])

    # 6. Grain
    if p["grain_enabled"]:
        img = add_grain(img, p["grain_intensity"], p["grain_type"])

    if draft:
        img = _draft_upscale(img, original_size)

    return img
