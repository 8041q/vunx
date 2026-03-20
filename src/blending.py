"""NumPy-based blend mode functions for image compositing.

All public functions accept:
    base  : np.ndarray float32, values in [0, 1]
    top   : np.ndarray float32, values in [0, 1]  (same shape as base)
    opacity : float in [0, 1]

All return np.ndarray float32, values clipped to [0, 1].
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def blend_screen(base: np.ndarray, top: np.ndarray, opacity: float = 1.0) -> np.ndarray:
    """Screen: 1 - (1-a)*(1-b).  Lightens the image."""
    blended = 1.0 - (1.0 - base) * (1.0 - top)
    return np.clip(base + (blended - base) * opacity, 0.0, 1.0)


def blend_linear_dodge(base: np.ndarray, top: np.ndarray, opacity: float = 1.0) -> np.ndarray:
    """Linear Dodge (Add): min(a + b, 1).  Strong lighten."""
    blended = np.minimum(base + top, 1.0)
    return np.clip(base + (blended - base) * opacity, 0.0, 1.0)


def blend_color_dodge(base: np.ndarray, top: np.ndarray, opacity: float = 1.0) -> np.ndarray:
    """Color Dodge: a / (1 - b), clamped.  Intense lighten on bright areas."""
    denom = np.maximum(1.0 - top, 1e-6)
    blended = np.minimum(base / denom, 1.0)
    return np.clip(base + (blended - base) * opacity, 0.0, 1.0)


def blend_soft_light(base: np.ndarray, top: np.ndarray, opacity: float = 1.0) -> np.ndarray:
    """Soft Light (Photoshop formula).  Gentle contrast boost."""
    lo = 2.0 * base * top + base * base * (1.0 - 2.0 * top)
    hi = 2.0 * base * (1.0 - top) + np.sqrt(base) * (2.0 * top - 1.0)
    blended = np.where(top <= 0.5, lo, hi)
    return np.clip(base + (blended - base) * opacity, 0.0, 1.0)


def blend_overlay(base: np.ndarray, top: np.ndarray, opacity: float = 1.0) -> np.ndarray:
    """Overlay: 2ab if a<0.5, else 1 - 2(1-a)(1-b).  Contrast enhancer."""
    lo = 2.0 * base * top
    hi = 1.0 - 2.0 * (1.0 - base) * (1.0 - top)
    blended = np.where(base < 0.5, lo, hi)
    return np.clip(base + (blended - base) * opacity, 0.0, 1.0)


BLEND_MODES: dict[str, Callable] = {
    "Screen": blend_screen,
    "Linear Dodge": blend_linear_dodge,
    "Color Dodge": blend_color_dodge,
    "Soft Light": blend_soft_light,
    "Overlay": blend_overlay,
}
