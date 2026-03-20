# Vunx

**Vunx** is a GUI-first image processing prototype built in Python using **PySide6**, **Pillow**, and **NumPy**.
It focuses on fast, high-quality image quantization and dithering with a retro aesthetic.

---

## Quickstart

### 1. Setup environment

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run the GUI

```bash
python -m app
```

---

## Features

* **Image Loading** – Open and preview images easily
* **Resampling** – High-quality resizing using Lanczos filtering
* **Preprocessing** – Gaussian blur for smoother quantization results
* **Adaptive Palette** – Intelligent color reduction
* **Vectorized Dithering** – Optional Bayer (ordered) dithering powered by NumPy (no pixel-by-pixel loops)
* **Emissive Dreamy Style** – Neon, dream-like image processing with glow, gradient maps, and film grain
* **Live Preview** – Instantly see results (draft mode for fast feedback)
* **Export** – Save processed images at full resolution

---

## High-Performance Dithering

Vunx uses a custom **Vectorized Bayer (Ordered) Dithering engine**. Unlike traditional methods that process images one pixel at a time, Vunx processes the entire image simultaneously using NumPy.

* **Instant Speed:** No slow Python loops. Even 4K images dither in milliseconds
* **Retro Aesthetic:** Classic grid-style dithering inspired by 90s hardware
* **Intelligent Palette:** Uses **K-Means Clustering** (when available) for optimal color selection

---

## Technical Architecture

The processing pipeline adapts based on your environment:

| Feature         | with scikit-learn (Pro)            | without scikit-learn (Lite) |
| --------------- | ---------------------------------- | --------------------------- |
| Color Selection | K-Means Clustering (Most accurate) | Median Cut (Standard)       |
| Dithering       | NumPy Bayer (Vectorized)           | Pillow FS (Fallback)        |
| Performance     | Optimized for large batches        | Standard single-threaded    |

---

## Project Status

> This is an early-stage prototype.

Planned improvements:

* Lower-level optimizations (closer to C performance)
* Better batching and pipeline control
* Expanded export and format support

---

## Building (Windows - Nuitka)

> Adjust the Python path to match your environment.

```bash
d:/Users/guivt/Documents/Projects/test/.venv/Scripts/python.exe -m nuitka \
    --standalone \
    --enable-plugin=numpy \
    --user-package-configuration-file=nuitka-package.config.yml \
    app.py
```

---

## Notes

* Designed as a **GUI-first experimentation tool**, not a finalized library
* Emphasis on **speed, simplicity, and visual output quality**


### Next improvements

- Expose Bayer matrix order as a user control [verify]
- Add Floyd–Steinberg option for users who prefer error-diffusion aesthetics [completed]

---

## Future: Graphic Overlays

The Emissive Dreamy style guide describes optional graphic overlay elements that are not yet implemented:

* **Warning labels** – Translucent caution/warning text overlays
* **UI dots** – Small circular elements scattered across the image
* **Geometric patterns** – Triangles, hexagons, grid lines as decorative layers
* **Tech textures** – Circuit-board or HUD-style texture overlays

These would be added as PNG overlays or generated shape layers composited at low opacity on top of the processed image. Planned approach:

1. Overlay asset loader (accept user-provided PNG files)
2. Built-in geometric shape generators (grids, triangles, circles)
3. Per-overlay controls: position, rotation, opacity, blend mode
4. Layer stack UI for ordering and toggling multiple overlays