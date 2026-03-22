"""IPC helpers for cross-process image transfer.

IMPORTANT: This module is imported in TWO contexts:
  1. The main GUI process  — only _img_to_bytes / _bytes_to_img are needed here.
     Heavy deps (PIL processing pipeline) must NOT be imported at module level
     so that importing _ipc in gui.py doesn't pull in numpy/PIL on cold start.
  2. The worker subprocess — _worker_process is the entry point.  The subprocess
     imports this module fresh; heavy imports there are fine because the subprocess
     only starts when the user first triggers processing.
"""
from __future__ import annotations

import io


# ---------------------------------------------------------------------------
# Lightweight serialisation helpers (used by main process — keep dep-free)
# ---------------------------------------------------------------------------

def _img_to_bytes(img) -> bytes:
    """Serialize a PIL Image to PNG bytes for cross-process transfer."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _bytes_to_img(data: bytes):
    """Deserialize PNG bytes back to a PIL Image."""
    from PIL import Image  # deferred — only needed when a result comes back
    return Image.open(io.BytesIO(data)).copy()


# ---------------------------------------------------------------------------
# Worker entry point (runs inside the subprocess — heavy imports are fine)
# ---------------------------------------------------------------------------

def _worker_process(mode: str, img_bytes: bytes, params: dict, draft: bool) -> bytes:
    """Called by ProcessPoolExecutor worker.  Imports are local to this process."""
    from PIL import Image
    from .emissive import process_emissive
    from .processing import process_image

    img = Image.open(io.BytesIO(img_bytes)).copy()

    if mode == "emissive":
        result = process_emissive(img, params, draft=draft)
    elif params.get("passthrough", False):
        # All quant params are at neutral — skip the pipeline entirely so
        # the image is returned pixel-perfect with no lossy round-trip.
        result = img
    else:
        result = process_image(
            img, 1.0, 0.0,
            params["colors"], params["dither"],
            use_pillow_dither=params.get("use_pillow_dither", False),
        )

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return buf.getvalue()
