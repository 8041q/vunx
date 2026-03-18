from PIL import Image, ImageFilter
import numpy as np
import traceback
import sys

# Availability probe — checked once, cached forever
_sklearn_available: bool | None = None


def sklearn_available() -> bool:
    """Return True if scikit-learn can be imported (cached after first call)."""
    global _sklearn_available
    if _sklearn_available is None:
        try:
            import sklearn  # noqa: F401
            _sklearn_available = True
        except Exception:
            _sklearn_available = False
    return _sklearn_available


def active_dither_method() -> str:
    """Return the name of the dither algorithm that will actually be used.

    'Bayer (ordered)'  — when sklearn is present and k-means path is active.
    'Floyd–Steinberg'  — when sklearn is absent and Pillow fallback is used.
    """
    return "Bayer (ordered)" if sklearn_available() else "Floyd–Steinberg"


def resample_image(img: Image.Image, scale: float) -> Image.Image:
    if scale == 1.0:
        return img
    w, h = img.size
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.resize(new_size, resample=Image.LANCZOS)


def blur_image(img: Image.Image, radius: float) -> Image.Image:
    if radius <= 0:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius))


def quantize_image(img: Image.Image, colors: int = 16, dither: bool = True) -> Image.Image:
    # Pillow's quantize() only honours the dither flag when a palette= reference
    # image is provided — in the normal (no palette) path the argument is silently
    # ignored.  The fix is a two-step approach:
    #   1. Build the palette with a plain quantize() call (dither irrelevant here).
    #   2. Re-quantize the original using that palette so the dither flag is used.
    img_rgb = img.convert("RGB")
    n = max(1, colors)
    dither_flag = Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE
    palette_img = img_rgb.quantize(colors=n, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    quantized = img_rgb.quantize(colors=n, palette=palette_img, dither=dither_flag)
    return quantized.convert("RGBA")


# --- New helpers and k-means quantizer ---

def _pil_to_numpy_rgb_alpha(img: Image.Image):
    """Return (rgb_flat (N,3) float32, alpha_flat (N,), (w,h))."""
    rgba = img.convert("RGBA")
    arr = np.asarray(rgba)
    h, w = arr.shape[:2]
    rgb = arr[..., :3].astype(np.float32)
    alpha = arr[..., 3].astype(np.uint8)
    return rgb.reshape(-1, 3), alpha.reshape(-1), (w, h)


def _numpy_rgba_to_pil(rgb_flat, alpha_flat, size):
    """rgb_flat: (N,3) uint8 or float; alpha_flat: (N,) uint8; size: (w,h)"""
    w, h = size
    rgb = np.clip(rgb_flat, 0, 255).astype(np.uint8).reshape((h, w, 3))
    alpha = alpha_flat.astype(np.uint8).reshape((h, w))
    out = np.empty((h, w, 4), dtype=np.uint8)
    out[..., :3] = rgb
    out[..., 3] = alpha
    return Image.fromarray(out, mode="RGBA")


_BAYER_CACHE: dict[int, np.ndarray] = {}


def _bayer_matrix(order: int) -> np.ndarray:
    """Return a normalized Bayer threshold matrix of shape (2^order, 2^order).

    Values are in [-0.5, 0.5), ready to be scaled and added to pixel values
    as a per-pixel threshold for ordered dithering.
    """
    if order in _BAYER_CACHE:
        return _BAYER_CACHE[order]
    m = np.array([[0.0]], dtype=np.float32)
    for _ in range(order):
        m = np.block([[4 * m,     4 * m + 2],
                      [4 * m + 3, 4 * m + 1]])
    n = (2 ** order) ** 2
    result = (m / n - 0.5).astype(np.float32)
    _BAYER_CACHE[order] = result
    return result


_DITHER_CHUNK = 65536  # max rows per batch when computing the (N, k) distance matrix


def _ordered_dither(rgb_hw3: np.ndarray, palette: np.ndarray, bayer_order: int = 3) -> np.ndarray:
    # Apply Bayer ordered dithering and snap each perturbed pixel to the nearest palette color.
    
    h, w = rgb_hw3.shape[:2]
    k = len(palette)

    # Tile the Bayer threshold matrix to cover the full image
    bayer = _bayer_matrix(bayer_order)               # (B, B)
    bh, bw = bayer.shape
    tiles_h = -(-h // bh)                            # ceil div
    tiles_w = -(-w // bw)
    tiled = np.tile(bayer, (tiles_h, tiles_w))[:h, :w]  # (h, w)

    # Perturb: spread adapts to palette density (denser palette → smaller nudge)
    spread = 255.0 / (2.0 * k)
    perturbed = rgb_hw3 + spread * tiled[:, :, np.newaxis]  # (h, w, 3)
    perturbed = np.clip(perturbed, 0.0, 255.0)

    # Flatten to (N, 3) for nearest-palette look-up
    flat = perturbed.reshape(-1, 3)                   # (N, 3) float32
    n = flat.shape[0]

    # Efficient nearest-color via: ||a-b||² = ||a||² - 2·a·bᵀ + ||b||²
    # Batched over _DITHER_CHUNK rows to keep memory bounded
    pal_sq = np.sum(palette ** 2, axis=1)             # (k,)
    labels = np.empty(n, dtype=np.int32)
    for start in range(0, n, _DITHER_CHUNK):
        chunk = flat[start: start + _DITHER_CHUNK]    # (c, 3)
        chunk_sq = np.sum(chunk ** 2, axis=1, keepdims=True)  # (c, 1)
        dists = chunk_sq - 2.0 * (chunk @ palette.T) + pal_sq  # (c, k)
        labels[start: start + _DITHER_CHUNK] = np.argmin(dists, axis=1)

    rgb_q = np.clip(palette[labels], 0, 255).astype(np.uint8)
    return rgb_q.reshape(h, w, 3)


_KMEANS_CACHE = {}


def quantize_kmeans(img: Image.Image, colors: int = 16, sample_rate: float = 0.1,
                    batch_size: int = 1000, random_state: int = 0, dither: bool = False,
                    cache: bool = True) -> Image.Image:
    """
    Quantize an image using MiniBatchKMeans on RGB pixels (fast for medium/large images).

    Falls back to Pillow adaptive palette when scikit-learn is not installed.
    When ``dither=True``, applies vectorized Bayer (ordered) dithering on top of the
    k-means palette — no sequential pixel loops, fully NumPy.

    Args:
        img: PIL Image
        colors: number of palette colors
        sample_rate: fraction of pixels sampled to fit the k-means (0<sample_rate<=1)
        batch_size: minibatch size for MiniBatchKMeans
        random_state: RNG seed
        dither: if True, apply Bayer ordered dithering after palette quantization

    Returns:
        Quantized PIL Image (RGBA)
    """
    if colors <= 0:
        raise ValueError("colors must be >= 1")

    print(f"[processing] quantize_kmeans start: colors={colors}, dither={dither}, sample_rate={sample_rate}")
    sys.stdout.flush()

    try:
        from sklearn.cluster import MiniBatchKMeans
    except Exception:
        print("[processing] sklearn not available, falling back to Pillow quantize")
        sys.stdout.flush()
        # sklearn not installed — fallback to Pillow quantize
        return quantize_image(img, colors=colors, dither=dither)

    rgb_flat, alpha_flat, (w, h) = _pil_to_numpy_rgb_alpha(img)
    n_pixels = rgb_flat.shape[0]
    print(f"[processing] image size: {w}x{h}, pixels={n_pixels}")
    sys.stdout.flush()

    # If only 1 color requested, shortcut
    if colors == 1:
        mean_color = np.mean(rgb_flat, axis=0, keepdims=True).astype(np.uint8)
        rgb_q = np.repeat(mean_color, n_pixels, axis=0)
        return _numpy_rgba_to_pil(rgb_q, alpha_flat, (w, h))

    # Determine sampled indices
    if sample_rate < 1.0:
        rng = np.random.RandomState(random_state)
        sample_n = max(min(int(n_pixels * sample_rate), n_pixels), 1)
        sample_idx = rng.choice(n_pixels, size=sample_n, replace=False)
        sample = rgb_flat[sample_idx]
    else:
        sample = rgb_flat

    # Try to reuse previously fitted centers for identical parameters + similar sampled data
    centers = None
    if cache:
        # create a lightweight summary of the sampled data (means/vars rounded)
        sample_mean = tuple(np.round(sample.mean(axis=0)).astype(int).tolist())
        sample_var = tuple(np.round(sample.var(axis=0)).astype(int).tolist())
        key = (colors, min(batch_size, max(1, sample.shape[0])), random_state, sample.shape[0], sample_mean, sample_var)
        cached = _KMEANS_CACHE.get(key)
        if cached is not None:
            centers = cached

    if centers is None:
        # Fit MiniBatchKMeans on sampled data (colors in 0..255)
        print(f"[processing] fitting MiniBatchKMeans on sample.shape={sample.shape}")
        sys.stdout.flush()
        try:
            kmeans = MiniBatchKMeans(n_clusters=colors, batch_size=min(batch_size, max(1, sample.shape[0])),
                                     random_state=random_state)
            kmeans.fit(sample)
            centers = kmeans.cluster_centers_.astype(np.uint8)
            if cache:
                _KMEANS_CACHE[key] = centers
            print("[processing] kmeans fit complete")
            sys.stdout.flush()
        except Exception as e:
            print("[processing] kmeans fit failed:", e)
            traceback.print_exc()
            sys.stdout.flush()
            return quantize_image(img, colors=colors, dither=dither)

    if dither:
        # Vectorized Bayer ordered dithering: perturb each pixel with the threshold
        # matrix then snap to the nearest palette color — no sequential loops.
        rgb_hw3 = rgb_flat.reshape(h, w, 3)
        rgb_dithered = _ordered_dither(rgb_hw3, centers.astype(np.float32))
        rgb_q = rgb_dithered.reshape(-1, 3)
    else:
        # Plain nearest-palette label assignment (no dithering)
        diffs = rgb_flat[:, None, :] - centers[None, :, :]
        labels = np.argmin(np.sum(diffs * diffs, axis=2), axis=1)
        rgb_q = centers[labels]

    print("[processing] quantize_kmeans completed reconstruction")
    sys.stdout.flush()

    return _numpy_rgba_to_pil(rgb_q, alpha_flat, (w, h))


def process_image(img: Image.Image, scale: float = 1.0, blur_radius: float = 0.0, colors: int = 16, dither: bool = True, use_pillow_dither: bool = False) -> Image.Image:
    """Run a minimal pipeline: resample -> blur -> quantize

    Args:
        img: PIL Image
        scale: scale factor (1.0 = native)
        blur_radius: gaussian blur radius
        colors: number of palette colors
        dither: apply dithering
        use_pillow_dither: if True, force Floyd–Steinberg via Pillow even when sklearn
            is available; if False, use Bayer k-means when sklearn is present.

    Returns:
        Processed PIL Image (RGBA)
    """
    img = img.copy()
    img = resample_image(img, scale)
    img = blur_image(img, blur_radius)
    if use_pillow_dither or not sklearn_available():
        img = quantize_image(img, colors=colors, dither=dither)
    else:
        try:
            img = quantize_kmeans(img, colors=colors, sample_rate=0.1, batch_size=1000, random_state=0, dither=dither)
        except Exception:
            img = quantize_image(img, colors=colors, dither=dither)
    return img
