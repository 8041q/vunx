import argparse
from PIL import Image
import numpy as np
from .processing import process_image


def generate_test_image(w=512, h=512):
    # simple RGB gradient
    x = np.linspace(0, 255, w, dtype=np.uint8)
    y = np.linspace(0, 255, h, dtype=np.uint8)
    xv = np.tile(x, (h, 1))
    yv = np.tile(y[:, None], (1, w))
    r = xv
    g = yv
    b = (255 - xv) // 2
    arr = np.dstack([r, g, b]).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def main():
    p = argparse.ArgumentParser(prog="app")
    p.add_argument("--input", "-i", help="Input image path (optional)")
    p.add_argument("--output", "-o", default="out.png", help="Output image path")
    p.add_argument("--scale", type=float, default=1.0, help="Scale factor (e.g., 0.5)")
    p.add_argument("--blur", type=float, default=0.0, help="Gaussian blur radius")
    p.add_argument("--colors", type=int, default=16, help="Number of palette colors")
    p.add_argument("--no-dither", dest="dither", action="store_false", help="Disable dithering")
    args = p.parse_args()

    if args.input:
        img = Image.open(args.input)
    else:
        img = generate_test_image()

    res = process_image(img, scale=args.scale, blur_radius=args.blur, colors=args.colors, dither=args.dither)
    res.save(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
