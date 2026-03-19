import time
import numpy as np
from PIL import Image

from src.processing import quantize_image, quantize_kmeans

def make_img(w, h):
    arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode='RGB')


def time_fn(fn, img, colors, trials=3):
    times = []
    out = None
    for _ in range(trials):
        t0 = time.perf_counter()
        out = fn(img, colors=colors)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times, out


if __name__ == '__main__':
    w, h = 800, 600
    img = make_img(w, h)
    trials = 3
    print(f"Image: {w}x{h}, trials={trials}")

    print('\nRunning Pillow quantize (adaptive palette)...')
    try:
        t_p, out_p = time_fn(quantize_image, img, 16, trials)
        print('Pillow times:', [round(t, 4) for t in t_p])
        out_p.save('out_pillow.png')
    except Exception as e:
        print('Pillow quantize failed:', e)

    print('\nRunning KMeans quantize...')
    try:
        t_k, out_k = time_fn(quantize_kmeans, img, 16, trials)
        print('KMeans times:', [round(t, 4) for t in t_k])
        out_k.save('out_kmeans.png')
    except Exception as e:
        print('KMeans quantize failed:', e)

    print('\nDone.')
