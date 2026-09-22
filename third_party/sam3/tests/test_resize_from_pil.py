#!/usr/bin/env python3
"""Feed PIL-decoded pixels through C++'s bilinear resize (replicated in Python
with double precision) and compare against torchvision resize.

This isolates the resize from the JPEG decoder.
"""
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

image_path = "tests/test_random.jpg"
img_size = 1008

# Load with PIL (same decoder as torchvision)
img = Image.open(image_path).convert("RGB")
src = np.array(img, dtype=np.uint8)  # [480, 640, 3] HWC
src_h, src_w = src.shape[:2]

# ═══ Torch resize (ground truth) ═══
img_tensor = v2.functional.to_image(img)
tv = v2.Resize(size=(img_size, img_size))(img_tensor)
tv_hwc = tv.permute(1, 2, 0).numpy()

# ═══ C++ resize replicated (double precision, same source pixels) ═══
sx = float(src_w) / img_size
sy = float(src_h) / img_size

cpp_hwc = np.zeros((img_size, img_size, 3), dtype=np.uint8)
for y in range(img_size):
    fy = (y + 0.5) * sy - 0.5
    if fy < 0: fy = 0.0
    y0 = int(fy)
    y1 = y0 + 1 if y0 < src_h - 1 else y0
    wy = fy - y0
    wy0 = 1.0 - wy
    for x in range(img_size):
        fx = (x + 0.5) * sx - 0.5
        if fx < 0: fx = 0.0
        x0 = int(fx)
        x1 = x0 + 1 if x0 < src_w - 1 else x0
        wx = fx - x0
        wx0 = 1.0 - wx
        for c in range(3):
            p00 = float(src[y0, x0, c])
            p01 = float(src[y0, x1, c])
            p10 = float(src[y1, x0, c])
            p11 = float(src[y1, x1, c])
            v = wy0 * (wx0 * p00 + wx * p01) + wy * (wx0 * p10 + wx * p11)
            iv = int(v + 0.5)
            if iv < 0: iv = 0
            if iv > 255: iv = 255
            cpp_hwc[y, x, c] = iv

# Compare
diff = np.abs(cpp_hwc.astype(np.int16) - tv_hwc.astype(np.int16))
print("═══ Same source pixels: C++ resize (double) vs torchvision ═══")
print(f"  max pixel diff: {diff.max()}")
print(f"  mean pixel diff: {diff.mean():.6f}")
print(f"  pixels differing: {(diff > 0).sum()} / {diff.size} ({100*(diff>0).sum()/diff.size:.2f}%)")
for t in [0, 1, 2]:
    n = (diff > t).sum()
    if n > 0:
        print(f"  pixels with diff > {t}: {n}")

if diff.max() == 0:
    print("\n  PERFECT MATCH! The resize algorithms are identical.")
    print("  The 8% difference was entirely from the JPEG decoder (stb_image vs libjpeg).")
else:
    print(f"\n  {(diff > 0).sum()} pixels still differ — residual rounding difference.")
