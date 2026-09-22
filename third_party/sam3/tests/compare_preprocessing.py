#!/usr/bin/env python3
"""Compare C++ vs Python image preprocessing.

Replicates the C++ sam3_preprocess_image() in Python and compares
against the torchvision pipeline used by SAM3 Python.
"""
import numpy as np
import sys
from PIL import Image
import torch
from torchvision.transforms import v2


def cpp_preprocess(image_path, img_size=1008):
    """Replicate C++ sam3_preprocess_image exactly.

    C++ does:
    1. Bilinear resize to img_size x img_size (half-pixel center mapping)
    2. Quantize back to uint8 with rounding
    3. Normalize: (pixel/255.0 - 0.5) / 0.5
    4. Layout: CHW
    """
    img = Image.open(image_path).convert("RGB")
    src = np.array(img, dtype=np.float32)  # [H, W, 3]
    src_h, src_w = src.shape[:2]

    # C++ bilinear resize (half-pixel center mapping)
    dst = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    sx = src_w / img_size
    sy = src_h / img_size

    for y in range(img_size):
        fy = (y + 0.5) * sy - 0.5
        y0 = max(0, int(fy))
        y1 = min(src_h - 1, y0 + 1)
        wy = fy - y0
        for x in range(img_size):
            fx = (x + 0.5) * sx - 0.5
            x0 = max(0, int(fx))
            x1 = min(src_w - 1, x0 + 1)
            wx = fx - x0
            for c in range(3):
                v = ((1 - wy) * ((1 - wx) * src[y0, x0, c] + wx * src[y0, x1, c]) +
                     wy * ((1 - wx) * src[y1, x0, c] + wx * src[y1, x1, c]))
                dst[y, x, c] = min(255, max(0, int(v + 0.5)))

    # Normalize and convert to CHW
    result = np.zeros((3, img_size, img_size), dtype=np.float32)
    for c in range(3):
        for y in range(img_size):
            for x in range(img_size):
                v = dst[y, x, c] / 255.0
                result[c, y, x] = (v - 0.5) / 0.5

    return result, dst


def python_preprocess(image_path, img_size=1008):
    """SAM3 Python preprocessing pipeline."""
    img = Image.open(image_path).convert("RGB")
    transform = v2.Compose([
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize(size=(img_size, img_size)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img_tensor = v2.functional.to_image(img)
    return transform(img_tensor).numpy()  # [3, H, W]


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "tests/test_random.jpg"

    print(f"Comparing preprocessing for: {image_path}")
    print()

    # Python preprocessing (torchvision)
    py = python_preprocess(image_path)
    print(f"Python shape: {py.shape}")
    print(f"  min={py.min():.6f} max={py.max():.6f} mean={py.mean():.6f}")

    # C++ preprocessing (replicated)
    print("\nReplicating C++ preprocessing (this is slow in Python)...")
    cpp, cpp_uint8 = cpp_preprocess(image_path)
    print(f"C++ shape: {cpp.shape}")
    print(f"  min={cpp.min():.6f} max={cpp.max():.6f} mean={cpp.mean():.6f}")

    # Compare
    diff = np.abs(py - cpp)
    print(f"\n═══ Comparison ═══")
    print(f"  max_diff:  {diff.max():.6e}")
    print(f"  mean_diff: {diff.mean():.6e}")
    cos = np.dot(py.flatten(), cpp.flatten()) / (np.linalg.norm(py.flatten()) * np.linalg.norm(cpp.flatten()))
    print(f"  cosine:    {cos:.10f}")

    # Pixel-level differences
    n_diff = (diff > 1e-6).sum()
    print(f"  pixels differing: {n_diff} / {diff.size} ({100*n_diff/diff.size:.2f}%)")

    n_diff_1pct = (diff > 0.01).sum()
    print(f"  pixels with |err| > 0.01: {n_diff_1pct}")

    # Sample some differences
    if diff.max() > 1e-6:
        worst = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"\n  Worst pixel at {worst}: py={py[worst]:.6f}, cpp={cpp[worst]:.6f}")

        # Check if the difference is due to resize or normalization
        # PIL resize
        img = Image.open(image_path).convert("RGB")
        pil_resized = img.resize((1008, 1008), Image.BILINEAR)
        pil_arr = np.array(pil_resized, dtype=np.uint8)

        # Compare uint8 resize outputs
        uint8_diff = np.abs(pil_arr.astype(np.int16) - cpp_uint8.astype(np.int16))
        print(f"\n  Resize comparison (uint8):")
        print(f"    max pixel diff: {uint8_diff.max()}")
        print(f"    pixels differing: {(uint8_diff > 0).sum()} / {uint8_diff.size}")
