#!/usr/bin/env python3
"""Compare C++ bilinear resize against Python torchvision resize at uint8 level.

Replicates the exact C++ sam3_resize_bilinear() in Python, then compares
against torchvision v2.Resize output.
"""
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import v2

image_path = "tests/test_random.jpg"
img_size = 1008

# Load original image
img = Image.open(image_path).convert("RGB")
src = np.array(img, dtype=np.float32)  # [H, W, 3]
src_h, src_w = src.shape[:2]
print(f"Original: {src_w}x{src_h}")

# ═══ Method 1: C++ bilinear resize (replicated in Python) ═══
print("\nReplicating C++ resize...")
dst_cpp = np.zeros((img_size, img_size, 3), dtype=np.uint8)
sx = src_w / img_size
sy = src_h / img_size

for y in range(img_size):
    fy = (y + 0.5) * sy - 0.5
    y0 = max(0, int(fy))
    y1 = min(src_h - 1, y0 + 1)
    wy = np.float32(fy - y0)  # Use f32 to match C++
    for x in range(img_size):
        fx = (x + 0.5) * sx - 0.5
        x0 = max(0, int(fx))
        x1 = min(src_w - 1, x0 + 1)
        wx = np.float32(fx - x0)
        for c in range(3):
            v = np.float32((1 - wy) * ((1 - wx) * src[y0, x0, c] + wx * src[y0, x1, c]) +
                           wy * ((1 - wx) * src[y1, x0, c] + wx * src[y1, x1, c]))
            dst_cpp[y, x, c] = min(255, max(0, int(v + 0.5)))

# ═══ Method 2: Python torchvision resize ═══
img_tensor = v2.functional.to_image(img)  # uint8 CHW tensor
step1 = v2.ToDtype(torch.uint8, scale=True)(img_tensor)
step2 = v2.Resize(size=(img_size, img_size))(step1)
dst_py = step2.permute(1, 2, 0).numpy()  # [H, W, 3] uint8

# ═══ Compare ═══
print("\n═══ C++ resize vs Python torchvision resize (uint8) ═══")
diff = np.abs(dst_cpp.astype(np.int16) - dst_py.astype(np.int16))
print(f"  max pixel diff: {diff.max()}")
print(f"  mean pixel diff: {diff.mean():.6f}")
n_diff = (diff > 0).sum()
print(f"  pixels differing: {n_diff} / {diff.size} ({100*n_diff/diff.size:.2f}%)")

# Histogram of differences
for threshold in [0, 1, 2, 3]:
    n = (diff > threshold).sum()
    print(f"  pixels with diff > {threshold}: {n}")

# ═══ Check: does the float normalization match? ═══
print("\n═══ Float normalization check ═══")

# C++ normalization: (pixel / 255.0 - 0.5) / 0.5
cpp_float = (dst_cpp.astype(np.float32) / 255.0 - 0.5) / 0.5  # [H, W, 3]
cpp_chw = cpp_float.transpose(2, 0, 1)  # [3, H, W]

# Python: v2.ToDtype(float32, scale=True) then Normalize(0.5, 0.5)
step3 = v2.ToDtype(torch.float32, scale=True)(step2)  # scale=True divides by 255
step4 = v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(step3)
py_chw = step4.numpy()

# If the uint8 images match, the float should also match
float_diff = np.abs(cpp_chw - py_chw)
print(f"  max float diff: {float_diff.max():.6e}")
print(f"  mean float diff: {float_diff.mean():.6e}")

# ═══ Check: what if we use the SAME uint8 but different normalization? ═══
# Python's ToDtype(float32, scale=True) does pixel / 255.0
# C++ does pixel / 255.0
# Then Python's Normalize does (x - 0.5) / 0.5
# C++ does (v - 0.5) / 0.5
# These should be identical
print("\n═══ Normalization formula check ═══")
# Take a single pixel value and check
test_val = np.uint8(128)
cpp_result = np.float32((np.float32(test_val) / np.float32(255.0) - np.float32(0.5)) / np.float32(0.5))
py_result = np.float32((np.float32(test_val) / np.float32(255.0) - np.float32(0.5)) / np.float32(0.5))
print(f"  test pixel=128: cpp={cpp_result:.10f}, py={py_result:.10f}, diff={abs(cpp_result-py_result):.10e}")

# Conclusion
print("\n═══ Conclusion ═══")
if diff.max() <= 1:
    print("  C++ and Python resizes differ by at most 1 uint8 value.")
    print("  This is a bilinear rounding difference, not a bug.")
    print("  To make them identical, the C++ resize must replicate")
    print("  the exact torchvision/PIL rounding behavior.")
else:
    print(f"  WARNING: C++ and Python resizes differ by up to {diff.max()} uint8 values!")
    print("  This suggests an algorithmic difference, not just rounding.")
