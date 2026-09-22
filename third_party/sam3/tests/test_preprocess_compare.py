#!/usr/bin/env python3
"""Quick comparison: does PIL resize match C++ bilinear resize?

Instead of replicating C++ in Python (slow), we compare:
1. Python torchvision preprocessing output (already saved as ref_phase3/preprocessed)
2. PIL BILINEAR resize → same normalization (what C++ tries to match)

If PIL BILINEAR matches torchvision, then C++ should too (if C++ matches PIL).
"""
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import v2

image_path = "tests/test_random.jpg"
img_size = 1008

# Method 1: torchvision pipeline (what dump_phase3_reference.py uses)
img = Image.open(image_path).convert("RGB")
transform = v2.Compose([
    v2.ToDtype(torch.uint8, scale=True),
    v2.Resize(size=(img_size, img_size)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
img_tensor = v2.functional.to_image(img)
py_tv = transform(img_tensor).numpy()  # [3, 1008, 1008]

# Method 2: PIL resize + same normalization
img_pil = img.resize((img_size, img_size), Image.BILINEAR)
pil_arr = np.array(img_pil, dtype=np.float32)  # [H, W, 3]
# Normalize same way: (pixel/255.0 - 0.5) / 0.5
pil_normalized = (pil_arr / 255.0 - 0.5) / 0.5  # [H, W, 3]
py_pil = pil_normalized.transpose(2, 0, 1)  # → [3, H, W]

# Compare
diff = np.abs(py_tv - py_pil)
print(f"torchvision vs PIL BILINEAR:")
print(f"  max_diff:  {diff.max():.6e}")
print(f"  mean_diff: {diff.mean():.6e}")
cos = np.dot(py_tv.flatten(), py_pil.flatten()) / (
    np.linalg.norm(py_tv.flatten()) * np.linalg.norm(py_pil.flatten()))
print(f"  cosine:    {cos:.10f}")
print(f"  pixels differing (>1e-6): {(diff > 1e-6).sum()} / {diff.size}")

# Method 3: What the actual SAM3 Python code does
# Check what v2.Resize actually uses under the hood
from torchvision.transforms import InterpolationMode
print(f"\n  v2.Resize default interpolation: {v2.Resize(size=(100,100)).interpolation}")
print(f"  (BILINEAR={InterpolationMode.BILINEAR})")

# The key question: does v2.Resize use PIL.BILINEAR or torch.nn.functional.interpolate?
# v2.Resize with antialias=True (default since torchvision 0.17) uses different algorithm
# Let's test with antialias=False to match pure bilinear
transform_no_aa = v2.Compose([
    v2.ToDtype(torch.uint8, scale=True),
    v2.Resize(size=(img_size, img_size), antialias=False),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
py_no_aa = transform_no_aa(v2.functional.to_image(img)).numpy()

diff2 = np.abs(py_tv - py_no_aa)
print(f"\ntorchvision (antialias=True) vs (antialias=False):")
print(f"  max_diff:  {diff2.max():.6e}")
print(f"  mean_diff: {diff2.mean():.6e}")

# If the default has antialias, the C++ simple bilinear will differ
if diff2.max() > 1e-6:
    print("  → antialias IS enabled by default, C++ bilinear will differ from Python")
