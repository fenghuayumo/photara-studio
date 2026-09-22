#!/usr/bin/env python3
"""Check torchvision's uint8 rounding behavior for bilinear resize."""
import torch
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode

# Create a simple test image where we know the exact bilinear result
# Use a 2x2 image resized to 4x4 - the bilinear result should be exactly deterministic
img = torch.tensor([[[100, 200], [50, 150]]], dtype=torch.uint8)  # [1, 2, 2]
print(f"Input: {img}")

# Resize to 4x4
out = resize(img, [4, 4], interpolation=InterpolationMode.BILINEAR, antialias=False)
print(f"Output (antialias=False): {out}")

out_aa = resize(img, [4, 4], interpolation=InterpolationMode.BILINEAR, antialias=True)
print(f"Output (antialias=True): {out_aa}")

# Check rounding: does torch round or truncate?
# For a pixel value that should be 127.5 after bilinear:
# Truncate → 127, Round → 128
img2 = torch.tensor([[[0, 255], [0, 255]]], dtype=torch.uint8)
out2 = resize(img2, [2, 3], interpolation=InterpolationMode.BILINEAR, antialias=False)
print(f"\nRounding test input: {img2}")
print(f"Rounding test output: {out2}")
# Middle column should be (0+255)/2 = 127.5 → check if 127 or 128

# Also check: does torchvision Resize on uint8 go through float internally?
import torchvision
print(f"\ntorchvision version: {torchvision.__version__}")
print(f"torch version: {torch.__version__}")
