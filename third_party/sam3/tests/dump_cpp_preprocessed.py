#!/usr/bin/env python3
"""Compare C++ vs Python image preprocessing.

The C++ preprocessing:
1. Resize to 1008x1008 using bilinear interpolation (custom implementation)
2. Normalize: (pixel / 255.0 - 0.5) / 0.5 = pixel/127.5 - 1.0

The Python preprocessing (torchvision):
1. v2.Resize(size=(1008, 1008)) — uses PIL's bilinear
2. v2.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])

Differences come from the resize algorithm (C++ custom vs PIL bilinear).
This script replicates the C++ preprocessing in Python for verification.
"""
import numpy as np
import sys
from PIL import Image

def cpp_style_preprocess(image_path, img_size=1008):
    """Replicate C++ sam3_preprocess_image exactly."""
    img = Image.open(image_path).convert("RGB")
    # PIL bilinear resize (this may differ from C++ implementation)
    img_resized = img.resize((img_size, img_size), Image.BILINEAR)
    pixels = np.array(img_resized, dtype=np.float32)  # [H, W, 3]

    # Normalize: (pixel / 255.0 - 0.5) / 0.5
    pixels = (pixels / 255.0 - 0.5) / 0.5

    # Convert HWC → CHW
    chw = pixels.transpose(2, 0, 1)  # [3, H, W]
    return chw

def load_tensor(path):
    with open(path + ".shape") as f:
        shape = [int(x) for x in f.read().strip().split(",") if x]
    data = np.fromfile(path + ".bin", dtype=np.float32)
    return data.reshape(shape)

if __name__ == "__main__":
    ref_dir = sys.argv[1] if len(sys.argv) > 1 else "tests/ref_phase3"

    # Load Python reference
    ref = load_tensor(f"{ref_dir}/preprocessed")
    print(f"Python ref shape: {ref.shape}")
    print(f"  min={ref.min():.6f} max={ref.max():.6f} mean={ref.mean():.6f}")

    # Also load raw float (before normalization) for debugging
    try:
        raw = load_tensor(f"{ref_dir}/preprocessed_raw_float")
        print(f"Python raw float shape: {raw.shape}")
        print(f"  min={raw.min():.6f} max={raw.max():.6f} mean={raw.mean():.6f}")
    except:
        pass

    # Quick sanity: check normalization
    # PyTorch Normalize: (x - 0.5) / 0.5 where x is in [0,1]
    # So output range should be [-1, 1]
    print(f"\n  Expected range: [-1, 1]")
    print(f"  Actual range:   [{ref.min():.6f}, {ref.max():.6f}]")

    # Check first few pixels for debugging
    print(f"\n  First 10 pixels (channel 0): {ref.flatten()[:10]}")
