#!/usr/bin/env python3
"""Compare the ACTUAL C++ uint8 dump against torch F.interpolate output."""
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

image_path = "tests/test_random.jpg"
img_size = 1008

# Load image for torch
img = Image.open(image_path).convert("RGB")
img_tensor = v2.functional.to_image(img)  # uint8 CHW [3, 480, 640]

# ═══ Ground truth: v2.Resize (antialias=True) ═══
tv_resized = v2.Resize(size=(img_size, img_size))(img_tensor)
tv_hwc = tv_resized.permute(1, 2, 0).numpy()  # [H, W, 3] uint8

# ═══ F.interpolate (antialias=False) ═══
float_input = img_tensor.float()
fi_resized = torch.nn.functional.interpolate(
    float_input.unsqueeze(0), size=(img_size, img_size),
    mode='bilinear', align_corners=False, antialias=False
).squeeze(0)
fi_uint8 = fi_resized.clamp(0, 255).round().to(torch.uint8)
fi_hwc = fi_uint8.permute(1, 2, 0).numpy()

# ═══ F.interpolate (antialias=True) ═══
fi_aa_resized = torch.nn.functional.interpolate(
    float_input.unsqueeze(0), size=(img_size, img_size),
    mode='bilinear', align_corners=False, antialias=True
).squeeze(0)
fi_aa_uint8 = fi_aa_resized.clamp(0, 255).round().to(torch.uint8)
fi_aa_hwc = fi_aa_uint8.permute(1, 2, 0).numpy()

# ═══ Load ACTUAL C++ dump ═══
cpp_hwc = np.fromfile("tests/ref_phase3/cpp_resized_uint8.bin", dtype=np.uint8).reshape(img_size, img_size, 3)

print("═══ Comparisons ═══\n")

for name, ref in [("v2.Resize", tv_hwc),
                   ("F.interp(aa=False)", fi_hwc),
                   ("F.interp(aa=True)", fi_aa_hwc)]:
    diff = np.abs(cpp_hwc.astype(np.int16) - ref.astype(np.int16))
    n_diff = (diff > 0).sum()
    print(f"  C++ vs {name:22s}: max={diff.max():2d}  n_diff={n_diff:7d} ({100*n_diff/diff.size:.2f}%)"
          f"  >1: {(diff>1).sum():6d}  >2: {(diff>2).sum():4d}")

# Also compare the torch methods against each other
for name, ref, comp in [("v2.Resize vs F(aa=F)", tv_hwc, fi_hwc),
                          ("v2.Resize vs F(aa=T)", tv_hwc, fi_aa_hwc),
                          ("F(aa=F) vs F(aa=T)", fi_hwc, fi_aa_hwc)]:
    diff = np.abs(ref.astype(np.int16) - comp.astype(np.int16))
    n_diff = (diff > 0).sum()
    print(f"  {name:22s}: max={diff.max():2d}  n_diff={n_diff:7d} ({100*n_diff/diff.size:.2f}%)")

# ═══ Save the F.interpolate(aa=True) + round output for C++ to compare against ═══
with open("tests/ref_phase3/resized_uint8_finterp_aa.bin", "wb") as f:
    f.write(fi_aa_hwc.tobytes())
with open("tests/ref_phase3/resized_uint8_finterp_aa.shape", "w") as f:
    f.write(f"{img_size},{img_size},3")
print(f"\nSaved F.interp(aa=True) uint8 output")
