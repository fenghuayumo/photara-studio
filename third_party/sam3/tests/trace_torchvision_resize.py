#!/usr/bin/env python3
"""Trace exactly what torchvision v2.Resize does on uint8 tensor input.

The goal: figure out the exact algorithm so C++ can replicate it identically.
"""
import numpy as np
import torch
from torchvision.transforms import v2
from PIL import Image

image_path = "tests/test_random.jpg"
img_size = 1008

# Load image
img = Image.open(image_path).convert("RGB")
img_tensor = v2.functional.to_image(img)  # uint8 CHW

# ═══ Test 1: Does v2.Resize on uint8 go through PIL or torch? ═══
# Create a unique pattern that would reveal the algorithm
print("═══ Test 1: PIL vs torch path detection ═══")

# Check if v2.Resize produces the same output as PIL resize
step1 = v2.ToDtype(torch.uint8, scale=True)(img_tensor)
tv_out = v2.Resize(size=(img_size, img_size))(step1)  # uint8 output

# PIL path: convert to PIL Image, resize, convert back
pil_img = Image.fromarray(step1.permute(1, 2, 0).numpy(), "RGB")
pil_resized = pil_img.resize((img_size, img_size), Image.BILINEAR)
pil_out = torch.from_numpy(np.array(pil_resized)).permute(2, 0, 1)

diff = (tv_out.to(torch.int16) - pil_out.to(torch.int16)).abs()
print(f"  v2.Resize vs PIL.resize(BILINEAR): max_diff={diff.max().item()}, n_diff={(diff>0).sum().item()}")

# ═══ Test 2: Does v2.Resize use float internally? ═══
# Resize through float path manually: uint8 → float → interpolate → round → uint8
print("\n═══ Test 2: Manual float interpolation ═══")
float_input = step1.float()  # [3, 480, 640] float32
float_resized = torch.nn.functional.interpolate(
    float_input.unsqueeze(0),  # [1, 3, 480, 640]
    size=(img_size, img_size),
    mode='bilinear',
    align_corners=False,
    antialias=False,
).squeeze(0)  # [3, 1008, 1008]

# Method A: clamp and truncate
manual_uint8_trunc = float_resized.clamp(0, 255).to(torch.uint8)
diff_a = (tv_out.to(torch.int16) - manual_uint8_trunc.to(torch.int16)).abs()
print(f"  vs float→interpolate→clamp→uint8(trunc): max={diff_a.max().item()}, n_diff={(diff_a>0).sum().item()}")

# Method B: clamp, round, then truncate
manual_uint8_round = float_resized.clamp(0, 255).round().to(torch.uint8)
diff_b = (tv_out.to(torch.int16) - manual_uint8_round.to(torch.int16)).abs()
print(f"  vs float→interpolate→clamp→round→uint8:  max={diff_b.max().item()}, n_diff={(diff_b>0).sum().item()}")

# Method C: with antialias=True
float_resized_aa = torch.nn.functional.interpolate(
    float_input.unsqueeze(0),
    size=(img_size, img_size),
    mode='bilinear',
    align_corners=False,
    antialias=True,
).squeeze(0)
manual_uint8_aa = float_resized_aa.clamp(0, 255).round().to(torch.uint8)
diff_c = (tv_out.to(torch.int16) - manual_uint8_aa.to(torch.int16)).abs()
print(f"  vs float→interpolate(aa)→clamp→round→uint8: max={diff_c.max().item()}, n_diff={(diff_c>0).sum().item()}")

# ═══ Test 3: Check exact matching path ═══
print("\n═══ Test 3: Finding exact match ═══")

# Try: PIL → resize directly
# torchvision v2 for uint8 on CPU typically dispatches to PIL
from torchvision.transforms.functional import resize as F_resize
from torchvision.transforms import InterpolationMode

# F.resize with different options
for aa in [True, False]:
    out = F_resize(step1, [img_size, img_size],
                  interpolation=InterpolationMode.BILINEAR, antialias=aa)
    diff = (tv_out.to(torch.int16) - out.to(torch.int16)).abs()
    print(f"  F.resize(antialias={aa}): max={diff.max().item()}, n_diff={(diff>0).sum().item()}")

# ═══ Test 4: Save the float interpolation result (before uint8 conversion) ═══
# This tells us the "ground truth" sub-pixel values
print("\n═══ Test 4: Float interpolation values ═══")
print(f"  float_resized range: [{float_resized.min():.4f}, {float_resized.max():.4f}]")
print(f"  Sample values at (0,0): {float_resized[:, 0, 0].tolist()}")
print(f"  tv_out values at (0,0): {tv_out[:, 0, 0].tolist()}")
print(f"  float rounded at (0,0): {float_resized[:, 0, 0].round().tolist()}")

# ═══ Test 5: What rounding does tv_out use? ═══
# Compare tv_out against the float values to determine rounding
print("\n═══ Test 5: Rounding analysis ═══")
float_vals = float_resized.flatten().numpy()
tv_vals = tv_out.flatten().numpy().astype(np.float32)

# For each pixel, check: is tv_val == floor(float_val) or round(float_val)?
n_floor = 0
n_round = 0
n_ceil = 0
n_other = 0
for i in range(min(len(float_vals), 100000)):
    f = float_vals[i]
    t = tv_vals[i]
    if t == np.floor(f):
        n_floor += 1
    elif t == np.round(f):
        n_round += 1
    elif t == np.ceil(f):
        n_ceil += 1
    else:
        n_other += 1

print(f"  floor: {n_floor}, round: {n_round}, ceil: {n_ceil}, other: {n_other}")
print(f"  (sampled first 100000 pixels)")

# Most likely: v2.Resize on uint8 goes through PIL which uses truncation (floor)
# Let's verify by computing floor explicitly
manual_floor = np.floor(float_vals).astype(np.uint8)
tv_flat = tv_out.flatten().numpy()
floor_match = (manual_floor == tv_flat).sum()
round_match = (np.round(float_vals).astype(np.uint8) == tv_flat).sum()
print(f"\n  Global floor match: {floor_match}/{len(tv_flat)} ({100*floor_match/len(tv_flat):.2f}%)")
print(f"  Global round match: {round_match}/{len(tv_flat)} ({100*round_match/len(tv_flat):.2f}%)")
