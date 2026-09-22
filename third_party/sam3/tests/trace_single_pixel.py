#!/usr/bin/env python3
"""Trace a single pixel through both C++ and torch bilinear resize."""
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

# Target pixel: (y=36, x=816, c=2) where cpp=110, py=113

image_path = "tests/test_random.jpg"
img = Image.open(image_path).convert("RGB")
src = np.array(img, dtype=np.uint8)  # [480, 640, 3] HWC
src_h, src_w = src.shape[:2]
img_size = 1008
c = 2

sx = src_w / img_size
sy = src_h / img_size

y, x = 36, 816

# ═══ C++ path (double precision, exact) ═══
fy = (y + 0.5) * sy - 0.5
if fy < 0: fy = 0.0
y0 = int(fy)
y1 = y0 + 1 if y0 < src_h - 1 else y0
wy = fy - y0
wy0 = 1.0 - wy

fx = (x + 0.5) * sx - 0.5
if fx < 0: fx = 0.0
x0 = int(fx)
x1 = x0 + 1 if x0 < src_w - 1 else x0
wx = fx - x0
wx0 = 1.0 - wx

p00 = float(src[y0, x0, c])
p01 = float(src[y0, x1, c])
p10 = float(src[y1, x0, c])
p11 = float(src[y1, x1, c])

v_double = wy0 * (wx0 * p00 + wx * p01) + wy * (wx0 * p10 + wx * p11)
result_cpp = int(v_double + 0.5)

print("═══ Pixel Trace: (y=36, x=816, c=2) ═══")
print(f"\nSource image: {src_w}x{src_h}")
print(f"sx={sx:.15f}, sy={sy:.15f}")
print(f"fy={fy:.15f}, fx={fx:.15f}")
print(f"y0={y0}, y1={y1}, wy={wy:.15f}, wy0={wy0:.15f}")
print(f"x0={x0}, x1={x1}, wx={wx:.15f}, wx0={wx0:.15f}")
print(f"p00={p00}, p01={p01}, p10={p10}, p11={p11}")
print(f"v(double)={v_double:.15f}")
print(f"result_cpp = int({v_double:.15f} + 0.5) = {result_cpp}")

# ═══ torch path ═══
img_tensor = v2.functional.to_image(img)  # uint8 CHW
float_input = img_tensor.float()  # [3, 480, 640]

# Check: what float values does torch see at the source positions?
print(f"\ntorch float input at (c={c}, y0={y0}, x0={x0}): {float_input[c, y0, x0].item()}")
print(f"torch float input at (c={c}, y0={y0}, x1={x1}): {float_input[c, y0, x1].item()}")
print(f"torch float input at (c={c}, y1={y1}, x0={x0}): {float_input[c, y1, x0].item()}")
print(f"torch float input at (c={c}, y1={y1}, x1={x1}): {float_input[c, y1, x1].item()}")

# Run torch F.interpolate
fi = torch.nn.functional.interpolate(
    float_input.unsqueeze(0), size=(img_size, img_size),
    mode='bilinear', align_corners=False, antialias=False
).squeeze(0)

torch_float_result = fi[c, y, x].item()
torch_uint8_result = fi[c, y, x].clamp(0, 255).round().to(torch.uint8).item()

print(f"\ntorch float result: {torch_float_result:.15f}")
print(f"torch uint8 result: {torch_uint8_result}")
print(f"C++ uint8 result:   {result_cpp}")
print(f"difference:         {torch_uint8_result - result_cpp}")

# ═══ Manual torch replication (float32 step by step) ═══
# Replicate torch's exact computation in float32
rheight = np.float64(src_h) / np.float64(img_size)
rwidth = np.float64(src_w) / np.float64(img_size)

h1r = rheight * (y + 0.5) - 0.5
if h1r < 0: h1r = 0.0
h1 = int(h1r)
h1p = 1 if h1 < src_h - 1 else 0
h1lambda = np.float32(h1r - h1)
h0lambda = np.float32(1.0) - h1lambda

w1r = rwidth * (x + 0.5) - 0.5
if w1r < 0: w1r = 0.0
w1 = int(w1r)
w1p = 1 if w1 < src_w - 1 else 0
w1lambda = np.float32(w1r - w1)
w0lambda = np.float32(1.0) - w1lambda

# Read as float32 (matching torch's float conversion)
pos1_0 = np.float32(src[h1, w1, c])
pos1_wp = np.float32(src[h1, w1 + w1p, c])
pos2_0 = np.float32(src[h1 + h1p, w1, c])
pos2_wp = np.float32(src[h1 + h1p, w1 + w1p, c])

# torch formula: h0lambda * (w0lambda * pos1[0] + w1lambda * pos1[wp]) + h1lambda * (w0lambda * pos2[0] + w1lambda * pos2[wp])
manual_val = np.float32(h0lambda * (w0lambda * pos1_0 + w1lambda * pos1_wp) +
                         h1lambda * (w0lambda * pos2_0 + w1lambda * pos2_wp))

print(f"\n═══ Manual float32 replication ═══")
print(f"h0lambda={h0lambda:.10f}, h1lambda={h1lambda:.10f}")
print(f"w0lambda={w0lambda:.10f}, w1lambda={w1lambda:.10f}")
print(f"pos: {pos1_0}, {pos1_wp}, {pos2_0}, {pos2_wp}")
print(f"manual float32 result: {manual_val:.10f}")
print(f"manual uint8: {int(round(float(manual_val)))}")

# ═══ Step by step float32 with explicit intermediate values ═══
print(f"\n═══ Step-by-step float32 ═══")
t1 = np.float32(w0lambda * pos1_0)
t2 = np.float32(w1lambda * pos1_wp)
row0 = np.float32(t1 + t2)
t3 = np.float32(w0lambda * pos2_0)
t4 = np.float32(w1lambda * pos2_wp)
row1 = np.float32(t3 + t4)
t5 = np.float32(h0lambda * row0)
t6 = np.float32(h1lambda * row1)
result_f32 = np.float32(t5 + t6)
print(f"w0*p00={t1}, w1*p01={t2}, row0={row0}")
print(f"w0*p10={t3}, w1*p11={t4}, row1={row1}")
print(f"h0*row0={t5}, h1*row1={t6}, result={result_f32}")
print(f"uint8(round): {int(round(float(result_f32)))}")
print(f"uint8(+0.5):  {int(result_f32 + np.float32(0.5))}")
