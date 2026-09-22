#!/usr/bin/env python3
"""Compare ACTUAL C++ resized uint8 (from sam3_preprocess_image dump)
against Python torchvision v2.Resize uint8 output."""
import numpy as np

# Load C++ resized uint8 [H, W, 3]
cpp = np.fromfile("tests/ref_phase3/cpp_resized_uint8.bin", dtype=np.uint8).reshape(1008, 1008, 3)

# Load Python torchvision resized uint8 [H, W, 3]
py = np.fromfile("tests/ref_phase3/resized_uint8.bin", dtype=np.uint8).reshape(1008, 1008, 3)

print("═══ Actual C++ resize vs Python torchvision resize (uint8) ═══")
print(f"  C++ shape: {cpp.shape}, range: [{cpp.min()}, {cpp.max()}]")
print(f"  Python shape: {py.shape}, range: [{py.min()}, {py.max()}]")

diff = np.abs(cpp.astype(np.int16) - py.astype(np.int16))
print(f"\n  max pixel diff: {diff.max()}")
print(f"  mean pixel diff: {diff.mean():.6f}")
n_diff = (diff > 0).sum()
print(f"  pixels differing: {n_diff} / {diff.size} ({100*n_diff/diff.size:.2f}%)")

for t in [0, 1, 2, 3, 5, 10]:
    n = (diff > t).sum()
    if n > 0:
        print(f"  pixels with diff > {t}: {n}")

# Find worst locations
if diff.max() > 0:
    worst_flat = np.argmax(diff.flatten())
    worst_pos = np.unravel_index(worst_flat, diff.shape)
    print(f"\n  Worst pixel at (y={worst_pos[0]}, x={worst_pos[1]}, c={worst_pos[2]})")
    print(f"    C++={cpp[worst_pos]}, Python={py[worst_pos]}, diff={diff[worst_pos]}")

    # Show some worst locations
    flat_diff = diff.flatten()
    worst_indices = np.argsort(flat_diff)[-10:][::-1]
    print(f"\n  Top 10 worst positions:")
    for idx in worst_indices:
        pos = np.unravel_index(idx, diff.shape)
        print(f"    y={pos[0]:4d} x={pos[1]:4d} c={pos[2]} cpp={cpp[pos]:3d} py={py[pos]:3d} diff={diff[pos]:2d}")

# After fix: compare normalized float outputs
print("\n═══ Normalized float comparison ═══")
cpp_float = (cpp.astype(np.float32) / 255.0 - 0.5) / 0.5
py_float = (py.astype(np.float32) / 255.0 - 0.5) / 0.5
float_diff = np.abs(cpp_float - py_float)
print(f"  max float diff: {float_diff.max():.6e}")
print(f"  mean float diff: {float_diff.mean():.6e}")
cos = np.dot(cpp_float.flatten(), py_float.flatten()) / (
    np.linalg.norm(cpp_float.flatten()) * np.linalg.norm(py_float.flatten()))
print(f"  cosine: {cos:.10f}")
