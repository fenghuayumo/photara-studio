#!/usr/bin/env python3
"""Compare C++ preprocessed float CHW against Python preprocessed float CHW."""
import numpy as np

ref_dir = "tests/ref_phase3"

# Load Python reference: [1, 3, 1008, 1008] NCHW
py = np.fromfile(f"{ref_dir}/preprocessed.bin", dtype=np.float32)
py_shape = [int(x) for x in open(f"{ref_dir}/preprocessed.shape").read().strip().split(",")]
py = py.reshape(py_shape)
print(f"Python preprocessed: shape={py.shape}")

# Load C++ preprocessed: [1, 3, 1008, 1008] NCHW
cpp = np.fromfile(f"{ref_dir}/cpp_preprocessed.bin", dtype=np.float32)
cpp_shape = [int(x) for x in open(f"{ref_dir}/cpp_preprocessed.shape").read().strip().split(",")]
cpp = cpp.reshape(cpp_shape)
print(f"C++ preprocessed: shape={cpp.shape}")

# Compare
diff = np.abs(py - cpp)
print(f"\n═══ Preprocessed Float Comparison ═══")
print(f"  max_diff:  {diff.max():.6e}")
print(f"  mean_diff: {diff.mean():.6e}")
cos = np.dot(py.flatten(), cpp.flatten()) / (np.linalg.norm(py.flatten()) * np.linalg.norm(cpp.flatten()))
print(f"  cosine:    {cos:.10f}")
n_diff = (diff > 0).sum()
print(f"  elements differing (>0): {n_diff} / {diff.size} ({100*n_diff/diff.size:.2f}%)")

# Since normalization is (x/255 - 0.5)/0.5, a 1 uint8 diff = 1/255 * 2 = 0.00784
expected_1px = 1.0 / 255.0 / 0.5  # 0.00784
print(f"\n  Expected diff for 1 uint8 pixel: {expected_1px:.6f}")
n_1px = (diff > expected_1px * 0.5).sum()
print(f"  Elements with diff > 0.5 pixel: {n_1px} ({100*n_1px/diff.size:.2f}%)")

# Look at where biggest differences are
worst = np.unravel_index(np.argmax(diff), diff.shape)
print(f"\n  Worst at {worst}: py={py[worst]:.6f}, cpp={cpp[worst]:.6f}, diff={diff[worst]:.6f}")
print(f"  That's {diff[worst] / expected_1px:.2f} uint8 pixel values")

# Compare uint8 values (undo normalization)
py_uint8 = np.round((py * 0.5 + 0.5) * 255).astype(np.uint8)
cpp_uint8 = np.round((cpp * 0.5 + 0.5) * 255).astype(np.uint8)
uint8_diff = np.abs(py_uint8.astype(np.int16) - cpp_uint8.astype(np.int16))
print(f"\n  Recovered uint8 max pixel diff: {uint8_diff.max()}")
print(f"  Recovered uint8 n_diff: {(uint8_diff > 0).sum()}")
