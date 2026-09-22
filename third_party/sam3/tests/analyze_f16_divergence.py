#!/usr/bin/env python3
"""Analyze where f16 precision breaks down in the ViT blocks."""
import numpy as np
import os

ref_dir = "tests/ref_phase3"
cpp_dir = "tests/cpp_out_f16"

print("═══ f16 C++ vs f32 Python — Block-by-Block Error Progression ═══\n")
print(f"{'Block':>6} {'MaxErr':>12} {'MeanErr':>12} {'MaxVal':>12} {'RelErr':>12} {'Cosine':>14} {'WorstIdx':>10}")
print("-" * 90)

for i in range(32):
    ref_path = f"{ref_dir}/block_{i}_out"
    cpp_path = f"{cpp_dir}/dbg_block_{i}_out"
    if not os.path.exists(ref_path + ".bin") or not os.path.exists(cpp_path + ".bin"):
        continue

    ref = np.fromfile(ref_path + ".bin", dtype=np.float32)
    cpp = np.fromfile(cpp_path + ".bin", dtype=np.float32)

    diff = np.abs(ref - cpp)
    max_err = diff.max()
    mean_err = diff.mean()
    max_val = max(np.abs(ref).max(), np.abs(cpp).max())
    rel_err = max_err / max_val if max_val > 0 else 0
    cos = np.dot(ref, cpp) / (np.linalg.norm(ref) * np.linalg.norm(cpp) + 1e-12)
    worst = np.argmax(diff)

    flag = " <<<" if i in [10, 11, 15, 16] or max_err > 1.0 else ""
    print(f"{i:>6} {max_err:>12.6e} {mean_err:>12.6e} {max_val:>12.4f} {rel_err:>12.6e} {cos:>14.10f} {worst:>10}{flag}")

# Detailed analysis of the jump between block 14 and 15
print("\n═══ Detailed Block 15 Analysis (first global block that breaks) ═══\n")

# Check block 15 intermediates
intermediates = [
    "block_15_after_norm1",
    "block_15_q_pre_rope", "block_15_k_pre_rope",
    "block_15_q_post_rope", "block_15_k_post_rope",
    "block_15_v",
    "block_15_attn_out",
    "block_15_proj_out",
    "block_15_after_attn_residual",
]

for name in intermediates:
    ref_path = f"{ref_dir}/{name}"
    if not os.path.exists(ref_path + ".bin"):
        continue
    ref = np.fromfile(ref_path + ".bin", dtype=np.float32)
    print(f"  {name:40s}  range=[{ref.min():.4f}, {ref.max():.4f}]  "
          f"max_abs={np.abs(ref).max():.4f}")

# Check the worst element across blocks
print("\n═══ Tracking the Worst Element ═══\n")
for i in [10, 11, 12, 13, 14, 15, 16, 20, 23, 30]:
    ref_path = f"{ref_dir}/block_{i}_out"
    cpp_path = f"{cpp_dir}/dbg_block_{i}_out"
    if not os.path.exists(ref_path + ".bin") or not os.path.exists(cpp_path + ".bin"):
        continue

    ref = np.fromfile(ref_path + ".bin", dtype=np.float32)
    cpp = np.fromfile(cpp_path + ".bin", dtype=np.float32)
    diff = np.abs(ref - cpp)
    worst = np.argmax(diff)

    # Convert flat index to spatial coords (ref is [1, 72, 72, 1024] NHWC)
    H, W, E = 72, 72, 1024
    h = worst // (W * E)
    w = (worst % (W * E)) // E
    e = worst % E

    print(f"  Block {i:2d}: worst at (h={h:2d}, w={w:2d}, e={e:4d})  "
          f"ref={ref[worst]:>12.4f}  cpp={cpp[worst]:>12.4f}  "
          f"diff={diff[worst]:>10.4f}  rel={diff[worst]/abs(ref[worst]) if ref[worst] != 0 else 0:.6f}")

# Check: how many elements have error > 1.0?
print("\n═══ Error Distribution by Block ═══\n")
print(f"{'Block':>6} {'> 0.01':>10} {'> 0.1':>10} {'> 1.0':>10} {'> 10.0':>10}")
for i in range(32):
    ref_path = f"{ref_dir}/block_{i}_out"
    cpp_path = f"{cpp_dir}/dbg_block_{i}_out"
    if not os.path.exists(ref_path + ".bin") or not os.path.exists(cpp_path + ".bin"):
        continue

    ref = np.fromfile(ref_path + ".bin", dtype=np.float32)
    cpp = np.fromfile(cpp_path + ".bin", dtype=np.float32)
    diff = np.abs(ref - cpp)

    n01 = (diff > 0.01).sum()
    n1 = (diff > 0.1).sum()
    n10 = (diff > 1.0).sum()
    n100 = (diff > 10.0).sum()
    flag = " <<<" if n10 > 0 else ""
    print(f"{i:>6} {n01:>10} {n1:>10} {n10:>10} {n100:>10}{flag}")
