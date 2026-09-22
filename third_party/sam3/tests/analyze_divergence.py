#!/usr/bin/env python3
"""Analyze divergence between C++ and Python ViT outputs.

Focuses on understanding the error distribution and whether it's
just f32 precision on large-magnitude values.
"""
import numpy as np
import os, sys

def load_tensor(path):
    with open(path + ".shape") as f:
        shape = [int(x) for x in f.read().strip().split(",") if x]
    data = np.fromfile(path + ".bin", dtype=np.float32)
    return data, shape

ref_dir = sys.argv[1] if len(sys.argv) > 1 else "tests/ref_phase3"
cpp_dir = sys.argv[2] if len(sys.argv) > 2 else "tests/cpp_out"

print("═══ Divergence Analysis ═══\n")

# Analyze per-block error progression
print("Block-by-block error progression:")
print(f"{'Block':>6} {'MaxErr':>12} {'MeanErr':>12} {'MaxVal':>12} {'RelErr':>12} {'Cosine':>12}")
print("-" * 78)

for i in range(32):
    ref_path = f"{ref_dir}/block_{i}_out"
    cpp_path = f"{cpp_dir}/dbg_block_{i}_out"
    if not os.path.exists(ref_path + ".bin") or not os.path.exists(cpp_path + ".bin"):
        continue

    ref, ref_shape = load_tensor(ref_path)
    cpp, cpp_shape = load_tensor(cpp_path)

    # ggml [E, W, H] → NHWC [H, W, E] is the same flat layout for comparison
    diff = np.abs(ref - cpp)
    max_err = diff.max()
    mean_err = diff.mean()
    max_val = max(np.abs(ref).max(), np.abs(cpp).max())
    rel_err = max_err / max_val if max_val > 0 else 0
    cos = np.dot(ref, cpp) / (np.linalg.norm(ref) * np.linalg.norm(cpp) + 1e-12)

    print(f"{i:>6} {max_err:>12.6e} {mean_err:>12.6e} {max_val:>12.4f} {rel_err:>12.6e} {cos:>12.10f}")

# Detailed analysis of the worst element
print("\n═══ Worst Element Analysis ═══\n")
for block in [14, 15, 23, 31]:
    ref_path = f"{ref_dir}/block_{block}_out"
    cpp_path = f"{cpp_dir}/dbg_block_{block}_out"
    if block == 31:
        cpp_path = f"{cpp_dir}/vit_output"
    if not os.path.exists(ref_path + ".bin") or not os.path.exists(cpp_path + ".bin"):
        continue

    ref, ref_shape = load_tensor(ref_path)
    cpp, cpp_shape = load_tensor(cpp_path)

    # For block 31 / vit_output, need NCHW→NHWC transpose
    if block == 31 and len(cpp_shape) == 3:
        # cpp is [C, W, H] ggml → ref is [1, H, W, C] NHWC
        C, W, H = cpp_shape
        cpp_3d = cpp.reshape(H, W, C)
        # ref is [1, H, W, C] NHWC
        ref_3d = ref.reshape(ref_shape[1], ref_shape[2], ref_shape[3])
        diff = np.abs(ref_3d.flatten() - cpp_3d.flatten())
    else:
        diff = np.abs(ref - cpp)

    worst_idx = np.argmax(diff)

    # Convert flat index to spatial coordinates
    # ref is [1, H=72, W=72, E=1024]
    if len(ref_shape) == 4:
        _, H, W, E = ref_shape
    else:
        H, W, E = ref_shape[-3], ref_shape[-2], ref_shape[-1]

    total = H * W * E
    h = worst_idx // (W * E)
    w = (worst_idx % (W * E)) // E
    e = worst_idx % E

    print(f"Block {block}: worst at (h={h}, w={w}, e={e})")
    print(f"  ref={ref[worst_idx]:.6e}, cpp={cpp[worst_idx]:.6e}, diff={diff[worst_idx]:.6e}")
    print(f"  magnitude={abs(ref[worst_idx]):.4f}, rel_err={diff[worst_idx]/abs(ref[worst_idx]):.6e}")

    # Check: how many elements have error > 1e-3?
    n_above_1e3 = (diff > 1e-3).sum()
    n_above_1e2 = (diff > 1e-2).sum()
    print(f"  elements with |err| > 1e-3: {n_above_1e3} / {len(diff)} ({100*n_above_1e3/len(diff):.4f}%)")
    print(f"  elements with |err| > 1e-2: {n_above_1e2} / {len(diff)} ({100*n_above_1e2/len(diff):.4f}%)")
    print()

# Compare block 15 intermediates
print("\n═══ Block 15 Intermediate Comparison ═══\n")
intermediates = [
    ("after_norm1", "norm1"),
    ("q_pre_rope", "Q pre-RoPE"),
    ("k_pre_rope", "K pre-RoPE"),
    ("q_post_rope", "Q post-RoPE"),
    ("k_post_rope", "K post-RoPE"),
    ("v", "V"),
    ("attn_out", "attention output"),
    ("proj_out", "projection output"),
    ("after_attn_residual", "after attn+residual"),
]

for suffix, label in intermediates:
    ref_path = f"{ref_dir}/block_15_{suffix}"
    if not os.path.exists(ref_path + ".bin"):
        continue
    ref, ref_shape = load_tensor(ref_path)
    max_val = np.abs(ref).max()
    print(f"  block_15_{suffix:25s} shape={ref_shape}  range=[{ref.min():.4f}, {ref.max():.4f}]  max_abs={max_val:.4f}")
