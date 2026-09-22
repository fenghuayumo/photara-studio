#!/usr/bin/env python3
"""
Compare C++ vs Python SAM2 video tracking outputs.

Compares:
1. Backbone features (neck_trk_2) — are images encoded the same?
2. Output masks — do they segment the same region?
3. Memory-conditioned features — is memory attention working?
"""
import numpy as np
import os, sys
from PIL import Image

py_dir = "/tmp/debug_sam2_pipeline"
cpp_dir = "/tmp/debug_sam2_cpp"

def load_tensor(path):
    data = np.fromfile(path + ".bin", dtype=np.float32)
    if os.path.exists(path + ".shape"):
        with open(path + ".shape") as f:
            shape = [int(x) for x in f.read().strip().split(",") if x]
        return data.reshape(shape) if shape else data
    return data

def compare(name, ref, cpp, atol=1e-3):
    ref_f = ref.flatten()
    cpp_f = cpp.flatten()
    if ref_f.size != cpp_f.size:
        print(f"  FAIL {name:50s} SIZE MISMATCH ref={ref_f.size} cpp={cpp_f.size}")
        return False
    diff = np.abs(ref_f - cpp_f)
    max_d = diff.max()
    mean_d = diff.mean()
    cos = np.dot(ref_f, cpp_f) / (np.linalg.norm(ref_f) * np.linalg.norm(cpp_f) + 1e-12)
    p95 = np.percentile(diff, 95)
    p99 = np.percentile(diff, 99)
    status = "PASS" if cos > 0.99 else "FAIL"
    print(f"  {status} {name:50s}")
    print(f"       mae={mean_d:.6e}  max={max_d:.6e}  cos={cos:.8f}  p95={p95:.6e}  p99={p99:.6e}")
    return status == "PASS"

def mask_iou(m1, m2):
    """Compute IoU between two binary masks."""
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return intersection / max(union, 1)

print("=" * 80)
print("SAM2 Video Tracking: Python vs C++ Comparison")
print("=" * 80)

# Compare vision features (backbone output)
# Python vision_feat_0 is [4096, 1, 256] (HW, B, C) = flattened (64, 64, 256)
# C++ neck_trk_2 is [256, 64, 64] in ggml
print("\n--- Backbone Features (neck_trk_2 / vision_feat_0) ---")
for fi in range(6):
    py_path = os.path.join(py_dir, f"f{fi}_vision_feat_0")
    cpp_path = os.path.join(cpp_dir, f"f{fi}_neck_trk_2")
    if not os.path.exists(py_path + ".bin") or not os.path.exists(cpp_path + ".bin"):
        print(f"  Frame {fi}: SKIP (missing tensors)")
        continue

    py_data = load_tensor(py_path)  # [4096, 1, 256]
    cpp_data = load_tensor(cpp_path)  # [256, 64, 64] in ggml

    print(f"  Frame {fi}: py={py_data.shape} ({py_data.size} floats), cpp={cpp_data.shape} ({cpp_data.size} floats)")

    # The Python tensor is [HW, B, C] = [4096, 1, 256]
    # The C++ tensor is ggml [D, H, W] = [256, 64, 64]
    # Both have 256*64*64 = 1048576 floats

    # Flatten and compare (cosine similarity should be near 1 if they match)
    compare(f"frame{fi}_backbone", py_data, cpp_data)

# Compare output masks
print("\n--- Output Masks ---")
for fi in range(1, 6):
    py_mask_path = os.path.join(py_dir, f"f{fi}_output_mask")
    cpp_mask_path = os.path.join(cpp_dir, f"f{fi}_output_mask_binary")

    if not os.path.exists(py_mask_path + ".bin") or not os.path.exists(cpp_mask_path + ".bin"):
        print(f"  Frame {fi}: SKIP")
        continue

    py_mask = load_tensor(py_mask_path)  # [1, 540, 960] logits
    cpp_mask = load_tensor(cpp_mask_path)  # [540, 960] binary

    py_binary = (py_mask.flatten() > 0.0).astype(float)
    cpp_binary = cpp_mask.flatten()

    iou = mask_iou(py_binary > 0.5, cpp_binary > 0.5)
    py_fg = (py_binary > 0.5).sum()
    cpp_fg = (cpp_binary > 0.5).sum()

    status = "PASS" if iou > 0.5 else "FAIL"
    print(f"  {status} frame{fi}_mask  IoU={iou:.4f}  py_fg={int(py_fg)}  cpp_fg={int(cpp_fg)}")

# Compare memory-conditioned features
print("\n--- Memory-Conditioned Features ---")
for fi in range(6):
    py_path = os.path.join(py_dir, f"f{fi}_mem_conditioned")
    if not os.path.exists(py_path + ".bin"):
        continue
    py_data = load_tensor(py_path)  # [1, 256, 64, 64]
    print(f"  Frame {fi}: py_mem_conditioned shape={py_data.shape} "
          f"range=[{py_data.min():.4f}, {py_data.max():.4f}]")

# Compare SAM decoder outputs
print("\n--- SAM Decoder Outputs ---")
for fi in range(6):
    for name in ["sam_pred_masks", "sam_ious", "sam_obj_ptr", "sam_obj_score"]:
        path = os.path.join(py_dir, f"f{fi}_{name}")
        if os.path.exists(path + ".bin"):
            data = load_tensor(path)
            print(f"  Frame {fi} {name:20s}: shape={data.shape} "
                  f"range=[{data.min():.4f}, {data.max():.4f}]")

# Print summary
print("\n--- Summary ---")
print("Python ref: 2000-2200 fg pixels/frame, IoU scores 0.88-0.90")
print("If C++ masks have similar fg counts but different locations, memory attention is wrong.")
print("If C++ masks are reasonable, the issue may be in longer sequences (memory drift).")
