#!/usr/bin/env python3
"""Side-by-side comparison: F32 vs F16 mask decoder precision against Python reference."""
import numpy as np
import os

REF_BASE = "tests/ref_phase6"
F32_BASE = "tests/ref_phase6/cpp_out_phase6"
F16_BASE = "tests/ref_phase6/cpp_out_phase6_f16/cpp_out_phase6"

CASES = ["point_single", "box_only", "box_and_points"]

TENSORS = [
    "sam_pe_sparse", "sam_pe_dense", "sam_pe_image_pe",
    "sam_dec_image_feats", "sam_dec_tokens_initial",
    "sam_dec_block0_queries", "sam_dec_block0_keys",
    "sam_dec_block1_queries", "sam_dec_block1_keys",
    "sam_dec_final_queries",
    "sam_dec_feat_s1_proj", "sam_dec_feat_s0_proj",
    "sam_dec_upscaled", "sam_dec_mask_tokens",
    "sam_dec_masks", "sam_dec_iou", "sam_dec_obj_score", "sam_dec_sam_token",
]

def load(path):
    with open(path + ".shape") as f:
        shape = [int(x) for x in f.read().strip().split(",") if x]
    return np.fromfile(path + ".bin", dtype=np.float32), shape

def metrics(ref, cpp):
    diff = np.abs(ref - cpp)
    eps = 1e-8
    return {
        "mae": float(diff.mean()),
        "max": float(diff.max()),
        "rel": float((diff / (np.abs(ref) + eps)).mean()),
        "cos": float(np.dot(ref, cpp) / (np.linalg.norm(ref) * np.linalg.norm(cpp) + 1e-12)),
        "p95": float(np.percentile(diff, 95)),
        "p99": float(np.percentile(diff, 99)),
    }

# Aggregate across all cases
all_f32 = {}
all_f16 = {}

for case in CASES:
    for t in TENSORS:
        ref_path = os.path.join(REF_BASE, case, t)
        f32_path = os.path.join(F32_BASE, case, t)
        f16_path = os.path.join(F16_BASE, case, t)
        if not all(os.path.exists(p + ".bin") for p in [ref_path, f32_path, f16_path]):
            continue
        ref_data, _ = load(ref_path)
        f32_data, _ = load(f32_path)
        f16_data, _ = load(f16_path)
        if ref_data.size != f32_data.size or ref_data.size != f16_data.size:
            continue
        key = (case, t)
        all_f32[key] = metrics(ref_data, f32_data)
        all_f16[key] = metrics(ref_data, f16_data)

# Print per-tensor summary (worst across cases)
print(f"{'Tensor':40s} | {'F32 max':>10s} {'F32 mae':>10s} {'F32 cos':>12s} | {'F16 max':>10s} {'F16 mae':>10s} {'F16 cos':>12s} | {'Ratio max':>10s}")
print("-" * 140)

for t in TENSORS:
    f32_worst_max = 0
    f32_worst_mae = 0
    f32_worst_cos = 1.0
    f16_worst_max = 0
    f16_worst_mae = 0
    f16_worst_cos = 1.0
    found = False
    for case in CASES:
        key = (case, t)
        if key in all_f32:
            found = True
            f32_worst_max = max(f32_worst_max, all_f32[key]["max"])
            f32_worst_mae = max(f32_worst_mae, all_f32[key]["mae"])
            f32_worst_cos = min(f32_worst_cos, all_f32[key]["cos"])
            f16_worst_max = max(f16_worst_max, all_f16[key]["max"])
            f16_worst_mae = max(f16_worst_mae, all_f16[key]["mae"])
            f16_worst_cos = min(f16_worst_cos, all_f16[key]["cos"])
    if not found:
        continue
    ratio = f16_worst_max / f32_worst_max if f32_worst_max > 0 else float('inf')
    print(f"{t:40s} | {f32_worst_max:10.2e} {f32_worst_mae:10.2e} {f32_worst_cos:12.8f} | {f16_worst_max:10.2e} {f16_worst_mae:10.2e} {f16_worst_cos:12.8f} | {ratio:10.1f}x")

# Print per-case detail for key tensors
print(f"\n{'='*120}")
print("Detailed per-case metrics for key decoder tensors")
print(f"{'='*120}")
key_tensors = ["sam_dec_block0_queries", "sam_dec_block1_keys", "sam_dec_final_queries",
               "sam_dec_upscaled", "sam_dec_masks", "sam_dec_iou", "sam_dec_obj_score"]

for case in CASES:
    print(f"\n  Case: {case}")
    print(f"  {'Tensor':34s} | {'':5s} {'mae':>10s} {'max':>10s} {'rel':>10s} {'cos':>12s} {'p95':>10s} {'p99':>10s}")
    print(f"  {'-'*105}")
    for t in key_tensors:
        key = (case, t)
        if key not in all_f32:
            continue
        f32 = all_f32[key]
        f16 = all_f16[key]
        print(f"  {t:34s} | {'f32':5s} {f32['mae']:10.2e} {f32['max']:10.2e} {f32['rel']:10.2e} {f32['cos']:12.8f} {f32['p95']:10.2e} {f32['p99']:10.2e}")
        print(f"  {'':34s} | {'f16':5s} {f16['mae']:10.2e} {f16['max']:10.2e} {f16['rel']:10.2e} {f16['cos']:12.8f} {f16['p95']:10.2e} {f16['p99']:10.2e}")
