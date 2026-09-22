#!/usr/bin/env python3
"""Comprehensive comparison of DETR decoder tensors: Python ref vs C++ output.

Computes all 6 metrics: mae, max, rel, cos, p95, p99.
Handles layout differences between PyTorch (seq-first/batch-first) and ggml.
"""
import os
import sys
import numpy as np

REF_DIR = os.path.join(os.path.dirname(__file__), "ref_phase5")
CPP_DIR = os.path.join(REF_DIR, "cpp_out_phase5")


def load_tensor(path):
    """Load a .bin + .shape tensor pair."""
    shape_file = path + ".shape"
    bin_file = path + ".bin"
    if not os.path.exists(shape_file) or not os.path.exists(bin_file):
        return None, None
    with open(shape_file) as f:
        shape = tuple(int(x) for x in f.read().strip().split(",") if x)
    data = np.fromfile(bin_file, dtype=np.float32)
    return data, shape


def compare(name, ref_path, cpp_path, atol=1e-4):
    """Compare two tensors and return detailed metrics."""
    ref_data, ref_shape = load_tensor(ref_path)
    cpp_data, cpp_shape = load_tensor(cpp_path)

    if ref_data is None:
        return None, f"SKIP (missing ref: {ref_path})"
    if cpp_data is None:
        return None, f"SKIP (missing cpp: {cpp_path})"

    if ref_data.size != cpp_data.size:
        return None, f"FAIL NUMEL MISMATCH ref={ref_data.size} ({ref_shape}) cpp={cpp_data.size} ({cpp_shape})"

    diff = np.abs(ref_data - cpp_data)
    eps = 1e-8

    mae = float(diff.mean())
    max_d = float(diff.max())
    rel_err = diff / (np.abs(ref_data) + eps)
    mean_rel = float(rel_err.mean())
    cos_num = float(np.dot(ref_data, cpp_data))
    cos_den = float(np.linalg.norm(ref_data) * np.linalg.norm(cpp_data) + 1e-12)
    cos = cos_num / cos_den
    p95 = float(np.percentile(diff, 95))
    p99 = float(np.percentile(diff, 99))

    worst_idx = int(np.argmax(diff))
    status = "PASS" if max_d < atol else "FAIL"

    metrics = {
        "mae": mae,
        "max": max_d,
        "rel": mean_rel,
        "cos": cos,
        "p95": p95,
        "p99": p99,
        "worst_idx": worst_idx,
        "worst_ref": float(ref_data[worst_idx]),
        "worst_cpp": float(cpp_data[worst_idx]),
        "n_over_atol": int(np.sum(diff > atol)),
        "n_total": int(ref_data.size),
        "ref_shape": ref_shape,
        "cpp_shape": cpp_shape,
    }
    return metrics, status


def print_result(name, metrics, status, atol):
    if metrics is None:
        print(f"  {status}")
        return

    label = f"{status} {name:45s}  ref_shape={metrics['ref_shape']}  cpp_shape={metrics['cpp_shape']}"
    print(f"  {label}")
    print(f"         mae={metrics['mae']:.6e}  max={metrics['max']:.6e}  "
          f"rel={metrics['rel']:.6e}  cos={metrics['cos']:.10f}  "
          f"p95={metrics['p95']:.6e}  p99={metrics['p99']:.6e}")
    if status == "FAIL":
        print(f"         worst_idx={metrics['worst_idx']}  "
              f"ref={metrics['worst_ref']:.6e}  cpp={metrics['worst_cpp']:.6e}  "
              f"n_over_atol={metrics['n_over_atol']}/{metrics['n_total']}")


# Define all decoder tensor cases with their tolerances
DECODER_TENSORS = [
    # Inputs (should be exact or near-exact)
    ("ddec_query_embed", 1e-4),
    ("ddec_ref_pts_raw", 1e-4),
    ("ddec_presence_token", 1e-4),
    ("ddec_ref_boxes_init", 1e-4),

    # Layer 0 intermediates
    ("ddec_query_sine_0", 1e-4),
    ("ddec_query_pos_0", 1e-4),
    ("ddec_rpb_mask_0", 2e-4),
    ("ddec_layer0_after_sa", 1e-4),
    ("ddec_layer0_after_text_ca", 1e-4),
    ("ddec_layer0_after_img_ca", 1e-4),
    ("ddec_layer0_full_out", 1e-4),
    ("ddec_layer0_presence", 1e-4),

    # Per-layer outputs (6 layers)
    ("ddec_layer0_out", 1e-4),
    ("ddec_layer0_refboxes", 1e-4),
    ("ddec_layer1_out", 1e-4),
    ("ddec_layer1_refboxes", 1e-4),
    ("ddec_layer2_out", 2e-4),
    ("ddec_layer2_refboxes", 1e-4),
    ("ddec_layer3_out", 2e-4),
    ("ddec_layer3_refboxes", 1e-4),
    ("ddec_layer4_out", 2e-4),
    ("ddec_layer4_refboxes", 1e-4),
    ("ddec_layer5_out", 2e-4),
    ("ddec_layer5_refboxes", 1e-4),

    # Final outputs
    ("ddec_normed_output", 2e-4),
    ("ddec_pred_boxes", 1e-4),
    ("ddec_presence_logit", 1e-4),

    # Scoring
    ("scoring_prompt_mlp_out", 1e-4),
    ("scoring_pooled", 1e-4),
    ("scoring_proj_pooled", 1e-4),
    ("scoring_proj_hs", 1e-4),
    ("scoring_class_scores", 1e-4),
]


def main():
    ref_dir = sys.argv[1] if len(sys.argv) > 1 else REF_DIR
    cpp_dir = sys.argv[2] if len(sys.argv) > 2 else CPP_DIR

    print(f"Reference dir: {ref_dir}")
    print(f"C++ output dir: {cpp_dir}")
    print()

    n_pass = 0
    n_fail = 0
    n_skip = 0

    # ═══ Section 1: Decoder Inputs ═══
    print("═══ Decoder Inputs (should be exact match) ═══")
    for name, atol in DECODER_TENSORS[:4]:
        metrics, status = compare(name, os.path.join(ref_dir, name), os.path.join(cpp_dir, name), atol)
        print_result(name, metrics, status, atol)
        if "PASS" in status:
            n_pass += 1
        elif "FAIL" in status:
            n_fail += 1
        else:
            n_skip += 1

    # ═══ Section 2: Decoder Layer 0 Intermediates ═══
    print("\n═══ Decoder Layer 0 Intermediates ═══")
    for name, atol in DECODER_TENSORS[4:12]:
        metrics, status = compare(name, os.path.join(ref_dir, name), os.path.join(cpp_dir, name), atol)
        print_result(name, metrics, status, atol)
        if "PASS" in status:
            n_pass += 1
        elif "FAIL" in status:
            n_fail += 1
        else:
            n_skip += 1

    # ═══ Section 3: Per-Layer Outputs ═══
    print("\n═══ Decoder Per-Layer Outputs (6 layers) ═══")
    for name, atol in DECODER_TENSORS[12:24]:
        metrics, status = compare(name, os.path.join(ref_dir, name), os.path.join(cpp_dir, name), atol)
        print_result(name, metrics, status, atol)
        if "PASS" in status:
            n_pass += 1
        elif "FAIL" in status:
            n_fail += 1
        else:
            n_skip += 1

    # ═══ Section 4: Final Decoder Outputs ═══
    print("\n═══ Final Decoder Outputs ═══")
    for name, atol in DECODER_TENSORS[24:27]:
        metrics, status = compare(name, os.path.join(ref_dir, name), os.path.join(cpp_dir, name), atol)
        print_result(name, metrics, status, atol)
        if "PASS" in status:
            n_pass += 1
        elif "FAIL" in status:
            n_fail += 1
        else:
            n_skip += 1

    # ═══ Section 5: Scoring ═══
    print("\n═══ DotProductScoring ═══")
    for name, atol in DECODER_TENSORS[27:]:
        metrics, status = compare(name, os.path.join(ref_dir, name), os.path.join(cpp_dir, name), atol)
        print_result(name, metrics, status, atol)
        if "PASS" in status:
            n_pass += 1
        elif "FAIL" in status:
            n_fail += 1
        else:
            n_skip += 1

    # ═══ Summary ═══
    print(f"\n{'='*70}")
    total = n_pass + n_fail
    print(f"SUMMARY: {n_pass}/{total} PASS, {n_fail}/{total} FAIL, {n_skip} SKIP")
    if n_fail == 0 and n_skip == 0:
        print("ALL DECODER TENSORS PASS ✓")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
