#!/usr/bin/env python3
"""
Comprehensive comparison of SAM3 mask decoder tensors: Python reference vs C++ output.
Reports mae, max, rel, cos, p95, p99 for every tensor in every test case.
"""
import os
import sys
import numpy as np

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument("--cpp-base", default="tests/ref_phase6/cpp_out_phase6")
_parser.add_argument("--ref-base", default="tests/ref_phase6")
_args = _parser.parse_args()

REF_BASE = _args.ref_base
CPP_BASE = _args.cpp_base

CASES = ["point_single", "box_only", "box_and_points"]

TENSORS = [
    "sam_pe_sparse",
    "sam_pe_dense",
    "sam_pe_image_pe",
    "sam_dec_image_feats",
    "sam_dec_tokens_initial",
    "sam_dec_block0_queries",
    "sam_dec_block0_keys",
    "sam_dec_block1_queries",
    "sam_dec_block1_keys",
    "sam_dec_final_queries",
    "sam_dec_feat_s1_proj",
    "sam_dec_feat_s0_proj",
    "sam_dec_upscaled",
    "sam_dec_mask_tokens",
    "sam_dec_masks",
    "sam_dec_iou",
    "sam_dec_obj_score",
    "sam_dec_sam_token",
]


def load_tensor(path):
    with open(path + ".shape") as f:
        shape = [int(x) for x in f.read().strip().split(",") if x]
    data = np.fromfile(path + ".bin", dtype=np.float32)
    return data, shape


def compare(name, ref_path, cpp_path):
    ref_data, ref_shape = load_tensor(ref_path)
    cpp_data, cpp_shape = load_tensor(cpp_path)

    if ref_data.size != cpp_data.size:
        return {
            "status": "FAIL",
            "note": f"SIZE MISMATCH ref={ref_data.size} cpp={cpp_data.size}",
            "ref_shape": ref_shape,
            "cpp_shape": cpp_shape,
        }

    diff = np.abs(ref_data - cpp_data)
    eps = 1e-8
    max_d = float(diff.max())
    mae = float(diff.mean())
    rel_err = diff / (np.abs(ref_data) + eps)
    mean_rel = float(rel_err.mean())
    cos = float(
        np.dot(ref_data, cpp_data)
        / (np.linalg.norm(ref_data) * np.linalg.norm(cpp_data) + 1e-12)
    )
    p95 = float(np.percentile(diff, 95))
    p99 = float(np.percentile(diff, 99))

    worst_idx = int(np.argmax(diff))
    worst_ref = float(ref_data[worst_idx])
    worst_cpp = float(cpp_data[worst_idx])

    return {
        "status": "OK",
        "ref_shape": ref_shape,
        "cpp_shape": cpp_shape,
        "mae": mae,
        "max": max_d,
        "rel": mean_rel,
        "cos": cos,
        "p95": p95,
        "p99": p99,
        "worst_idx": worst_idx,
        "worst_ref": worst_ref,
        "worst_cpp": worst_cpp,
        "n_elements": ref_data.size,
    }


def main():
    all_pass = True

    for case in CASES:
        ref_dir = os.path.join(REF_BASE, case)
        cpp_dir = os.path.join(CPP_BASE, case)

        print(f"\n{'='*80}")
        print(f"  Case: {case}")
        print(f"{'='*80}")

        if not os.path.isdir(ref_dir):
            print(f"  [SKIP] Reference dir missing: {ref_dir}")
            continue
        if not os.path.isdir(cpp_dir):
            print(f"  [SKIP] C++ output dir missing: {cpp_dir}")
            continue

        for tensor_name in TENSORS:
            ref_path = os.path.join(ref_dir, tensor_name)
            cpp_path = os.path.join(cpp_dir, tensor_name)

            if not os.path.exists(ref_path + ".bin"):
                print(f"  [SKIP] {tensor_name:40s}  missing ref")
                continue
            if not os.path.exists(cpp_path + ".bin"):
                print(f"  [SKIP] {tensor_name:40s}  missing cpp")
                continue

            r = compare(tensor_name, ref_path, cpp_path)

            if r["status"] != "OK":
                print(f"  [FAIL] {tensor_name:40s}  {r['note']}")
                all_pass = False
                continue

            # Determine pass/fail thresholds
            is_fail = False
            if r["cos"] < 0.999:
                is_fail = True
            if r["max"] > 0.1:
                is_fail = True

            status = "FAIL" if is_fail else "PASS"
            if is_fail:
                all_pass = False

            print(
                f"  [{status}] {tensor_name:40s}  shape={r['ref_shape']}"
            )
            print(
                f"         mae={r['mae']:.6e}  max={r['max']:.6e}  "
                f"rel={r['rel']:.6e}  cos={r['cos']:.8f}  "
                f"p95={r['p95']:.6e}  p99={r['p99']:.6e}"
            )
            if is_fail:
                print(
                    f"         worst_idx={r['worst_idx']}  "
                    f"ref={r['worst_ref']:.6e}  cpp={r['worst_cpp']:.6e}  "
                    f"n_elements={r['n_elements']}"
                )

    print(f"\n{'='*80}")
    print(f"  Overall: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}")
    print(f"{'='*80}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
