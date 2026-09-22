#!/usr/bin/env python3
"""Compare the full pipeline: C++ encode_image (with C++ preprocessing)
vs Python preprocessing → Python ViT.

If the C++ bilinear resize matches PIL closely enough, the ViT outputs
should also be close.
"""
import numpy as np
import sys, os

def load_tensor(path):
    with open(path + ".shape") as f:
        shape = [int(x) for x in f.read().strip().split(",") if x]
    data = np.fromfile(path + ".bin", dtype=np.float32)
    return data, shape

def compare(name, ref, cpp, ref_shape, cpp_shape):
    """Compare flat arrays, handling NCHW vs ggml [C, W, H] transposition."""
    # If shapes suggest NCHW vs ggml, transpose
    if len(cpp_shape) == 3 and len(ref_shape) == 4:
        C, W, H = cpp_shape
        # ggml [C, W, H] flat: c + w*C + h*C*W
        # NCHW flat: c*H*W + h*W + w
        cpp_transposed = np.zeros(C * W * H, dtype=np.float32)
        cpp_3d = cpp.reshape(H, W, C)  # ggml column-major
        # Transpose to [C, H, W] for NCHW comparison
        cpp_nchw = cpp_3d.transpose(2, 0, 1).flatten()
        cpp = cpp_nchw
        ref = ref[:len(cpp)]

    diff = np.abs(ref - cpp)
    max_d = diff.max()
    mean_d = diff.mean()
    cos = np.dot(ref, cpp) / (np.linalg.norm(ref) * np.linalg.norm(cpp) + 1e-12)
    print(f"  {name:40s}  max={max_d:.6e}  mean={mean_d:.6e}  cos={cos:.8f}")


if __name__ == "__main__":
    ref_dir = sys.argv[1] if len(sys.argv) > 1 else "tests/ref_phase3"
    cpp_dir = sys.argv[2] if len(sys.argv) > 2 else "tests/cpp_out_full"

    if not os.path.exists(cpp_dir):
        print(f"C++ output dir '{cpp_dir}' not found")
        print("Run: ./build/tests/test_debug_encoder <model> <ref_dir> <out_dir> --full")
        sys.exit(1)

    print("═══ Full Pipeline Comparison (C++ preprocess + ViT vs Python preprocess + ViT) ═══\n")

    for name in ["vit_output", "neck_det_0", "neck_det_1", "neck_det_2", "neck_det_3"]:
        ref_path = ref_dir + "/" + ("vit_output_bchw" if name == "vit_output" else name)
        cpp_path = cpp_dir + "/" + name
        if not os.path.exists(ref_path + ".bin") or not os.path.exists(cpp_path + ".bin"):
            print(f"  [SKIP] {name}")
            continue
        ref, ref_shape = load_tensor(ref_path)
        cpp, cpp_shape = load_tensor(cpp_path)
        compare(name, ref, cpp, ref_shape, cpp_shape)
