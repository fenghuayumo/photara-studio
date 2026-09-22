#!/usr/bin/env python3
"""Compare tensors from Python reference vs C++ output."""
import numpy as np
import sys, os

def load_tensor(path):
    data = np.fromfile(path + ".bin", dtype=np.float32)
    shape_file = path + ".shape"
    if os.path.exists(shape_file):
        with open(shape_file) as f:
            shape = [int(x) for x in f.read().strip().split(",") if x]
        return data.reshape(shape) if shape else data
    return data

def compare(name, ref_path, cpp_path, atol=1e-4):
    ref = load_tensor(ref_path) if isinstance(ref_path, str) else ref_path
    cpp = load_tensor(cpp_path) if isinstance(cpp_path, str) else cpp_path
    ref_f = ref.flatten()
    cpp_f = cpp.flatten()
    if ref_f.size != cpp_f.size:
        print(f"  FAIL {name:40s}  SIZE MISMATCH ref={ref_f.size} cpp={cpp_f.size}")
        return False
    diff = np.abs(ref_f - cpp_f)
    eps = 1e-8
    max_d = diff.max()
    mean_d = diff.mean()
    rel_err = (diff / (np.abs(ref_f) + eps)).mean()
    cos = np.dot(ref_f, cpp_f) / (np.linalg.norm(ref_f) * np.linalg.norm(cpp_f) + 1e-12)
    p95 = np.percentile(diff, 95)
    p99 = np.percentile(diff, 99)
    status = "PASS" if max_d < atol else "FAIL"
    print(f"  {status} {name:40s}  shape={ref.shape}")
    print(f"         mae={mean_d:.6e}  max={max_d:.6e}  rel={rel_err:.6e}  cos={cos:.8f}  p95={p95:.6e}  p99={p99:.6e}")
    if status == "FAIL":
        worst = np.argmax(diff)
        print(f"         worst idx={worst}  ref={ref_f[worst]:.6e}  cpp={cpp_f[worst]:.6e}")
    return status == "PASS"

if __name__ == "__main__":
    ref_dir = sys.argv[1]  # Python dump dir
    cpp_dir = sys.argv[2]  # C++ dump dir
    atol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-4
    ref_tensors = {f[:-4] for f in os.listdir(ref_dir) if f.endswith(".bin")}
    cpp_tensors = {f[:-4] for f in os.listdir(cpp_dir) if f.endswith(".bin")}
    common = sorted(ref_tensors & cpp_tensors)
    if not common:
        print("No common tensors found!")
        ref_only = sorted(ref_tensors - cpp_tensors)
        cpp_only = sorted(cpp_tensors - ref_tensors)
        if ref_only: print(f"  Ref only: {ref_only[:10]}")
        if cpp_only: print(f"  Cpp only: {cpp_only[:10]}")
        sys.exit(1)
    n_pass = 0
    for name in common:
        if compare(name, os.path.join(ref_dir, name), os.path.join(cpp_dir, name), atol):
            n_pass += 1
    print(f"\n{n_pass}/{len(common)} passed (atol={atol})")
