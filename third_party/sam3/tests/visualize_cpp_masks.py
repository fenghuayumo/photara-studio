#!/usr/bin/env python3
"""
Generate overlay images from C++ SAM2 tracking masks.
Also compares with Python masks side-by-side.
"""
import numpy as np
import os, sys
from PIL import Image

sam2_root = os.path.expanduser("~/Documents/sam2")
video_dir = os.path.join(sam2_root, "notebooks/videos/bedroom")
py_dir = "/tmp/debug_sam2_pipeline"
cpp_dir = "/tmp/debug_sam2_cpp"
out_dir = os.path.join(os.path.dirname(__file__), "debug_pipeline")
os.makedirs(out_dir, exist_ok=True)

def load_tensor(path):
    data = np.fromfile(path + ".bin", dtype=np.float32)
    if os.path.exists(path + ".shape"):
        with open(path + ".shape") as f:
            shape = [int(x) for x in f.read().strip().split(",") if x]
        return data.reshape(shape) if shape else data
    return data

for fi in [1, 2, 5, 10, 20, 30, 40, 50]:
    frame_path = os.path.join(video_dir, f"{fi:05d}.jpg")
    if not os.path.exists(frame_path):
        continue
    img = np.array(Image.open(frame_path))

    # C++ binary mask
    cpp_path = os.path.join(cpp_dir, f"f{fi}_output_mask_binary")
    if os.path.exists(cpp_path + ".bin"):
        cpp_mask = load_tensor(cpp_path)
        cpp_binary = cpp_mask > 0.5
        if cpp_binary.shape[0] == img.shape[0] and cpp_binary.shape[1] == img.shape[1]:
            overlay_cpp = img.copy()
            overlay_cpp[cpp_binary] = (overlay_cpp[cpp_binary] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
            Image.fromarray(overlay_cpp).save(os.path.join(out_dir, f"cpp_frame{fi:03d}.png"))
            fg = cpp_binary.sum()
            print(f"Frame {fi:3d}: C++ fg={fg:6d} ({100.0*fg/cpp_binary.size:.1f}%)")

    # Python mask (logits)
    py_path = os.path.join(py_dir, f"f{fi}_output_mask")
    if os.path.exists(py_path + ".bin"):
        py_logits = load_tensor(py_path).squeeze()
        py_binary = py_logits > 0.0
        if py_binary.shape[0] == img.shape[0] and py_binary.shape[1] == img.shape[1]:
            overlay_py = img.copy()
            overlay_py[py_binary] = (overlay_py[py_binary] * 0.5 + np.array([0, 0, 255]) * 0.5).astype(np.uint8)
            Image.fromarray(overlay_py).save(os.path.join(out_dir, f"py_frame{fi:03d}.png"))

    # IoU comparison
    if os.path.exists(cpp_path + ".bin") and os.path.exists(py_path + ".bin"):
        cpp_mask2 = load_tensor(cpp_path)
        py_logits2 = load_tensor(py_path).squeeze()
        c = cpp_mask2.flatten() > 0.5
        p = py_logits2.flatten() > 0.0
        if c.size == p.size:
            inter = np.logical_and(c, p).sum()
            union = np.logical_or(c, p).sum()
            iou = inter / max(union, 1)
            print(f"         IoU(py,cpp)={iou:.4f}")

print(f"\nOverlays saved to {out_dir}/")
