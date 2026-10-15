"""Validate real RGB -> native GeometryPredictor against official MoGe-2 infer.

Uses local weights/source only; never downloads or modifies the official model.
RGB is decoded once using Pillow and shared losslessly with both paths, so codec
differences cannot hide preprocessing/network errors. Reports retain source and
checkpoint hashes, settings, masks, depth/point errors and normal angles.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def invoke(args, rgb, directory, backend="vulkan", focal=None, preprocess_only=False, lifecycle=False):
    height, width = rgb.shape[:2]
    directory.mkdir(parents=True, exist_ok=True)
    rgb.tofile(directory / "input.rgb")
    command = [str(args.executable.resolve()), str(args.gguf.resolve()),
               str((directory / "input.rgb").resolve()), str(width), str(height),
               str(args.tokens), str(directory.resolve()), backend]
    if focal is not None:
        command += ["--focal", str(focal)]
    if preprocess_only:
        command += ["--preprocess-only"]
    elif lifecycle:
        command += ["--lifecycle"]
    environment = os.environ.copy()
    if backend == "vulkan" and args.vulkan_precision == "strict":
        for key in ("GGML_VK_DISABLE_F16", "GGML_VK_DISABLE_COOPMAT", "GGML_VK_DISABLE_COOPMAT2"):
            environment[key] = "1"
    start = time.perf_counter()
    subprocess.run(command, check=True, env=environment)
    return time.perf_counter() - start


def preprocessing(rgb, directory, tokens):
    height, width = rgb.shape[:2]
    rows, columns = round(math.sqrt(tokens * height / width)), round(math.sqrt(tokens * width / height))
    actual_grid = np.loadtxt(directory / "grid.txt", dtype=int)
    if list(actual_grid) != [rows, columns]:
        raise AssertionError(f"Token grid mismatch: {actual_grid} vs {[rows, columns]}")
    image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255
    expected = F.interpolate(image, (rows * 14, columns * 14), mode="bilinear",
                             align_corners=False, antialias=True)
    expected = (expected - torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]) / \
        torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    actual = np.fromfile(directory / "normalized.bin", dtype="<f4").reshape(expected.shape)
    error = float(np.max(np.abs(actual - expected.numpy())))
    if error > 3e-5 or not np.isfinite(actual).all():
        raise AssertionError(f"Antialias preprocessing max absolute error {error}")
    return {"grid": [rows, columns], "max_absolute": error}


def tight_known_recovery(model, image, tokens, focal):
    """Independent convergence control; does not modify official infer/code.

    Upstream known-focal LM stops at ftol=1e-3. On imperfect real point maps
    this can differ from native's converged scalar minimum. Keep the untouched
    infer comparison AND check against a float64, tightly converged SciPy fit.
    """
    from scipy.optimize import least_squares
    from moge.utils.geometry_torch import normalized_view_plane_uv
    height, width = image.shape[-2:]
    with torch.inference_mode():
        output = model.forward(image[None], tokens)
    points = output["points"][0]
    mask = output["mask"][0] > .5
    xyz = F.interpolate(points.permute(2, 0, 1)[None], (64, 64), mode="nearest")[0].permute(1, 2, 0)
    uv = normalized_view_plane_uv(width, height)
    uv = F.interpolate(uv.permute(2, 0, 1)[None], (64, 64), mode="nearest")[0].permute(1, 2, 0)
    valid = F.interpolate(mask.float()[None, None], (64, 64), mode="nearest")[0, 0] > 0
    xyz, uv = xyz[valid].numpy().astype(np.float64), uv[valid].numpy().astype(np.float64)
    normalized_focal = 2 * focal / math.hypot(width, height)

    def residual(shift):
        return (normalized_focal * xyz[:, :2] / (xyz[:, 2] + shift)[:, None] - uv).reshape(-1)

    fit = least_squares(residual, 0., method="lm", ftol=1e-13, xtol=1e-13, gtol=1e-13)
    if not fit.success or not np.isfinite(fit.x).all():
        raise AssertionError("Independent known-focal convergence control failed")
    scale = output["metric_scale"][0].item()
    depth = (points[..., 2].numpy().astype(np.float64) + fit.x[0]) * scale
    return depth, {"shift": float(fit.x[0]), "objective": float(np.mean(residual(fit.x[0]) ** 2))}


def compare(expected, directory, height, width, args, tight_depth=None):
    valid = np.fromfile(directory / "valid.bin", dtype=np.uint8).reshape(height, width).astype(bool)
    reference_mask = expected["mask"]
    mismatch = float(np.mean(valid != reference_mask))
    common = valid & reference_mask
    if not common.any():
        raise AssertionError("No common valid pixels")
    metrics = {"mask_mismatch_fraction": mismatch, "valid_fraction": float(valid.mean())}
    passed = mismatch <= args.max_mask_mismatch
    for field, name, channels in (("depth", "depth", 1), ("points", "points", 3), ("normals", "normal", 3)):
        shape = (height, width) if channels == 1 else (height, width, channels)
        actual = np.fromfile(directory / (field + ".bin"), dtype="<f4").reshape(shape)
        a, b = actual[common].astype(np.float64), expected[name][common].astype(np.float64)
        relative = float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))
        maximum = float(np.max(np.abs(a - b)))
        metrics[field] = {"relative_l2": relative, "max_absolute": maximum}
        limit = args.max_recovery_relative_l2 if tight_depth is not None and field != "normals" else args.max_relative_l2
        passed &= bool(np.isfinite(a).all() and relative <= limit)
        if field == "depth" and tight_depth is not None:
            control = tight_depth[common]
            error = float(np.linalg.norm(a - control) / max(np.linalg.norm(control), 1e-12))
            metrics["tight_solver_depth_relative_l2"] = error
            passed &= error <= args.max_relative_l2
        if field == "normals":
            a /= np.linalg.norm(a, axis=-1, keepdims=True)
            b /= np.linalg.norm(b, axis=-1, keepdims=True)
            angles = np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a * b, axis=-1)))
            metrics[field].update(mean_degrees=float(angles.mean()), p99_degrees=float(np.quantile(angles, .99)),
                                  max_degrees=float(angles.max()))
            passed &= bool(np.quantile(angles, .99) <= args.max_normal_p99_degrees)
        if field == "depth":
            pixels = np.zeros((height, width), dtype=np.uint8)
            low, high = np.quantile(actual[valid], [.02, .98])
            pixels[valid] = (255 * np.clip((actual[valid] - low) / max(high - low, 1e-12), 0, 1)).astype(np.uint8)
            Image.fromarray(pixels).save(directory / "depth.png")
        elif field == "normals":
            pixels = np.zeros((height, width, 3), dtype=np.uint8)
            pixels[valid] = (255 * np.clip(actual[valid] * .5 + .5, 0, 1)).astype(np.uint8)
            Image.fromarray(pixels).save(directory / "normals.png")
    native_k = np.loadtxt(directory / "intrinsics.txt")
    k = expected["intrinsics"]
    reference_k = np.array([k[0, 0] * width, k[1, 1] * height, k[0, 2] * width - .5, k[1, 2] * height - .5])
    metrics["intrinsics_max_absolute"] = float(np.max(np.abs(native_k - reference_k)))
    passed &= bool(np.allclose(native_k, reference_k, rtol=args.max_relative_l2, atol=2e-3))
    Image.fromarray(valid.astype(np.uint8) * 255).save(directory / "valid.png")
    return metrics, bool(passed)


def write_preview(output: Path, report: dict):
    cases = [case for case in report["cases"] if case["known_focal"] is None]
    tile_width, tile_height, label_height = 240, 360, 28
    sheet = Image.new("RGB", (tile_width * 3, label_height + len(cases) * (tile_height + label_height)), "#182027")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    for column, title in enumerate(("Input RGB", "Depth (near dark)", "Camera-space normals")):
        draw.text((column * tile_width + 8, 4), title, font=font, fill="white")
    for row, case in enumerate(cases):
        y = label_height + row * (tile_height + label_height)
        directory = output / case["output_directory"]
        for column, path in enumerate((Path(case["image"]), directory / "depth.png", directory / "normals.png")):
            with Image.open(path) as original:
                image = original.convert("RGB")
                image.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
                sheet.paste(image, (column * tile_width + (tile_width - image.width) // 2, y))
        draw.text((8, y + tile_height + 3), Path(case["image"]).name, font=font, fill="white")
    sheet.save(output / "preview.jpg", quality=92)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("executable", type=Path)
    parser.add_argument("--moge-source", type=Path, required=True)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=1200)
    parser.add_argument("--backends", nargs="+", choices=("vulkan",), default=["vulkan"])
    parser.add_argument("--vulkan-precision", choices=("strict", "default"), default="strict")
    parser.add_argument("--max-relative-l2", type=float, default=5e-4)
    parser.add_argument("--max-recovery-relative-l2", type=float, default=1e-3,
                        help="Known-focal depth/point tolerance for official LM (ftol=1e-3); tight independent fit must also pass")
    parser.add_argument("--max-mask-mismatch", type=float, default=1e-4)
    parser.add_argument("--max-normal-p99-degrees", type=float, default=.1)
    args = parser.parse_args()
    if not 1 <= args.tokens <= 4096:
        parser.error("Tokens must lie in 1..4096")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    sys.path.insert(0, str(args.moge_source.resolve()))
    from moge.model.v2 import MoGeModel
    model = MoGeModel.from_pretrained(args.checkpoint).float().cpu().eval()
    report = {"source_commit": subprocess.check_output(["git", "-C", str(args.moge_source), "rev-parse", "HEAD"], text=True).strip(),
              "checkpoint_sha256": sha256(args.checkpoint), "gguf_sha256": sha256(args.gguf),
              "torch": torch.__version__, "tokens": args.tokens, "vulkan_precision": args.vulkan_precision,
              "thresholds": {"relative_l2": args.max_relative_l2, "known_focal_recovery_relative_l2": args.max_recovery_relative_l2,
                             "mask_mismatch": args.max_mask_mismatch,
                             "normal_p99_degrees": args.max_normal_p99_degrees},
              "preprocessing": [], "cases": []}
    # Isolate resize math: down/up, identity, mixed-axis, singleton, tie-to-even.
    rng = np.random.default_rng(42)
    for width, height in ((901, 599), (35, 20), (7, 5), (1, 19), (490, 490), (9, 777)):
        rgb = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        directory = args.output / f"resize-{width}x{height}"
        invoke(args, rgb, directory, preprocess_only=True)
        report["preprocessing"].append({"size": [width, height], **preprocessing(rgb, directory, args.tokens)})
    for index, path in enumerate(args.images):
        rgb = np.asarray(Image.open(path).convert("RGB"))
        height, width = rgb.shape[:2]
        image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255
        for known in (False, True):
            focal = .75 * width if known else None
            kwargs = {"fov_x": math.degrees(2 * math.atan(width / (2 * focal)))} if known else {}
            start = time.perf_counter()
            with torch.inference_mode():
                expected = {name: value.cpu().numpy() for name, value in
                            model.infer(image, num_tokens=args.tokens, use_fp16=False, **kwargs).items()}
            reference_seconds = time.perf_counter() - start
            tight_depth, solver_control = tight_known_recovery(model, image, args.tokens, focal) if known else (None, None)
            for backend in args.backends:
                directory = args.output / f"{index}-{path.stem}-{'known' if known else 'unknown'}-{backend}"
                seconds = invoke(args, rgb, directory, backend, focal, lifecycle=index == 0 and not known)
                preprocess = preprocessing(rgb, directory, args.tokens)
                metrics, passed = compare(expected, directory, height, width, args, tight_depth)
                case = {"image": str(path), "image_sha256": sha256(path), "size": [width, height],
                        "output_directory": directory.name,
                        "known_focal": focal, "backend": backend, "preprocessing": preprocess,
                        "reference_seconds": reference_seconds, "native_seconds_including_load_and_io": seconds,
                        "metrics": metrics, "solver_control": solver_control, "passed": passed}
                report["cases"].append(case)
                (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                print(json.dumps(case), flush=True)
    report["passed"] = all(case["passed"] for case in report["cases"])
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_preview(args.output, report)
    if not report["passed"]:
        raise SystemExit("Real-image prediction comparison FAILED; see report.json")


if __name__ == "__main__":
    main()
