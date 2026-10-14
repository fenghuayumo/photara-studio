"""Check native GeometryPrior recovery against official MoGe forward/infer.

Fixed raw heads isolate output resizing, remapping and camera recovery from
network arithmetic. Native uses a positive-depth bounded scalar fit; upstream
uses scipy LM, so fit parameters are compared with numeric tolerances.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import torch


def official(raw: dict, width: int, height: int, source: Path, focal: float | None) -> dict:
    sys.path.insert(0, str(source.resolve()))
    from moge.model.v2 import MoGeModel

    class Encoder(torch.nn.Module):
        def forward(self, image, rows, columns, return_class_token=False):
            return torch.zeros(1, 768, rows, columns), torch.zeros(1, 768)

    class Head(torch.nn.Module):
        def __init__(self, values):
            super().__init__()
            self.register_buffer("values", torch.from_numpy(values).permute(2, 0, 1).unsqueeze(0))

        def forward(self, features):
            return [self.values]

    class Scale(torch.nn.Module):
        def forward(self, cls):
            return torch.tensor([[raw["log_metric_scale"]]], dtype=torch.float32)

    # Bypass network construction/weights only. The official forward and infer
    # methods still execute all output postprocessing and geometry recovery.
    model = MoGeModel.__new__(MoGeModel)
    torch.nn.Module.__init__(model)
    model.register_parameter("anchor", torch.nn.Parameter(torch.zeros(1)))
    model.remap_output = "exp"
    model.num_tokens_range = [24, 24]
    model.encoder = Encoder()
    model.neck = torch.nn.Identity()
    model.points_head = Head(raw["points"])
    model.normal_head = Head(raw["normals"])
    model.mask_head = Head(raw["mask_logits"][..., None])
    model.scale_head = Scale()
    kwargs = {"fov_x": math.degrees(2 * math.atan(width / (2 * focal)))} if focal else {}
    model.eval()
    result = model.infer(torch.zeros(3, height, width), use_fp16=False, **kwargs)
    return {name: value.numpy() for name, value in result.items()}


def fixture(width: int, height: int, shift: float, focal: float) -> dict:
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    depth = 2 + 0.013 * xx + 0.021 * yy + 0.0003 * xx * yy
    z = depth - shift
    px = (xx - (width - 1) / 2) / focal * depth
    py = (yy - (height - 1) / 2) / focal * depth
    raw_points = np.stack((px / z, py / z, np.log(z)), axis=-1).astype(np.float32)
    normals = np.broadcast_to(np.array([0.2, -0.1, -3], dtype=np.float32), (height, width, 3)).copy()
    mask = np.full((height, width), 5, dtype=np.float32)
    mask[3:6, 8:12] = -5
    return {"points": raw_points, "normals": normals, "mask_logits": mask,
            "log_metric_scale": np.float32(math.log(2))}


def compare(raw: dict, width: int, height: int, source: Path, executable: Path,
            focal: float | None, name: str) -> dict:
    expected = official(raw, width, height, source, focal)
    with tempfile.TemporaryDirectory(prefix="photara-moge-geometry-") as temporary:
        directory = Path(temporary)
        raw_height, raw_width = raw["points"].shape[:2]
        for field in ("points", "normals", "mask_logits", "log_metric_scale"):
            np.asarray(raw[field], dtype="<f4").tofile(directory / (field + ".bin"))
        output = directory / "native"
        command = [str(executable.resolve()), str(directory), str(raw_width), str(raw_height),
                   str(width), str(height), str(output)]
        if focal:
            command.extend(map(str, (focal, focal, (width - 1) / 2, (height - 1) / 2)))
        subprocess.run(command, check=True)
        recovery = list(map(float, (output / "recovery.txt").read_text().split()))
        if recovery[0] != 0:
            raise AssertionError(f"Native recovery failed: {name}")
        valid = np.fromfile(output / "valid.bin", dtype=np.uint8).reshape(height, width) > 0
        if not np.array_equal(valid, expected["mask"]):
            raise AssertionError(f"Validity mismatch: {name}")
        errors = {}
        for field, reference_field, channels in (("depth", "depth", 1), ("points", "points", 3), ("normals", "normal", 3)):
            values = np.fromfile(output / (field + ".bin"), dtype="<f4")
            values = values.reshape(height, width) if channels == 1 else values.reshape(height, width, channels)
            target = expected[reference_field]
            actual_valid = values[valid].astype(np.float64)
            expected_valid = target[valid].astype(np.float64)
            if not len(actual_valid):
                raise AssertionError("Comparison fixture has no valid pixels")
            relative = float(np.linalg.norm(actual_valid - expected_valid) / max(np.linalg.norm(expected_valid), 1e-12))
            maximum = float(np.max(np.abs(actual_valid - expected_valid)))
            if relative > 2e-4 or maximum > 2e-3:
                raise AssertionError(f"{name}/{field}: relative L2={relative}, max abs={maximum}")
            errors[field] = {"relative_l2": relative, "max_absolute": maximum}
        k = expected["intrinsics"]
        expected_k = np.array([k[0, 0] * width, k[1, 1] * height, k[0, 2] * width - 0.5, k[1, 2] * height - 0.5])
        native_k = np.array(recovery[5:9])
        if not np.allclose(native_k, expected_k, rtol=2e-4, atol=2e-3):
            raise AssertionError(f"{name}: recovered intrinsics mismatch")
        print(name, json.dumps(errors))
        return {"name": name, "known_focal": focal, "errors": errors,
                "native_shift": recovery[1], "native_intrinsics": native_k.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("--moge-source", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--raw-dir", type=Path, help="Also check saved native decoder outputs")
    parser.add_argument("--raw-width", type=int, default=96)
    parser.add_argument("--raw-height", type=int, default=64)
    parser.add_argument("--width", type=int, default=84)
    parser.add_argument("--height", type=int, default=56)
    args = parser.parse_args()
    torch.set_num_threads(4)
    results = []
    for shift in (-0.8, 0.0, 0.7):
        raw = fixture(40, 32, shift, 50)
        for known in (False, True):
            results.append(compare(raw, 40, 32, args.moge_source, args.executable,
                                   50 if known else None, f"shift-{shift}-{'known' if known else 'unknown'}"))
    # Exercise output interpolation before the nonlinear exp/normal/sigmoid
    # operations, at both upsampled and downsampled target sizes.
    raw = fixture(20, 16, 0.4, 25)
    for width, height in ((37, 29), (15, 11)):
        results.append(compare(raw, width, height, args.moge_source, args.executable,
                               25 * width / 20, f"resize-{width}x{height}"))
    if args.raw_dir:
        if min(args.raw_width, args.raw_height, args.width, args.height) <= 0:
            parser.error("Map dimensions must be positive")
        raw = {field: np.fromfile(args.raw_dir / (field + ".bin"), dtype="<f4").reshape(
            args.raw_height, args.raw_width, channels)
               for field, channels in (("points", 3), ("normals", 3), ("mask_logits", 1))}
        raw["mask_logits"] = raw["mask_logits"][..., 0]
        raw["log_metric_scale"] = np.fromfile(args.raw_dir / "log_metric_scale.bin", dtype="<f4")[0]
        for known in (False, True):
            results.append(compare(raw, args.width, args.height, args.moge_source, args.executable,
                                   0.75 * args.width if known else None,
                                   "native-heads-known" if known else "native-heads-unknown"))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"torch_version": torch.__version__, "cases": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
