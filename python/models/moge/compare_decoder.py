"""Compare native neck/heads against official MoGe modules on identical features.

This gate isolates decoder arithmetic from upstream encoder error. It does not
validate raw-image preprocessing, output remapping or camera/depth recovery.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import torch


def reference(checkpoint: Path, source: Path, rows: int, columns: int,
              aspect: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    sys.path.insert(0, str(source.resolve()))
    from moge.model.modules.dinov2_encoder import DINOv2Encoder
    from moge.model.modules.conv_stack import ConvStack
    from moge.model.modules.mlp import MLP
    content = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config, weights = content["model_config"], content["model"]
    encoder = DINOv2Encoder(**config["encoder"]).eval()
    encoder.load_state_dict({name[len("encoder."):]: tensor for name, tensor in weights.items()
                             if name.startswith("encoder.")}, strict=True)
    torch.manual_seed(seed)
    image = torch.rand(1, 3, rows * 14, columns * 14)
    with torch.inference_mode():
        features, class_token = encoder(image, rows, columns, return_class_token=True)
    normalized = ((image - encoder.image_mean) / encoder.image_std).numpy()
    stages: dict[str, np.ndarray] = {}

    def record(name: str, value: torch.Tensor, spatial: bool = True) -> None:
        if spatial:
            value = value.permute(0, 2, 3, 1)
        stages[name] = value.detach().float().cpu().contiguous().numpy().reshape(-1)

    stacks = {}
    hooks = []
    for prefix in ("neck", "points_head", "normal_head", "mask_head"):
        module = ConvStack(**config[prefix]).eval()
        module.load_state_dict({name[len(prefix) + 1:]: tensor for name, tensor in weights.items()
                                if name.startswith(prefix + ".")}, strict=True)
        for level in range(5):
            for members, label in ((module.input_blocks, "input"), (module.res_blocks, "residual"),
                                   (module.output_blocks, "output")):
                hooks.append(members[level].register_forward_hook(
                    lambda m, i, o, name=f"{prefix}.{label}.{level}": record(name, o)))
        for level, resampler in enumerate(module.resamplers):
            for index in range(2):
                hooks.append(resampler[index].register_forward_hook(
                    lambda m, i, o, name=f"{prefix}.resamplers.{level}.{index}": record(name, o)))
        stacks[prefix] = module
    scale_head = MLP(**config["scale_head"]).eval()
    scale_head.load_state_dict({name[len("scale_head."):]: tensor for name, tensor in weights.items()
                               if name.startswith("scale_head.")}, strict=True)
    for stage, index in enumerate((1, 3, 4)):
        hooks.append(scale_head[index].register_forward_hook(
            lambda m, i, o, name=f"scale_head.{stage}": record(name, o, False)))
    # The official geometry helper imports optional utils3d dependencies. The
    # UV definition itself is analytic and independent of network arithmetic.
    span_y = 1 / (1 + aspect ** 2) ** 0.5
    span_x = aspect * span_y
    pyramid = []
    for level in range(5):
        width, height = columns * 2 ** level, rows * 2 ** level
        u = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width)
        v = torch.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height)
        uu, vv = torch.meshgrid(u, v, indexing="xy")
        uv = torch.stack((uu, vv), dim=0).unsqueeze(0)
        record(f"uv.{level}", uv)
        pyramid.append(torch.cat((features, uv), dim=1) if level == 0 else uv)
    with torch.inference_mode():
        neck = stacks["neck"](pyramid)
        points = stacks["points_head"](neck)[-1]
        normals = stacks["normal_head"](neck)[-1]
        mask = stacks["mask_head"](neck)[-1]
        scale = scale_head(class_token)
    record("points", points)
    record("normals", normals)
    record("mask_logits", mask)
    record("log_metric_scale", scale, False)
    for hook in hooks:
        hook.remove()
    input_features = features.permute(0, 2, 3, 1).contiguous().numpy()
    return input_features, class_token.numpy(), normalized, stages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("executable", type=Path)
    parser.add_argument("--moge-source", required=True, type=Path)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--columns", type=int, default=6)
    parser.add_argument("--aspect", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vulkan", action="store_true")
    parser.add_argument("--native-encoder", action="store_true",
                        help="Run the native encoder too, instead of isolating the decoder with reference features")
    parser.add_argument("--vulkan-precision", choices=("strict", "default"), default="strict")
    parser.add_argument("--max-relative-l2", type=float, default=2e-4)
    parser.add_argument("--max-absolute", type=float, default=2e-3)
    parser.add_argument("--max-relative-max", type=float, default=2e-4)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.rows <= 0 or args.columns <= 0 or args.rows * args.columns > 4096:
        parser.error("Grid must contain 1..4096 patches")
    aspect = args.aspect if args.aspect is not None else args.columns / args.rows
    if not np.isfinite(aspect) or aspect <= 0:
        parser.error("Aspect ratio must be finite and positive")
    torch.set_num_threads(8)
    features, cls, normalized, expected = reference(args.checkpoint, args.moge_source,
                                         args.rows, args.columns, aspect, args.seed)
    results = []
    with tempfile.TemporaryDirectory(prefix="photara-moge-decoder-") as temporary:
        directory = Path(temporary)
        features.astype("<f4").tofile(directory / "features.bin")
        cls.astype("<f4").tofile(directory / "class.bin")
        normalized.astype("<f4").tofile(directory / "normalized.bin")
        outputs = directory / "native"
        command = [str(args.executable.resolve()), str(args.gguf.resolve()), str(args.rows), str(args.columns),
                   str(aspect), str(directory / "features.bin"), str(directory / "class.bin"), str(outputs)]
        if args.vulkan:
            command.append("--vulkan")
        if args.native_encoder:
            command.extend(("--encoder-image", str(directory / "normalized.bin")))
        environment = os.environ.copy()
        if args.vulkan and args.vulkan_precision == "strict":
            for option in ("GGML_VK_DISABLE_F16", "GGML_VK_DISABLE_COOPMAT", "GGML_VK_DISABLE_COOPMAT2"):
                environment[option] = "1"
        subprocess.run(command, check=True, env=environment)
        for name, target in expected.items():
            actual = np.fromfile(outputs / (name + ".bin"), dtype="<f4")
            if actual.shape != target.shape or not np.isfinite(actual).all():
                raise AssertionError(f"Invalid native output: {name}")
            difference = actual.astype(np.float64) - target.astype(np.float64)
            relative = float(np.linalg.norm(difference) / max(np.linalg.norm(target.astype(np.float64)), 1e-12))
            maximum = float(np.max(np.abs(difference)))
            reference_maximum = float(np.max(np.abs(target)))
            passed = relative <= args.max_relative_l2 and maximum <= args.max_absolute + args.max_relative_max * reference_maximum
            results.append({"stage": name, "relative_l2": relative, "max_absolute": maximum,
                            "reference_maximum": reference_maximum, "passed": passed})
            print(f"{name:32} relative_L2={relative:.3e} max_abs={maximum:.3e} {'PASS' if passed else 'FAIL'}")
    source_revision = subprocess.run(["git", "-C", str(args.moge_source), "rev-parse", "HEAD"],
                                     capture_output=True, text=True, check=True).stdout.strip()
    report = {"source_revision": source_revision, "torch_version": torch.__version__,
              "backend": "vulkan" if args.vulkan else "cpu", "vulkan_precision": args.vulkan_precision,
              "rows": args.rows, "columns": args.columns, "aspect": aspect, "seed": args.seed,
              "native_encoder": args.native_encoder,
              "thresholds": {"relative_l2": args.max_relative_l2, "absolute": args.max_absolute,
                             "relative_max": args.max_relative_max}, "stages": results}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not all(result["passed"] for result in results):
        raise SystemExit("Decoder parity failed")


if __name__ == "__main__":
    main()
