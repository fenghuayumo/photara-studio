"""Gate the native ViT encoder against the official MoGe PyTorch modules.

Requires a local official MoGe checkout; no downloads or upstream modification.
Outputs are compared at patch embedding, token preparation, every transformer
block, normalized taps, output projections, features and the final class token.
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
              seed: int, precision: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    sys.path.insert(0, str(source.resolve()))
    from moge.model.modules.dinov2_encoder import DINOv2Encoder
    content = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = content["model_config"]["encoder"]
    model = DINOv2Encoder(**config).eval()
    state = {name[len("encoder."):]: tensor for name, tensor in content["model"].items()
             if name.startswith("encoder.")}
    if precision == "f16":
        # Match converted weight values, but keep PyTorch arithmetic F32 so
        # quantization error is separated from graph/layout differences.
        state = {name: tensor.half().float() if name.endswith(".weight") and tensor.ndim >= 2
                 else tensor for name, tensor in state.items()}
    model.load_state_dict(state, strict=True)
    torch.manual_seed(seed)
    image = torch.rand(1, 3, rows * 14, columns * 14)
    normalized = (image - model.image_mean) / model.image_std
    stages: dict[str, np.ndarray] = {}

    def record(name: str, value: torch.Tensor, spatial: bool = False) -> None:
        if spatial:
            value = value.permute(0, 2, 3, 1)
        stages[name] = value.detach().float().cpu().contiguous().numpy().reshape(-1)

    hooks = []
    hooks.append(model.backbone.patch_embed.register_forward_hook(
        lambda module, inputs, output: record("patch_embed", output)))
    for index, block in enumerate(model.backbone.blocks):
        hooks.append(block.register_forward_hook(
            lambda module, inputs, output, i=index: record(f"block.{i}", output)))
    norm_calls = iter(config["intermediate_layers"])
    hooks.append(model.backbone.norm.register_forward_hook(
        lambda module, inputs, output: record(f"normalized.{next(norm_calls)}", output)))
    for index, projection in enumerate(model.output_projections):
        hooks.append(projection.register_forward_hook(
            lambda module, inputs, output, i=index: record(f"projection.{i}", output, True)))
    prepare = model.backbone.prepare_tokens_with_masks

    def traced_prepare(*args, **kwargs):
        output = prepare(*args, **kwargs)
        record("prepared_tokens", output)
        return output

    model.backbone.prepare_tokens_with_masks = traced_prepare
    interpolate = model.backbone.interpolate_pos_encoding

    def traced_position(*args, **kwargs):
        output = interpolate(*args, **kwargs)
        record("position_encoding", output)
        return output

    model.backbone.interpolate_pos_encoding = traced_position
    with torch.inference_mode():
        features, class_token = model(image, rows, columns, return_class_token=True)
    record("features", features, True)
    record("class_token", class_token)
    for hook in hooks:
        hook.remove()
    return normalized.numpy(), stages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("executable", type=Path)
    parser.add_argument("--moge-source", required=True, type=Path)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--columns", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=("f32", "f16"), default="f32")
    parser.add_argument("--vulkan", action="store_true")
    parser.add_argument("--vulkan-precision", choices=("strict", "default"), default="strict",
                        help="Strict mode disables ggml's implicit half-precision Vulkan kernels in the child process")
    parser.add_argument("--max-relative-l2", type=float, default=2e-4)
    parser.add_argument("--max-absolute", type=float, default=2e-3)
    parser.add_argument("--max-relative-max", type=float, default=2e-4,
                        help="Max-error allowance proportional to each reference stage's largest magnitude")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.rows <= 0 or args.columns <= 0 or args.rows * args.columns > 4096:
        parser.error("Grid must contain 1..4096 patches")
    torch.set_num_threads(8)
    normalized, expected = reference(args.checkpoint, args.moge_source,
                                     args.rows, args.columns, args.seed, args.precision)
    results = []
    with tempfile.TemporaryDirectory(prefix="photara-moge-encoder-") as temporary:
        directory = Path(temporary)
        image = directory / "normalized.bin"
        normalized.astype("<f4").tofile(image)
        outputs = directory / "native"
        command = [str(args.executable.resolve()), str(args.gguf.resolve()), str(args.rows),
                   str(args.columns), str(image), str(outputs)]
        if args.vulkan:
            command.append("--vulkan")
        environment = os.environ.copy()
        if args.vulkan and args.vulkan_precision == "strict":
            # Child-only diagnostic policy; never changes the desktop process.
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
            # Late DINOv2 residual blocks can contain values above 300. Report
            # absolute errors, but also scale the maximum-error gate by the
            # stage amplitude; a single fixed absolute tolerance is unsuitable.
            passed = relative <= args.max_relative_l2 and maximum <= \
                args.max_absolute + args.max_relative_max * reference_maximum
            results.append({"stage": name, "relative_l2": relative, "max_absolute": maximum,
                            "reference_maximum": reference_maximum,
                            "passed": passed})
            print(f"{name:20} relative_L2={relative:.3e} max_abs={maximum:.3e} {'PASS' if passed else 'FAIL'}")
    source_revision = subprocess.run(["git", "-C", str(args.moge_source), "rev-parse", "HEAD"],
                                     capture_output=True, text=True, check=True).stdout.strip()
    report = {"source_revision": source_revision, "precision": args.precision,
              "torch_version": torch.__version__, "checkpoint": str(args.checkpoint.resolve()),
              "gguf": str(args.gguf.resolve()),
              "backend": "vulkan" if args.vulkan else "cpu", "rows": args.rows,
              "columns": args.columns, "seed": args.seed,
              "vulkan_precision": args.vulkan_precision if args.vulkan else None,
              "thresholds": {"relative_l2": args.max_relative_l2,
                             "absolute": args.max_absolute, "relative_max": args.max_relative_max},
              "stages": results}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not all(result["passed"] for result in results):
        raise SystemExit("Encoder parity failed")


if __name__ == "__main__":
    main()
