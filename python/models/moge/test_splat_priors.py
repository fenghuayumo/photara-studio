"""Controlled RGB/depth/normal-prior ablation on a local COLMAP capture.

Native MoGe runs on Vulkan in a separate preparation process. No MVS, held-out
RGB supervision, DN, normal field, or multi-view loss is enabled. Reports PSNR,
SSIM, training time and sampled device-memory usage; a gain is not assumed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import time


def subset_colmap(source: Path, output: Path, count: int):
    entries = []
    with (source / "images.bin").open("rb") as stream:
        total, = struct.unpack("<Q", stream.read(8))
        for _ in range(total):
            start = stream.tell()
            header = stream.read(64)
            image_id, = struct.unpack("<I", header[:4])
            name = bytearray()
            while True:
                char = stream.read(1)
                if not char:
                    raise ValueError("Truncated COLMAP image name")
                if char == b"\0":
                    break
                name.extend(char)
            points, = struct.unpack("<Q", stream.read(8))
            stream.seek(points * 24, 1)
            end = stream.tell()
            stream.seek(start)
            entries.append((image_id, bytes(name).decode("utf-8"), stream.read(end-start)))
    entries.sort(key=lambda entry: entry[1])
    count = min(count, len(entries))
    selected = [entries[round(i * (len(entries)-1) / (count-1))] for i in range(count)]
    ids = {entry[0] for entry in selected}
    output.mkdir(parents=True, exist_ok=True)
    (output / "cameras.bin").write_bytes((source / "cameras.bin").read_bytes())
    (output / "images.bin").write_bytes(struct.pack("<Q", len(selected)) + b"".join(entry[2] for entry in selected))
    points = []
    with (source / "points3D.bin").open("rb") as stream:
        total, = struct.unpack("<Q", stream.read(8))
        for _ in range(total):
            header = stream.read(43)
            tracks, = struct.unpack("<Q", stream.read(8))
            observations = stream.read(tracks*8)
            kept = [observations[i:i+8] for i in range(0, len(observations), 8)
                    if struct.unpack("<I", observations[i:i+4])[0] in ids]
            if kept:
                points.append(header + struct.pack("<Q", len(kept)) + b"".join(kept))
    (output / "points3D.bin").write_bytes(struct.pack("<Q", len(points)) + b"".join(points))
    return [entry[1] for entry in selected]


def run(command, directory, environment=None):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "command.json").write_text(json.dumps(command, ensure_ascii=False, indent=2), encoding="utf-8")
    start = time.perf_counter()
    with (directory / "run.log").open("w", encoding="utf-8") as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    text = (directory / "run.log").read_text(encoding="utf-8", errors="replace")
    if process.returncode:
        raise RuntimeError(f"Command failed ({process.returncode}): {text[-4000:]}")
    return text, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="COLMAP binary model directory")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="MoGe-2 ViT-B normal F32 GGUF")
    parser.add_argument("--prepare", type=Path, default=Path("build/photara/Release/photara_splat_prior_prepare.exe"))
    parser.add_argument("--trainer", type=Path, default=Path("build/photara/Release/photara.exe"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--views", type=int, default=96)
    parser.add_argument("--iterations", type=int, default=6000)
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--tokens", type=int, default=1200)
    parser.add_argument("--backend", choices=("cuda", "vulkan"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.views < 16 or args.iterations <= 0 or args.resolution <= 0:
        parser.error("Need at least 16 views and positive iterations/resolution")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_dir = output / "dataset" / "sparse" / "0"
    names = subset_colmap(args.dataset, model_dir, args.views)
    environment = os.environ.copy()
    for key in ("GGML_VK_DISABLE_F16", "GGML_VK_DISABLE_COOPMAT", "GGML_VK_DISABLE_COOPMAT2"):
        environment[key] = "1"
    summary = {"training_backend": args.backend, "inference_backend": "vulkan-f32", "seed": args.seed,
               "iterations": args.iterations, "resolution": args.resolution, "tokens": args.tokens,
               "images": names, "eval_split_every": 8, "initial_budget": 100000, "densification_cap": 300000,
               "trainer_sha256": hashlib.sha256(args.trainer.read_bytes()).hexdigest(), "runs": []}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["preparation_seconds"] = 0.
    variants = [("baseline", 0., 0.), ("depth", .01, 0.), ("normal", 0., .01), ("depth_normal", .01, .01)]
    for name, depth, normal in variants:
        directory = output / name
        prior_dir = directory / "priors"
        preparation_seconds = 0.
        if depth > 0 or normal > 0:
            print("Preparing requested priors for " + name, flush=True)
            _, preparation_seconds = run([str(args.prepare.resolve()), str(model_dir), str(args.images.resolve()),
                str(args.model.resolve()), str(prior_dir), str(args.resolution), str(args.tokens), "8",
                str(depth), str(normal)], directory / "preparation", environment)
            summary["preparation_seconds"] += preparation_seconds
        command = [str(args.trainer.resolve()), "--images", str(args.images.resolve()), "--splat-dataset", str(model_dir),
            "--output", str(directory / "model.ply"), "--splat", "--mesh=false", "--backend", args.backend,
            "--splat-mode", "standard", "--splat-iterations", str(args.iterations), "--splat-log-interval", "500",
            "--splat-use-mask=false", "--splat-focus-region=false", "--splat-max-resolution", str(args.resolution),
            "--splat-progressive-resolution=false", "--splat-sh-degree", "3", "--splat-eval-checkpoints=false",
            "--splat-eval-split-every", "8", "--splat-densification=true", "--splat-init-point-budget", "100000",
            "--splat-densification-cap", "300000", "--splat-ppisp=false", "--splat-bilateral-grid=false",
            "--splat-checkpoint-seconds", "0", "--splat-seed", str(args.seed), "--splat-geometry-loss=false",
            "--splat-normal-field=false", "--splat-filter-mode=classic",
            "--splat-lambda-d", str(depth), "--splat-lambda-n", str(normal)]
        if depth > 0 or normal > 0:
            command += ["--splat-prior-dir", str(prior_dir)]
        print("Training " + name, flush=True)
        text, wall_seconds = run(command, directory)
        match = re.search(r"splat_final_evaluation_views=(\d+) held_out=(\d+) average_psnr=([\d.eE+-]+) average_ssim=([\d.eE+-]+)", text)
        if not match or match[2] != "1":
            raise AssertionError("Missing held-out evaluation")
        psnr, ssim = float(match[3]), float(match[4])
        if not math.isfinite(psnr + ssim):
            raise AssertionError("Nonfinite evaluation")
        training = re.search(r"splat_training_completed iterations=(\d+) training_s=([\d.eE+-]+)", text)
        if not training or int(training[1]) != args.iterations:
            raise AssertionError("Training did not finish requested iterations")
        losses = re.findall(r"\b(?:loss|rgb|depth|normal)=([-+.eE\d]+)", text)
        if not all(math.isfinite(float(value)) for value in losses):
            raise AssertionError("Nonfinite training loss")
        record = {"variant": name, "lambda_d": depth, "lambda_n": normal, "eval_views": int(match[1]),
                  "psnr": psnr, "ssim": ssim, "preparation_seconds": preparation_seconds,
                  "training_seconds": float(training[2]), "wall_seconds": wall_seconds,
                  "sampled_device_used_bytes_max": max((int(value) for value in re.findall(r"\bvram=(\d+)/", text)), default=0)}
        summary["runs"].append(record)
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(record), flush=True)
    baseline = summary["runs"][0]
    for record in summary["runs"]:
        record["psnr_delta"] = record["psnr"] - baseline["psnr"]
        record["ssim_delta"] = record["ssim"] - baseline["ssim"]
        record["training_time_ratio"] = record["training_seconds"] / baseline["training_seconds"]
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Ablation complete; compare summary.json without assuming an improvement", flush=True)


if __name__ == "__main__":
    main()
