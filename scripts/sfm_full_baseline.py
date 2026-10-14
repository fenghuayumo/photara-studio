"""Run fixed-track COMPLETE alignment baselines and reference-only evaluation.

Usage: python scripts/sfm_full_baseline.py INPUT_SCENARIOS.json NEW_OUTPUT_DIR
Scenario fields: name, tracks, images, reference; optional positioning/ba
backends (default automatic). Requires psutil for OS peak working-set sampling.
Optional mode (global by default) and cluster_capacity select hierarchical runs.
Never sends reference paths to the reconstruction process. Runs sequentially.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta

import psutil

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "build/photara/Release-fixed-deps"
NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gpu_sample():
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True, creationflags=NO_WINDOW, timeout=10,
        )
        return [[float(value.strip()) for value in line.split(",")] for line in raw.strip().splitlines()]
    except (OSError, subprocess.SubprocessError, ValueError):
        return []


def metrics(path):
    result = {}
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        if re.match(r"^[a-z][a-z0-9_]*(?:\[\d+\])?=", line):
            key, value = line.split("=", 1)
            try:
                result[key] = float(value)
            except ValueError:
                result[key] = value
    return result


def run(command, log, resources, env):
    if log.exists() or resources.exists():
        raise RuntimeError(f"Refusing overwrite: {log}")
    samples, peak = [], 0
    start = time.perf_counter()
    with log.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(list(map(str, command)), cwd=ROOT, env=env,
                                   stdout=output, stderr=subprocess.STDOUT, creationflags=NO_WINDOW)
        observed = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                memory = observed.memory_info()
                peak = max(peak, getattr(memory, "peak_wset", memory.rss))
                samples.append({"seconds": time.perf_counter() - start, "rss_bytes": memory.rss,
                                "private_bytes": getattr(memory, "private", None), "gpu": gpu_sample()})
            except psutil.Error:
                break
            time.sleep(0.5)
        code = process.wait()
    resources.write_text(json.dumps(samples, indent=2), encoding="utf-8")
    return {"command": list(map(str, command)), "exit_code": code,
            "process_wall_seconds": time.perf_counter() - start, "os_peak_working_set_bytes": peak,
            "peak_device_gpu_memory_mib": max((gpu[1] for sample in samples for gpu in sample["gpu"]), default=None)}


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    scenarios = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
    out = Path(sys.argv[2]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PHOTARA_LOG_LEVEL="debug")
    env["PATH"] = str(BIN) + os.pathsep + env["PATH"]
    benchmark = BIN / "photara_sfm_reconstruction_benchmark.exe"
    evaluator = BIN / "photara_sfm_reference_eval.exe"
    diagnostics = BIN / "photara_sfm_position_diagnostics.exe"
    report = {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "started": datetime.now(timezone(timedelta(hours=8))).isoformat(),
              "scope": "Complete alignment from fixed tracks, including final recovery, BA and audits; excludes frontend extraction/matching.",
              "reference_use": "Post-reconstruction evaluation only; reference is a comparison, not absolute ground truth.",
              "gpu_memory_scope": "Device-wide sampled usage, including other applications; not process-exclusive VRAM.",
              "binary_sha256": {path.name: sha256(path) for path in [benchmark, evaluator, diagnostics]}, "runs": []}
    report["source_sha256"] = {name: sha256(ROOT / name) for name in [
        "photara/src/sfm/global_positioning.cpp", "photara/src/sfm/reconstruct.cpp",
        "photara/src/sfm/submap_recovery.cpp", "photara/src/sfm/hierarchical.cpp",
        "photara/src/sfm/rotation_recovery.cpp", "photara/src/tools/sfm_reconstruction_benchmark.cpp",
        "photara/src/tools/sfm_position_diagnostics.cpp", "scripts/sfm_full_baseline.py"]}

    def save():
        temporary = out / "manifest.tmp"
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(out / "manifest.json")

    save()
    failures = 0
    for case in scenarios:
        name = case["name"]
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
            raise ValueError("Scenario name must be a simple filename")
        tracks = (ROOT / case["tracks"]).resolve()
        model = out / (name + ".asfm")
        row = dict(case, tracks_sha256=sha256(tracks), gpu_before=gpu_sample())
        report["runs"].append(row); save()
        print("Starting " + name, flush=True)
        command = [benchmark, tracks, case.get("positioning", "automatic"), model,
                   case.get("ba", "automatic"), case.get("mode", "global")]
        if "cluster_capacity" in case:
            command.append(case["cluster_capacity"])
        reconstruction = run(command, out / (name + ".log"), out / (name + "_resources.json"), env)
        row["reconstruction"] = reconstruction
        row["summary"] = metrics(out / (name + ".log"))
        log = (out / (name + ".log")).read_text(encoding="utf-8", errors="replace")
        row["stages"] = [{"name": match[0], "seconds": float(match[1])} for match in
                         re.findall(r"stage finished: ([\w.]+) elapsed_s=([\d.]+)", log)]
        row["acceptance"] = [line for line in log.splitlines() if "warm-start geometry:" in line]
        row["actual_positioning_paths"] = {
            "cuda_bearing": "CUDA bearing pass=" in log,
            "vulkan_bearing": "Vulkan bearing pass=" in log,
            "cpu_primary": "CPU pass timing: attempt=primary/" in log,
            "cpu_bearing": "CPU pass timing: attempt=only_points/bearing_schur" in log,
        }
        row["actual_ba_backends"] = sorted(set(re.findall(r"bundle.*?backend=(cpu|cuda|vulkan)\b", log)))
        save()
        if reconstruction["exit_code"]:
            failures += 1
        else:
            row["evaluation"] = run([evaluator, case["reference"], case["images"], model, out / (name + "_reference.csv")],
                out / (name + "_eval.txt"), out / (name + "_eval_resources.json"), env)
            row["reference_metrics"] = metrics(out / (name + "_eval.txt"))
            graph_files = list((out / (name + ".graph")).glob("reconstruction-*.bin"))
            if len(graph_files) != 1:
                raise RuntimeError("Expected one final graph checkpoint")
            row["diagnostics"] = run([diagnostics, graph_files[0], out / name], out / (name + "_diagnostics.log"),
                                     out / (name + "_diagnostics_resources.json"), env)
            failures += row["evaluation"]["exit_code"] != 0 or row["diagnostics"]["exit_code"] != 0
        save()
        print(json.dumps({"name": name, "summary": row["summary"],
                          "reference": row.get("reference_metrics", {}),
                          "peak_rss_mib": reconstruction["os_peak_working_set_bytes"] / 1048576}), flush=True)
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
