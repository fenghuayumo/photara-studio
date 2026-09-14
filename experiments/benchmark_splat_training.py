"""Run an isolated splat training benchmark with timestamped NVIDIA samples.

Example (run before/after serially, with other GPU workloads stopped):
  python experiments/benchmark_splat_training.py --exe build/aetherscan/Release/aetherscan.exe \
    --images D:/ScanVideo/ori_img/images --splat-dataset D:/ScanVideo/ori_img \
    --output artifacts/splat_perf/after --iterations 6000 --strategy adc_plus

Utilization is the driver's sampled busy percentage, not kernel occupancy.
The steady window begins at the first progress report at/after --warmup and
ends at the last progress report, excluding dataset ingestion/final export.
Scheduled refinement/evaluation inside that window stays in the results.
"""

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
import re
import statistics
import subprocess
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument(
        "--splat-dataset", type=Path,
        help="Optional external camera dataset; omit it to reuse internal SfM")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=6000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--strategy", default="adc_plus")
    parser.add_argument("--device-cache-mb", type=int)
    parser.add_argument("--view-cache-mb", type=int)
    parser.add_argument(
        "--cache-auto", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--prefetch-views", type=int)
    parser.add_argument("--working-sfm", type=Path)
    parser.add_argument("--cap", type=int, default=1000000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # Internal SfM caches are keyed by the output stem. Preserve the source
    # directory stem so repeated benchmarks reuse the existing reconstruction.
    output_name = "scene.ply" if args.splat_dataset is not None else (
        args.images.stem + ".ply")
    command = [str(args.exe.resolve()), "--images", str(args.images),
               "--output", str(args.output / output_name),
               "--splat-iterations", str(args.iterations),
               "--splat-strategy", args.strategy,
               "--splat-densification-cap", str(args.cap),
               "--splat-progressive-resolution=false", "--splat-log-interval", "100",
               "--splat-profile-cuda=true", "--splat-profile-interval", "100"]
    if args.splat_dataset is not None:
        command += ["--splat-dataset", str(args.splat_dataset)]
    else:
        command += ["--splat"]
    if args.device_cache_mb is not None:
        command += ["--splat-device-cache-mb", str(args.device_cache_mb)]
    if args.view_cache_mb is not None:
        command += ["--splat-view-cache-mb", str(args.view_cache_mb)]
    if args.cache_auto is not None:
        command += [f"--splat-cache-auto={str(args.cache_auto).lower()}"]
    if args.prefetch_views is not None:
        command += ["--splat-prefetch-views", str(args.prefetch_views)]
    if args.working_sfm is not None:
        command += ["--working-sfm", str(args.working_sfm)]
    (args.output / "command.json").write_text(json.dumps(command, indent=2))
    stop = threading.Event()
    samples = []
    monitor_errors = []

    def monitor():
        with (args.output / "gpu.csv").open("w", newline="") as out:
            writer = csv.writer(out)
            writer.writerow(["timestamp", "gpu_percent", "memory_mib", "watts"])
            while not stop.is_set():
                try:
                    result = subprocess.run(
                        ["nvidia-smi", "--id=0",
                         "--query-gpu=timestamp,utilization.gpu,memory.used,power.draw",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, check=True, timeout=5,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    row = [s.strip() for s in result.stdout.strip().split(",")]
                    timestamp = dt.datetime.strptime(row[0], "%Y/%m/%d %H:%M:%S.%f")
                    sample = (timestamp, *map(float, row[1:]))
                    samples.append(sample)
                    writer.writerow(row)
                    out.flush()
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    monitor_errors.append(str(error))
                stop.wait(0.25)

    thread = threading.Thread(target=monitor)
    started = time.perf_counter()
    thread.start()
    try:
        with (args.output / "train.log").open("w") as out:
            process = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT)
    finally:
        stop.set()
        thread.join()
    elapsed = time.perf_counter() - started
    log = (args.output / "train.log").read_text(errors="replace")
    progress = []
    for line in log.splitlines():
        match = re.search(r"splat iteration=(\d+)/(\d+)", line)
        if match:
            timestamp = dt.datetime.strptime(line[:23], "%Y-%m-%d %H:%M:%S.%f")
            progress.append((int(match[1]), timestamp))
    window = [(i, t) for i, t in progress if i >= args.warmup]
    selected = [s for s in samples if window and window[0][1] <= s[0] <= window[-1][1]]
    util = [s[1] for s in selected]
    training = re.search(r"training_s=([0-9.]+)", log)
    psnr = re.search(r"splat_final_evaluation_views=.*?average_psnr=([0-9.]+)", log)
    stats = {"exit_code": process.returncode, "wall_seconds": elapsed,
             "training_seconds": float(training[1]) if training else None,
             "average_psnr": float(psnr[1]) if psnr else None,
             "steady_iterations": [window[0][0], window[-1][0]] if window else None,
             "gpu_samples": len(util), "monitor_errors": monitor_errors}
    if len(window) >= 2:
        seconds = (window[-1][1] - window[0][1]).total_seconds()
        stats["steady_seconds"] = seconds
        stats["steady_iterations_per_second"] = (window[-1][0] - window[0][0]) / seconds
    if util:
        stats.update(gpu_mean=statistics.mean(util), gpu_min=min(util),
                     gpu_p10=sorted(util)[int(0.1 * (len(util) - 1))],
                     gpu_samples_ge90_percent=100 * sum(x >= 90 for x in util) / len(util),
                     peak_memory_mib=max(s[2] for s in selected))
    (args.output / "summary.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2), flush=True)
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
