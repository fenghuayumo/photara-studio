"""Measure one isolated training arm with GPU telemetry and provenance."""
import argparse
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("exe", "dataset", "images", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--prefetch", type=int, default=4)
    parser.add_argument("--start-iteration", type=int, default=16000)
    parser.add_argument("--max-resolution", type=int)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.steps <= 10000 or args.log_interval < 1 or args.steps % args.log_interval:
        parser.error("steps must be 1..10000 and divisible by log-interval")
    if args.start_iteration < 0 or (args.max_resolution is not None and args.max_resolution < 0):
        parser.error("start-iteration and max-resolution must be nonnegative")
    out = args.output.resolve()
    if out.exists() or out.with_suffix(".log").exists():
        parser.error("Output already exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    exe = args.exe.resolve()
    command = [str(exe), str(args.dataset.resolve()), str(args.images.resolve()),
               str(args.model.resolve()), str(out), str(args.steps),
               str(args.log_interval), str(int(args.profile)), str(args.prefetch)]
    if args.start_iteration != 16000 or args.max_resolution is not None:
        command.append(str(args.start_iteration))
    if args.max_resolution is not None:
        command.append(str(args.max_resolution))
    with out.with_suffix(".gpu.csv").open("wb") as gpu, out.with_suffix(".log").open("wb") as log:
        monitor = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,memory.used,clocks.sm",
                                    "--format=csv,nounits", "--loop-ms=200"], stdout=gpu,
                                   stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
        finally:
            monitor.terminate()
            monitor.wait()
    log_text = out.with_suffix(".log").read_text(encoding="utf-8", errors="replace")
    timing = re.search(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+) \[info\] splat_optimization_s=([\d.]+)", log_text)
    result = {"command": command, "exit_code": code,
              "executable_sha256": hashlib.sha256(exe.read_bytes()).hexdigest()}
    if timing:
        end = datetime.fromisoformat(timing[1])
        seconds = float(timing[2])
        # Skip the first two seconds so initial clocks and loading do not
        # contaminate the training utilization sample.
        start = end - timedelta(seconds=seconds) + timedelta(seconds=2)
        samples = []
        for row in out.with_suffix(".gpu.csv").read_text().splitlines()[1:]:
            fields = [field.strip() for field in row.split(",")]
            try:
                stamp = datetime.strptime(fields[0], "%Y/%m/%d %H:%M:%S.%f")
                if start <= stamp <= end:
                    samples.append([float(field) for field in fields[1:]])
            except (ValueError, IndexError):
                continue
        result.update(optimization_seconds=seconds, milliseconds_per_step=1000 * seconds / args.steps,
                      iterations_per_second=args.steps / seconds, gpu_samples=len(samples))
        if samples:
            result["gpu_training_mean"] = dict(zip(
                ("utilization_percent", "memory_utilization_percent", "power_watts", "memory_mib", "sm_clock_mhz"),
                [sum(column) / len(samples) for column in zip(*samples)]))
    metrics = out / "result.json"
    if metrics.exists():
        result["evaluation"] = json.loads(metrics.read_text())
    out.with_suffix(".summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
