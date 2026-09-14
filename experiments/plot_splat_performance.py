"""Plot two benchmark_splat_training.py runs against optimizer iteration."""

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


def read_run(directory):
    iterations, times = [], []
    for line in (directory / "train.log").read_text(errors="replace").splitlines():
        match = re.search(r"splat iteration=(\d+)/", line)
        if match:
            iterations.append(int(match[1]))
            times.append(dt.datetime.strptime(line[:23], "%Y-%m-%d %H:%M:%S.%f").timestamp())
    with (directory / "gpu.csv").open() as source:
        rows = list(csv.DictReader(source))
    sample_times = [dt.datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f").timestamp()
                    for row in rows]
    valid = np.array([(times[0] <= t <= times[-1]) for t in sample_times])
    sample_steps = np.interp(sample_times, times, iterations)[valid]
    util = np.array([float(row["gpu_percent"]) for row in rows])[valid]
    mean_ms = 1000 * np.diff(times) / np.diff(iterations)
    return sample_steps, util, np.array(iterations[1:]), mean_ms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--warmup", type=int, default=10000)
    args = parser.parse_args()
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.6), sharex=True, layout="constrained")
    for directory, label, color in [(args.before, "Before", "#b15d4a"),
                                     (args.after, "After", "#167b80")]:
        steps, util, progress_steps, ms = read_run(directory)
        valid = steps >= args.warmup
        axes[0].plot(steps[valid], util[valid], label=label, color=color, linewidth=1.5)
        valid = progress_steps >= args.warmup
        axes[1].plot(progress_steps[valid], ms[valid], label=label, color=color, linewidth=1.1)
    axes[0].axhline(90, color="#566573", linestyle="--", linewidth=1, label="90% target")
    axes[0].set_ylabel("GPU busy (%)")
    axes[0].set_ylim(0, 102)
    axes[0].legend(loc="lower right", ncol=3)
    axes[1].set_ylabel("Wall time / step (ms)\n100-step windows")
    axes[1].set_xlabel("Training iteration")
    axes[1].set_xlim(left=args.warmup)
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    command = json.loads((args.after / "command.json").read_text())
    strategy = command[command.index("--splat-strategy") + 1]
    cap = command[command.index("--splat-densification-cap") + 1]
    fig.suptitle(f"3DGS steady training: {strategy}, Gaussian cap {int(cap):,}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)


if __name__ == "__main__":
    main()
