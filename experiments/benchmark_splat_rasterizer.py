"""Validate and time reference_compare on a fixed captured scene.

Build using scripts/build_ab_compare.ps1. Captures come from
AETHERSCAN_SPLAT_GRAD_DUMP; width/height must match the captured image.
Runs are serial, with warm-up in the CUDA harness. Event times include host
submission gaps, and are not sums of isolated kernel durations.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--height', type=int, required=True)
    parser.add_argument('--geometry', action='store_true')
    parser.add_argument('--iterations', type=int, default=300)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rtol', type=float, default=1e-4)
    parser.add_argument('--atol', type=float, default=1e-5)
    args = parser.parse_args()
    if min(args.iterations, args.repeats, args.width, args.height) <= 0:
        parser.error('counts and dimensions must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    command = [str(args.exe.resolve()), 'geometry' if args.geometry else 'rgb',
               str(args.capture.resolve()), str(args.width), str(args.height)]
    env = dict(os.environ, SPLAT_BENCH_ITERS=str(args.iterations))
    runs = []
    failures = []
    required = {'color', 'alpha', 'depth', 'normal', 'mean', 'scale',
                'rotation', 'opacity', 'sh', 'refine'}
    for repeat in range(args.repeats):
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        log = result.stdout + result.stderr
        (args.output / f'run_{repeat}.log').write_text(log)
        if result.returncode:
            failures.append(f'run {repeat}: exit {result.returncode}')
        counts = re.search(r'instances ref=(\d+) new=(\d+)', log)
        if not counts or counts[1] != counts[2]:
            failures.append(f'run {repeat}: instance count mismatch/missing')
        channels = {}
        for name, relative, maximum in re.findall(r'^(\w+) relative_l2=(\S+) max=(\S+)', log, re.M):
            if name not in required:
                continue
            relative, maximum = float(relative), float(maximum)
            channels[name] = dict(relative_l2=relative, maximum=maximum)
            if (not math.isfinite(relative) or not math.isfinite(maximum)
                    or (relative > args.rtol and maximum > args.atol)):
                failures.append(f'run {repeat}: {name} exceeds tolerance')
        if channels.keys() != required:
            failures.append(f'run {repeat}: missing correctness channels')
        timings = {name: float(wall) for name, wall in re.findall(
            r'^BENCH (\w+).* wall_ms=(\S+)', log, re.M)}
        expected = {f'{backend}_{stage}' for backend in ('reference', 'new')
                    for stage in ('forward', 'backward', 'pair')}
        if timings.keys() != expected or any(not math.isfinite(t) or t <= 0 for t in timings.values()):
            failures.append(f'run {repeat}: invalid/missing timings')
        runs.append(dict(channels=channels, wall_ms=timings))
    summary = dict(command=command, iterations=args.iterations, repeats=args.repeats,
                   rtol=args.rtol, atol=args.atol, failures=failures, runs=runs)
    if all(r['wall_ms'].keys() == expected for r in runs):
        summary['median_wall_ms'] = {key: statistics.median(r['wall_ms'][key] for r in runs)
                                     for key in sorted(expected)}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != 'runs'}, indent=2))
    raise SystemExit(bool(failures))


if __name__ == '__main__':
    main()
