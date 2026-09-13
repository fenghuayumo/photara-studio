"""Run an SfM command and retain wall time plus whole-device GPU samples."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New evidence filename stem')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    stem = Path(args.output).resolve()
    stem.parent.mkdir(parents=True, exist_ok=True)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('Missing command')
    log_path = stem.with_suffix('.log')
    csv_path = stem.with_suffix('.gpu.csv')
    report_path = stem.with_suffix('.profile.json')
    if any(p.exists() for p in (log_path, csv_path, report_path)):
        parser.error('Evidence already exists; choose a new stem')
    samples = []
    start = time.perf_counter()
    with log_path.open('w', encoding='utf-8') as log, csv_path.open('w', newline='') as output:
        writer = csv.writer(output)
        writer.writerow(['elapsed_seconds', 'gpu_percent', 'memory_MiB', 'power_W'])
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--id=0', '--query-gpu=utilization.gpu,memory.used,power.draw',
                     '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
                values = [float(v.strip()) for v in result.stdout.strip().split(',')]
                if len(values) == 3:
                    sample = [time.perf_counter()-start, *values]
                    writer.writerow(sample)
                    output.flush()
                    samples.append(sample)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
            try:
                process.wait(timeout=.5)
            except subprocess.TimeoutExpired:
                pass
    report = {'command': command, 'exit_code': process.returncode,
              'wall_seconds': time.perf_counter()-start,
              'gpu_samples': len(samples),
              'gpu_scope': 'Whole device including other applications; not per-process',
              'gpu_average_percent': sum(s[1] for s in samples)/len(samples) if samples else None,
              'gpu_peak_percent': max((s[1] for s in samples), default=None),
              'device_peak_memory_MiB': max((s[2] for s in samples), default=None)}
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return process.returncode


if __name__ == '__main__':
    raise SystemExit(main())
