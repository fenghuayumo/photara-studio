"""Compare matched progressive-training windows using timestamped log boundaries.

GPU memory is device-wide, sampled every 200 ms; this cannot prove an
instantaneous or process-specific peak. Missing boundaries are reported explicitly.
"""
import argparse
import csv
from datetime import datetime, timedelta
import json
from pathlib import Path
import re

STAMP = r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)"


def summarize(prefix):
    log = prefix.with_suffix('.log').read_text(encoding='utf-8', errors='replace')
    summary = json.loads(prefix.with_suffix('.summary.json').read_text())
    command = summary['command']
    start_iteration = int(command[9]) if len(command) > 9 else 16000
    finish = re.search(STAMP + r" \[info\] splat_optimization_s=([\d.]+)", log)
    end = datetime.fromisoformat(finish[1])
    anchors = {start_iteration: end - timedelta(seconds=float(finish[2]))}
    stages = []
    memory = []
    for line in log.splitlines():
        stamp = re.match(STAMP, line)
        iteration = re.search(r'\biteration=(\d+)', line)
        if not stamp or not iteration:
            continue
        iteration = int(iteration[1])
        when = datetime.fromisoformat(stamp[1])
        fields = dict(re.findall(r'(\w+)=([^ ]+)', line))
        # Transition logs precede training step N, not the end of that step.
        if 'splat_data_stages ' in line:
            # before_scale is the tail of the previous resolution's window.
            stages.append((iteration - 1 if fields.get('boundary') == 'before_scale' else iteration, fields))
        if 'splat_data_memory ' in line:
            memory.append((iteration, fields))
        if any(name in line for name in ('splat_memory ', 'splat_data_pipeline ')):
            anchors[iteration] = when
    anchors[start_iteration + int(command[5])] = end
    samples = []
    with prefix.with_suffix('.gpu.csv').open() as stream:
        for row in list(csv.reader(stream))[1:]:
            samples.append((datetime.strptime(row[0].strip(), '%Y/%m/%d %H:%M:%S.%f'),
                            float(row[1]), float(row[4])))
    windows = {}
    for lo, hi in [(2800, 3000), (3000, 3800), (3800, 6000), (6000, 6800), (6800, 7800), (6000, 7000), (7000, 7800)]:
        key = f'{lo}-{hi}'
        if lo not in anchors or hi not in anchors:
            windows[key] = {'missing_boundary': True}
            continue
        rows = [row for row in samples if anchors[lo] <= row[0] <= anchors[hi]]
        stage_sums = {}
        for iteration, fields in stages:
            if lo < iteration <= hi:
                for name, value in fields.items():
                    if name.endswith('_ms') or name in ('decodes', 'restores'):
                        stage_sums[name] = stage_sums.get(name, 0) + float(value)
        windows[key] = dict(
            wall_seconds=(anchors[hi] - anchors[lo]).total_seconds(),
            samples=len(rows),
            sampled_peak_mib=max((row[2] for row in rows), default=None),
            gpu_utilization_mean=sum(row[1] for row in rows) / len(rows) if rows else None,
            # These are completed jobs reported at diagnostic boundaries;
            # a job/log window can straddle the wall-time phase boundary.
            completed_job_stage_sums=stage_sums)
    transitions = [dict(iteration=i, **{k: v for k, v in f.items() if k != 'iteration'})
                   for i, f in memory if f.get('boundary') in ('before_scale', 'after_scale')]
    return dict(executable_sha256=summary['executable_sha256'], command=command,
                optimization_seconds=summary['optimization_seconds'],
                windows=windows, transitions=transitions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prefixes', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = {str(prefix): summarize(prefix) for prefix in args.prefixes}
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
