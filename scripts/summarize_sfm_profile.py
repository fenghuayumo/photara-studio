"""Summarize SfM stage wall times and backend decisions from a Photara log.

Stage times are inclusive and may overlap (including CPU/GPU pipelining).
They must not be added to obtain total runtime or GPU utilization.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re


NUMBER = r"[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
STAGE = re.compile(rf"stage finished: (\S+) elapsed_s=({NUMBER})(?: backend=(\S+))?")
FIELD = re.compile(r"([a-zA-Z_][a-zA-Z_0-9]*)=([^\s]+)")


def summarize(lines):
    stages = defaultdict(list)
    decisions = Counter()
    checkpoints = Counter()
    warnings = Counter()
    transfers = defaultdict(float)
    transfer_samples = 0
    pipeline = defaultdict(float)
    pipeline_samples = 0
    feature_selections = Counter()
    for line in lines:
        match = STAGE.search(line)
        if match and (match[1].startswith('sfm.') or match[1] == 'bundle_pack'):
            stages[(match[1], match[3] or 'unspecified')].append(float(match[2]))
        if 'sfm backend selection:' in line:
            fields = dict(FIELD.findall(line))
            decisions[(fields.get('stage', '?'), fields.get('requested', '?'),
                       fields.get('selected', '?'), fields.get('reason', '?'))] += 1
        if 'sfm feature selection:' in line:
            feature_selections[line.split('sfm feature selection:', 1)[1].strip()] += 1
        if 'checkpoint hit:' in line:
            checkpoints[line.split('checkpoint hit:', 1)[1].strip()] += 1
        if any(token in line for token in ('falling back', 'using CPU:',
                                           'continuing with CPU', 'stays on CPU:',
                                           'stays on the CPU:')):
            # Remove timestamp and severity while preserving the actual failure.
            warnings[line.split('] ', 1)[-1].strip()] += 1
        fields = dict(FIELD.findall(line))
        if 'match/verify timing:' in line:
            pipeline_samples += 1
            for key in ('pairs', 'match_wall_s', 'geometry_work_sum_s',
                        'wall_s', 'throttle_wait_s'):
                if key in fields:
                    pipeline[key] += float(fields[key])
        if all(key in fields for key in ('upload_ms', 'compute_ms', 'download_ms')):
            transfer_samples += 1
            for key in ('upload_ms', 'compute_ms', 'download_ms'):
                transfers[key] += float(fields[key])
    return {
        'timing_semantics': 'inclusive wall time; stages may overlap; do not sum',
        'stages': sorted([
            dict(stage=stage, backend=backend, calls=len(times),
                 total_s=sum(times), max_s=max(times))
            for (stage, backend), times in stages.items()
        ], key=lambda row: (-row['total_s'], row['stage'], row['backend'])),
        'backend_decisions': [dict(stage=k[0], requested=k[1], selected=k[2],
                                   reason=k[3], calls=v)
                              for k, v in sorted(decisions.items())],
        'checkpoints': dict(checkpoints),
        'feature_selections': dict(feature_selections),
        'fallback_messages': dict(warnings),
        'matcher_timing': dict(samples=transfer_samples, **transfers),
        'match_verify_pipeline': dict(samples=pipeline_samples, **pipeline),
    }


def render(report):
    lines = ['SfM profile: inclusive wall times; stages can overlap. Do not sum.',
             'Seconds shown are elapsed time, not CPU/GPU utilization.', '',
             f"{'Stage':38} {'Backend':12} {'Calls':>6} {'Total(s)':>10} {'Max(s)':>10}"]
    for row in report['stages']:
        lines.append(f"{row['stage']:38} {row['backend']:12} {row['calls']:6} "
                     f"{row['total_s']:10.3f} {row['max_s']:10.3f}")
    if not report['stages']:
        lines.append('No SfM stage timings found. Use a log with info/debug logging enabled.')
    lines += ['', 'Backend decisions:']
    for selection, count in report['feature_selections'].items():
        lines.append(f'  features: {selection} calls={count}')
    for row in report['backend_decisions']:
        lines.append(f"  {row['stage']}: {row['requested']} -> {row['selected']} "
                     f"reason={row['reason']} calls={row['calls']}")
    lines += ['', 'Checkpoint hits: ' + (str(report['checkpoints']) or 'none')]
    if report['fallback_messages']:
        lines += ['', 'Fallback messages:']
        lines += [f'  {count}x {message}'
                  for message, count in report['fallback_messages'].items()]
    timing = report['matcher_timing']
    if timing['samples']:
        lines += ['', 'Matcher GPU event timings (reported samples only):',
                  f"  samples={timing['samples']} upload_ms={timing['upload_ms']:.3f} "
                  f"compute_ms={timing['compute_ms']:.3f} download_ms={timing['download_ms']:.3f}"]
    else:
        lines += ['', 'Matcher transfer timings unavailable in this log.']
    pipeline = report['match_verify_pipeline']
    if pipeline['samples']:
        lines += ['', 'Match/verify pipeline (geometry worker time can exceed wall time):',
                  '  ' + ' '.join(f'{k}={v:g}' for k, v in pipeline.items())]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path, help='One run per log is recommended')
    parser.add_argument('--json', action='store_true', help='Print a machine-readable report')
    args = parser.parse_args()
    with args.log.open(encoding='utf-8-sig', errors='replace') as source:
        report = summarize(source)
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else render(report))


if __name__ == '__main__':
    main()
