"""Serial before/after training profiles, including topology changes.

Use matching Release executables. CUDA event stage times include submission
gaps; training trajectories can diverge after densification. No GPU workloads
or builds should run alongside this benchmark.
"""
import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before', type=Path, required=True)
    p.add_argument('--after', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--splat-dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--iterations', type=int, default=3000)
    p.add_argument('--interval', type=int, default=500)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--geometry', action='store_true')
    p.add_argument('--strategy', default='default')
    p.add_argument('--graph', action='store_true', help='Also test experimental SSIM graph mode')
    p.add_argument('--device-points', action='store_true', help='Also test fixed-size point lists without count readback')
    a = p.parse_args()
    if min(a.iterations, a.interval, a.repeats) <= 0:
        p.error('iteration, interval and repeat counts must be positive')
    a.output.mkdir(parents=True, exist_ok=True)
    variants = [('before', a.before), ('after', a.after)]
    if a.graph:
        variants.append(('graph', a.after))
    if a.device_points:
        variants.append(('device_points', a.after))
    runs = []
    fields = ('cuda_timeline_avg_ms', 'data_load_ms', 'raster_forward_ms',
              'training_loss_ms', 'multi_view_ms', 'raster_backward_ms',
              'densification_stats_ms', 'optimizer_ms', 'refinement_ms')
    for repeat in range(a.repeats):
        # Reverse every other round to reduce consistent ordering bias.
        for name, exe in variants[::1 if repeat % 2 == 0 else -1]:
            stem = a.output / f'{name}_{repeat}'
            command = [str(exe.resolve()), '--images', str(a.images),
                       '--splat-dataset', str(a.splat_dataset),
                       '--output', str(stem.with_suffix('.ply')), '--splat-iterations', str(a.iterations),
                       '--splat-strategy', a.strategy, '--splat-progressive-resolution=false',
                       '--splat-log-interval', str(a.interval), '--splat-profile-cuda=true',
                       '--splat-profile-interval', str(a.interval)]
            if a.geometry:
                command += ['--splat', '--mesh', '--mesh-method', 'tsdf', '--splat-geometry-from-iter', '10']
            else:
                command += ['--splat-depth-normal-weight', '0', '--splat-mv-geo-weight', '0',
                            '--splat-mv-ncc-weight', '0', '--splat-normal-field=false']
            env = dict(os.environ, AETHERSCAN_SPLAT_SSIM_GRAPH='1' if name == 'graph' else '0',
                       AETHERSCAN_SPLAT_DEVICE_POINTS='1' if name == 'device_points' else '0')
            with stem.with_suffix('.log').open('w', encoding='utf-8') as out:
                subprocess.run(command, env=env, stdout=out, stderr=subprocess.STDOUT, check=True)
            log = stem.with_suffix('.log').read_text(encoding='utf-8', errors='replace')
            def scalar(key):
                matches = re.findall(r'\b' + key + r'=([0-9.eE+-]+)', log)
                if not matches:
                    raise RuntimeError(f'Missing {key} in {stem}')
                return float(matches[-1])
            profiles = []
            for line in log.splitlines():
                if 'splat_cuda_profile iterations=' in line:
                    samples = int(re.search(r'\bsamples=(\d+)', line)[1])
                    profiles.append((samples, {f: float(re.search(r'\b' + f + r'=([0-9.eE+-]+)', line)[1]) for f in fields}))
            if not profiles:
                raise RuntimeError(f'Missing CUDA profile in {stem}')
            row = dict(variant=name, repeat=repeat, command=command, training_s=scalar('training_s'),
                       psnr=scalar('average_psnr'), gaussians=scalar('gaussians'),
                       stages_ms={f: sum(n * d[f] for n, d in profiles) / sum(n for n, _ in profiles) for f in fields},
                       first_window_ms=profiles[0][1])
            runs.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            (a.output / 'runs.json').write_text(json.dumps(runs, indent=2), encoding='utf-8')
    medians = {}
    for name, _ in variants:
        subset = [r for r in runs if r['variant'] == name]
        medians[name] = dict(training_s=statistics.median(r['training_s'] for r in subset),
                             stages_ms={f: statistics.median(r['stages_ms'][f] for r in subset) for f in fields})
    (a.output / 'summary.json').write_text(json.dumps(medians, indent=2), encoding='utf-8')
    print(json.dumps(medians, indent=2), flush=True)


if __name__ == '__main__':
    main()
