"""Replay structure-less, stable track completion and boundary-window diagnostics.

Usage: python scripts/sfm_boundary_sequence_trial.py NEW_OUTPUT [london|alameda]
Original geometry is read-only; pose hypotheses do not register parent cameras.
"""
import csv
import json
import os
from pathlib import Path
import shutil
import sys

from sfm_full_baseline import run, sha256


def main():
    root = Path(__file__).resolve().parents[1]
    output = Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    binary_dir = root / 'build/photara/Release'
    executable = output / 'photara_sfm_stable_resection_probe.exe'
    shutil.copy2(binary_dir / executable.name, executable)
    sources = ['experiments/sfm_structureless_probe.hpp', 'experiments/sfm_track_completion.hpp',
               'experiments/sfm_boundary_windows.hpp', 'experiments/sfm_local_proposals.hpp',
               'experiments/sfm_ray_group_alignment.hpp', 'experiments/sfm_pair_group_init.hpp',
               'photara/src/tools/sfm_stable_resection_probe.cpp',
               'photara/tests/submap_recovery_test.cpp', 'scripts/sfm_boundary_sequence_trial.py']
    manifest = dict(scope='Read-only parent. Calibrated cached proposals. Separate hypothesis validation; no default integration.',
                    binary_sha256=sha256(executable), sources={p: sha256(root / p) for p in sources}, runs=[])
    only = sys.argv[2] if len(sys.argv) > 2 else None
    if only not in [None, 'london', 'alameda']:
        raise ValueError('Unknown scene filter')
    for scene, variant in [('london', 'control'), ('london', 'completion'), ('alameda', 'completion-windows')]:
        if only and scene != only:
            continue
        directory = output / scene / variant
        directory.mkdir(parents=True)
        source = next((root / 'artifacts/sfm-resection-analysis-20261018' / scene /
                       'baseline/model.graph').glob('reconstruction-*.bin'))
        cache = root / 'artifacts/sfm-group-frozen-20261018' / scene / 'frozen/frontend-cache'
        env = dict(os.environ, PATH=str(binary_dir) + os.pathsep + os.environ['PATH'])
        for key in list(env):
            if key.startswith(('PHOTARA_GROUP_', 'PHOTARA_PROBE_', 'PHOTARA_TRACK_', 'PHOTARA_BOUNDARY_')):
                del env[key]
        env['PHOTARA_GROUP_FRONTEND_CACHE'] = str(cache)
        if variant != 'control':
            env['PHOTARA_TRACK_COMPLETION'] = '1'
        if variant == 'completion-windows':
            env['PHOTARA_BOUNDARY_WINDOWS'] = '1'
        row = dict(scene=scene, variant=variant, input_sha256=sha256(source),
                   cache_sha256={str(p.relative_to(cache)): sha256(p) for p in sorted(cache.rglob('*.bin'))})
        manifest['runs'].append(row)
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print('START', scene, variant, flush=True)
        model = directory / 'model.asfm'
        row['execution'] = run([executable, source, model, '--structureless'], directory / 'run.log',
                               directory / 'resources.json', env)
        if row['execution']['exit_code'] == 0:
            for label, path in [('hypotheses', Path(str(model) + '.rounds.csv')),
                                ('completion', Path(str(model) + '.tracks.csv'))]:
                if path.exists():
                    with path.open(newline='', encoding='utf-8') as stream:
                        row[label] = list(csv.DictReader(stream))
            row['window_summaries'] = {}
            for path in sorted(directory.glob('model.asfm.windows/*/windows.csv')):
                with path.open(newline='', encoding='utf-8') as stream:
                    row['window_summaries'][path.parent.name] = list(csv.DictReader(stream))
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        if row['execution']['exit_code']:
            raise RuntimeError(f'{scene}/{variant} failed; see run.log')
        print('DONE', scene, variant, 'accepted hypotheses', sum(int(r['accepted']) for r in row['hypotheses']), flush=True)


if __name__ == '__main__':
    main()
