"""Replay reference-free, component-disjoint precision diagnostics on frozen caches.

Usage: python scripts/sfm_cross_anchor_trial.py NEW_OUTPUT [london|alameda]
No candidate is committed or exported as a validated reconstruction.
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
    only = sys.argv[2] if len(sys.argv) > 2 else None
    if only not in (None, 'london', 'alameda'):
        raise ValueError('Unknown scene filter')
    output = Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    binary_dir = root / 'build/photara/Release'
    executable = output / 'photara_sfm_stable_resection_probe.exe'
    shutil.copy2(binary_dir / executable.name, executable)
    sources = ['experiments/sfm_cross_anchor_precision.hpp', 'experiments/sfm_local_proposals.hpp',
               'photara/src/tools/sfm_stable_resection_probe.cpp', 'photara/src/sfm/submap_recovery.cpp',
               'photara/tests/submap_recovery_test.cpp', 'scripts/sfm_cross_anchor_trial.py']
    manifest = dict(scope='Frozen parent; no references; all reserved component pixels excluded before fitting. '
                    'Control metrics overlap the pose fit and are descriptive only. No new descriptor matching.',
                    binary_sha256=sha256(executable), sources={p: sha256(root / p) for p in sources}, runs=[])
    for scene in ['london', 'alameda']:
        if only and scene != only:
            continue
        directory = output / scene
        directory.mkdir()
        source = next((root / 'artifacts/sfm-resection-analysis-20261018' / scene /
                       'baseline/model.graph').glob('reconstruction-*.bin'))
        cache = root / 'artifacts/sfm-group-frozen-20261018' / scene / 'frozen/frontend-cache'
        env = dict(os.environ, PATH=str(binary_dir) + os.pathsep + os.environ['PATH'])
        for key in list(env):
            if key.startswith(('PHOTARA_GROUP_', 'PHOTARA_PROBE_', 'PHOTARA_TRACK_', 'PHOTARA_BOUNDARY_')):
                del env[key]
        env['PHOTARA_GROUP_FRONTEND_CACHE'] = str(cache)
        row = dict(scene=scene, input_sha256=sha256(source),
                   cache_sha256={str(p.relative_to(cache)): sha256(p) for p in sorted(cache.rglob('*.bin'))})
        manifest['runs'].append(row)
        manifest_path = output / 'manifest.json'
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print('START', scene, flush=True)
        model = directory / 'audit.asfm'
        row['execution'] = run([executable, source, model, '--cross-anchor-precision'], directory / 'run.log',
                               directory / 'resources.json', env)
        if row['execution']['exit_code'] == 0:
            with Path(str(model) + '.rounds.csv').open(newline='', encoding='utf-8') as stream:
                reader = csv.DictReader(stream)
                if 'eligible_reserved_features' not in reader.fieldnames:
                    raise RuntimeError('Stale probe binary: rebuild before replaying')
                row['precision'] = list(reader)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        if row['execution']['exit_code']:
            raise RuntimeError(f'{scene} failed; see run.log')
        print('DONE', scene, flush=True)


if __name__ == '__main__':
    main()
