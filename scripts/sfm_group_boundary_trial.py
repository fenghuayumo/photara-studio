"""Compare independent missing-run recovery with bounded covisible anchors.

Run from the repository root: python scripts/sfm_group_boundary_trial.py NEW_DIR [VARIANT ...]
References are never provided to the recovery process. Existing output is refused.
"""
import json
import csv
import os
from pathlib import Path
import shutil
import sys

from sfm_full_baseline import metrics, run, sha256


def main():
    root = Path(__file__).resolve().parents[1]
    output = Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    binary_dir = root / 'build/photara/Release'
    executable = output / 'photara_sfm_stable_resection_probe.exe'
    shutil.copy2(binary_dir / executable.name, executable)
    manifest = {
        'scope': 'Independent local reconstruction; frozen parent; unchanged validation gates.',
        'binary_sha256': sha256(executable),
        'sources': {p: sha256(root / p) for p in [
            'experiments/sfm_missing_group_probe.hpp',
            'experiments/sfm_pair_group_init.hpp',
            'photara/src/sfm/submap_recovery.cpp']},
        'runs': [],
    }
    variants = sys.argv[2:] or ['control', 'covisible', 'frozen']
    if len(set(variants)) != len(variants) or any(v not in ['control', 'covisible', 'frozen'] for v in variants):
        raise ValueError('Unknown boundary trial variant')
    for name in ['alameda', 'london']:
        source = next((root / 'artifacts/sfm-resection-analysis-20261018' /
                       name / 'baseline/model.graph').glob('reconstruction-*.bin'))
        for variant in variants:
            directory = output / name / variant
            directory.mkdir(parents=True)
            env = dict(os.environ, PATH=str(binary_dir) + os.pathsep + os.environ['PATH'])
            for key in list(env):
                if key.startswith(('PHOTARA_GROUP_', 'PHOTARA_PROBE_')):
                    del env[key]
            env['PHOTARA_GROUP_PAIR_INITIALIZER'] = '1'
            if variant in ['covisible', 'frozen']:
                env['PHOTARA_GROUP_COVISIBLE_ANCHORS'] = '1'
            if variant == 'frozen':
                env['PHOTARA_GROUP_FROZEN_ANCHOR_GEOMETRY'] = '1'
                env['PHOTARA_GROUP_FRONTEND_CACHE'] = str(directory / 'frontend-cache')
            env['PHOTARA_PROBE_EXPORT_COLMAP'] = str(directory / 'colmap')
            row = dict(scene=name, variant=variant, input_sha256=sha256(source))
            manifest['runs'].append(row)
            (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            print('START', name, variant, flush=True)
            row['execution'] = run([executable, source, directory / 'model.asfm', '--groups'],
                                   directory / 'run.log', directory / 'resources.json', env)
            row['metrics'] = metrics(directory / 'run.log')
            if row['execution']['exit_code'] == 0:
                with (directory / 'model.asfm.rounds.csv').open(newline='', encoding='utf-8') as stream:
                    row['groups'] = list(csv.DictReader(stream))
            (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            if row['execution']['exit_code']:
                raise RuntimeError(f'{name}/{variant} failed')
            print('DONE', name, variant, row['metrics'], flush=True)


if __name__ == '__main__':
    main()
