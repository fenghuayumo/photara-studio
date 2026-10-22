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
        'scope': ('Independent local reconstruction; frozen parent; no default integration. '
                  'Existing 3D validator unchanged; ray variants use a separate multi-camera holdout validator.'),
        'binary_sha256': sha256(executable),
        'sources': {p: sha256(root / p) for p in [
            'experiments/sfm_missing_group_probe.hpp',
            'experiments/sfm_pair_group_init.hpp',
            'photara/src/sfm/submap_recovery.cpp',
            'experiments/sfm_local_proposals.hpp',
            'experiments/sfm_ray_group_alignment.hpp']},
        'runs': [],
    }
    variants = sys.argv[2:] or ['control', 'covisible', 'frozen']
    replay_variants = ['replay-control', 'native-raw', 'native-cycle', 'native-star', 'ray', 'ray-ba']
    if len(set(variants)) != len(variants) or any(v not in ['control', 'covisible', 'frozen'] + replay_variants for v in variants):
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
            if variant in ['covisible', 'frozen'] + replay_variants:
                env['PHOTARA_GROUP_COVISIBLE_ANCHORS'] = '1'
            if variant in replay_variants:
                replay = root / 'artifacts/sfm-group-frozen-20261018' / name / 'frozen/frontend-cache'
                env['PHOTARA_GROUP_REPLAY_ROOT'] = str(replay)
                # Control repeats the preceding frozen-anchor path using exactly
                # the same serialized proposals as all native-raw variants.
                env['PHOTARA_GROUP_FRONTEND_CACHE'] = str(replay)
                if variant == 'replay-control':
                    env['PHOTARA_GROUP_FROZEN_ANCHOR_GEOMETRY'] = '1'
                else:
                    env['PHOTARA_GROUP_NATIVE_RAW'] = '1'
                if variant == 'native-raw':
                    env['PHOTARA_GROUP_NO_WEAK_CYCLES'] = '1'
                if variant == 'native-star':
                    env.pop('PHOTARA_GROUP_PAIR_INITIALIZER')
                if variant in ['ray', 'ray-ba']:
                    env['PHOTARA_GROUP_RAY_ALIGNMENT'] = '1'
                    # These weak pairs remain tentative. The independently
                    # reconstructed local points and multi-anchor track/pixel
                    # holdout validator provide the multi-view check here.
                    env['PHOTARA_GROUP_NO_WEAK_CYCLES'] = '1'
                    if variant == 'ray':
                        env['PHOTARA_GROUP_NO_SEGMENT_BA'] = '1'
            if variant == 'frozen':
                env['PHOTARA_GROUP_FROZEN_ANCHOR_GEOMETRY'] = '1'
                env['PHOTARA_GROUP_FRONTEND_CACHE'] = str(directory / 'frontend-cache')
            env['PHOTARA_PROBE_EXPORT_COLMAP'] = str(directory / 'colmap')
            row = dict(scene=name, variant=variant, input_sha256=sha256(source))
            row['validator'] = 'shared-3d then independent ray holdout' if variant in ['ray', 'ray-ba'] else 'shared-3d'
            if variant in replay_variants:
                row['replay_sha256'] = {str(p.relative_to(replay)): sha256(p) for p in replay.rglob('*.bin')}
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
