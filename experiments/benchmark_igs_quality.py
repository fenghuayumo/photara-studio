"""Train and canonically evaluate IGS binaries serially on one COLMAP dataset."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--evaluator', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=30000)
    parser.add_argument('--cap', type=int, default=1000000)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--split', type=int, default=8)
    args = parser.parse_args()
    if min(args.iterations, args.cap, args.repeats) < 1 or args.split < 2:
        parser.error('Positive iterations/cap/repeats and split >= 2 required')
    args.output.mkdir(parents=True, exist_ok=False)
    sparse = args.dataset / 'sparse' / '0'
    manifest = {str(p): sha256(p) for p in sparse.glob('*.bin')}
    manifest['evaluator'] = sha256(args.evaluator)
    (args.output / 'inputs.json').write_text(json.dumps(manifest, indent=2))
    results = []
    for repeat in range(args.repeats):
        for label, exe in [('reference', args.reference), ('candidate', args.candidate)]:
            out = args.output / f'{label}_{repeat + 1}'
            out.mkdir()
            command = [str(exe.resolve()), '--images', str(args.dataset / 'images'),
                       '--splat-dataset', str(sparse), '--output', str(out / 'model.ply'),
                       '--splat-strategy', 'adc_igs', '--splat-iterations', str(args.iterations),
                       '--splat-densification-cap', str(args.cap),
                       '--splat-progressive-resolution=false', '--splat-use-mask=false',
                       '--splat-depth-normal-weight', '0', '--splat-mv-geo-weight', '0',
                       '--splat-mv-ncc-weight', '0', '--splat-log-interval', '1000',
                       '--splat-eval-split-every', str(args.split)]
            metadata = {'command': command, 'sha256': sha256(exe)}
            (out / 'command.json').write_text(json.dumps(metadata, indent=2))
            print(f'Training {out.name}', flush=True)
            with (out / 'train.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            evaluation = [str(args.evaluator.resolve()), str(sparse),
                          str(args.dataset / 'images'), str(out / 'model_splat.ply'),
                          str(out / 'evaluation'), str(args.split)]
            (out / 'evaluation_command.json').write_text(json.dumps(evaluation, indent=2))
            with (out / 'evaluation.log').open('w') as log:
                subprocess.run(evaluation, stdout=log, stderr=subprocess.STDOUT, check=True)
            with (out / 'evaluation' / 'metrics.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            result = {'label': label, 'repeat': repeat + 1, 'views': rows}
            result.update({metric: sum(float(r[metric]) for r in rows) / len(rows)
                           for metric in ['psnr', 'ssim', 'full_psnr', 'full_ssim']})
            results.append(result)
            (args.output / 'results.json').write_text(json.dumps(results, indent=2))
            print({k: v for k, v in result.items() if k != 'views'}, flush=True)
    means = {label: {metric: sum(r[metric] for r in results if r['label'] == label) / args.repeats
                     for metric in ['psnr', 'ssim', 'full_psnr', 'full_ssim']}
             for label in ['reference', 'candidate']}
    delta = {m: means['candidate'][m] - means['reference'][m]
             for m in ['psnr', 'ssim', 'full_psnr', 'full_ssim']}
    verdict = {'means': means, 'delta': delta,
               'passed': delta['psnr'] > 0 and delta['ssim'] > 0,
               'note': 'PSNR/SSIM use the same camera-derived valid pixel mask for both models. '
                       'Full metrics retain undefined black undistortion borders. Same-seed CUDA repeats; '
                       'photometric metrics do not prove floater removal.'}
    (args.output / 'comparison.json').write_text(json.dumps(verdict, indent=2))
    print(verdict)
    raise SystemExit(0 if verdict['passed'] else 1)


if __name__ == '__main__':
    main()
