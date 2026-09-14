"""Serial, matched 30k-step RGB training comparison of two AetherScan binaries."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--new', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=30000)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--split', type=int, default=8)
    parser.add_argument('--max-regression-db', type=float, default=0.2,
                        help='Fail if mean new PSNR falls below reference by more than this amount')
    args = parser.parse_args()
    if min(args.iterations, args.repeats) <= 0 or args.split < 0 or args.split == 1:
        parser.error('iterations/repeats must be positive; split must be 0 or >= 2')
    if args.max_regression_db < 0:
        parser.error('max-regression-db must be nonnegative')
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for repeat in range(args.repeats):
        for backend, exe in [('reference', args.reference), ('new', args.new)]:
            name = f'{backend}_split{args.split}_run{repeat + 1}'
            log_path = args.output / f'{name}.log'
            command = [str(exe.resolve()), '--images', str(args.dataset / 'images'),
                       '--splat-dataset', str(args.dataset / 'sparse' / '0'),
                       '--output', str(args.output / f'{name}.ply'),
                       '--splat-iterations', str(args.iterations),
                       '--splat-strategy', 'adc_plus',
                       '--splat-progressive-resolution=false', '--splat-use-mask=false',
                       '--splat-depth-normal-weight', '0', '--splat-mv-geo-weight', '0',
                       '--splat-mv-ncc-weight', '0', '--splat-log-interval', '1000',
                       '--splat-eval-split-every', str(args.split)]
            metadata = {'command': command, 'binary_sha256': hashlib.sha256(exe.read_bytes()).hexdigest()}
            (args.output / f'{name}.command.json').write_text(json.dumps(metadata, indent=2))
            print(f'Starting {name}', flush=True)
            with log_path.open('w') as output:
                run = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
            log = log_path.read_text(errors='replace')
            final = re.search(r'splat_final_evaluation_views=(\d+) average_psnr=([\d.]+) average_masked_psnr=([\d.]+)', log)
            if run.returncode or not final:
                raise RuntimeError(f'{name} failed ({run.returncode}); see {log_path}')
            views = [{'view': int(m[1]), 'psnr': float(m[2])} for m in
                     re.finditer(r'splat_render=.*? view=(\d+) foreground_psnr=([\d.]+)', log)]
            model = re.search(r'splat_model=.*? gaussians=(\d+) training_s=([\d.]+)', log)
            results.append({'name': name, 'backend': backend, 'repeat': repeat + 1,
                            'views': views, 'average_psnr': float(final[2]),
                            'gaussians': int(model[1]) if model else None,
                            'training_seconds': float(model[2]) if model else None})
            (args.output / 'summary.json').write_text(json.dumps(results, indent=2))
            print(json.dumps({k: v for k, v in results[-1].items() if k != 'views'}), flush=True)
    means = {backend: statistics.mean(r['average_psnr'] for r in results if r['backend'] == backend)
             for backend in ('new', 'reference')}
    delta = means['new'] - means['reference']
    verdict = {'mean_psnr': means, 'new_minus_reference_db': delta,
               'max_regression_db': args.max_regression_db,
               'passed': delta >= -args.max_regression_db,
               'note': 'Same-seed repeats measure numerical variability, not variation across random seeds.'}
    (args.output / 'comparison.json').write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict), flush=True)
    raise SystemExit(0 if verdict['passed'] else 1)


if __name__ == '__main__':
    main()
