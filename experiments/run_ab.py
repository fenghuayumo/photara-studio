"""Train and canonically evaluate labelled configurations serially."""
import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary', type=Path,
                        default=Path('build/aetherscan/Release/aetherscan.exe'))
    parser.add_argument('--evaluator', type=Path,
                        default=Path('build/aetherscan/Release/aetherscan_splat_quality_eval.exe'))
    parser.add_argument('--iterations', type=int, default=30000)
    parser.add_argument('--cap', type=int, default=1000000)
    parser.add_argument('--split', type=int, default=8)
    parser.add_argument('--config', action='append', required=True,
                        help='label:flag=value[,flag=value]')
    parser.add_argument('--camera-audit', action='store_true',
                        help='also record the camera-proximity floater diagnostic')
    parser.add_argument('--visibility-audit', action='store_true',
                        help='also record the contribution-visibility audit')
    args = parser.parse_args()
    sparse = args.dataset / 'sparse' / '0'
    images = args.dataset / 'images'
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {'dataset': str(args.dataset), 'binary': sha256(args.binary.resolve()),
               'configurations': {}}
    for specification in args.config:
        label, _, flags = specification.partition(':')
        out = args.output / label
        out.mkdir(parents=True, exist_ok=True)
        command = [str(args.binary.resolve()), '--images', str(images),
                   '--splat-dataset', str(sparse),
                   '--output', str(out / 'model.ply'),
                   '--splat-strategy', 'adc_igs',
                   '--splat-iterations', str(args.iterations),
                   '--splat-densification-cap', str(args.cap),
                   '--splat-progressive-resolution=false',
                   '--splat-use-mask=false',
                   '--splat-depth-normal-weight', '0',
                   '--splat-mv-geo-weight', '0', '--splat-mv-ncc-weight', '0',
                   '--splat-log-interval', '1000',
                   '--splat-eval-split-every', str(args.split)]
        for flag in flags.split(','):
            if flag:
                # cxxopts only binds a value to a bool flag in the
                # `--flag=value` form; `--flag value` silently drops the value
                # and the flag falls back to its implicit true.
                command.append(flag if '=' in flag else f'{flag}=true')
        (out / 'command.json').write_text(json.dumps(command, indent=2))
        print(f'Training {out}', flush=True)
        with (out / 'train.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                           check=True)
        evaluation = [str(args.evaluator.resolve()), str(sparse), str(images),
                      str(out / 'model_splat.ply'), str(out / 'evaluation'),
                      str(args.split)]
        with (out / 'evaluation.log').open('w') as log:
            subprocess.run(evaluation, stdout=log, stderr=subprocess.STDOUT,
                           check=True)
        with (out / 'evaluation' / 'metrics.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        if args.camera_audit:
            audit = subprocess.run(
                ['python', 'experiments/igs_camera_audit.py',
                 '--sparse', str(sparse),
                 '--model', str(out / 'model_splat.ply'),
                 '--output', str(out / 'camera_audit.json')],
                capture_output=True, text=True)
            print(audit.stdout.strip() or audit.stderr.strip(), flush=True)
        if args.visibility_audit:
            visibility = subprocess.run(
                [str(args.evaluator.resolve()), str(sparse), str(images),
                 str(out / 'model_splat.ply'), str(out / 'visibility'),
                 str(args.split), '--visibility-audit'],
                capture_output=True, text=True)
            (out / 'visibility_audit.log').write_text(
                visibility.stdout + visibility.stderr)
            print('visibility:', visibility.stdout.strip(), flush=True)
        entry = {'views': rows}
        entry.update({metric: sum(float(row[metric]) for row in rows) / len(rows)
                      for metric in rows[0] if metric != 'view'})
        log_text = (out / 'train.log').read_text(errors='replace')
        entry['appearance_lines'] = [
            line for line in log_text.splitlines() if 'splat_appearance ' in line]
        entry['gaussian_line'] = next(
            (line for line in reversed(log_text.splitlines())
             if 'splat_model=' in line), '')
        summary['configurations'][label] = entry
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(label, {key: value for key, value in entry.items()
                      if key not in ('views', 'appearance_lines')}, flush=True)
        if entry['appearance_lines']:
            print(entry['appearance_lines'][-1], flush=True)


if __name__ == '__main__':
    main()
