"""Run one complete dataset with an explicit timeout and retained evidence.

Example:
  python experiments/run_sfm_acceptance.py --exe build/aetherscan/Release/aetherscan.exe \
    --images D:/ScanVideo/chuan/images --reference D:/ScanVideo/chuan/sparse/0 \
    --output-dir artifacts/sfm-regression --name chuan -- --mode global
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time

from sfm_acceptance import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', required=True)
    parser.add_argument('--images', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--cache-dir')
    parser.add_argument('--timeout-seconds', type=float, default=900)
    parser.add_argument('extra', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in ('.', '..'):
        parser.error('--name must be a filename stem')
    if args.timeout_seconds <= 0:
        parser.error('--timeout-seconds must be positive')
    directory = Path(args.output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    stem = directory / args.name
    log = stem.with_suffix('.console.log')
    report = stem.with_suffix('.run.json')
    if log.exists() or report.exists() or stem.with_suffix('.asfm').exists():
        parser.error('Run name already exists; choose another name to preserve evidence')
    cache = Path(args.cache_dir).resolve() if args.cache_dir else directory / 'cache' / args.name
    extra = args.extra[1:] if args.extra[:1] == ['--'] else args.extra
    if any(v.split('=')[0] in ('--images', '--output', '--cache-dir', '--gui') for v in extra):
        parser.error('Output, images, cache and diagnostics must be controlled by this runner')
    command = [str(Path(args.exe).resolve()), '--images', str(Path(args.images).resolve()),
               '--output', str(stem.with_suffix('.asfm')), '--cache-dir', str(cache), *extra]
    result = dict(command=command, reference=str(Path(args.reference).resolve()),
                  cache_preexisting=cache.exists() and any(cache.iterdir()),
                  timeout_seconds=args.timeout_seconds)
    start = time.perf_counter()
    with log.open('w', encoding='utf-8') as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
        try:
            result['exit_code'] = process.wait(timeout=args.timeout_seconds)
            result['timed_out'] = False
        except subprocess.TimeoutExpired:
            process.kill()
            result['exit_code'] = process.wait()
            result['timed_out'] = True
    result['wall_seconds'] = time.perf_counter()-start
    text = log.read_text(encoding='utf-8', errors='replace')
    result['stages'] = [dict(stage=m[0], seconds=float(m[1])) for m in
                        re.findall(r'stage finished: (\S+) elapsed_s=([\d.]+)', text)]
    result['checkpoint_hits'] = re.findall(r'checkpoint hit: (\w+)', text)
    memory = re.findall(r'peak_working_set_mb=([\d.]+)', text)
    result['peak_working_set_mb'] = float(memory[-1]) if memory else None
    result['passed'] = False
    if result['exit_code'] == 0:
        try:
            result['quality'] = evaluate(str(stem)+'_sfm_diagnostics.csv', args.reference)
            result['passed'] = result['quality']['passed']
        except Exception as error:
            result['evaluation_error'] = str(error)
    report.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: v for k,v in result.items() if k != 'quality'}, indent=2))
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
