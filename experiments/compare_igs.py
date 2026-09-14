"""Paired COLMAP holdout benchmark. Logs, commands and metrics stay per run."""
import argparse, hashlib, json, re, subprocess, time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--iterations', type=int, default=10000)
parser.add_argument('--resolution', type=int, default=1280)
parser.add_argument('--scenes', nargs='+', choices=['train','nyc'], default=['train','nyc'])
parser.add_argument('--strategies', nargs='+', choices=['adc_plus','adc_igs'], default=['adc_plus','adc_igs'])
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
exe = root / 'build/aetherscan/Release/aetherscan.exe'
datasets = {'train': Path('D:/Models/tandt_db/tandt/train'), 'nyc': Path('D:/Models/nyc')}
args.output.mkdir(parents=True, exist_ok=True)
results_path = args.output / 'results.json'
results = json.loads(results_path.read_text()) if results_path.exists() else []
for scene in args.scenes:
    dataset = datasets[scene]
    for strategy in args.strategies:
        out = args.output / f'{scene}_{strategy}'
        out.mkdir(exist_ok=False)
        command = [str(exe), '--images', str(dataset / 'images'), '--output', str(out / 'reconstruction.ply'),
            '--splat', '--splat-dataset', str(dataset / 'sparse'),
            '--splat-strategy', strategy, '--splat-iterations', str(args.iterations),
            '--splat-max-resolution', str(args.resolution), '--splat-eval-split-every', '8',
            '--splat-progressive-resolution=false', '--splat-use-mask=false',
            '--splat-densification-cap', '2000000', '--splat-log-interval', '500']
        (out / 'command.json').write_text(json.dumps(command, indent=2))
        with exe.open('rb') as binary:
            binary_hash = hashlib.file_digest(binary, 'sha256').hexdigest()
        (out / 'executable_sha256.txt').write_text(binary_hash)
        print(f'START {scene} {strategy}', flush=True)
        started = time.time()
        with (out / 'run.log').open('w') as log:
            run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        text = (out / 'run.log').read_text(errors='replace')
        match = re.search(r'splat_final_evaluation_views=(\d+) average_psnr=([\d.]+)', text)
        result = dict(scene=scene, strategy=strategy, exit_code=run.returncode, seconds=time.time()-started,
            holdout_views=int(match[1]) if match else None, psnr=float(match[2]) if match else None)
        results.append(result)
        (args.output / 'results.json').write_text(json.dumps(results, indent=2))
        print(json.dumps(result), flush=True)
        if run.returncode or not match:
            raise RuntimeError(f'Failed run: {out / "run.log"}')
