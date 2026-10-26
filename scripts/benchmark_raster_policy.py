"""Serial, resumable raster/filter ablations with immutable camera initialization."""
import argparse
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

VARIANTS = {
    'legacy': ('legacy', 'legacy', []),
    'sort_z': ('z', 'legacy', ['--splat-depth-sort', 'z']),
    'alpha_999': ('alpha', 'legacy', ['--splat-alpha-clip', '0.999']),
    'near_001': ('near', 'legacy', ['--splat-near-plane', '0.01']),
    'pixel_half': ('pixel', 'legacy', ['--splat-pixel-center', '0.5']),
    'brush_legacy': ('brush', 'legacy', ['--splat-raster-profile', 'brush']),
    'brush_classic': ('brush', 'classic', ['--splat-raster-profile', 'brush', '--splat-filter-mode', 'classic']),
    'brush_mip2d': ('brush', 'mip2d', ['--splat-raster-profile', 'brush', '--splat-filter-mode', 'mip2d']),
    'brush_mip3d': ('brush', 'mip3d', ['--splat-raster-profile', 'brush', '--splat-filter-mode', 'mip3d']),
    'brush_mip': ('brush', 'mip', ['--splat-raster-profile', 'brush', '--splat-filter-mode', 'mip']),
}


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def run(command, logfile):
    start = time.perf_counter()
    with logfile.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            time.sleep(1)
    return process.returncode, time.perf_counter()-start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--colmap', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--exe', type=Path, default=Path('build/photara/Release/photara.exe'))
    parser.add_argument('--audit-exe', type=Path, default=Path('build/photara/Release/photara_splat_raster_audit.exe'))
    parser.add_argument('--iterations', type=int, default=8000)
    parser.add_argument('--progressive-interval', type=int, default=1000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--variants', nargs='+', choices=list(VARIANTS),
                        default=[name for name in VARIANTS if name != 'brush_mip'])
    parser.add_argument('--mesh', action='store_true')
    parser.add_argument('--geometry', action='store_true')
    parser.add_argument('--common-audit', action='store_true')
    args = parser.parse_args()
    for name in ('cameras.txt', 'images.txt', 'points3D.txt'):
        if not (args.colmap/name).is_file(): raise ValueError('Missing '+name)
    args.output.mkdir(parents=True, exist_ok=True)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.colmap.glob('*.txt'))}
    manifest = dict(images=str(args.images.resolve()), colmap=str(args.colmap.resolve()),
                    input_hashes=hashes, iterations=args.iterations, progressive_interval=args.progressive_interval,
                    seeds=args.seeds, variants=args.variants, mesh=args.mesh, geometry=args.geometry,
                    common_audit=args.common_audit,
                    exe_sha256=hashlib.sha256(args.exe.read_bytes()).hexdigest(),
                    audit_sha256=hashlib.sha256(args.audit_exe.read_bytes()).hexdigest())
    manifest_path = args.output/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Experiment manifest changed; choose a fresh output folder')
    write(manifest_path, manifest)
    results = []
    for seed in args.seeds:
        for variant in args.variants:
            profile, filter_mode, flags = VARIANTS[variant]
            folder = args.output/(variant+'_'+str(seed))
            folder.mkdir(exist_ok=True)
            result_file = folder/'result.json'
            if result_file.exists():
                result = json.loads(result_file.read_text())
                if result.get('complete'):
                    results.append(result)
                    continue
            command = [str(args.exe.resolve()), '--images', str(args.images.resolve()),
                '--splat-dataset', str(args.colmap.resolve()), '--output', str((folder/'model.ply').resolve()),
                '--splat', '--mesh=false', '--backend', 'cuda', '--splat-strategy', 'adc_igs', '--capture-mode', 'scene',
                '--splat-iterations', str(args.iterations), '--splat-seed', str(seed),
                '--splat-max-resolution', '0', '--splat-progressive-resolution=true',
                '--splat-progressive-interval', str(args.progressive_interval),
                '--splat-eval-split-every', '8', '--splat-eval-checkpoints=false',
                '--splat-use-mask=false', '--splat-preview-interval', '0',
                '--splat-densification-cap', '1000000', '--splat-log-interval', '500',
                '--splat-speedy-pruning=false', *flags]
            if args.geometry:
                command += ['--splat-geometry-loss', '--splat-geometry-from-iter', '2000',
                            '--splat-mv-geo-weight', '0', '--splat-mv-ncc-weight', '0']
            write(folder/'command.json', command)
            write(args.output/'status.json', dict(stage='training', variant=variant, seed=seed, start=time.time()))
            print('START', variant, seed, flush=True)
            try:
                snapshot = subprocess.check_output(['nvidia-smi',
                    '--query-gpu=timestamp,name,driver_version,temperature.gpu,clocks.gr,power.draw,utilization.gpu,memory.used',
                    '--format=csv'], text=True)
                (folder/'gpu-before.txt').write_text(snapshot, encoding='utf-8')
            except (OSError, subprocess.CalledProcessError):
                pass
            # Resume a completed training when only the audit was interrupted.
            if not (folder/'training.json').exists():
                code, wall = run(command, folder/'train.log')
                if code: raise RuntimeError('Training failed: '+str(folder/'train.log'))
                text = (folder/'train.log').read_text(errors='replace')
                result = dict(variant=variant, seed=seed, iterations=args.iterations, process_s=wall)
                patterns = {
                    'optimization_s': r'splat_optimization_s=([\d.]+)',
                    'psnr': r'splat_final_evaluation_views=.*average_psnr=([\d.]+)',
                    'valid_psnr': r'splat_final_evaluation_views=.*average_foreground_psnr=([\d.]+)',
                    'ssim': r'splat_final_evaluation_views=.*average_ssim=([\d.]+)',
                    'gaussians': r'splat_model=.*gaussians=(\d+)',
                }
                for key, pattern in patterns.items():
                    found = re.findall(pattern, text)
                    if not found: raise RuntimeError('Missing '+key+' in '+str(folder))
                    result[key] = float(found[-1])
                memory = re.findall(r'vram=(\d+)/(\d+)', text)
                result['peak_sampled_device_vram_bytes'] = max((int(m[0]) for m in memory), default=0)
                write(folder/'training.json', result)
            result = json.loads((folder/'training.json').read_text())
            audit = [str(args.audit_exe.resolve()), str(args.colmap.resolve()), str(args.images.resolve()),
                     str((folder/'model_splat.ply').resolve()), str((folder/'audit').resolve()), profile, filter_mode]
            # A .ply output is a pipeline stem; the trainer writes its Gaussian sidecar.
            if not Path(audit[3]).is_file():
                audit[3] = str((folder/'model.ply').resolve())
            if args.mesh: audit.append('--mesh')
            write(folder/'audit-command.json', audit)
            write(args.output/'status.json', dict(stage='audit', variant=variant, seed=seed, start=time.time()))
            code, seconds = run(audit, folder/'audit.log')
            if code: raise RuntimeError('Audit failed: '+str(folder/'audit.log'))
            result['audit_s'] = seconds
            if args.common_audit:
                common = audit[:]
                common[4] = str((folder/'audit_common').resolve())
                common[5:7] = ['legacy', 'legacy']
                write(folder/'audit-common-command.json', common)
                code, seconds = run(common, folder/'audit-common.log')
                if code: raise RuntimeError('Common audit failed: '+str(folder/'audit-common.log'))
                result['common_audit_s'] = seconds
            result['complete'] = True
            write(result_file, result)
            results.append(result)
            write(args.output/'results.json', results)
            print('DONE', json.dumps(result), flush=True)
    write(args.output/'results.json', results)
    write(args.output/'status.json', dict(stage='complete', count=len(results)))


if __name__ == '__main__':
    main()
