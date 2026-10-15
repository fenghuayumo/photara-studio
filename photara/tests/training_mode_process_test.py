"""Check mode defaults and the RGB-only policy through the training CLI."""
import argparse
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile

from training_resume_process_test import png


def run(executable, backend):
    with tempfile.TemporaryDirectory(prefix='photara-mode-loss-') as directory:
        root = Path(directory)
        images = root / 'images'; images.mkdir()
        sparse = root / 'sparse' / '0'; sparse.mkdir(parents=True)
        png(images / 'frame.png')
        (sparse / 'cameras.txt').write_text('1 PINHOLE 32 32 20 20 15.5 15.5\n')
        (sparse / 'images.txt').write_text(''.join(
            f'{i+1} 1 0 0 0 {-i * .1} 0 0 1 frame.png\n13.5 15.5 1 17.5 15.5 2\n' for i in range(3)))
        (sparse / 'points3D.txt').write_text(
            '1 -0.2 0 2 128 128 128 0.01 1 0 2 0 3 0\n'
            '2 0.2 0 2 128 128 128 0.01 1 1 2 1 3 1\n')
        base = [str(executable), '--images', str(images), '--splat-dataset', str(root),
                '--splat', '--mesh=false', '--backend', backend, '--splat-iterations', '3',
                '--splat-log-interval', '1', '--splat-progressive-resolution=false',
                '--splat-max-resolution', '32', '--splat-sh-degree', '1',
                '--splat-use-mask=false', '--splat-focus-region=false',
                '--splat-densification=false', '--splat-eval-checkpoints=false']
        stale = ['--splat-scale-reg', '1', '--splat-shape-erank-reg', '1',
                 '--splat-sh-reg', '1', '--splat-opacity-reg', '1',
                 '--splat-shape-flatness-reg', '1', '--splat-shape-rotation-reg', '1',
                 '--splat-shape-axis-ratio-reg', '1', '--splat-screen-size-limit', '.2',
                 '--splat-screen-size-penalty', '1', '--splat-match-alpha-weight', '1',
                 '--splat-geometry-loss=true', '--splat-geometry-from-iter', '1',
                 '--splat-normal-field=true', '--splat-normal-field-from-iter', '1',
                 '--splat-ppisp=true', '--splat-bilateral-grid=true',
                 '--splat-photometric-weight', '.7', '--splat-ssim-weight', '.35']
        cases = [('standard', []), ('high-quality', []), ('fast', []), ('fast', stale),
                 ('standard', ['--splat-scale-reg', '0', '--splat-shape-erank-reg', '0'])]
        for index, (mode, extra) in enumerate(cases):
            kwargs = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}
            result = subprocess.run(base + ['--output', str(root / f'{index}.ply'),
                '--splat-mode', mode] + extra, capture_output=True, text=True,
                encoding='utf-8', errors='replace', timeout=45, **kwargs)
            log = result.stdout + result.stderr
            assert result.returncode == 0, log[-4000:]
            objective = re.search(r'splat_objective ([^\r\n]+)', log)
            assert objective, log
            weights = {k: float(v) for k, v in (item.split('=') for item in objective[1].split())}
            assert all(math.isfinite(v) for v in weights.values())
            if mode == 'fast':
                for key, value in weights.items():
                    if key not in ('splat-photometric-weight', 'splat-ssim-weight', 'splat-shape-max-axis-ratio'):
                        assert value == 0, (mode, key, value)
                for token in ('opacity_decay=0 ', 'scale_decay=0 ', 'ppisp=0 ',
                              'bilateral_grid=0 ', 'depth_normal_loss=0 ', 'normal_field=0 '):
                    assert token in log, (mode, token)
                assert math.isclose(weights['splat-photometric-weight'], .7 if extra else 1)
                assert math.isclose(weights['splat-ssim-weight'], .35 if extra else .2)
            else:
                assert weights['splat-scale-reg'] == (0 if extra else .01)
                assert weights['splat-shape-erank-reg'] == (0 if extra else .001)
            assert 'splat iteration=3/3' in log, log[-4000:]
        print(f'{backend}: Standard/High quality defaults, explicit disable, and Fast override suppression passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('executable', type=Path)
    parser.add_argument('--backend', choices=('cuda', 'vulkan'), default='cuda')
    args = parser.parse_args()
    run(args.executable.resolve(), args.backend)
