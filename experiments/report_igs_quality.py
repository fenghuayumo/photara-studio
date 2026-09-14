"""Full-frame and fixed central-90% PNG comparison; no test-dependent crops."""
import argparse
import csv
import html
import json
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference-label', default='AetherScan v1')
    p.add_argument('--candidate-label', default='revised ADC-IGS')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    rows = []
    content = ['<!doctype html><meta charset="utf-8"><title>IGS quality comparison</title>',
               '<style>body{font:16px system-ui;background:#15171b;color:#eee;margin:30px}'
               'img{width:32%;height:auto}section{margin:30px 0}p{color:#ccc}</style>',
               f'<h1>Ground truth / {html.escape(a.reference_label)} / {html.escape(a.candidate_label)}</h1>',
               '<p>Same held-out cameras. Primary valid-pixel metrics use the same camera-derived '
               'source validity mask for both models, excluding undefined undistortion borders. '
               'Full-frame metrics include those borders. '
               'Central-90% metrics remove a fixed 5% on each edge for every view.</p>']
    for target in sorted(a.reference.glob('target_*.png'), key=lambda f: int(f.stem.split('_')[-1])):
        view = int(target.stem.split('_')[-1])
        gt = np.asarray(Image.open(target).convert('RGB'))
        row = {'view': view}
        for name, folder in [('reference', a.reference), ('candidate', a.candidate)]:
            pred = np.asarray(Image.open(folder / f'view_{view}.png').convert('RGB'))
            if pred.shape != gt.shape:
                raise ValueError('Mismatched image dimensions')
            y, x = int(gt.shape[0] * .05), int(gt.shape[1] * .05)
            for region, target_pixels, prediction in [('full', gt, pred),
                    ('central90', gt[y:-y, x:-x], pred[y:-y, x:-x])]:
                mse = np.mean((target_pixels.astype(float) - prediction.astype(float)) ** 2)
                row[f'{name}_{region}_psnr'] = float(10 * np.log10(255 ** 2 / max(mse, 1e-20)))
                row[f'{name}_{region}_ssim'] = float(structural_similarity(
                    target_pixels, prediction, data_range=255, channel_axis=2,
                    gaussian_weights=True, sigma=1.5, use_sample_covariance=False))
        rows.append(row)
        content.append(f'<section><h2>View {view}</h2><p>Central PSNR: '
                       f'{row["reference_central90_psnr"]:.3f} → {row["candidate_central90_psnr"]:.3f} dB</p>')
        for label, path in [('target', target), ('v1', a.reference / f'view_{view}.png'),
                            ('revised', a.candidate / f'view_{view}.png')]:
            # Portable local report: thumbnails are copied into its own directory.
            out = a.output / f'{label}_{view}.jpg'
            Image.open(path).save(out, quality=95)
            content.append(f'<img src="{html.escape(out.name)}" alt="{label}">')
        content.append('</section>')
        novel_ref = a.reference / f'novel_{view}.png'
        novel_new = a.candidate / f'novel_{view}.png'
        if novel_ref.exists() and novel_new.exists():
            content.append('<details><summary>Novel midpoint camera: reference / revised (no ground truth)</summary>')
            for label, path in [('v1', novel_ref), ('revised', novel_new)]:
                out = a.output / f'novel_{label}_{view}.jpg'
                Image.open(path).save(out, quality=95)
                content.append(f'<img src="{out.name}" alt="novel {label}">')
            content.append('</details>')
        for percent in [-40, -20, 20, 40]:
            paths = [(label, folder / f'dolly_{percent}_{view}.png')
                     for label, folder in [('reference', a.reference), ('revised', a.candidate)]]
            if not all(path.exists() for _, path in paths):
                continue
            content.append(f'<details><summary>Optical-axis displacement {percent}% '
                           '(sparse median depth; no ground truth): reference / revised</summary>')
            for label, path in paths:
                out = a.output / f'dolly_{percent}_{label}_{view}.jpg'
                Image.open(path).save(out, quality=95)
                content.append(f'<a href="{out.name}"><img src="{out.name}" alt="{label}"></a>')
            content.append('</details>')
    if not rows:
        raise ValueError('No target images')
    means = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != 'view'}
    for name, folder in [('reference', a.reference), ('candidate', a.candidate)]:
        with (folder / 'metrics.csv').open() as stream:
            native = list(csv.DictReader(stream))
        for metric in ['psnr', 'ssim']:
            means[f'{name}_valid_{metric}'] = float(np.mean([float(r[metric]) for r in native]))
    result = {'means': means, 'views': rows,
              'note': 'Valid metrics are float-render measurements on camera-defined source pixels; '
                      'full/crop metrics use PNGs. Central crop is fixed before comparison. '
                      'Novel midpoint images have no ground truth; none of these metrics counts floaters.'}
    content.insert(4, '<p>Valid-pixel PSNR: '
                   f'{means["reference_valid_psnr"]:.4f} → {means["candidate_valid_psnr"]:.4f} dB; '
                   'SSIM: '
                   f'{means["reference_valid_ssim"]:.6f} → {means["candidate_valid_ssim"]:.6f}. '
                   'Native float metrics; PNG full/crop details are in metrics.json.</p>')
    (a.output / 'metrics.json').write_text(json.dumps(result, indent=2))
    (a.output / 'comparison.html').write_text('\n'.join(content), encoding='utf-8')
    print(json.dumps(means, indent=2))


if __name__ == '__main__':
    main()
