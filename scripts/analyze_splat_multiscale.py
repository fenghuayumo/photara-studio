"""Compare scale-aware renders with Lanczos reductions of held-out photographs."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

LANCZOS = getattr(Image, 'Resampling', Image).LANCZOS


def ssim(a, b, mask):
    # Local 11x11 Gaussian statistics, channel independent, in [0, 1].
    blur = lambda x: gaussian_filter(x, sigma=(1.5, 1.5, 0), truncate=10/3, mode='reflect')
    ma, mb = blur(a), blur(b)
    va, vb = blur(a*a)-ma*ma, blur(b*b)-mb*mb
    cov = blur(a*b)-ma*mb
    field = ((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))
    return float(field[mask].mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folders', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for folder in args.folders:
        metadata = json.loads((folder/'multiscale.json').read_text())
        native_file = Path(metadata['native_audit'])
        native = json.loads(native_file.read_text())
        views = {v['index']: v for v in native['views']}
        samples = []
        for item in metadata['views']:
            index = item['index']
            view = views[index]
            size = (item['width'], item['height'])
            target = np.asarray(Image.open(native_file.parent/f'target_{index}.png').convert('RGB').resize(size, LANCZOS), dtype=np.float64)/255
            render = np.asarray(Image.open(folder/item['file']).convert('RGB'), dtype=np.float64)/255
            if view['has_mask']:
                valid = np.fromfile(native_file.parent/f'valid_{index}.f32', dtype='<f4').reshape(view['height'], view['width'])
                mask = np.asarray(Image.fromarray(valid, mode='F').resize(size, LANCZOS)) > .999
            else:
                mask = np.ones((item['height'], item['width']), bool)
            mse = np.mean((target[mask]-render[mask])**2)
            samples.append(dict(index=index, width=item['width'], height=item['height'],
                                psnr=float(-10*np.log10(max(mse, 1e-20))), ssim=ssim(target, render, mask)))
        for width in sorted({s['width'] for s in samples}, reverse=True):
            group = [s for s in samples if s['width']==width]
            rows.append(dict(variant=folder.parent.name, width=width,
                             valid_psnr=float(np.mean([s['psnr'] for s in group])),
                             ssim=float(np.mean([s['ssim'] for s in group]))))
        (folder/'analysis.json').write_text(json.dumps(samples, indent=2), encoding='utf-8')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'multiscale.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    with (args.output/'multiscale.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, ['variant','width','valid_psnr','ssim'])
        writer.writeheader()
        writer.writerows(rows)


if __name__=='__main__':
    main()
