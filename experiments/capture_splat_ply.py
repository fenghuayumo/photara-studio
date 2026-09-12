"""Build fixed rasterizer inputs from a trained 3DGS PLY and captured camera.

Supports AetherScan's binary little-endian all-float Gaussian PLY. Scales,
rotations and opacity are activated; optional training-only fields are ignored.
Use a camera capture from the same dataset and pass its dimensions to the
A/B harness. This creates a benchmark scene, not a resumed training checkpoint.
"""
import argparse
from pathlib import Path
import shutil
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ply', type=Path, required=True)
    parser.add_argument('--camera-capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.ply.open('rb') as stream:
        names = []
        count = None
        if stream.readline().strip() != b'ply':
            raise ValueError('not a PLY file')
        while True:
            line = stream.readline()
            if not line:
                raise ValueError('incomplete PLY header')
            words = line.decode('ascii').strip().split()
            if words == ['end_header']:
                break
            if words[:1] == ['format'] and words[1:] != ['binary_little_endian', '1.0']:
                raise ValueError('requires binary little-endian PLY')
            if words[:1] == ['element']:
                if words[1] != 'vertex':
                    raise ValueError('only Gaussian vertex elements supported')
                count = int(words[2])
            if words[:1] == ['property']:
                if words[1] != 'float':
                    raise ValueError('only float properties supported')
                names.append(words[2])
        if not count or len(names) != len(set(names)):
            raise ValueError('invalid Gaussian header')
        data = np.fromfile(stream, dtype=np.dtype([(name, '<f4') for name in names]), count=count)
        if len(data) != count:
            raise ValueError('truncated Gaussian data')
    def columns(fields):
        return np.stack([data[name] for name in fields], axis=-1)
    sh = np.zeros((count, 16, 3), dtype=np.float32)
    sh[:, 0] = columns([f'f_dc_{i}' for i in range(3)])
    rest = [name for name in names if name.startswith('f_rest_')]
    if len(rest) not in (0, 9, 24, 45):
        raise ValueError('unsupported SH coefficient count')
    if rest:
        bases = len(rest) // 3
        sh[:, 1:bases+1] = columns([f'f_rest_{i}' for i in range(len(rest))]).reshape(count, 3, bases).transpose(0, 2, 1)
    rotation = columns([f'rot_{i}' for i in range(4)])
    rotation /= np.maximum(np.linalg.norm(rotation, axis=1, keepdims=True), 1e-12)
    arrays = {'.inputmeans': columns(['x', 'y', 'z']), '.inputsh': sh,
              '.inputrot': rotation,
              '.inputscales': np.exp(columns([f'scale_{i}' for i in range(3)])),
              '.inputopa': 1.0 / (1.0 + np.exp(-np.clip(data['opacity'], -80, 80)))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix, array in arrays.items():
        if not np.isfinite(array).all():
            raise ValueError(f'non-finite {suffix}')
        np.asarray(array, dtype='<f4').tofile(str(args.output) + suffix)
    for suffix in ('.inputcamera', '.hostcamera', '.gc', '.ga'):
        shutil.copyfile(str(args.camera_capture) + suffix, str(args.output) + suffix)
    print(f'captured {count} Gaussians: {args.output}')


if __name__ == '__main__':
    main()
