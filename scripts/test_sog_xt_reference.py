"""Cross-decode a degree-3 export using the paper's image/range conventions.

Independent NumPy decoder: no Photara codec calls. Compare to the PLY produced
by the C++ decoder. Also emit ordinary block YAML without native-only fields.
"""
import argparse
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import yaml


def check(archive, native, output):
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as package:
        def image(field):
            with package.open(field + '.webp') as stream:
                return np.array(Image.open(stream), dtype=np.float32)
        metadata = yaml.safe_load(package.read('container_meta.yaml'))
        ranges = {}
        for operation in metadata['ops']:
            for transform in operation['transforms']:
                if 'simple_quantize' in transform:
                    ranges[operation['input_fields'][0]] = transform['simple_quantize']

        def expand(field, codes):
            config = ranges[field]
            fraction = (codes - codes.min()) / max(float(codes.max() - codes.min()), 1e-8)
            low = np.asarray(config['min_values'], dtype=np.float32)
            high = np.asarray(config['max_values'], dtype=np.float32)
            return low + fraction * (high - low)

        mask = image('active_mask')[..., 0].reshape(-1) != 0
        side = metadata['meta']['f_rest_centroids_side']
        position = expand('means', image('means_bytes_0') + image('means_bytes_1') * 256)
        position = np.sign(position) * np.expm1(np.abs(position))
        scale = expand('scales', image('scales'))
        opacity = expand('opacities', image('opacities')[..., 0])
        rotation = expand('quaternions', image('quaternions'))
        rotation /= np.linalg.norm(rotation, axis=-1, keepdims=True).clip(min=1e-12)
        dc = expand('f_dc', image('f_dc'))
        tiled = image('f_rest_centroids')
        codes = np.stack([tiled[(b // 5) * side:(b // 5 + 1) * side,
                               (b % 5) * side:(b % 5 + 1) * side] for b in range(15)], axis=-1)
        # [H,W,RGB,15] is the reference's channel-major coefficient order.
        centers = expand('f_rest_centroids', codes.reshape(side, side, 45))
        labels = image('f_rest_labels').astype(np.int64)
        ac = centers[labels[..., 1], labels[..., 0]].reshape(-1, 3, 15)
        sh = np.concatenate([dc.reshape(-1, 3, 1), ac], axis=2)
        fields = {}
        for d, name in enumerate(('x', 'y', 'z')):
            fields[name] = position.reshape(-1, 3)[mask, d]
            fields['scale_' + str(d)] = scale.reshape(-1, 3)[mask, d]
            fields['f_dc_' + str(d)] = sh[mask, d, 0]
        fields['opacity'] = opacity.reshape(-1)[mask]
        for d in range(4):
            fields['rot_' + str(d)] = rotation.reshape(-1, 4)[mask, d]
        for d in range(45):
            fields['f_rest_' + str(d)] = sh[mask, d // 15, d % 15 + 1]

        # A plain upstream-style directory, for testing the C++ reader too.
        for entry in package.namelist():
            if entry.endswith('.webp') and Path(entry).name == entry:
                (output / entry).write_bytes(package.read(entry))
        metadata.pop('format', None)
        metadata.pop('version', None)
        for name in ('count', 'sh_degree', 'antialiased'):
            metadata['meta'].pop(name, None)
        for operation in metadata['ops']:
            for transform in operation['transforms']:
                if 'simple_quantize' in transform:
                    transform['simple_quantize'].pop('levels', None)
        (output / 'container_meta.yaml').write_text(yaml.safe_dump(metadata), encoding='utf-8')

    with native.open('rb') as stream:
        names = []
        count = 0
        while True:
            raw = stream.readline()
            if not raw:
                raise ValueError('Truncated PLY')
            line = raw.decode('ascii').strip()
            if line.startswith('element vertex '):
                count = int(line.split()[-1])
            if line.startswith('property '):
                _, kind, name = line.split()
                assert kind == 'float', kind
                names.append(name)
            if line == 'end_header':
                break
        offset = stream.tell()
    matrix = np.memmap(native, dtype='<f4', mode='r', offset=offset, shape=(count, len(names)))
    errors = {}
    for name, expected in fields.items():
        actual = matrix[:, names.index(name)]
        assert len(actual) == len(expected)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5, err_msg=name)
        errors[name] = float(np.max(np.abs(actual - expected)))
    report = {'gaussians': count, 'compared_fields': len(fields),
              'maximum_absolute_difference': max(errors.values()), 'field_errors': errors}
    (output / 'reference-check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'field_errors'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    parser.add_argument('native_decoded_ply', type=Path)
    parser.add_argument('output_directory', type=Path)
    args = parser.parse_args()
    check(args.archive, args.native_decoded_ply, args.output_directory)
