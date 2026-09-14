"""Measure camera-proximity diagnostics; these are proxies, not floater ground truth."""
import argparse
import json
import struct
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def read_scene(sparse):
    cameras = []
    with (sparse / 'images.bin').open('rb') as f:
        for _ in range(struct.unpack('<Q', f.read(8))[0]):
            ident, *rest = struct.unpack('<i7di', f.read(64))
            q, t = np.array(rest[:4]), np.array(rest[4:7])
            rotation = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
            name = bytearray()
            while (ch := f.read(1)) != b'\0':
                if not ch:
                    raise ValueError('Truncated image name')
                name.extend(ch)
            observations = struct.unpack('<Q', f.read(8))[0]
            f.seek(observations * 24, 1)
            cameras.append((ident, name.decode(), -rotation.T @ t, rotation))
    points = []
    with (sparse / 'points3D.bin').open('rb') as f:
        for _ in range(struct.unpack('<Q', f.read(8))[0]):
            row = struct.unpack('<Q3d3Bd', f.read(43))
            tracks = struct.unpack('<Q', f.read(8))[0]
            f.seek(tracks * 8, 1)
            points.append(row[1:4])
    return sorted(cameras, key=lambda x: x[1]), np.asarray(points)


def read_ply(path):
    with path.open('rb') as f:
        header = []
        while (line := f.readline().decode().strip()) != 'end_header':
            if not line:
                raise ValueError('Truncated PLY')
            header.append(line)
        names = [s.split()[2] for s in header if s.startswith('property float ')]
        count = int(next(s for s in header if s.startswith('element vertex ')).split()[-1])
        data = np.fromfile(f, dtype='<f4').reshape(count, len(names))
    return names, data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sparse', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    cameras, sparse = read_scene(a.sparse)
    centers = np.array([c[2] for c in cameras])
    clearance = cKDTree(sparse).query(centers)[0]
    names, model = read_ply(a.model)
    xyz = model[:, [names.index(x) for x in ['x', 'y', 'z']]]
    distance, nearest = cKDTree(centers).query(xyz)
    ratio = distance / clearance[nearest]
    opacity = 1 / (1 + np.exp(-np.clip(model[:, names.index('opacity')], -80, 80)))
    result = {'gaussians': len(xyz), 'camera_clearance_quantiles': np.quantile(clearance, [0, .1, .5, .9, 1]).tolist(),
              'near_camera_counts': {str(t): int(np.sum(ratio < t)) for t in [.1, .25, .5, .75, 1.]},
              'near_camera_opacity_sum': {str(t): float(opacity[ratio < t].sum()) for t in [.1, .25, .5, .75, 1.]},
              'note': 'Distance relative to nearest sparse point clearance is a diagnostic only, not proof of empty space.'}
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
