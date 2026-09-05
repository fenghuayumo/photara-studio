"""Compare SfM diagnostics against an independent COLMAP model (numpy only).

Reference models are comparison baselines, not surveyed ground truth. All matched
cameras are reported after a single proper Sim(3); no outliers are removed.
"""
import argparse
import csv
import json
from pathlib import Path
import struct

import numpy as np


def rotation(q):
    q = np.asarray(q, dtype=float)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def reference_poses(directory):
    directory = Path(directory)
    poses = {}
    def add(name, q, t):
        r = rotation(q)
        if name in poses:
            raise ValueError(f"Duplicate reference image: {name}")
        poses[name] = (-r.T @ np.asarray(t), r)
    if (directory / 'images.bin').exists():
        with (directory / 'images.bin').open('rb') as f:
            count, = struct.unpack('<Q', f.read(8))
            for _ in range(count):
                values = struct.unpack('<i7di', f.read(64))
                name = bytearray()
                while True:
                    b = f.read(1)
                    if not b:
                        raise ValueError('Truncated COLMAP image name')
                    if b == b'\0':
                        break
                    name.extend(b)
                add(name.decode('utf-8'), values[1:5], values[5:8])
                points, = struct.unpack('<Q', f.read(8))
                f.seek(points * 24, 1)
    else:
        with (directory / 'images.txt').open(encoding='utf-8') as f:
            for line in f:
                if not line.strip() or line.startswith('#'):
                    continue
                p = line.split(maxsplit=9)
                add(p[9].strip(), list(map(float, p[1:5])), list(map(float, p[5:8])))
                next(f)  # POINTS2D line, which may be empty
    return poses


def similarity(x, y):
    xc, yc = x - x.mean(0), y - y.mean(0)
    u, singular, vt = np.linalg.svd(yc.T @ xc / len(x))
    signs = np.ones(3)
    signs[-1] = np.linalg.det(u @ vt)
    r = (u * signs) @ vt
    scale = (singular @ signs) / np.mean(np.sum(xc * xc, axis=1))
    return scale, r, y.mean(0) - scale * r @ x.mean(0)


def stats(values):
    values = np.asarray(values)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Empty or nonfinite metric')
    return dict(min=float(np.min(values)), median=float(np.median(values)), p95=float(np.percentile(values, 95)),
                max=float(np.max(values)), rms=float(np.sqrt(np.mean(values**2))))


def evaluate(diagnostics, reference):
    with open(diagnostics, encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    ref = reference_poses(reference)
    registered = [r for r in rows if int(r['registered'])]
    matched = [r for r in registered if r['name'] in ref]
    eligible_reference = set(ref).intersection(r['name'] for r in rows)
    if len(matched) < 3:
        raise ValueError('Need at least three reference correspondences')
    names = [r['name'] for r in matched]
    x = np.array([[float(r['center_'+axis]) for axis in 'xyz'] for r in matched])
    y = np.array([ref[n][0] for n in names])
    if np.linalg.matrix_rank(x-x.mean(0)) < 2 or np.linalg.matrix_rank(y-y.mean(0)) < 2:
        raise ValueError('Degenerate camera layout for Sim(3)')
    scale, align, translation = similarity(x, y)
    distances = np.linalg.norm(scale * x @ align.T + translation - y, axis=1)
    radius = np.sqrt(np.mean(np.sum((y-y.mean(0))**2, axis=1)))
    angles = []
    for row, name in zip(matched, names):
        estimated = rotation([float(row[k]) for k in ('qw','qx','qy','qz')])
        delta = estimated @ align.T @ ref[name][1].T
        angles.append(np.degrees(np.arccos(np.clip((np.trace(delta)-1)/2, -1, 1))))
    obs = np.array([int(r['observations']) for r in registered])
    rms = np.array([float(r['reprojection_rms_px']) for r in registered])
    weighted_rms = float(np.sqrt(np.sum(obs*rms*rms)/obs.sum()))
    result = dict(images=len(rows), registered=len(registered), reference_images=len(ref),
                  matched_reference=len(matched), registration_fraction=len(registered)/len(rows),
                  reference_match_fraction=len(matched)/len(registered),
                  reference_registration_fraction=len(matched)/len(eligible_reference),
                  observations=int(obs.sum()), observations_per_camera=stats(obs),
                  reprojection_rms_px=weighted_rms,
                  camera_reprojection_p95_px=stats([float(r['reprojection_p95_px']) for r in registered]),
                  center_error_percent_reference_radius=stats(100*distances/radius),
                  rotation_error_deg=stats(angles), similarity_scale=float(scale),
                  unregistered=[r['name'] for r in rows if not int(r['registered'])],
                  weak_cameras=[dict(name=r['name'], observations=int(r['observations']))
                                for r in registered if int(r['observations']) < 30],
                  worst_cameras=[dict(name=names[i], center_error_percent_radius=float(100*distances[i]/radius),
                                      rotation_error_deg=float(angles[i]))
                                 for i in np.argsort(distances)[-10:][::-1]])
    # Explicit provisional engineering gates, not a claim of universal product quality.
    result['gates'] = dict(registration=result['registration_fraction'] >= .98,
                           reference_coverage=result['reference_registration_fraction'] >= .98,
                           reprojection=weighted_rms <= 1.0,
                           center_p95=result['center_error_percent_reference_radius']['p95'] <= 5.0,
                           rotation_p95=result['rotation_error_deg']['p95'] <= 3.0,
                           camera_support=bool(np.min(obs) >= 30))
    result['passed'] = all(result['gates'].values())
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostics', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = evaluate(args.diagnostics, args.reference)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('unregistered','worst_cameras')}, indent=2))
    raise SystemExit(0 if result['passed'] else 2)
