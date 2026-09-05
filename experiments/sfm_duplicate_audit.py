"""Audit byte-identical image poses in both reconstruction and COLMAP reference."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from sfm_acceptance import reference_poses, rotation, stats


def audit(images, diagnostics, reference):
    groups = defaultdict(list)
    for path in sorted(Path(images).iterdir()):
        if path.is_file() and path.suffix.lower() in ('.jpg','.jpeg','.png','.tif','.tiff'):
            with path.open('rb') as f:
                digest = hashlib.file_digest(f, 'sha256').hexdigest()
            groups[digest].append(path.name)
    with open(diagnostics, encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    estimated = {r['name']:(np.array([float(r['center_'+a]) for a in 'xyz']),
                            rotation([float(r[a]) for a in ('qw','qx','qy','qz')]))
                 for r in rows if int(r['registered'])}
    def compare(poses):
        values = []
        centers = np.array([v[0] for v in poses.values()])
        radius = np.sqrt(np.mean(np.sum((centers-centers.mean(0))**2, axis=1)))
        for group in groups.values():
            valid = [n for n in group if n in poses]
            for i, a in enumerate(valid):
                for b in valid[i+1:]:
                    ca, ra = poses[a]; cb, rb = poses[b]
                    angle = np.degrees(np.arccos(np.clip((np.trace(ra@rb.T)-1)/2,-1,1)))
                    values.append(dict(first=a, second=b,
                                       center_percent_radius=float(100*np.linalg.norm(ca-cb)/radius),
                                       rotation_deg=float(angle)))
        return dict(compared_pairs=len(values),
                    center_percent_radius=stats([v['center_percent_radius'] for v in values]) if values else None,
                    rotation_deg=stats([v['rotation_deg'] for v in values]) if values else None,
                    worst_pairs=sorted(values,key=lambda v:v['center_percent_radius'], reverse=True)[:10])
    return dict(images=sum(map(len,groups.values())), unique_images=len(groups),
                duplicate_groups=sum(len(g)>1 for g in groups.values()),
                estimated=compare(estimated), reference=compare(reference_poses(reference)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('images','diagnostics','reference','output'):
        p.add_argument('--'+key,required=True)
    a = p.parse_args()
    result = audit(a.images,a.diagnostics,a.reference)
    Path(a.output).write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,indent=2))
