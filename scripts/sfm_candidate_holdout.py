"""Evaluate added cameras on reference-selected tracks fitted only in original views.

Usage: python scripts/sfm_candidate_holdout.py REFERENCE CANDIDATE CONTROL NEW_OUTPUT.json
Reference files are read after reconstruction; fresh point fits exclude added views.
Target pixels may overlap measurements used for the already estimated camera pose.
"""
import json
from pathlib import Path
import sys

import numpy as np
from sfm_quality_compare import load, normalize_reference_grid, fit_point, project, stats


def main():
    reference_path, candidate_path, control_path, output_path = map(Path, sys.argv[1:])
    if output_path.exists():
        raise ValueError('Output already exists')
    reference, tracks = load(reference_path)
    native, _ = load(candidate_path)
    control, _ = load(control_path)
    grid = normalize_reference_grid(reference, native)
    rb = {im['name']: im for im in reference.values()}
    nb = {im['name']: im for im in native.values()}
    cb = {im['name']: im for im in control.values()}
    added = sorted(nb.keys() - cb.keys())
    original = cb.keys() & nb.keys() & rb.keys()
    for name in original:
        if not np.allclose(nb[name]['R'], cb[name]['R'], atol=1e-12, rtol=0) or not np.allclose(nb[name]['C'], cb[name]['C'], atol=1e-12, rtol=0):
            raise ValueError('Original camera pose changed: ' + name)
    errors = {n: {'reference': [], 'native': []} for n in added}
    for _, track in tracks.values():
        observations = [(reference[int(i)]['name'], reference[int(i)]['obs'][int(j), :2]) for i, j in track]
        tests = [(n, xy) for n, xy in observations if n in errors]
        anchors = [(n, xy) for n, xy in observations if n in original]
        if not tests or len({n for n, _ in anchors}) < 3:
            continue
        for label, model in [('reference', rb), ('native', nb)]:
            point = fit_point([(model[n], xy) for n, xy in anchors])
            for n, xy in tests:
                errors[n][label].append(float(np.linalg.norm(project(point, model[n])[0][0] - xy)))
    report = dict(scope='Every reference track seen by an added camera and at least three original registered cameras; added cameras excluded from all point fits; no residual/depth filtering. Reference-selected observations are a comparison, not ground truth.',
                  original_poses_equal=True, pixel_grid=grid, added_views={})
    for name, labels in errors.items():
        report['added_views'][name] = {label: dict(stats(values), total_observations=len(values),
            nonfinite_observations=sum(not np.isfinite(v) for v in values)) for label, values in labels.items()}
    with output_path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
