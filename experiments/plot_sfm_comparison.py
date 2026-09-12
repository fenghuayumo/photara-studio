"""Plot every shared camera after a single proper Sim(3), without outlier removal."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from sfm_acceptance import reference_poses, rotation, similarity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostics', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--title', default='SfM camera comparison')
    args = parser.parse_args()
    ref = reference_poses(args.reference)
    with open(args.diagnostics, encoding='utf-8-sig', newline='') as f:
        rows = sorted((r for r in csv.DictReader(f)
                       if int(r['registered']) and r['name'] in ref), key=lambda r: r['name'])
    if len(rows) < 3:
        raise ValueError('At least three shared cameras required')
    x = np.array([[float(r['center_'+axis]) for axis in 'xyz'] for r in rows])
    y = np.array([ref[r['name']][0] for r in rows])
    if np.linalg.matrix_rank(x-x.mean(0)) < 2 or np.linalg.matrix_rank(y-y.mean(0)) < 2:
        raise ValueError('Degenerate camera layout')
    scale, align, translation = similarity(x, y)
    aligned = scale*x@align.T + translation
    radius = np.sqrt(np.mean(np.sum((y-y.mean(0))**2, axis=1)))
    distances = 100*np.linalg.norm(aligned-y, axis=1)/radius
    angles = np.array([np.degrees(np.arccos(np.clip((np.trace(
        rotation([float(r[k]) for k in ('qw','qx','qy','qz')]) @ align.T @
        ref[r['name']][1].T)-1)/2, -1, 1))) for r in rows])
    _, _, axes = np.linalg.svd(y-y.mean(0), full_matrices=False)
    figure, panels = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    figure.suptitle(f'{args.title}\n{len(rows)} shared cameras; one Sim(3); no outlier removal')
    for panel, dims in zip(panels[0], [(0, 1), (0, 2)]):
        projected_ref = (y-y.mean(0))@axes.T/radius
        projected_est = (aligned-y.mean(0))@axes.T/radius
        panel.scatter(projected_ref[:, dims[0]], projected_ref[:, dims[1]],
                      s=10, c='#64748b', label='Reference')
        panel.scatter(projected_est[:, dims[0]], projected_est[:, dims[1]],
                      s=5, c='#0891b2', alpha=.65, label='AetherScan')
        panel.set(xlabel=f'PC{dims[0]+1} / reference radius',
                  ylabel=f'PC{dims[1]+1} / reference radius')
        panel.set_aspect('equal', adjustable='datalim')
        panel.legend()
    for panel, values, unit in zip(panels[1], [distances, angles],
                                   ['Position difference (% reference radius)', 'Rotation difference (degrees)']):
        panel.plot(values, linewidth=.8, color='#0891b2')
        panel.axhline(np.percentile(values, 95), color='#e08820', linestyle='--',
                      label=f'P95 {np.percentile(values, 95):.3f}; max {values.max():.3f}')
        worst = int(np.argmax(values))
        panel.annotate(rows[worst]['name'], (worst, values[worst]),
                       xytext=(-8, -18), textcoords='offset points', ha='right', fontsize=8)
        panel.set(xlabel='Shared camera index (filename order)', ylabel=unit)
        panel.set_ylim(bottom=0)
        panel.legend()
    for panel in panels.flat:
        panel.grid(alpha=.15)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    with output.with_suffix('.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'center_error_percent_reference_radius', 'rotation_error_deg'])
        writer.writerows((r['name'], d, a) for r, d, a in zip(rows, distances, angles))


if __name__ == '__main__':
    main()
