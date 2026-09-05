"""Plot all matched camera centers using one Sim(3), without outlier removal."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from sfm_acceptance import reference_poses, similarity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',required=True)
    p.add_argument('--diagnostics',required=True,nargs='+')
    p.add_argument('--output',required=True)
    a = p.parse_args()
    reference = reference_poses(a.reference)
    fig, axes = plt.subplots(1,len(a.diagnostics),figsize=(7*len(a.diagnostics),6),squeeze=False)
    for ax, filename in zip(axes[0],a.diagnostics):
        with open(filename,encoding='utf-8-sig',newline='') as f:
            rows=[r for r in csv.DictReader(f) if int(r['registered']) and r['name'] in reference]
        x=np.array([[float(r['center_'+k]) for k in 'xyz'] for r in rows])
        y=np.array([reference[r['name']][0] for r in rows])
        scale,r,t=similarity(x,y)
        z=scale*x@r.T+t
        _,_,vt=np.linalg.svd(y-y.mean(0),full_matrices=False)
        yr=(y-y.mean(0))@vt[:2].T
        zr=(z-y.mean(0))@vt[:2].T
        ax.scatter(yr[:,0],yr[:,1],s=12,c='#73839b',label='COLMAP reference',alpha=.6)
        ax.scatter(zr[:,0],zr[:,1],s=9,c='#e87535',label='AetherScan',alpha=.8)
        for b,c in zip(yr,zr):
            ax.plot([b[0],c[0]],[b[1],c[1]],c='#e87535',alpha=.1,linewidth=.6)
        ax.set_aspect('equal',adjustable='datalim')
        ax.set_title(Path(filename).stem.replace('_sfm_diagnostics',''))
        ax.set_xlabel('Reference principal axis 1 (arbitrary units)')
        ax.set_ylabel('Reference principal axis 2 (arbitrary units)')
        ax.legend()
        ax.grid(alpha=.15)
    fig.suptitle('Single proper Sim(3) alignment; all matched cameras; reference is not ground truth')
    fig.tight_layout()
    fig.savefig(a.output,dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
