"""Fresh CPU SIFT correspondences; camera models are never used to select matches."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from compare_sfm_evidence import load_reference,load_asfm,undistort,metrics
from sfm_acceptance import rotation


def residual(first,second,xy1,xy2):
    # Sampson distance in undistorted pixel coordinates, in the original
    # image's pixel scale. Both models see exactly the same fresh matches.
    def k(im):
        fx,fy,cx,cy=im['intr'][:4]
        return np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
    r=second['r']@first['r'].T
    t=second['r']@(first['c']-second['c'])
    tx=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])
    f=np.linalg.inv(k(second)).T@tx@r@np.linalg.inv(k(first))
    x=np.column_stack([undistort(xy1,first['intr']),np.ones(len(xy1))])@k(first).T
    y=np.column_stack([undistort(xy2,second['intr']),np.ones(len(xy2))])@k(second).T
    fx=x@f.T; fy=y@f
    return np.abs(np.sum(y*fx,axis=1))/np.sqrt(np.sum(fx[:,:2]**2+fy[:,:2]**2,axis=1))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--asfm',type=Path,required=True)
    p.add_argument('--images',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume-log',type=Path)
    p.add_argument('--extra-diagnostics',type=Path)
    args=p.parse_args();cv2.setNumThreads(8)
    targets=['DCIM5783.jpg','DCIM5833.jpg','DCIM5630-HDR.jpg','DCIM5012-HDR.jpg','DCIM6228.jpg','DCIM5732.jpg']
    ref,rt=load_reference(args.reference);ours,ot=load_asfm(args.asfm)
    models={'existing':{im['name']:im for im in ref.values()},'aetherscan':{im['name']:im for im in ours.values()}}
    if args.extra_diagnostics:
        with args.extra_diagnostics.open(encoding='utf-8-sig') as f:
            models['aetherscan_fast']={r['name']:dict(
                r=rotation([float(r[k]) for k in ['qw','qx','qy','qz']]),
                c=np.array([float(r[k]) for k in ['center_x','center_y','center_z']]),
                intr=np.array([float(r[k]) for k in ['fx','fy','cx','cy','k1','k2','p1','p2']]))
                for r in csv.DictReader(f) if int(r['registered'])}
    pairs=set()
    for images,tracks in [(ref,rt),(ours,ot)]:
        neighbors={t:Counter() for t in targets}
        for _,obs in tracks:
            names=[images[int(i)]['name'] for i,_ in obs if i in images]
            for t in set(names)&set(targets): neighbors[t].update(n for n in names if n!=t)
        for t,counts in neighbors.items():
            for n,count in counts.most_common(3):
                if count>=20: pairs.add(tuple(sorted([t,n])))
    cache={};sift=cv2.SIFT_create(nfeatures=8000,contrastThreshold=.025)
    def features(name):
        if name not in cache:
            raw=np.fromfile(args.images/name,dtype=np.uint8)
            image=cv2.imdecode(raw,cv2.IMREAD_GRAYSCALE)
            kp,d=sift.detectAndCompute(image,None)
            cache[name]=(np.array([k.pt for k in kp]),d)
        return cache[name]
    matcher=cv2.FlannBasedMatcher(dict(algorithm=1,trees=4),dict(checks=128))
    completed={}
    if args.resume_log:
        for line in args.resume_log.read_text().splitlines():
            if line.startswith('{'):
                row=json.loads(line);completed[(row['first'],row['second'])]=row
    result=[]
    for first,second in sorted(pairs):
        if (first,second) in completed:
            result.append(completed[(first,second)]);continue
        if any(first not in model or second not in model for model in models.values()):
            result.append(dict(first=first,second=second,skipped='not registered in both models'));continue
        x,dx=features(first);y,dy=features(second)
        forward=matcher.knnMatch(dx,dy,k=2);reverse=matcher.knnMatch(dy,dx,k=2)
        rev={m.queryIdx:m.trainIdx for m,n in reverse if m.distance<.7*n.distance}
        matches=[(m.queryIdx,m.trainIdx) for m,n in forward
            if m.distance<.7*n.distance and rev.get(m.trainIdx)==m.queryIdx]
        row={'first':first,'second':second,'mutual_ratio_matches':len(matches)}
        if len(matches)>=30:
            a=x[[m[0] for m in matches]];b=y[[m[1] for m in matches]]
            cv2.setRNGSeed(20260913)
            f,mask=cv2.findFundamentalMat(a,b,cv2.USAC_MAGSAC,2.0,.999,10000)
            if mask is not None:
                ids=mask.ravel()!=0;a=a[ids];b=b[ids]
                row['independent_F_inliers']=len(a)
                if len(a)>=30:
                    row['models']={label:metrics(residual(model[first],model[second],a,b)) for label,model in models.items()}
        result.append(row);print(json.dumps(row),flush=True)
    args.output.write_text(json.dumps({'method':__doc__,'targets':targets,'pairs':result},indent=2),encoding='utf8')
