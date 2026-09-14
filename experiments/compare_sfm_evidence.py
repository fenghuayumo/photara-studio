"""Symmetric leave-one-view geometry audit. Neither model is ground truth.

Use each reconstruction's feature tracks as a separate correspondence bank.
For every target observation, triangulate with three OTHER images under each
model, then predict the same target pixel. The target is withheld from point
triangulation, not from the original camera optimization: this is an internal
consistency audit, not an independent test-set accuracy measurement.
"""
import argparse
import csv
import json
import struct
from pathlib import Path

import cv2
import numpy as np
from sfm_acceptance import rotation


def unpack(f, fmt):
    return struct.unpack('<'+fmt, f.read(struct.calcsize('<'+fmt)))


def load_reference(root):
    cameras = {}
    with (root/'cameras.bin').open('rb') as f:
        for _ in range(unpack(f,'Q')[0]):
            cid, model, w, h = unpack(f,'iiQQ')
            if model != 4:
                raise ValueError('This audit requires OPENCV reference cameras')
            cameras[cid] = np.array(unpack(f,'8d'))
    images = {}
    with (root/'images.bin').open('rb') as f:
        for _ in range(unpack(f,'Q')[0]):
            values = unpack(f,'i7di')
            name = bytearray()
            while (b := f.read(1)) != b'\0':
                if not b: raise ValueError('Truncated name')
                name.extend(b)
            n = unpack(f,'Q')[0]
            obs = np.frombuffer(f.read(n*24),dtype=[('xy','<f8',2),('point','<i8')])
            r = rotation(values[1:5])
            images[values[0]] = dict(name=name.decode(), r=r,
                c=-r.T@np.array(values[5:8]), intr=cameras[values[8]], xy=obs['xy'])
    tracks = []
    with (root/'points3D.bin').open('rb') as f:
        for _ in range(unpack(f,'Q')[0]):
            _,x,y,z,_,_,_,error = unpack(f,'Q3d3Bd')
            n = unpack(f,'Q')[0]
            obs = np.frombuffer(f.read(n*8),dtype='<i4').reshape(-1,2)
            tracks.append((np.array([x,y,z]),obs))
    return images, tracks


class Reader:
    def __init__(self,b): self.b=b; self.p=0
    def get(self,fmt):
        value=struct.unpack_from('<'+fmt,self.b,self.p)
        self.p+=struct.calcsize('<'+fmt)
        return value
    def text(self):
        n=self.get('Q')[0]; s=bytes(self.b[self.p:self.p+n]).decode();self.p+=n;return s
    def record(self):
        n=self.get('Q')[0]; r=Reader(self.b[self.p:self.p+n]);self.p+=n;return r
    def array(self,dtype,n):
        a=np.frombuffer(self.b,dtype=dtype,count=n,offset=self.p)
        self.p+=a.nbytes;return a


def load_asfm(path):
    data=path.read_bytes()
    if data[:8]!=b'AETHSFM\0': raise ValueError('Bad ASFM')
    if struct.unpack_from('<Q',data,16)[0]!=len(data)-32: raise ValueError('Bad payload length')
    f=Reader(memoryview(data)[32:]); cameras={}
    for _ in range(f.get('Q')[0]):
        r=f.record(); cid,w,h=r.get('III'); intr=np.array(r.get('9d')[:8])
        trusted,model=r.get('BI')
        if model!=0: raise ValueError('This audit requires pinhole ASFM cameras')
        cameras[cid]=intr
    images={}
    for _ in range(f.get('Q')[0]):
        r=f.record(); iid,cid=r.get('II'); name=Path(r.text()).name
        rot=np.array(r.get('9d')).reshape(3,3); c=np.array(r.get('3d'))
        registered,w,h=r.get('BII'); r.text(); n=r.get('Q')[0]
        xy=r.array('<f4',n*5).reshape(-1,5)[:,:2]
        if registered: images[iid]=dict(name=name,r=rot,c=c,intr=cameras[cid],xy=xy)
    tracks=[]
    for _ in range(f.get('Q')[0]):
        r=f.record(); point=np.array(r.get('3d'));n=r.get('Q')[0]
        obs=r.array('<u4',n*2).reshape(-1,2); count=r.get('I')[0]
        tracks.append((point,obs[:count]))
    return images,tracks


def undistort(xy,intr):
    fx,fy,cx,cy,k1,k2,p1,p2=intr
    k=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
    return cv2.undistortPointsIter(np.asarray(xy,dtype=float).reshape(-1,1,2),k,
        np.array([k1,k2,p1,p2]),None,None,
        (cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,40,1e-12)).reshape(-1,2)


def project(points,im):
    p=(points-im['c'])@im['r'].T
    with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
        x,y=(p[:,:2]/p[:,2:]).T
        fx,fy,cx,cy,k1,k2,p1,p2=im['intr'];r2=x*x+y*y
        radial=1+k1*r2+k2*r2*r2
        out=np.column_stack([fx*(x*radial+2*p1*x*y+p2*(r2+2*x*x))+cx,
            fy*(y*radial+p1*(r2+2*y*y)+2*p2*x*y)+cy])
    out[p[:,2]<=0]=np.nan
    return out


def metrics(errors):
    e=np.asarray(errors); finite=np.isfinite(e)
    # Invalid depth/projection is counted as a failure, not dropped.
    safe=np.where(finite,e,np.inf)
    return dict(n=len(e),valid_fraction=float(finite.mean()),
        median_px=float(np.median(safe)) if finite.mean()>.5 else None,
        p95_px=float(np.percentile(safe,95)) if finite.mean()>.95 else None,
        within_2px=float(np.mean(safe<=2)),within_4px=float(np.mean(safe<=4)),
        within_10px=float(np.mean(safe<=10)),clipped_rms_20px=float(np.sqrt(np.mean(np.minimum(safe,20)**2))))


def audit(bank_images,tracks,models,limit):
    common=set.intersection(*(set(m) for m in models.values()))
    by_image={i:[] for i,im in bank_images.items() if im['name'] in common}
    for t,(_,obs) in enumerate(tracks):
        if len(obs)<4: continue
        if len({int(o[0]) for o in obs})!=len(obs): continue
        for image,feature in obs:
            if image in by_image: by_image[image].append((t,int(feature)))
    result={}; all_errors={name:[] for name in models}
    for image,entries in by_image.items():
        if not entries: continue
        rng=np.random.default_rng(20260913+int(image))
        selected=rng.choice(len(entries),min(limit,len(entries)),replace=False)
        target=[]; anchors=[]; pixels=[]
        for idx in selected:
            t,feature=entries[idx]
            others=sorted((bank_images[int(i)]['name'],int(i),int(k)) for i,k in tracks[t][1]
                if i!=image and i in bank_images and bank_images[int(i)]['name'] in common)
            if len(others)<3: continue
            chosen=[others[j] for j in [0,len(others)//2,len(others)-1]]
            anchors.append([o[0] for o in chosen]);pixels.append([bank_images[o[1]]['xy'][o[2]] for o in chosen])
            target.append(bank_images[image]['xy'][feature])
        if not target: continue
        name=bank_images[image]['name']; pixels=np.asarray(pixels);target=np.asarray(target)
        per_model={}
        for label,model in models.items():
            a=np.zeros((len(target),3,3));b=np.zeros((len(target),3))
            for j in range(3):
                for anchor in sorted(set(v[j] for v in anchors)):
                    ids=np.array([i for i,v in enumerate(anchors) if v[j]==anchor])
                    im=model[anchor];u=undistort(pixels[ids,j],im['intr'])
                    rays=np.column_stack([u,np.ones(len(u))])@im['r']
                    rays/=np.linalg.norm(rays,axis=1)[:,None]
                    normal=np.eye(3)[None]-rays[:,:,None]*rays[:,None,:]
                    a[ids]+=normal;b[ids]+=normal@im['c']
            # Same three anchors for both models; no reference-based outlier removal.
            points=(np.linalg.pinv(a,rcond=1e-10)@b[:,:,None])[:,:,0]
            error=np.linalg.norm(project(points,model[name])-target,axis=1)
            error[np.linalg.eigvalsh(a)[:,0]<1e-8]=np.inf
            for j in range(3):
                for anchor in sorted(set(v[j] for v in anchors)):
                    ids=np.array([i for i,v in enumerate(anchors) if v[j]==anchor])
                    im=model[anchor]
                    depth=((points[ids]-im['c'])@im['r'].T)[:,2]
                    error[ids[depth<=0]]=np.inf
            per_model[label]=metrics(error);all_errors[label].extend(error.tolist())
        result[name]=per_model
    return dict(pooled={k:metrics(v) for k,v in all_errors.items()},images=result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--asfm',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--samples',type=int,default=400)
    args=p.parse_args()
    ref,rt=load_reference(args.reference); ours,ot=load_asfm(args.asfm)
    models={'existing':{i['name']:i for i in ref.values()},'aetherscan':{i['name']:i for i in ours.values()}}
    out={'method':__doc__,'sample_cap_per_image':args.samples,'registered':{k:len(v) for k,v in models.items()}}
    for label,images,tracks in [('existing_tracks',ref,rt),('aetherscan_tracks',ours,ot)]:
        print('Auditing',label,'tracks',len(tracks),flush=True)
        out[label]=audit(images,tracks,models,args.samples)
        print(out[label]['pooled'],flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf8')
