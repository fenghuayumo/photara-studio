"""Supplemental SSIM on exported PNGs and bilinearly resized pinhole inputs."""
import argparse,json,struct,re
from pathlib import Path
import cv2
import numpy as np
from skimage.metrics import structural_similarity
p=argparse.ArgumentParser()
p.add_argument('directory',type=Path)
p.add_argument('--baseline',type=Path)
p.add_argument('--scene',choices=['train','nyc'],required=True)
a=p.parse_args()
root=Path('D:/Models/tandt_db/tandt/train' if a.scene=='train' else 'D:/Models/nyc')
names=[]
with next((root/'sparse').rglob('images.bin')).open('rb') as f:
    count=struct.unpack('<Q',f.read(8))[0]
    for _ in range(count):
        f.read(64)
        name=bytearray()
        while (c:=f.read(1))!=b'\0':
            if not c: raise EOFError('COLMAP image name')
            name.extend(c)
        names.append(name.decode('utf-8'))
        f.seek(24*struct.unpack('<Q',f.read(8))[0],1)
names.sort()
rows=[]
for i,name in enumerate(names):
    if i%8: continue
    source=cv2.imread(str(root/'images'/name))
    if source is None: raise ValueError(name)
    row=dict(view=i,name=name)
    for strategy in ['adc_plus','adc_igs']:
        directory=a.baseline if strategy=='adc_plus' and a.baseline else a.directory
        path=directory/f'{a.scene}_{strategy}'/f'reconstruction_splat_view_{i}.png'
        pred=cv2.imread(str(path))
        if pred is None: raise ValueError(str(path))
        target=cv2.resize(source,(pred.shape[1],pred.shape[0]),interpolation=cv2.INTER_LINEAR_EXACT)
        row[strategy]=float(structural_similarity(target,pred,channel_axis=2,data_range=255,
            gaussian_weights=True,sigma=1.5,use_sample_covariance=False))
    rows.append(row)
result=dict(scene=a.scene,views=len(rows),adc_plus_ssim=float(np.mean([r['adc_plus'] for r in rows])),
    igs_ssim=float(np.mean([r['adc_igs'] for r in rows])),per_view=rows)
(a.directory/f'{a.scene}_ssim.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='per_view'}))
