"""Image, depth-consistency and mesh diagnostics; none is ground-truth geometry."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def read_map(folder, kind, view, channels=1):
    shape = (channels, view['height'], view['width']) if channels > 1 else (view['height'], view['width'])
    return np.fromfile(folder/(kind+'_'+str(view['index'])+'.f32'), dtype='<f4').reshape(shape)


def percentile(values, p):
    return float(np.percentile(values, p)) if len(values) else None


def geometry(folder, audit):
    maps = {}
    angles, coverages = [], []
    for view in audit['views']:
        depth = read_map(folder, 'depth', view)
        alpha = read_map(folder, 'alpha', view)
        mask = read_map(folder, 'valid', view) > 0 if view['has_mask'] else np.ones_like(depth, bool)
        valid = mask & np.isfinite(depth) & (depth > 0) & (alpha >= .5)
        maps[view['index']] = depth, alpha, mask, valid
        coverages.append(float(valid.sum()/max(mask.sum(), 1)))
        yy, xx = np.indices(depth.shape, dtype=np.float32)
        position = np.stack(((xx-view['cx'])*depth/view['fx'],
                             (yy-view['cy'])*depth/view['fy'], depth), axis=-1)
        dy = position[2:,1:-1]-position[:-2,1:-1]
        dx = position[1:-1,2:]-position[1:-1,:-2]
        normal = np.cross(dy, dx)
        norm = np.linalg.norm(normal, axis=-1)
        learned = read_map(folder, 'normal', view, 3).transpose(1,2,0)[1:-1,1:-1]
        learned_norm = np.linalg.norm(learned, axis=-1)
        good = valid[1:-1,1:-1] & valid[2:,1:-1] & valid[:-2,1:-1] & valid[1:-1,2:] & valid[1:-1,:-2]
        good &= (norm > 1e-10) & (learned_norm > 1e-10)
        good &= (np.abs(depth[2:,1:-1]-depth[:-2,1:-1]) < .05*depth[1:-1,1:-1])
        good &= (np.abs(depth[1:-1,2:]-depth[1:-1,:-2]) < .05*depth[1:-1,1:-1])
        cosine = np.sum(normal*learned, axis=-1)/np.maximum(norm*learned_norm,1e-20)
        angles.append(np.rad2deg(np.arccos(np.clip(np.abs(cosine[good][::4]), 0, 1))))
    errors, visible_errors = [], []
    for view in audit['views']:
        others = sorted((v for v in audit['views'] if v['index'] != view['index']),
                        key=lambda v: np.linalg.norm(np.array(v['position'])-view['position']))[:3]
        d, _, _, valid = maps[view['index']]
        yy, xx = np.indices(d[::4,::4].shape)
        yy, xx = yy*4, xx*4
        keep = valid[yy,xx]
        x, y, z = xx[keep], yy[keep], d[yy,xx][keep]
        camera = np.vstack(((x-view['cx'])*z/view['fx'], (y-view['cy'])*z/view['fy'], z))
        transform = np.array(view['world_to_camera_column_major']).reshape(4,4,order='F')
        world = transform[:3,:3].T@(camera-transform[:3,3:4])
        for other in others:
            t = np.array(other['world_to_camera_column_major']).reshape(4,4,order='F')
            point = t[:3,:3]@world+t[:3,3:4]
            u = other['fx']*point[0]/np.maximum(point[2],1e-12)+other['cx']
            v = other['fy']*point[1]/np.maximum(point[2],1e-12)+other['cy']
            nd, na, nm, _ = maps[other['index']]
            sample = lambda data: map_coordinates(data.astype(np.float32), [v,u], order=1, mode='constant', cval=0)
            sampled = sample(nd)
            good = (point[2]>0) & (sampled>0) & (sample(na)>=.5) & (sample(nm)>.999)
            relative = np.abs(point[2][good]-sampled[good])/sampled[good]
            errors.append(relative)
            visible_errors.append(relative[point[2][good] <= sampled[good]*1.01])
    angles = np.concatenate(angles)
    errors = np.concatenate(errors)
    visible_errors = np.concatenate(visible_errors)
    return dict(depth_coverage=float(np.mean(coverages)),
                depth_normal_median_deg=percentile(angles,50), depth_normal_p90_deg=percentile(angles,90),
                multiview_relative_median=percentile(errors,50), multiview_relative_p90=percentile(errors,90),
                multiview_within_1pct=float(np.mean(errors<=.01)) if len(errors) else None,
                multiview_visible_relative_median=percentile(visible_errors,50),
                multiview_samples=len(errors), multiview_visible_samples=len(visible_errors))


def gaussian_stats(model, cameras):
    with model.open('rb') as source:
        dtype, count = [], None
        while True:
            line = source.readline().decode('ascii').strip()
            if line.startswith('element vertex '): count=int(line.split()[-1])
            elif line.startswith('property '):
                _, type_name, name = line.split()
                dtype.append((name, {'float':'<f4','double':'<f8','uchar':'u1','int':'<i4','uint':'<u4'}[type_name]))
            elif line == 'end_header': break
            elif not line: raise ValueError('Invalid Gaussian PLY')
        data = np.fromfile(source, dtype=np.dtype(dtype), count=count)
    position = np.column_stack([data[a] for a in 'xyz'])
    scales = np.exp(np.column_stack([data['scale_'+str(a)] for a in range(3)]).astype(np.float64))
    opacity = 1/(1+np.exp(-data['opacity'].astype(np.float64)))
    if 'filter_3D' in data.dtype.names:
        filtered = np.sqrt(scales*scales+data['filter_3D'][:,None]**2)
        opacity *= np.prod(scales/filtered,axis=1)
        scales = filtered
    nearest = cKDTree(np.asarray(cameras)).query(position, workers=2)[0]
    center = np.mean(cameras,axis=0)
    extent = 1.1*np.max(np.linalg.norm(np.asarray(cameras)-center,axis=1))
    active = opacity>=.01
    ratios = scales.max(axis=1)/np.maximum(scales.min(axis=1),1e-20)
    return dict(effective_opacity_median=float(np.median(opacity)),
                scale_axis_ratio_p95=percentile(ratios[active],95), scale_axis_ratio_p99=percentile(ratios[active],99),
                near_camera_5pct_extent=int(np.sum(active & (nearest<.05*extent))),
                near_camera_10pct_extent=int(np.sum(active & (nearest<.10*extent))),
                active_gaussians=int(active.sum()), camera_extent=float(extent))


def mesh_stats(path):
    import trimesh
    mesh = trimesh.load_mesh(path, process=False)
    counts = np.bincount(mesh.edges_unique_inverse)
    edges = mesh.edges_unique
    adjacency = coo_matrix((np.ones(len(edges),np.uint8),(edges[:,0],edges[:,1])),
                           shape=(len(mesh.vertices),len(mesh.vertices))).tocsr()
    _, labels = connected_components(adjacency, directed=False)
    face_components = np.bincount(labels[np.asarray(mesh.faces)[:,0]])
    populated = face_components[face_components>0]
    return dict(mesh_vertices=len(mesh.vertices), mesh_faces=len(mesh.faces),
                mesh_components=len(populated), mesh_largest_face_fraction=float(populated.max()/len(mesh.faces)),
                mesh_boundary_edges=int(np.sum(counts==1)), mesh_nonmanifold_edges=int(np.sum(counts>2)),
                mesh_area=float(mesh.area), mesh_extents=mesh.extents.tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lpips', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    metric = None
    if args.lpips:
        import torch
        import lpips
        torch.set_num_threads(2)
        metric = lpips.LPIPS(net='squeeze', spatial=True, verbose=False).eval()
    rows = []
    for runs in args.runs:
        manifest = json.loads((runs/'manifest.json').read_text())
        # The camera centers in an audit are a fixed subset. Use all 76 centers
        # from COLMAP for the near-camera diagnostic, without fitting to a model.
        from scipy.spatial.transform import Rotation
        cameras = []
        lines=(Path(manifest['colmap'])/'images.txt').read_text().splitlines()
        for line in lines:
            p=line.split()
            if len(p)<10 or line.startswith('#'): continue
            try:
                q=list(map(float,p[1:5])); t=np.array(list(map(float,p[5:8])))
                if abs(np.linalg.norm(q)-1)>.01: continue
                r=Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix()
                cameras.append((-r.T@t).tolist())
            except ValueError: continue
        for result in json.loads((runs/'results.json').read_text()):
            folder=runs/(result['variant']+'_'+str(result['seed']))
            for name in ('audit','audit_common'):
                audit_folder=folder/name
                if not (audit_folder/'audit.json').exists(): continue
                cache=audit_folder/'analysis.json'
                if cache.exists():
                    extra=json.loads(cache.read_text())
                else:
                    audit=json.loads((audit_folder/'audit.json').read_text())
                    extra=geometry(audit_folder,audit)
                    for key in ('psnr','valid_psnr','ssim'):
                        extra['audit_'+key]=float(np.mean([v[key] for v in audit['views']]))
                    for geometry_pass, label in ((False,'rgb_forward_ms'),(True,'geometry_forward_ms')):
                        extra[label]=float(np.median([v['median_ms'] for v in audit['timing'] if v['geometry']==geometry_pass]))
                    if (audit_folder/'mesh.ply').exists():
                        extra.update(mesh_stats(audit_folder/'mesh.ply'))
                        extra['mesh_seconds']=audit['mesh']['seconds']
                    model=folder/'model_splat.ply'
                    if not model.exists(): model=folder/'model.ply'
                    extra.update(gaussian_stats(model,cameras))
                if metric is not None and 'lpips_squeeze_valid' not in extra:
                    audit=json.loads((audit_folder/'audit.json').read_text())
                    scores=[]
                    for view in audit['views']:
                        stem=str(view['index'])
                        load=lambda kind: torch.from_numpy(np.asarray(Image.open(audit_folder/(kind+'_'+stem+'.png')).convert('RGB')).copy()).permute(2,0,1).float().unsqueeze(0)/127.5-1
                        with torch.no_grad(): field=metric(load('render'),load('target'))[0,0].numpy()
                        mask=read_map(audit_folder,'valid',view)>0 if view['has_mask'] else np.ones_like(field,bool)
                        scores.append(float(np.mean(field[mask])))
                    extra['lpips_squeeze_valid']=float(np.mean(scores))
                cache.write_text(json.dumps(extra,indent=2),encoding='utf-8')
                row=dict(result, experiment=runs.name, renderer=name, folder=str(folder.resolve()), **extra)
                rows.append(row)
                print('ANALYZED',runs.name,result['variant'],name,flush=True)
    (args.output/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with (args.output/'results.csv').open('w',newline='',encoding='utf-8-sig') as out:
        writer=csv.DictWriter(out,keys); writer.writeheader(); writer.writerows(rows)


if __name__=='__main__': main()
