"""Compare two COLMAP models without using reference geometry in reconstruction.

Usage: python scripts/sfm_quality_compare.py REFERENCE NATIVE NEW_OUTPUT_DIRECTORY
Requires numpy, scipy, matplotlib. Binary models take precedence over text models.
The shared-track test fixes each camera solution,
fits a fresh 3D point on alternating reference camera groups, then evaluates the
remaining observations. Reference points are only used for their observation IDs.
"""
from pathlib import Path
import csv
import json
import sys
import struct

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def stats(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return dict(count=len(a), mean=float(np.mean(a)), median=float(np.median(a)),
                rms=float(np.sqrt(np.mean(a*a))), p95=float(np.percentile(a, 95)),
                max=float(np.max(a))) if len(a) else dict(count=0)


def load(directory):
    directory = Path(directory)
    if all((directory / (n+".bin")).exists() for n in ("cameras", "images", "points3D")):
        return load_binary(directory)
    cameras, images, points = {}, {}, {}
    for line in (directory / "cameras.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        v = line.split()
        model, p = v[1], np.array(v[4:], dtype=float)
        if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            intr = np.array([p[0], p[0], p[1], p[2]])
            dist = np.pad(p[3:], (0, 4-len(p[3:])))
        elif model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE"):
            intr, dist = p[:4], np.pad(p[4:], (0, 4-len(p[4:])))
        else:
            raise ValueError(f"Unsupported camera model: {model}")
        cameras[int(v[0])] = dict(model=model, intr=intr, dist=dist,
                                  width=int(v[2]), height=int(v[3]))
    with (directory / "images.txt").open() as stream:
        while True:
            line = stream.readline()
            if not line:
                break
            if not line.strip() or line.startswith("#"):
                continue
            v = line.split()
            q = np.array(v[1:5], dtype=float)
            R = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
            t = np.array(v[5:8], dtype=float)
            observation_line = stream.readline().strip()
            obs = (np.fromstring(observation_line, sep=" ").reshape(-1, 3)
                   if observation_line else np.empty((0, 3)))
            images[int(v[0])] = dict(name=Path(v[9]).name, R=R, t=t, C=-R.T@t,
                                     camera=cameras[int(v[8])], obs=obs)
    for line in (directory / "points3D.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        v = line.split()
        points[int(v[0])] = (np.array(v[1:4], dtype=float),
                             np.array(v[8:], dtype=int).reshape(-1, 2))
    return images, points


def load_binary(directory):
    def read(f, fmt):
        return struct.unpack("<"+fmt, f.read(struct.calcsize("<"+fmt)))
    models = {0:("SIMPLE_PINHOLE",3), 1:("PINHOLE",4), 2:("SIMPLE_RADIAL",4),
              3:("RADIAL",5), 4:("OPENCV",8), 5:("OPENCV_FISHEYE",8)}
    cameras, images, points = {}, {}, {}
    with (directory/"cameras.bin").open("rb") as f:
        for _ in range(read(f,"Q")[0]):
            cid, mid, width, height = read(f,"iiQQ")
            model, count = models[mid]
            p = np.array(read(f,"d"*count))
            if mid in (0,2,3):
                intr, dist = p[[0,0,1,2]], np.pad(p[3:],(0,7-count))
            else:
                intr, dist = p[:4], np.pad(p[4:],(0,8-count))
            cameras[cid] = dict(model=model, intr=intr, dist=dist, width=width, height=height)
    with (directory/"images.bin").open("rb") as f:
        for _ in range(read(f,"Q")[0]):
            v = read(f,"idddddddi")
            iid, cid = v[0], v[8]
            R = Rotation.from_quat(np.array(v[1:5])[[1,2,3,0]]).as_matrix()
            t = np.array(v[5:8])
            name = bytearray()
            while True:
                c = f.read(1)
                if c == b"\0": break
                if not c: raise ValueError("Truncated image name")
                name.extend(c)
            count = read(f,"Q")[0]
            raw = np.frombuffer(f.read(count*24), dtype=np.dtype([("xy","<f8",2),("pid","<i8")]))
            obs = np.column_stack((raw["xy"], raw["pid"]))
            images[iid] = dict(name=Path(name.decode("utf-8")).name, R=R, t=t, C=-R.T@t,
                               camera=cameras[cid], obs=obs)
    with (directory/"points3D.bin").open("rb") as f:
        for _ in range(read(f,"Q")[0]):
            v = read(f,"QdddBBBdQ")
            track = np.frombuffer(f.read(v[-1]*8), dtype="<u4").reshape(-1,2)
            points[v[0]] = (np.array(v[1:4]), track)
    return images, points


def normalize_reference_grid(reference, native):
    """Use the native image grid for both cameras and observed feature pixels."""
    native_by_name = {im["name"]: im for im in native.values()}
    targets = {}
    for im in reference.values():
        if im["name"] in native_by_name:
            cam = native_by_name[im["name"]]["camera"]
            targets.setdefault(id(im["camera"]), set()).add((cam["width"], cam["height"]))
    transforms, unmatched = {}, []
    for im in reference.values():
        cam = im["camera"]
        if im["name"] in native_by_name:
            target = native_by_name[im["name"]]["camera"]
            width, height = target["width"], target["height"]
        elif len(targets.get(id(cam), set())) == 1:
            width, height = next(iter(targets[id(cam)]))
        else:
            unmatched.append(im["name"])
            continue
        key = (cam["width"], cam["height"], width, height)
        transforms[key] = transforms.get(key, 0)+1
        if key[:2] == key[2:]:
            continue
        sx, sy = width/cam["width"], height/cam["height"]
        im["camera"] = dict(cam, width=width, height=height,
                            intr=cam["intr"]*np.array([sx, sy, sx, sy]))
        im["obs"] = im["obs"].copy()
        im["obs"][:, :2] *= np.array([sx, sy])
    return dict(pixel_units="native image grid; reference intrinsics and observations rescaled together",
                transforms=[dict(source_size=list(k[:2]), target_size=list(k[2:]), images=n)
                            for k,n in sorted(transforms.items())],
                unmatched_reference_views_without_inferred_grid=unmatched)


def project(xyz, image):
    p = np.atleast_2d(xyz) @ image["R"].T + image["t"]
    xy = p[:, :2] / p[:, 2:3]
    x, y = xy.T
    r2 = x*x + y*y
    k1, k2, p1, p2 = image["camera"]["dist"]
    if image["camera"]["model"] == "OPENCV_FISHEYE":
        r = np.sqrt(r2)
        theta = np.arctan(r)
        d = theta*(1+k1*theta**2+k2*theta**4+p1*theta**6+p2*theta**8)
        xy *= (d/np.maximum(r, 1e-15))[:, None]
    else:
        radial = 1+k1*r2+k2*r2*r2
        xy = np.column_stack((x*radial+2*p1*x*y+p2*(r2+2*x*x),
                              y*radial+p1*(r2+2*y*y)+2*p2*x*y))
    intr = image["camera"]["intr"]
    return xy*intr[:2]+intr[2:], p[:, 2]


def unproject(xy, image):
    intr = image["camera"]["intr"]
    target = (xy-intr[2:])/intr[:2]
    # Solve the actual projection, including tangential or fisheye distortion.
    def residual(z):
        p = image["C"] + image["R"].T @ np.r_[z, 1.]
        return (project(p, image)[0][0]-xy)/intr[:2]
    uv = least_squares(residual, target, max_nfev=30).x
    return image["R"].T @ np.r_[uv, 1.]


def fit_point(observations):
    A, b = np.zeros((3, 3)), np.zeros(3)
    for image, xy in observations:
        ray = unproject(xy, image)
        ray /= np.linalg.norm(ray)
        m = np.eye(3)-np.outer(ray, ray)
        A += m
        b += m@image["C"]
    point = np.linalg.lstsq(A, b, rcond=None)[0]
    def residual(x):
        return np.concatenate([project(x, im)[0][0]-xy for im, xy in observations])
    return least_squares(residual, point, max_nfev=40).x


def own_metrics(images, points):
    errors, lengths, behind = [], [], 0
    for image in images.values():
        obs = image["obs"]
        valid = np.array([int(pid) in points for pid in obs[:, 2]])
        obs = obs[valid]
        if len(obs):
            xyz = np.array([points[int(pid)][0] for pid in obs[:, 2]])
            uv, depth = project(xyz, image)
            errors.extend(np.linalg.norm(uv-obs[:, :2], axis=1))
            behind += int(np.sum(depth <= 0))
    lengths = [len(v[1]) for v in points.values()]
    return dict(views=len(images), points=len(points), reprojection_px=stats(errors),
                track_length=stats(lengths), behind_camera_observations=behind)


def main():
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    ref, native, output = sys.argv[1:]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    ri, rp = load(ref)
    ni, np_ = load(native)
    normalization = normalize_reference_grid(ri, ni)
    rb = {im["name"]: im for im in ri.values()}
    nb = {im["name"]: im for im in ni.values()}
    names = sorted(rb.keys() & nb.keys())
    if len(names) < 3:
        raise ValueError("At least three common cameras are required")
    src = np.array([nb[n]["C"] for n in names])
    dst = np.array([rb[n]["C"] for n in names])
    x, y = src-src.mean(0), dst-dst.mean(0)
    u, s, vt = np.linalg.svd(y.T@x)
    d = np.diag([1., 1., np.linalg.det(u@vt)])
    rotation = u@d@vt
    scale = np.sum(s*np.diag(d))/np.sum(x*x)
    aligned = scale*x@rotation.T+dst.mean(0)
    step = np.median(np.linalg.norm(np.diff(dst, axis=0), axis=1))
    center = np.linalg.norm(aligned-dst, axis=1)
    camera_radius = float(np.sqrt(np.mean(np.sum(y*y, axis=1))))
    angles = np.array([Rotation.from_matrix(nb[n]["R"]@rotation.T@rb[n]["R"].T).magnitude()*180/np.pi for n in names])
    # Relative rotations cancel the world-frame gauge. This also exposes local
    # jumps when a nearly straight camera path weakly constrains Sim(3) roll.
    relative_angles = np.array([
        Rotation.from_matrix(nb[b]["R"]@nb[a]["R"].T@rb[a]["R"]@rb[b]["R"].T).magnitude()*180/np.pi
        for a, b in zip(names, names[1:])])
    rf = np.array([rb[n]["camera"]["intr"][:2].mean() for n in names])
    nf = np.array([nb[n]["camera"]["intr"][:2].mean() for n in names])
    report = dict(reference=own_metrics(ri, rp), native=own_metrics(ni, np_),
                  coordinate_normalization=normalization,
                  common_views=len(names), missing_native=sorted(rb.keys()-nb.keys()),
                  alignment="all-common-camera least-squares Sim(3); no outlier removal",
                  center_error_steps=stats(center/step), rotation_error_deg=stats(angles),
                  center_error_camera_radius_fraction=stats(center/max(camera_radius, 1e-12)),
                  reference_camera_radius=camera_radius,
                  adjacent_relative_rotation_error_deg=stats(relative_angles),
                  focal_relative_difference=stats(np.abs(nf/rf-1)), reference_step=float(step))
    rows, candidates, coverage_candidates = [], [], {}
    for pid, (_, track) in sorted(rp.items()):
        common = [(ri[int(i)]["name"], ri[int(i)]["obs"][int(j), :2]) for i, j in track
                  if ri[int(i)]["name"] in nb]
        by_view = {}
        for name, xy in common:
            by_view.setdefault(name, []).append((name, xy))
        # A COLMAP track can contain multiple feature observations in one view.
        # Keep these together so a test camera never also enters this point fit.
        if len(by_view) >= 4:
            candidates.append((pid, [by_view[n] for n in sorted(by_view)]))
            for j, n in enumerate(sorted(by_view)):
                coverage_candidates.setdefault(n, (len(candidates)-1, j % 2))
    rng = np.random.default_rng(20261016)
    if not candidates:
        raise ValueError("No common tracks with at least four observations")
    selected = sorted(rng.choice(len(candidates), min(2000, len(candidates)), replace=False))
    # Randomize the test parity, then supplement the random sample so every
    # camera with eligible tracks is tested. Selection never uses either score.
    jobs = {(int(i), int(rng.integers(2))) for i in selected}
    covered = set()
    for i, parity in jobs:
        covered.update(group[0][0] for group in candidates[i][1][parity::2])
    for name in sorted(coverage_candidates):
        if name in covered:
            continue
        i, parity = coverage_candidates[name]
        jobs.add((i, parity))
        covered.update(group[0][0] for group in candidates[i][1][parity::2])
    errors = {"reference": [], "native": []}
    camera_errors = {label: {n: [] for n in names} for label in errors}
    failed_depth = {"reference": 0, "native": 0}
    for index, test_parity in sorted(jobs):
        pid, groups = candidates[index]
        fit_groups, test_groups = groups[1-test_parity::2], groups[test_parity::2]
        fit_obs = [obs for group in fit_groups for obs in group]
        test_obs = [obs for group in test_groups for obs in group]
        row = dict(point_id=pid, test_camera_parity=test_parity,
                   fit_views=len(fit_groups), held_out_views=len(test_groups),
                   fit_observations=len(fit_obs), held_out_observations=len(test_obs))
        for label, by_name in (("reference", rb), ("native", nb)):
            fit = [(by_name[n], xy) for n, xy in fit_obs]
            test = [(by_name[n], xy) for n, xy in test_obs]
            xyz = fit_point(fit)
            err = []
            for image, xy in test:
                uv, depth = project(xyz, image)
                err.append(float(np.linalg.norm(uv[0]-xy)))
                camera_errors[label][image["name"]].append(err[-1])
                failed_depth[label] += int(depth[0] <= 0)
            errors[label].extend(err)
            row[label+"_held_out_mean_px"] = float(np.mean(err))
        rows.append(row)
    report["shared_tracks"] = dict(seed=20261016, eligible_tracks=len(candidates),
        random_base_tracks=len(selected), sampled_tracks=len({i for i, _ in jobs}), point_fit_splits=len(jobs),
        eligible_test_views=len(coverage_candidates),
        method="Random 2000 tracks plus camera-coverage supplements; randomized alternating camera-group fit/test; same-view observations stay together; fixed cameras; fresh linear-ray + nonlinear point fit; no reference XYZ; no error/depth filtering",
        reprojection_px={k:stats(v) for k,v in errors.items()}, behind_camera=failed_depth)
    report["shared_tracks"]["covered_test_views"] = sum(bool(v) for v in camera_errors["native"].values())
    with (output/"shared_camera_errors.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "test_observations", "reference_mean_px", "reference_p95_px", "native_mean_px", "native_p95_px", "native_max_px"])
        for n in names:
            rs, ns = stats(camera_errors["reference"][n]), stats(camera_errors["native"][n])
            writer.writerow([n, ns["count"], rs.get("mean"), rs.get("p95"), ns.get("mean"), ns.get("p95"), ns.get("max")])
    (output/"metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output/"shared_tracks.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with (output/"cameras.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "center_error_steps", "rotation_error_deg", "reference_focal", "native_focal"])
        writer.writerows(zip(names, center/step, angles, rf, nf))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    _, _, v = np.linalg.svd(dst-dst.mean(0), full_matrices=False)
    for points, label in ((dst, "COLMAP"), (aligned, "Native aligned")):
        uv = (points-dst.mean(0))@v[:2].T
        axes[0,0].plot(*uv.T, ".-", label=label)
    axes[0,0].set(title="Camera trajectory (PCA plane)", aspect="equal")
    axes[0,1].plot(rf, label="COLMAP"); axes[0,1].plot(nf, label="Native")
    axes[0,1].set(title="Mean focal length", xlabel="Image order", ylabel="Pixels")
    axes[1,0].plot(angles, label="Rotation difference")
    axes[1,0].set(title="Rotation after Sim(3)", xlabel="Image order", ylabel="Degrees")
    for label, values in errors.items():
        a = np.sort(values)
        axes[1,1].plot(a, np.arange(1,len(a)+1)/len(a), label=label)
    axes[1,1].set(title="Shared tracks: held-out reprojection CDF", xlabel="Pixels (log scale)", ylabel="Fraction", xscale="log", xlim=(.01, 100))
    for ax in axes.flat:
        ax.legend(); ax.grid(alpha=.25)
    fig.savefig(output/"comparison.png", dpi=160)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
