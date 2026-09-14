"""Compare camera geometry on fresh CPU SIFT matches, independent of SfM tracks.

Temporal neighbors are selected by filename, never by either model's residuals.
This checks relative epipolar geometry, not absolute pose or metric scale.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from sfm_acceptance import rotation


def load_model(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        result = {}
        for row in csv.DictReader(source):
            if not int(row["registered"]):
                continue
            fx, fy, cx, cy = [float(row[k]) for k in ("fx", "fy", "cx", "cy")]
            if row.get("camera_model", "pinhole") != "pinhole":
                raise ValueError("This audit currently supports pinhole cameras only")
            result[row["name"]] = dict(
                k=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]]),
                distortion=np.array([float(row[k]) for k in ("k1", "k2", "p1", "p2")]),
                r=rotation([float(row[k]) for k in ("qw", "qx", "qy", "qz")]),
                c=np.array([float(row["center_" + k]) for k in "xyz"]))
        return result


def residual(a, b, x, y):
    r = b["r"] @ a["r"].T
    t = b["r"] @ (a["c"] - b["c"])
    if np.linalg.norm(t) < 1e-12:
        return None
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    f = np.linalg.inv(b["k"]).T @ tx @ r @ np.linalg.inv(a["k"])
    def pixels(im, xy):
        uv = cv2.undistortPoints(xy.reshape(-1, 1, 2), im["k"], im["distortion"], P=im["k"])
        return np.column_stack((uv.reshape(-1, 2), np.ones(len(xy))))
    x, y = pixels(a, x), pixels(b, y)
    fx, fy = x @ f.T, y @ f
    denominator = np.sqrt(np.sum(fx[:, :2]**2 + fy[:, :2]**2, axis=1))
    if np.any(denominator < 1e-15):
        return None
    error = np.abs(np.sum(y * fx, axis=1)) / denominator
    return dict(median_px=float(np.median(error)), p95_px=float(np.percentile(error, 95)),
                fraction_under_2px=float(np.mean(error < 2)))


def heldout_projection(model, target, target_xy, observations):
    rows = []
    for name, xy in observations.items():
        im = model[name]
        uv = cv2.undistortPoints(np.asarray(xy).reshape(1, 1, 2), im["k"], im["distortion"]).ravel()
        p = np.column_stack((im["r"], -im["r"] @ im["c"]))
        rows.extend((uv[0] * p[2] - p[0], uv[1] * p[2] - p[1]))
    _, _, vt = np.linalg.svd(rows)
    if abs(vt[-1, 3]) < 1e-12:
        return None
    point = vt[-1, :3] / vt[-1, 3]
    def error(name, xy):
        im = model[name]
        local = im["r"] @ (point - im["c"])
        if local[2] <= 0:
            return None
        uv, _ = cv2.projectPoints(local.reshape(1, 3), np.zeros(3), np.zeros(3), im["k"], im["distortion"])
        return float(np.linalg.norm(uv.ravel() - xy))
    anchor_errors = [error(name, xy) for name, xy in observations.items()]
    target_error = error(target, target_xy)
    if target_error is None or any(e is None for e in anchor_errors):
        return None
    return dict(target_px=target_error, anchor_max_px=max(anchor_errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, help="LABEL=diagnostics.csv")
    parser.add_argument("--target", action="append", required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    models = {}
    for item in args.model:
        label, path = item.split("=", 1)
        if label in models:
            raise ValueError("Duplicate model label")
        models[label] = load_model(Path(path))
    names = sorted(p.name for p in args.images.iterdir() if p.suffix.lower() in (".jpeg", ".jpg", ".png"))
    pairs = set()
    for target in args.target:
        sequence = [n for n in names if n.rsplit("_", 1)[0] == target.rsplit("_", 1)[0]]
        index = sequence.index(target)
        for offset in (-2, -1, 1, 2):
            if 0 <= index + offset < len(sequence):
                pairs.add(tuple(sorted((target, sequence[index + offset]))))
    cv2.setNumThreads(8)
    sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=.025)
    cache = {}
    def features(name):
        if name not in cache:
            image = cv2.imdecode(np.fromfile(args.images / name, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError("Cannot decode " + name)
            h, w = image.shape
            scale = min(1., 2000. / max(h, w))
            resized = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
            kp, desc = sift.detectAndCompute(resized, None)
            xy = np.array([k.pt for k in kp], dtype=float).reshape(-1, 2)
            xy *= np.array([w / resized.shape[1], h / resized.shape[0]])
            cache[name] = xy, desc
        return cache[name]
    results = []
    fresh_tracks = {target: {} for target in args.target}
    for first, second in sorted(pairs):
        row = dict(first=first, second=second)
        x, dx = features(first)
        y, dy = features(second)
        if dx is None or dy is None or min(len(dx), len(dy)) < 2:
            row["skipped"] = "insufficient descriptors"
        else:
            cv2.setRNGSeed(20260914)
            matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=128))
            forward, reverse = matcher.knnMatch(dx, dy, k=2), matcher.knnMatch(dy, dx, k=2)
            rev = {m.queryIdx: m.trainIdx for m, n in reverse if m.distance < .7 * n.distance}
            matches = [(m.queryIdx, m.trainIdx) for m, n in forward
                       if m.distance < .7 * n.distance and rev.get(m.trainIdx) == m.queryIdx]
            row["mutual_matches"] = len(matches)
            if len(matches) >= 30:
                a, b = x[[i for i, _ in matches]], y[[j for _, j in matches]]
                cv2.setRNGSeed(20260914)
                _, mask = cv2.findFundamentalMat(a, b, cv2.USAC_MAGSAC, 2., .999, 10000)
                if mask is not None:
                    keep = mask.ravel() != 0
                    row["independent_F_inliers"] = int(keep.sum())
                    if keep.sum() >= 30:
                        row["models"] = {label: residual(model[first], model[second], a[keep], b[keep])
                                         if first in model and second in model else None
                                         for label, model in models.items()}
                        for index in np.flatnonzero(keep):
                            i, j = matches[index]
                            if first in fresh_tracks:
                                fresh_tracks[first].setdefault(i, {})[second] = y[j]
                            if second in fresh_tracks:
                                fresh_tracks[second].setdefault(j, {})[first] = x[i]
        results.append(row)
        print(json.dumps(row), flush=True)
    multiview = {}
    for target, tracks in fresh_tracks.items():
        if any(target not in model for model in models.values()):
            multiview[target] = dict(skipped="target unregistered in a compared model")
            continue
        samples = []
        proposed = 0
        for feature, observations in tracks.items():
            observations = {name: xy for name, xy in observations.items()
                            if all(name in model for model in models.values())}
            if len(observations) < 3:
                continue
            proposed += 1
            scores = {label: heldout_projection(model, target, features(target)[0][feature], observations)
                      for label, model in models.items()}
            if all(score is not None for score in scores.values()):
                samples.append(scores)
        multiview[target] = dict(fresh_tracks_with_3_other_views=proposed, positive_in_all_models=len(samples), models={})
        for label in models:
            if samples:
                multiview[target]["models"][label] = {
                    key: dict(median=float(np.median([s[label][key] for s in samples])),
                              p95=float(np.percentile([s[label][key] for s in samples], 95)))
                    for key in ("target_px", "anchor_max_px")}
    args.output.write_text(json.dumps(dict(method=__doc__, targets=args.target, pairs=results,
        multiview_note="Fresh shared SIFT features triangulated from at least three OTHER cameras; no target pixel in triangulation. All jointly positive depths reported without residual trimming.",
        multiview=multiview), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
