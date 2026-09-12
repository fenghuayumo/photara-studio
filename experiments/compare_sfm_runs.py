"""Compare timed SfM runs and all common camera poses without rejecting outliers."""
import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np

from sfm_acceptance import rotation, similarity, stats


def read_run(log):
    log = Path(log)
    text = log.read_text(encoding="utf-8-sig", errors="replace")
    final = re.findall(r"valid=\d registered=(\d+)/(\d+) landmarks=(\d+) "
                       r"reprojection_mean_px=([\d.e+-]+) reprojection_rms_px=([\d.e+-]+) "
                       r"reprojection_observations=(\d+) peak_working_set_mb=([\d.e+-]+) "
                       r"failed=\d+ elapsed_s=([\d.e+-]+)", text)
    if not final:
        raise ValueError(f"No completed reconstruction summary in {log}")
    r = final[-1]
    stages = {}
    for name, seconds in re.findall(r"stage finished: (\S+) elapsed_s=([\d.e+-]+)", text):
        stages.setdefault(name, []).append(float(seconds))
    with log.with_name(log.stem + "_sfm_diagnostics.csv").open(
            encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    if len({row["name"] for row in rows}) != len(rows):
        raise ValueError("Duplicate camera names")
    poses = {row["name"]: row for row in rows if int(row["registered"])}
    return dict(log=str(log), registered=int(r[0]), images=int(r[1]), landmarks=int(r[2]),
                rms_px=float(r[4]), observations=int(r[5]), peak_working_set_mb=float(r[6]),
                elapsed_s=float(r[7]), stages_seconds=stages,
                cuda_ba_calls=len(stages.get("ba.cuda", [])),
                cuda_ba_seconds=sum(stages.get("ba.cuda", [])),
                unregistered=[row["name"] for row in rows if not int(row["registered"])]), poses


def compare(before, after):
    a, pa = read_run(before)
    b, pb = read_run(after)
    names = sorted(pa.keys() & pb.keys())
    if len(names) < 3:
        raise ValueError("Fewer than three common cameras")
    centers = lambda poses: np.array([[float(poses[name]["center_" + axis])
                                      for axis in "xyz"] for name in names])
    x, y = centers(pb), centers(pa)
    if min(np.linalg.matrix_rank(x - x.mean(0)), np.linalg.matrix_rank(y - y.mean(0))) < 2:
        raise ValueError("Degenerate camera layout")
    scale, align, translation = similarity(x, y)
    distance = np.linalg.norm(scale * x @ align.T + translation - y, axis=1)
    radius = np.sqrt(np.mean(np.sum((y - y.mean(0)) ** 2, axis=1)))
    angles = []
    for name in names:
        ra, rb = [rotation([float(p[name][k]) for k in ("qw", "qx", "qy", "qz")])
                  for p in (pa, pb)]
        delta = rb @ align.T @ ra.T
        angles.append(np.degrees(np.arccos(np.clip((np.trace(delta)-1)/2, -1, 1))))
    return dict(before=a, after=b, speedup=a["elapsed_s"]/b["elapsed_s"],
                elapsed_reduction_percent=100*(1-b["elapsed_s"]/a["elapsed_s"]),
                common_cameras=len(names), added=sorted(pb.keys()-pa.keys()),
                removed=sorted(pa.keys()-pb.keys()),
                center_change_percent_before_radius=stats(100*distance/radius),
                rotation_change_deg=stats(angles),
                note="All common cameras, one positive proper Sim(3), no outlier removal; changes are not ground-truth errors.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = compare(args.before, args.after)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))
