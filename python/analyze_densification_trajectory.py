#!/usr/bin/env python3
"""Summarize Photara and pygsplat grow/prune trajectories."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ITERATION = re.compile(
    r"splat iteration=(\d+)/(\d+).*?gaussians=(\d+)"
    r".*?grown=(\d+) pruned=(\d+)"
)
PYGSPLAT_INITIAL = re.compile(r"Model initialized\. Number of GS:\s*(\d+)")
PYGSPLAT_EVENT = re.compile(
    r"Step\s+(\d+)\s+\[(growth|prune-only)\]:\s*"
    r"prune=(\d+),\s*refine=(\d+),\s*before=(\d+),\s*after=(\d+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aether-log", type=Path, required=True)
    parser.add_argument("--pygsplat-log", type=Path)
    parser.add_argument("--pygsplat-cfg", type=Path, required=True)
    parser.add_argument("--pygsplat-initial-ply", type=Path, required=True)
    parser.add_argument("--pygsplat-final-ply", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def ply_vertex_count(path: Path) -> int:
    with path.open("rb") as stream:
        for raw_line in stream:
            line = raw_line.decode("ascii", errors="strict").strip()
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            if line == "end_header":
                break
    raise ValueError(f"No PLY vertex count in {path}")


def cfg_scalar(text: str, name: str) -> str:
    match = re.search(rf"(?m)^{re.escape(name)}:\s*(.+?)\s*$", text)
    if not match:
        raise ValueError(f"Missing {name} in cfg")
    return match.group(1)


def read_log_text(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16", errors="replace")
    if raw[:4096].count(b"\x00") > 32:
        return raw.decode("utf-16-le", errors="replace")
    return raw.decode("utf-8", errors="replace")


def summarize_rows(
    rows: list[dict[str, int]], initial_count: int
) -> dict[str, object]:
    bins = [
        (0, 3000),
        (3000, 7000),
        (7000, 15000),
        (15000, 25000),
        (25000, 30000),
    ]
    stages = []
    for lower, upper in bins:
        selected = [
            row for row in rows if lower < row["iteration"] <= upper
        ]
        stages.append(
            {
                "iteration_range": [lower, upper],
                "grown": sum(row["grown"] for row in selected),
                "pruned": sum(row["pruned"] for row in selected),
                "ending_count": selected[-1]["count"] if selected else None,
            }
        )
    milestone_iterations = [
        1,
        1000,
        3000,
        5000,
        7000,
        10000,
        15000,
        20000,
        25000,
        30000,
    ]
    milestones = {
        str(iteration): min(
            rows, key=lambda row: abs(row["iteration"] - iteration)
        )
        for iteration in milestone_iterations
    }
    total_grown = sum(row["grown"] for row in rows)
    total_pruned = sum(row["pruned"] for row in rows)
    return {
        "initial_count": initial_count,
        "final_count": rows[-1]["count"],
        "maximum_count": max(row["count"] for row in rows),
        "total_grown": total_grown,
        "total_pruned": total_pruned,
        "net_growth": rows[-1]["count"] - initial_count,
        "gross_pruned_fraction": total_pruned
        / (initial_count + total_grown),
        "structural_event_rows": sum(
            row["grown"] > 0 or row["pruned"] > 0 for row in rows
        ),
        "prune_event_rows": sum(row["pruned"] > 0 for row in rows),
        "stages": stages,
        "milestones": milestones,
    }


def main() -> None:
    args = parse_args()
    rows = []
    for line in read_log_text(args.aether_log).splitlines():
        match = ITERATION.search(line)
        if match:
            iteration, total, count, grown, pruned = map(
                int, match.groups()
            )
            rows.append(
                {
                    "iteration": iteration,
                    "total_iterations": total,
                    "count": count,
                    "grown": grown,
                    "pruned": pruned,
                }
            )
    if not rows:
        raise ValueError("No Photara iteration rows found")

    cfg_text = args.pygsplat_cfg.read_text(
        encoding="utf-8", errors="replace"
    )
    pyg_initial = ply_vertex_count(args.pygsplat_initial_ply)
    pyg_final = ply_vertex_count(args.pygsplat_final_ply)
    pyg_report: dict[str, object]
    if args.pygsplat_log:
        pyg_text = read_log_text(args.pygsplat_log)
        initial_matches = PYGSPLAT_INITIAL.findall(pyg_text)
        pyg_initial_from_log = (
            int(initial_matches[0]) if initial_matches else pyg_initial
        )
        pyg_rows = []
        for match in PYGSPLAT_EVENT.finditer(pyg_text):
            iteration, phase, pruned, grown, before, after = match.groups()
            pyg_rows.append(
                {
                    "iteration": int(iteration),
                    "count": int(after),
                    "grown": int(grown),
                    "pruned": int(pruned),
                    "before": int(before),
                    "phase": phase,
                }
            )
        if not pyg_rows:
            raise ValueError("No pygsplat ADCPlus trajectory rows found")
        pyg_report = summarize_rows(pyg_rows, pyg_initial_from_log)
        pyg_report.update(
            {
                "log": str(args.pygsplat_log.resolve()),
                "cfg": str(args.pygsplat_cfg.resolve()),
                "final_ply_count": pyg_final,
                "trajectory_matches_final_ply": (
                    pyg_rows[-1]["count"] == pyg_final
                ),
            }
        )
    else:
        refine_stop = int(cfg_scalar(cfg_text, "  refine_stop_iter"))
        speedy_pruning = (
            cfg_scalar(cfg_text, "speedysplat_pruning").lower() == "true"
        )
        no_structural_updates = (
            refine_stop <= 0
            and not speedy_pruning
            and pyg_initial == pyg_final
        )
        pyg_report = {
            "cfg": str(args.pygsplat_cfg.resolve()),
            "initial_count": pyg_initial,
            "final_count": pyg_final,
            "refine_stop_iter": refine_stop,
            "speedysplat_pruning": speedy_pruning,
            "deduced_total_grown": 0 if no_structural_updates else None,
            "deduced_total_pruned": 0 if no_structural_updates else None,
            "constant_count_trajectory_proven_by_config": no_structural_updates,
            "evidence": (
                "DefaultStrategy returns immediately for every step when "
                "refine_stop_iter=0; independent Speedy-Splat pruning is off; "
                "initial and final PLY vertex counts are identical."
            ),
        }

    report = {
        "aether": {
            "log": str(args.aether_log.resolve()),
            **summarize_rows(rows, rows[0]["count"]),
        },
        "pygsplat": pyg_report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
