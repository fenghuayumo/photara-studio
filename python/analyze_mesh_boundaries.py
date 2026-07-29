#!/usr/bin/env python3
"""Measure open boundary components in triangle meshes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, action="append", required=True)
    parser.add_argument("--label", action="append")
    parser.add_argument("--world-scale", type=float, action="append")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def analyze(path: Path, world_scale: float) -> dict[str, object]:
    mesh = o3d.io.read_triangle_mesh(str(path))
    vertices = np.asarray(mesh.vertices, dtype=np.float64) * world_scale
    faces = np.asarray(mesh.triangles, dtype=np.int64)
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    edges.sort(axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique_edges[counts == 1]

    parent: dict[int, int] = {}

    def find(value: int) -> int:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left, right in boundary:
        union(int(left), int(right))

    components: dict[int, list[int]] = {}
    for index, (left, right) in enumerate(boundary):
        root = find(int(left))
        components.setdefault(root, []).append(index)

    summaries = []
    for edge_indices in components.values():
        component_edges = boundary[np.asarray(edge_indices)]
        component_vertices = np.unique(component_edges)
        lengths = np.linalg.norm(
            vertices[component_edges[:, 0]]
            - vertices[component_edges[:, 1]],
            axis=1,
        )
        summaries.append(
            {
                "edge_count": int(len(component_edges)),
                "vertex_count": int(len(component_vertices)),
                "perimeter": float(lengths.sum()),
                "bbox_min": vertices[component_vertices].min(axis=0).tolist(),
                "bbox_max": vertices[component_vertices].max(axis=0).tolist(),
            }
        )
    summaries.sort(key=lambda item: item["perimeter"], reverse=True)
    edge_counts = np.asarray(
        [item["edge_count"] for item in summaries], dtype=np.int64
    )
    perimeters = np.asarray(
        [item["perimeter"] for item in summaries], dtype=np.float64
    )
    return {
        "path": str(path.resolve()),
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "unique_edges": int(len(unique_edges)),
        "nonmanifold_edges": int(np.count_nonzero(counts > 2)),
        "boundary_edges": int(len(boundary)),
        "boundary_components": int(len(summaries)),
        "boundary_component_count_by_min_edges": {
            str(threshold): int(np.count_nonzero(edge_counts >= threshold))
            for threshold in [3, 8, 16, 32, 64, 128, 256]
        },
        "boundary_component_count_by_min_perimeter": {
            str(threshold): int(np.count_nonzero(perimeters >= threshold))
            for threshold in [0.001, 0.002, 0.004, 0.008, 0.016, 0.032]
        },
        "boundary_components_perimeter_top20": summaries[:20],
    }


def main() -> None:
    args = parse_args()
    labels = args.label or [path.stem for path in args.mesh]
    if len(labels) != len(args.mesh):
        raise ValueError("--label count must match --mesh count")
    scales = args.world_scale or [1.0] * len(args.mesh)
    if len(scales) != len(args.mesh):
        raise ValueError("--world-scale count must match --mesh count")
    report = {
        label: analyze(path, scale)
        for label, path, scale in zip(labels, args.mesh, scales)
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    for label, item in report.items():
        print(
            label,
            f"vertices={item['vertices']}",
            f"faces={item['faces']}",
            f"boundary_edges={item['boundary_edges']}",
            f"boundary_components={item['boundary_components']}",
            f"nonmanifold_edges={item['nonmanifold_edges']}",
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
