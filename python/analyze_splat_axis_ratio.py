#!/usr/bin/env python3
"""Axis-ratio distribution report for trained Gaussian PLY files.

Reads standard 3DGS binary PLY output (Photara layout included) with numpy
only, and reports the longest/shortest per-axis scale ratio quantiles plus
the fraction of Gaussians above configurable ratio thresholds. Used to
verify the effect of --splat-max-scale-ratio.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


QUANTILES = (50, 90, 99, 99.9, 100)
DEFAULT_THRESHOLDS = (10, 100, 1000, 10000)

NUMPY_TYPES = {
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "uchar": "u1",
    "uint8": "u1",
    "char": "i1",
    "int8": "i1",
    "ushort": "u2",
    "uint16": "u2",
    "short": "i2",
    "int16": "i2",
    "uint": "u4",
    "int": "i4",
}


def parse_ply_header(path: Path) -> tuple[dict[str, str], list[tuple[str, str]], int]:
    """Return (element counts, vertex [(name, type)], data offset)."""
    with path.open("rb") as handle:
        offset = 0
        header_lines: list[bytes] = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"{path}: PLY header end not found")
            offset += len(line)
            header_lines.append(line.strip())
            if header_lines[-1] == b"end_header":
                break
        text_lines = [line.decode("ascii") for line in header_lines]
    if text_lines[0] != "ply":
        raise ValueError(f"{path}: not a PLY file")
    if "format binary_little_endian 1.0" not in text_lines:
        raise ValueError(f"{path}: only binary_little_endian PLY is supported")
    counts: dict[str, str] = {}
    properties: list[tuple[str, str]] = []
    element = None
    for line in text_lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "element":
            element = parts[1]
            counts[element] = parts[2]
        elif parts[0] == "property" and element == "vertex":
            properties.append((parts[2], parts[1]))
    if "vertex" not in counts:
        raise ValueError(f"{path}: vertex element missing")
    return counts, properties, offset


def load_scales(path: Path) -> np.ndarray:
    counts, properties, offset = parse_ply_header(path)
    count = int(counts["vertex"])
    dtype = np.dtype(
        [(name, NUMPY_TYPES[type_name]) for name, type_name in properties]
    )
    with path.open("rb") as handle:
        handle.seek(offset)
        data = np.fromfile(handle, dtype=dtype, count=count)
    if len(data) != count:
        raise ValueError(f"{path}: truncated PLY body")
    names = set(data.dtype.names or ())
    if not {"scale_0", "scale_1", "scale_2"}.issubset(names):
        raise ValueError(f"{path}: scale properties missing")
    log_scales = np.stack(
        [data["scale_0"], data["scale_1"], data["scale_2"]], axis=1
    ).astype(np.float64)
    return np.exp(log_scales)


def summarize(path: Path, thresholds: list[float]) -> dict[str, object]:
    scales = load_scales(path)
    smallest = scales.min(axis=1)
    largest = scales.max(axis=1)
    ratio = largest / np.maximum(smallest, 1e-30)
    return {
        "path": str(path.resolve()),
        "count": int(len(ratio)),
        "axis_ratio_quantiles": {
            str(q): float(value)
            for q, value in zip(QUANTILES, np.percentile(ratio, QUANTILES))
        },
        "fraction_above": {
            str(t): float(np.mean(ratio > t)) for t in thresholds
        },
        "count_above": {
            str(t): int(np.sum(ratio > t)) for t in thresholds
        },
        "smallest_axis_quantiles": {
            str(q): float(value)
            for q, value in zip(QUANTILES, np.percentile(smallest, QUANTILES))
        },
        "largest_axis_quantiles": {
            str(q): float(value)
            for q, value in zip(QUANTILES, np.percentile(largest, QUANTILES))
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plies", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_THRESHOLDS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = [summarize(path, args.thresholds) for path in args.plies]
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
