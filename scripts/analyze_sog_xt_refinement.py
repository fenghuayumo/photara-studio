"""Compare fixed-model SOG-XT encoding and encoding-aware adaptation sweeps."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ply", type=Path)
    parser.add_argument("old_codec", type=Path)
    parser.add_argument("global_codec", type=Path)
    parser.add_argument("adapted", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    paths = {name: getattr(args, name) for name in ("ply", "old_codec", "global_codec", "adapted")}
    inputs = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    baseline = {row["view"]: row for row in inputs["ply"]["metrics"]}
    report = {}
    for name, data in inputs.items():
        assert {row["view"] for row in data["metrics"]} == set(baseline)
        assert data["output_count"] == inputs["ply"]["output_count"]
        for key in ("max_image_dimension", "compaction_holdout_every"):
            assert data[key] == inputs["ply"][key]
        item = {key: value for key, value in data.items() if key != "metrics"}
        for split in ("all", "holdout"):
            rows = [row for row in data["metrics"] if split == "all" or row["holdout"]]
            changes = [row["psnr"] - baseline[row["view"]]["psnr"] for row in rows]
            item[split] = {
                "views": len(rows),
                "psnr": statistics.mean(row["psnr"] for row in rows),
                "ssim": statistics.mean(row["ssim"] for row in rows),
                "psnr_delta_from_ply": statistics.mean(changes),
                "worst_psnr_delta_from_ply": min(changes),
                "views_losing_over_0_5_db_from_ply": sum(delta < -0.5 for delta in changes),
            }
        report[name] = item
    args.output.mkdir(exist_ok=True, parents=True)
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# SOG-XT fixed-model refinement",
        "",
        "Same 347,718-Gaussian input, Mip2D rendering at maximum dimension 960. "
        "Every eighth camera is excluded from adaptation and snapshot selection. "
        "PPISP and the bilateral grid remain enabled during optimization; "
        "metrics use decoded canonical RGB.",
        "",
        "| Stage | All PSNR | All SSIM | Holdout PSNR | Holdout SSIM | SOG-XT MB | Seconds |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, data in report.items():
        mb = f'{data["encoded_bytes"] / 1e6:.3f}' if "encoded_bytes" in data else "-"
        lines.append(f'| {name} | {data["all"]["psnr"]:.4f} | {data["all"]["ssim"]:.6f} | '
                     f'{data["holdout"]["psnr"]:.4f} | {data["holdout"]["ssim"]:.6f} | '
                     f'{mb} | {data["optimization_seconds"]:.2f} |')
    lines += ["", "PSNR deltas are against the same input PLY, not against the pre-POPSpa reconstruction. "
              "Snapshot selection uses twelve fixed training cameras only. "
              "This is one scene/seed; per-view preservation is not guaranteed."]
    (args.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
