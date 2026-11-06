"""Compare complete, identically ordered POPSpa evaluation sweeps."""

import argparse
import json
import statistics
from pathlib import Path


def summarize(rows, baseline):
    changes = sorted(row["psnr"] - baseline[row["view"]]["psnr"] for row in rows)
    ssim_changes = [row["ssim"] - baseline[row["view"]]["ssim"] for row in rows]
    worst = sorted(rows, key=lambda row: row["psnr"] - baseline[row["view"]]["psnr"])[:10]
    return {
        "views": len(rows),
        "mean_psnr": statistics.mean(row["psnr"] for row in rows),
        "mean_ssim": statistics.mean(row["ssim"] for row in rows),
        "mean_psnr_change": statistics.mean(changes),
        "mean_ssim_change": statistics.mean(ssim_changes),
        "p05_psnr_change": changes[int(0.05 * (len(changes) - 1))],
        "min_psnr_change": changes[0],
        "views_losing_over_0_5_db": sum(change < -0.5 for change in changes),
        "views_losing_over_1_db": sum(change < -1 for change in changes),
        "worst_views": [
            {
                "view": row["view"],
                "psnr_before": baseline[row["view"]]["psnr"],
                "psnr_after": row["psnr"],
                "ssim_before": baseline[row["view"]]["ssim"],
                "ssim_after": row["ssim"],
            }
            for row in worst
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--s3-reference", type=Path, help="Physical-min-scale comparison sweep")
    args = parser.parse_args()
    base = json.loads((args.directory / "baseline/result.json").read_text(encoding="utf-8"))
    baseline = {row["view"]: row for row in base["metrics"]}
    implementation_path = args.directory / "implementation.json"
    implementation = json.loads(implementation_path.read_text(encoding="utf-8-sig")) if implementation_path.exists() else {}
    reports = {}
    for path in sorted(args.directory.glob("*/result.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data.setdefault("s3_variant", implementation.get("s3_variant", "unspecified"))
        rows = data.pop("metrics")
        assert len(rows) == len(baseline)
        assert {row["view"] for row in rows} == set(baseline)
        assert data["max_image_dimension"] == base["max_image_dimension"]
        assert data["compaction_holdout_every"] == base["compaction_holdout_every"]
        assert data.get("filter_mode", implementation.get("evaluation_filter")) == base.get(
            "filter_mode", implementation.get("evaluation_filter")
        ), "Mismatched raster filter; metrics cannot be compared"
        data["all"] = summarize(rows, baseline)
        data["compaction_holdout"] = summarize([row for row in rows if row["holdout"]], baseline)
        data["compaction_train"] = summarize([row for row in rows if not row["holdout"]], baseline)
        reports[path.parent.name] = data
    for encoded, source in (("codec20", "popspa20"), ("codec40", "popspa40"), ("codecbase", "baseline"), ("codecpair", "s3pair20")):
        if encoded in reports and source in reports:
            encoded_rows = json.loads((args.directory / encoded / "result.json").read_text(encoding="utf-8"))["metrics"]
            source_rows = json.loads((args.directory / source / "result.json").read_text(encoding="utf-8"))["metrics"]
            reports[encoded]["encoding_only"] = summarize(encoded_rows, {row["view"]: row for row in source_rows})
    if args.s3_reference and "s3pair20" in reports:
        reference = json.loads((args.s3_reference / "popspa20/result.json").read_text(encoding="utf-8"))
        current = json.loads((args.directory / "s3pair20/result.json").read_text(encoding="utf-8"))
        for key in ("output_count", "max_image_dimension", "compaction_holdout_every", "steps_per_stage", "prune_cycles", "finetune_lr_scale", "use_ppisp", "use_bilateral_grid"):
            assert current[key] == reference[key], f"Unmatched s3 settings: {key}"
        assert current.get("geometry_lr_scale", 1) == reference.get("geometry_lr_scale", 1)
        assert current.get("seed", 42) == reference.get("seed", 42)
        reference_rows = {row["view"]: row for row in reference["metrics"]}
        reports["s3pair20"]["versus_physical_min_scale"] = summarize(current["metrics"], reference_rows)
    (args.directory / "comparison.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    lines = [
        "# POPSpa / SOG-XT indoor iPhone review",
        "",
        "Metrics cover all 739 registered views at maximum dimension 960. "
        "Every eighth view is excluded from compaction scoring and fitting. "
        "These views may have been used to train the original model. "
        "Every optimization run enables both PPISP and the bilateral grid. "
        "Metrics use exported canonical RGB without either training-only transform.",
        f'Shape s3 variant: {implementation.get("s3_variant", "see individual result metadata")}.',
        "",
        "| Run | Gaussians | PSNR all | PSNR compaction holdout | SSIM holdout | Worst delta dB | Views losing >1 dB | SOG-XT MB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, data in reports.items():
        all_views, held = data["all"], data["compaction_holdout"]
        encoded = f'{data["encoded_bytes"] / 1e6:.3f}' if "encoded_bytes" in data else "-"
        lines.append(
            f'| {name} | {data["output_count"]:,} | {all_views["mean_psnr"]:.4f} | '
            f'{held["mean_psnr"]:.4f} | {held["mean_ssim"]:.6f} | '
            f'{all_views["min_psnr_change"]:.4f} | {all_views["views_losing_over_1_db"]} | {encoded} |'
        )
    for name, data in reports.items():
        if "encoding_only" in data:
            delta = data["encoding_only"]
            lines += ["", f'{name}: encoding-only mean PSNR change {delta["mean_psnr_change"]:+.4f} dB, '
                      f'mean SSIM change {delta["mean_ssim_change"]:+.6f}.']
    if "s3pair20" in reports and "versus_physical_min_scale" in reports["s3pair20"]:
        delta = reports["s3pair20"]["versus_physical_min_scale"]
        lines += ["", f'Matched s3 comparison (same 0.001 coefficient, scene units, seed and optimizer settings): '
                  f'normalized-minus-physical mean PSNR {delta["mean_psnr_change"]:+.4f} dB, '
                  f'mean SSIM {delta["mean_ssim_change"]:+.6f}. '
                  'The two penalties have different units and gradient strengths; this compares these configurations, not universally optimal hyperparameters.']
    (args.directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
