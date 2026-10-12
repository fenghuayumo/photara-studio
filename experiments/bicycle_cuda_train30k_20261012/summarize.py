"""Summarize completed production-training runs without rerunning training."""
import json
from datetime import datetime
from pathlib import Path
import re

root = Path(__file__).resolve().parent
report = json.loads((root / "report.json").read_text(encoding="utf-8"))
for run in report["runs"]:
    log = (root / run["label"] / "train.log").read_text(encoding="utf-8-sig")
    milestones = {}
    for line in log.splitlines():
        match = re.search(r"splat iteration=(\d+)/30000.*?gaussians=(\d+).*?image=(\d+x\d+)", line)
        if match:
            iteration, count, resolution = match.groups()
            milestones[int(iteration)] = (datetime.fromisoformat(line[:23]), int(count), resolution)
    windows = []
    for start, end in [(1, 3000), (3000, 6000), (6000, 10000), (10000, 15000), (15000, 30000)]:
        seconds = (milestones[end][0] - milestones[start][0]).total_seconds()
        windows.append({"start_iteration": start, "end_iteration": end,
                        "seconds": seconds, "mean_ms": seconds * 1000 / (end - start),
                        "end_gaussians": milestones[end][1], "end_resolution": milestones[end][2]})
    run["progress_windows"] = windows
    run["evaluation_views"] = [
        {"view": int(view), "psnr": float(psnr), "ssim": float(ssim)}
        for view, psnr, ssim in re.findall(r'splat_render=.*? view=(\d+) psnr=([\d.eE+-]+) ssim=([\d.eE+-]+)', log)
    ]
    run["model"] = str(root / run["label"] / "bicycle_splat.ply")

report["method"] = {
    "backend": "CUDA", "initialization": "same COLMAP sparse points, fresh training",
    "train_views": 169, "held_out_views": 25, "max_resolution": "1920x1276",
    "progressive_resolution": True, "gaussian_cap": 3000000,
    "timing": "production Trainer::train wall clock including initialization, optimization and densification; excluding final model export and evaluation",
    "profile_enabled": False, "changed_flag": "--splat-fuse-sh-adam=true versus false",
    "limitations": "One sequential pair with seed 42; GPU reductions and densification are not bitwise deterministic. This pair does not establish CUDA versus Vulkan training speed.",
}
(root / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
b, c = report["runs"]
rows = ["# Bicycle CUDA 30,000-iteration training A/B", "",
        "Same images_2 / COLMAP sparse initialization, seed 42, 169 training views and 25 held-out evaluation views. Progressive resolution reaches 1920×1276; cap 3,000,000 Gaussians. Production profiling and intermediate evaluation disabled.", "",
        "| Configuration | Training seconds | Held-out PSNR | Held-out SSIM | Final Gaussians |",
        "|---|---:|---:|---:|---:|"]
for run in (b, c):
    rows.append(f"| {run['label']} | {run['training_s']:.3f} | {run['average_psnr']:.4f} | {run['average_ssim']:.6f} | {int(run['gaussians'])} |")
rows += ["", f"Training time reduction: {report['training_time_reduction_percent']:.2f}%; speedup: {report['training_speedup']:.4f}×. Candidate minus baseline PSNR: {report['psnr_delta']:+.4f} dB; SSIM: {report['ssim_delta']:+.6f}.", "",
         "Training seconds include initialization, optimization, densification and temporary-buffer management. Model export and final evaluation are excluded. Logged step_ms is an exponential moving average of progress-window time; it is not a GPU duration for the displayed view.", "",
         "| Iteration window | Fused mean ms | Split SH mean ms | Time reduction |", "|---|---:|---:|---:|"]
for bw, cw in zip(b["progress_windows"], c["progress_windows"]):
    reduction = 100 * (1 - cw["seconds"] / bw["seconds"])
    rows.append(f"| {bw['start_iteration']}–{bw['end_iteration']} | {bw['mean_ms']:.3f} | {cw['mean_ms']:.3f} | {reduction:.2f}% |")
rows += ["", "One sequential pair does not establish a statistical confidence interval. Same seed does not guarantee identical learned models because GPU reductions and densification can diverge. These runs compare two CUDA trainer binaries; they do not measure Vulkan full-training performance.", "",
         "Both current runs use the automatic fused optimizer with identical CLI options; the executable is the compared variable. Historical split-SH reports in this directory were generated before the manual optimizer switch was removed."]
(root / "report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
print("\n".join(rows))
