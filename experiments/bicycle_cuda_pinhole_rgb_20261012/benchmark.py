"""Sequential, alternating-order comparison against the saved generic binary."""
import csv
import json
from pathlib import Path
import statistics
import subprocess

root = Path(__file__).resolve().parent
workspace = root.parents[1]
exe_dir = workspace / "build/photara/Release"
data = Path("D:/Models/360_extra_scenes_1/bicycle")
runs = []
for round_index in range(3):
    for view in (0, 27, 132):
        for optimizer in ("fused",):
            variants = ("generic", "specialized") if round_index % 2 == 0 else ("specialized", "generic")
            for variant in variants:
                out = root / f"r{round_index+1}_v{view}_{optimizer}_{variant}"
                out.mkdir(exist_ok=False)
                executable = "photara_splat_vulkan_parity_generic.exe" if variant == "generic" else "photara_splat_vulkan_parity.exe"
                args = [str(exe_dir / executable), "--bench-rgb", str(data / "images_2.cache/splat.ply"),
                        str(data / "images_2"), str(out), str(data / "images_2.cache/sfm.bin"),
                        "61", str(view), "streamed", "cuda-first" if round_index % 2 == 0 else "vulkan-first",
                        "compact"]
                (out / "command.json").write_text(json.dumps(args, indent=2), encoding="utf-8")
                with (out / "run.log").open("wb") as log:
                    subprocess.run(args, cwd=workspace, stdout=log, stderr=subprocess.STDOUT, check=True)
                with (out / f"rgb_summary_view_{view}.csv").open() as result:
                    rows = list(csv.DictReader(result))
                values = {row["backend"]: float(row["loop_average_ms"]) for row in rows}
                run = {"round": round_index+1, "view": view, "optimizer": optimizer,
                       "variant": variant, **values}
                runs.append(run)
                (root / "benchmark_runs.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
                print(json.dumps(run), flush=True)

summary = []
for view in (0, 27, 132):
    for optimizer in ("fused",):
        matched = [r for r in runs if r["view"] == view and r["optimizer"] == optimizer]
        b = statistics.median(r["CUDA"] for r in matched if r["variant"] == "generic")
        c = statistics.median(r["CUDA"] for r in matched if r["variant"] == "specialized")
        summary.append({"view": view, "optimizer": optimizer, "generic_ms": b,
                        "specialized_ms": c, "time_reduction_percent": 100*(1-c/b),
                        "vulkan_control_ms": statistics.median(r["Vulkan"] for r in matched)})
(root / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2), flush=True)
