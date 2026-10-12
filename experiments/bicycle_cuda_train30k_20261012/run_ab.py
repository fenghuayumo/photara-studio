"""Sequential 30k-step CUDA A/B using the production trainer and existing CLI."""
import json
import os
from pathlib import Path
import re
import subprocess
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
EXE = WORKSPACE / "build/photara/Release/photara.exe"
DATA = Path("D:/Models/360_extra_scenes_1/bicycle")
FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0)
COMMON = [
    "--images", str(DATA / "images_2"),
    "--splat-dataset", str(DATA / "sparse/0"),
    "--splat", "--backend", "cuda", "--splat-mode", "standard",
    "--splat-iterations", "30000", "--splat-seed", "42",
    "--splat-max-resolution", "1920",
    "--splat-eval-split-every", "8", "--splat-eval-checkpoints=false",
    "--splat-preview-interval", "0", "--splat-log-interval", "1000",
    "--splat-profile-cuda=false",
]
state = {"status": "starting", "dataset": str(DATA / "images_2"),
         "iterations_per_run": 30000, "seed": 42, "runs": []}

def save_state():
    temporary = HERE / "state.tmp.json"
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, HERE / "state.json")

def number(text, pattern):
    found = re.findall(pattern, text)
    return float(found[-1]) if found else None

for label, fuse in [("baseline_fused", "true"), ("split_sh_adam", "false")]:
    out = HERE / label
    out.mkdir(parents=True, exist_ok=True)
    log = out / "train.log"
    if log.exists():
        raise RuntimeError(f"Refusing to overwrite an existing experiment: {log}")
    argv = [str(EXE), *COMMON, "--output", str(out / "bicycle.ply"),
            f"--splat-fuse-sh-adam={fuse}"]
    (out / "command.json").write_text(json.dumps(argv, indent=2), encoding="utf-8")
    run = {"label": label, "fuse_sh_adam": fuse == "true", "command": argv,
           "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running"}
    state["runs"].append(run)
    state.update(status="running", active_run=label)
    started = time.perf_counter()
    with log.open("wb") as output, (out / "gpu_samples.jsonl").open("w", encoding="utf-8") as samples:
        process = subprocess.Popen(argv, cwd=WORKSPACE, stdout=output,
                                   stderr=subprocess.STDOUT, creationflags=FLAGS)
        run["pid"] = process.pid
        save_state()
        print(f"started {label} pid={process.pid}", flush=True)
        previous_iteration = -1
        while process.poll() is None:
            elapsed = time.perf_counter() - started
            text = log.read_text(encoding="utf-8-sig", errors="replace")
            matches = re.findall(r"splat iteration=(\d+)/30000[^\r\n]*", text)
            iteration = int(matches[-1]) if matches else 0
            run.update(elapsed_s=elapsed, latest_iteration=iteration)
            if iteration != previous_iteration:
                print(f"{label}: {iteration}/30000 elapsed_s={elapsed:.1f}", flush=True)
                previous_iteration = iteration
            try:
                gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                     timeout=5, creationflags=FLAGS)
                samples.write(json.dumps({"elapsed_s": elapsed, "iteration": iteration,
                                          "gpu": gpu.stdout.strip()}) + "\n")
                samples.flush()
            except (OSError, subprocess.TimeoutExpired):
                pass
            save_state()
            time.sleep(10)
        run.update(returncode=process.returncode, wall_s=time.perf_counter() - started)
    text = log.read_text(encoding="utf-8-sig", errors="replace")
    run["training_s"] = number(text, r"splat_training_completed iterations=30000 training_s=([\d.eE+-]+)")
    run["average_psnr"] = number(text, r"splat_final_evaluation_views=\d+ held_out=1 average_psnr=([\d.eE+-]+)")
    run["average_ssim"] = number(text, r"splat_final_evaluation_views=\d+ held_out=1 average_psnr=[\d.eE+-]+ average_ssim=([\d.eE+-]+)")
    run["gaussians"] = number(text, r"splat iteration=30000/30000 view=\d+ gaussians=(\d+)")
    if process.returncode != 0 or run["training_s"] is None or run["average_psnr"] is None:
        run["status"] = "failed"
        state["status"] = "failed"
        save_state()
        print("\n".join(text.splitlines()[-30:]), flush=True)
        raise RuntimeError(f"Training or final evaluation failed: {label}")
    run["status"] = "complete"
    run["latest_iteration"] = 30000
    save_state()
    print(f"completed {label}: training_s={run['training_s']} psnr={run['average_psnr']} ssim={run['average_ssim']}", flush=True)

baseline, candidate = state["runs"]
state.update(status="complete", active_run=None,
             training_speedup=baseline["training_s"] / candidate["training_s"],
             training_time_reduction_percent=100 * (1 - candidate["training_s"] / baseline["training_s"]),
             psnr_delta=candidate["average_psnr"] - baseline["average_psnr"],
             ssim_delta=candidate["average_ssim"] - baseline["average_ssim"])
save_state()
(HERE / "report.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
