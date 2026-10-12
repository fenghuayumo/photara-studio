"""Validate the specialized fused CUDA path with the same 30k configuration."""
import json
import os
from pathlib import Path
import re
import subprocess
import time

root = Path(__file__).resolve().parent
workspace = root.parents[1]
previous = workspace / "experiments/bicycle_cuda_train30k_20261012/report.json"
baseline = json.loads(previous.read_text(encoding="utf-8"))["runs"][0]
out = root / "train_specialized_fused"
out.mkdir(exist_ok=False)
args = [str(workspace / "build/photara/Release/photara.exe"),
        "--images", "D:/Models/360_extra_scenes_1/bicycle/images_2",
        "--splat-dataset", "D:/Models/360_extra_scenes_1/bicycle/sparse/0",
        "--splat", "--backend", "cuda", "--splat-mode", "standard",
        "--splat-iterations", "30000", "--splat-seed", "42",
        "--splat-max-resolution", "1920", "--splat-eval-split-every", "8",
        "--splat-eval-checkpoints=false", "--splat-preview-interval", "0",
        "--splat-log-interval", "1000", "--splat-profile-cuda=false",
        "--output", str(out / "bicycle.ply")]
args[args.index("--output")+1] = str(out / "bicycle.ply")
(out / "command.json").write_text(json.dumps(args, indent=2), encoding="utf-8")
state = {"status": "running", "baseline": baseline, "command": args}
def save():
    temporary = root / "training_state.tmp.json"
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(temporary, root / "training_state.json")

with (out / "train.log").open("wb") as log, (out / "gpu_samples.jsonl").open("w") as samples:
    process = subprocess.Popen(args, cwd=workspace, stdout=log, stderr=subprocess.STDOUT,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    state["pid"] = process.pid
    last = -1
    started = time.perf_counter()
    while process.poll() is None:
        text = (out / "train.log").read_text(encoding="utf-8-sig", errors="replace")
        iterations = re.findall(r"splat iteration=(\d+)/30000", text)
        iteration = int(iterations[-1]) if iterations else 0
        state.update(iteration=iteration, elapsed_s=time.perf_counter()-started)
        if iteration != last:
            print(f"specialized_fused {iteration}/30000 elapsed_s={state['elapsed_s']:.1f}", flush=True)
            last = iteration
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        samples.write(json.dumps({"elapsed_s": state["elapsed_s"], "iteration": iteration,
                                  "gpu": gpu.stdout.strip()})+"\n")
        samples.flush()
        save()
        time.sleep(10)
state["returncode"] = process.returncode
text = (out / "train.log").read_text(encoding="utf-8-sig", errors="replace")
def metric(pattern):
    match = re.findall(pattern, text)
    return float(match[-1]) if match else None
state["training_s"] = metric(r"splat_training_completed iterations=30000 training_s=([\d.eE+-]+)")
state["average_psnr"] = metric(r"splat_final_evaluation_views=25 held_out=1 average_psnr=([\d.eE+-]+)")
state["average_ssim"] = metric(r"splat_final_evaluation_views=25 held_out=1 average_psnr=[\d.eE+-]+ average_ssim=([\d.eE+-]+)")
state["gaussians"] = metric(r"splat iteration=30000/30000 view=\d+ gaussians=(\d+)")
state["status"] = "complete" if process.returncode == 0 and all(state[k] is not None for k in ("training_s", "average_psnr", "average_ssim", "gaussians")) else "failed"
if state["status"] == "complete":
    state["training_time_reduction_percent"] = 100*(1-state["training_s"]/baseline["training_s"])
    state["psnr_delta"] = state["average_psnr"]-baseline["average_psnr"]
    state["ssim_delta"] = state["average_ssim"]-baseline["average_ssim"]
save()
print(json.dumps(state, indent=2), flush=True)
if state["status"] != "complete":
    raise RuntimeError("Specialized training or evaluation failed; see train.log")
