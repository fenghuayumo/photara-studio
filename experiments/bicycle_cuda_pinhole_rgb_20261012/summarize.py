"""Collect kernel, repeated throughput and completed training evidence."""
import json
from pathlib import Path
import re
from datetime import datetime

root = Path(__file__).resolve().parent
training = json.loads((root / "training_state.json").read_text(encoding="utf-8"))
if training["status"] != "complete":
    raise RuntimeError("Training has not completed")
benchmarks = json.loads((root / "benchmark_summary.json").read_text(encoding="utf-8"))
baseline = training["baseline"]
baseline_log = root.parent / "bicycle_cuda_train30k_20261012/baseline_fused/train.log"
candidate_log = root / "train_specialized_fused/train.log"

def timestamps(path):
    result = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        match = re.search(r"splat iteration=(\d+)/30000", line)
        if match:
            result[int(match[1])] = datetime.fromisoformat(line[:23])
    return result

b_times, c_times = timestamps(baseline_log), timestamps(candidate_log)
windows = []
for start, end in ((1, 3000), (3000, 6000), (6000, 10000), (10000, 15000), (15000, 30000)):
    b = (b_times[end]-b_times[start]).total_seconds()
    c = (c_times[end]-c_times[start]).total_seconds()
    windows.append({"start": start, "end": end, "generic_mean_ms": b*1000/(end-start),
                    "specialized_mean_ms": c*1000/(end-start), "time_reduction_percent": 100*(1-c/b)})
report = {"benchmark": benchmarks, "training": training, "training_windows": windows,
          "kernel": {"generic_registers": 255, "specialized_fused_registers": 109,
                     "specialized_split_registers": 110,
                     "generic_achieved_occupancy_percent": 15.96,
                     "specialized_achieved_occupancy_percent": 32.01},
          "validation": {"splat_test": "passed", "geometry_gradient_test": "passed",
                         "rgb_gradient_cases": 128, "max_gradient_scaled_error": 3.96782e-5},
          "scope": "SH, scale/quaternion Gaussian parameterization, pinhole camera, absent depth/normal gradient scratch; other inputs retain generic kernels",
          "limitations": "Three alternating-order rounds per fixed view and optimizer, 61 measured steps after 3 warmups. Full training compares one new specialized-fused run against the earlier same-day generic-fused 30k run, not a randomized repeated full-training study. GPU reductions and densification are not bitwise deterministic. No Vulkan full-training run."}
(root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
rows = ["# CUDA 针孔 RGB 反向内核优化", "",
        "针对针孔相机、SH 颜色、scale/quaternion 参数化且无深度/法线梯度缓冲的场景，使用编译期专用反向路径。共享原有 conic/协方差求导公式，其余相机和几何训练继续使用通用路径。", "",
        "每线程寄存器：融合路径 255 → 109，拆分路径 255 → 110。Nsight 实际占用率：融合路径 15.96% → 32.01%；理论占用率 16.67% → 33.33%。", "",
        "完整 splat 测试及几何有限差分测试通过。128 组针孔 RGB 梯度对照涵盖相机旋转、裁剪均值、各向异性尺度、不同 kernel footprint；最大 |error|/(1+|reference|) 为 3.96782e-5，FP32 混合绝对/相对容差为 1e-4。结果不逐位一致。", "",
        "## 固定模型连续执行耗时", "",
        "同一约 300 万高斯模型、1920×1276，三个视角，每方案三轮、每轮 61 次测量、3 次预热；交替二进制和后端执行顺序。下表取整段循环平均耗时的跨轮中位数，包含临时对象释放及末尾 GPU 等待。", "",
        "| 视角 | Adam | 通用 CUDA ms | 专用 CUDA ms | 耗时减少 |", "|---|---|---:|---:|---:|"]
for b in benchmarks:
    rows.append(f"| {b['view']} | {b['optimizer']} | {b['generic_ms']:.4f} | {b['specialized_ms']:.4f} | {b['time_reduction_percent']:.2f}% |")
rows += ["", "融合路径明显改善；拆分路径本次中位数略慢，继续保留融合 Adam 默认设置。", "",
         "## Bicycle 完整 30,000 次训练", "",
         "从同一 COLMAP 稀疏点初始化，seed 42，169 个训练视角、25 个留出视角，渐进分辨率最高 1920×1276，上限 300 万高斯。两组均融合 SH/结构 Adam，关闭中途评估、预览和性能采样。训练计时包含初始化、优化、增密和缓冲管理，排除最终模型导出和评估。", "",
         "| 方案 | 训练秒数 | PSNR | SSIM | 最终高斯数 |", "|---|---:|---:|---:|---:|",
         f"| 原通用融合 | {baseline['training_s']:.3f} | {baseline['average_psnr']:.4f} | {baseline['average_ssim']:.6f} | {int(baseline['gaussians'])} |",
         f"| 专用融合 | {training['training_s']:.3f} | {training['average_psnr']:.4f} | {training['average_ssim']:.6f} | {int(training['gaussians'])} |", "",
         f"完整训练耗时减少 {training['training_time_reduction_percent']:.2f}%，PSNR 差值 {training['psnr_delta']:+.4f} dB，SSIM 差值 {training['ssim_delta']:+.6f}。", "",
         "| 迭代窗口 | 通用平均 ms | 专用平均 ms | 耗时减少 |", "|---|---:|---:|---:|"]
for w in windows:
    rows.append(f"| {w['start']}–{w['end']} | {w['generic_mean_ms']:.3f} | {w['specialized_mean_ms']:.3f} | {w['time_reduction_percent']:.2f}% |")
rows += ["", "完整训练对照复用了当天此前完成的原通用融合基线，未重复多轮完整训练。相同种子不保证 GPU 归约及增密轨迹逐位相同。该结果验证本次 CUDA 实现改善，不能据此断言 Vulkan 完整训练的性能。"]
(root / "report.md").write_text("\n".join(rows)+"\n", encoding="utf-8")
print("\n".join(rows[-14:]))
