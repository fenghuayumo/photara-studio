# splat_drender 前向优化实验（2026-09-12，bucket 版本之后）

本轮保留一项小改动：bucket 偏移的手写 Hillis–Steele 扫描替换为 CUB BlockScan。没有获得稳定、明显的 blend 加速；域裁剪、局部缓存、launch bounds 实验均撤回。不要把本轮约 1% 的整步波动表述成已证明的训练提速。

## 最终实现

`third_party/splat_drender/src/render_forward.cu` 的 `bucket_offsets_kernel` 使用 CUB inclusive integer scan，并从 block aggregate 更新 carry。尾批无效线程仍输入 0；每批结束保留同步，以便安全复用 TempStorage。bucket 编号、快照布局、排序、前后向公式及主机回读路径不变。

Nsight Compute 单次 kernel 测量：ori bucket 扫描 16.58 → 13.57 μs，约减少 3 μs。该结果是辅助定位，不是重复计时统计，也不足以解释整步耗时变化。

## 固定输入复测

RTX 5090 D v2，CUDA 12.8，Windows WDDM；构建方式为 `scripts/build_ab_compare.ps1`。修改前可执行文件保存在 `build/splat_perf_20260912/before.exe`。每组先校验全部主输出和模型梯度，再预热 10 次、计时 300 次，串行重复 3 轮取中位数。最终 `verified_*` 测量期间没有本任务的并行编译、训练或 profiling。

| 场景 | 前向修改前 / 后 ms | pair 修改前 / 后 ms | pair 变化 | 后测同轮 reference pair ms |
|---|---:|---:|---:|---:|
| capture_step100 RGB 640×360 | 0.451490 / 0.452417 | 0.588031 / 0.588184 | +0.03% | 0.743764 |
| capture_step100 geometry 640×360 | 1.186970 / 1.163770 | 1.492214 / 1.486003 | −0.42% | 1.713772 |
| ori_step100 geometry 1000×1000 | 2.161892 / 2.135024 | 3.112299 / 3.082504 | −0.96% | 3.636940 |

reference 本身也随测量轮次波动：ori reference pair 为 3.677018 → 3.636940 ms。因此不能把 new pair 的约 1% 下降全部归因于改动。RGB 基本持平。

三组均通过现有验收（relative L2 ≤ 1e-4 或 max absolute ≤ 1e-5），实例数相同，无非有限通道。修改后三组最大主通道 relative L2 分别为 2.34e-5、4.82e-5、6.42e-6。这里是与 reference 的误差，不是改动前后逐位比较。

## 已撤回的实验

以下均为单轮 300 次探索，不能与最终复测表混为统计结论；记录用于避免重复尝试。ori 几何场景均通过现有数值容差。

| 实验 | new forward ms | new pair ms | 结论 |
|---|---:|---:|---|
| 深度 refinement 的 16×2 warp 屏幕盒 pending bitmap | 2.226356 | 3.164183 | 没有改善前向 |
| 首轮缓存每像素最多 32 项 (alpha, t_peak, rsigma)，溢出走原遍历 | 2.158603 | 3.213148 | 没有稳定收益 |
| 缓存容量扩大至 128 项 | 3.201697 | 4.137528 | 明显退化 |
| CUB scan + blend launch bounds 最少 3 blocks/SM | 2.136937 | 3.118550 | 未证实改善；NCU 寄存器 64 → 72，活跃 warp 比例下降 |
| CUB scan + blend launch bounds 最少 5 blocks/SM | 2.246238 | 3.197556 | 没有改善 |

warp 裁剪仅用于 median 探针，不改 pass 1 或快照，使用原始实例位置检查 last_contributor；全景相机绕过屏幕盒拒绝。缓存路径完整保留超容量回退，没有截断贡献。最终源码不包含这些实验路径。

## 功能验证

- Release `photara`、`photara_splat_test` 构建成功；完整 splat tests passed。
- 随机 RGB / geometry reference_compare 执行完成；随机 geometry Compute Sanitizer synccheck：0 errors。
- WeChat_20250712175936：3000 步，1280×720，显式关闭几何损失；日志 depth/normal/mv_geo/mv_ncc 均为 0。82,738 Gaussians，training_s=7.07818；3 视图 average_psnr=20.0482、masked=27.6239。
- ori_img：1000 步，1000×1000，`--splat --mesh --splat-geometry-from-iter 10`；depth/normal/mv_geo/mv_ncc 均非零。53,071 Gaussians，training_s=16.1091；TSDF 输出 867,929 顶点、1,722,559 面。

短训只验证流程，未作配对训练耗时或重建质量评估，不据此推断收敛质量改善。

## 复现与产物

```powershell
./scripts/build_ab_compare.ps1
python experiments/benchmark_splat_rasterizer.py --exe build/reference_compare.exe --capture build/capture_step100 --width 640 --height 360 --output build/splat_perf_20260912/verified_rgb_after
python experiments/benchmark_splat_rasterizer.py --exe build/reference_compare.exe --capture build/capture_step100 --width 640 --height 360 --geometry --output build/splat_perf_20260912/verified_geo_after
python experiments/benchmark_splat_rasterizer.py --exe build/reference_compare.exe --capture build/splat_perf_20260911/ori_step100 --width 1000 --height 1000 --geometry --output build/splat_perf_20260912/verified_ori_after
```

原始日志和 summary.json 在 `build/splat_perf_20260912/verified_{rgb,geo,ori}_{before,after}`；训练日志为 `rgb_verified.log`、`geo_verified.log`；完整测试为 `tests.log`；同步检查为 `synccheck.log`。`ori_final` 与编译有重叠，不用于最终表。其余 `ori_*` 是探索结果。改动前源码及两个撤回方案保存在该目录的 `render_forward.*.cu` 中。

## 仍未解决

深度二分仍是前向主项。准确说，当前配置执行 1 次首轮探针 + 4 次 refinement，共 5 次深度列表遍历；加最初 color/normal/depth-seed pass 才是 6 次。`kDepthSplit=8` 表示 8 段、7 个内部探针，首轮额外检查两端点。

此前记录的约 185 μs WDDM 提交间隙本轮未重新测量，也未实现 CUDA Graph。当前主机回读的计数决定 workspace、排序路径及 CUB 排序长度，不能仅在现有 forward 外套一次 graph capture 就消除该依赖。后续更值得单独评估容量固定的设备计数管线，以及避免大规模每线程局部数组的深度贡献缓存；需要继续测量，不能先承诺收益。
