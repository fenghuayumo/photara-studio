# splat_drender bucket 并行反向性能验证（2026-09-12）

本轮实现 FasterGS 的 bucket 并行反向（warp-per-bucket），三个固定场景的反向耗时降低 35%–56%，forward+backward pair 全面超过 gggs_reference。前一轮文档中"删除 blenders warp 裁剪"的结论经复核有误：裁剪本身仍应保留于 classic 形态路径，本轮回退相关探索并保留 tile 级编号语义。

## 固定输入 A/B（RTX 5090 D v2, CUDA 12.8, WDDM）

同一 `reference_compare.exe` 内链接 gggs_reference 与 splat_drender，先全通道校验再测时；每组 300 次迭代、3 轮取中位数。脚本与捕获同 `SPLAT_RENDERER_PERFORMANCE_20260911.md`。

| 场景 | new_pair | ref_pair | pair 变化 | new_bwd | ref_bwd | bwd 变化 |
|---|---:|---:|---:|---:|---:|---:|
| RGB 640×360（capture_step100, 22,229 Gaussians / 60,591 instances） | 0.590 | 0.758 | **−22.1%** | 0.140 | 0.316 | **−55.8%** |
| RGB+深度/法线 640×360 | 1.519 | 1.748 | **−13.1%** | 0.336 | 0.516 | **−34.8%** |
| ori_img 1000×1000 geometry（48,197 Gaussians / 330,086 instances） | 3.110 | 3.681 | **−15.5%** | 0.940 | 1.513 | **−37.9%** |

正确性：全部输出/梯度通道通过 `benchmark_splat_rasterizer.py` 验收（relative L2 ≤ 1e-4 或 max ≤ 1e-5）。随机场景实测约 1e-6 量级；误差来自"state-after"公式与原子求和顺序，而非遗漏梯度。

## 实现要点

- **前向快照**：blend_tile 在每个 32-instance bucket 边界为 tile 的 256 像素存储 (rgb, T[, normal])，并在结尾保存无背景的累计色 total_color；bucket 数量与每 tile 偏移由排序后的 tile range 在 device 端单 block 扫描得出，backward 启动用确定性上界（无第二次主机回读）。
- **bucket 并行反向**（blend_bucket_backward）：每个 warp 拥有一个 bucket 的 32 个实例；每 lane 一次性载入自己的 Gaussian 数据并遍历 tile 的全部 256 像素。像素状态 (T, color-after, normal-after) 从 lane 0 的快照进入，沿 warp shuffle 对角波前传播；参考实现的 accum_rec 链改用等价 "state-after" 公式（T·c_dot − state·grad/(1−α)）。每 lane 梯度驻留寄存器，整 tile 只做一组 atomicAdd，消除逐实例 warp 规约、共享内存批装载与逐 warp 原子。像素屏幕盒（alpha≥floor 椭圆的精确外接盒）用于廉价跳过。
- **几何 median 预核**（median_scale_walk）：dL_dmt = dL_dmedian·ray_z / max(−dT_dtm, 1e-7) 以独立 thread-per-pixel 核写出，供 bucket 核读取；上游深度梯度为零的像素立即短路。
- **探索性裁剪回退**：pass-1/pending-bitmap 的 16×2 warp 域裁剪在真实捕获上引入难定位的梯度偏差（前向输出位精确但快照链路异常），refine 裁剪正确但小幅变慢，均已回退。classic backward 的裁剪结论见下。

## 对上一轮文档的更正

`SPLAT_RENDERER_PERFORMANCE_20260911.md` 声称删除 blenders 的 warp 范围裁剪带来提速，同表声称 pair 0.752 ms。本轮复测发现该状态实际为 **2.06 ms**（blend_tile_backward 1.56 ms）：裁剪删除使 thread-per-pixel 反退化为逐像素全实例遍历（ncu：occupancy 21.6%、计算吞吐 6.6%）。文档中的比较数字与最终提交代码不一致，应以本轮为准。bucket 反向取代该路径后问题不再存在。

## 实际训练

- WeChat_20250712175936（RGB，3000 步，1280×720，几何损失显式关闭）：training_s **6.13**（上轮 7.00），3 视图 average_psnr 19.94 / masked 27.51，82,980 Gaussians。
- ori_img（geometry，1000 步，1000×1000，--splat --mesh）：training_s 15.71（上轮 15.63，噪声范围内持平；端到端含采样/细化/网格导出），depth/normal/mv_geo/mv_ncc 损失全部非零，53,103 Gaussians，TSDF 86.6 万顶点输出正常。

短训受原子梯度差异导致的致密化轨迹影响，固定捕获 pair 才是 rasterizer 层的口径。

## 复现

```powershell
./scripts/build_ab_compare.ps1
python experiments/benchmark_splat_rasterizer.py --exe build/reference_compare.exe --capture build/capture_step100 --width 640 --height 360 --output build/splat_perf_final/rgb
python experiments/benchmark_splat_rasterizer.py --exe build/reference_compare.exe --capture build/capture_step100 --width 640 --height 360 --geometry --output build/splat_perf_final/geo
python experiments/benchmark_splat_rasterizer.py --exe build/reference_compare.exe --capture build/splat_perf_20260911/ori_step100 --width 1000 --height 1000 --geometry --output build/splat_perf_final/ori
```

原始数据：`build/splat_perf_final/{rgb,geo,ori}/run_N.log + summary.json`。内核分解（nsys，ori）：blend_bucket_backward 0.68 ms vs 参考 renderCUDA 1.42 ms（2.1×）；median_scale_walk 0.17 ms。前向 blend 1.20 ms 与参考持平——深度二分 6 次全实例重走仍是前向主项，域裁剪需在不破坏 bucket 快照编号的前提下重试（待办）。
