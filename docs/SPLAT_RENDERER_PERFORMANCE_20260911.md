# splat_drender 性能验证（2026-09-11）

本轮将固定场景的 RGB 前后向耗时降低约 16%，含深度/法线约 26%。当前基本追平 gggs_reference，尚不能宣称全面超过参考实现。FasterGS 的 bucket 并行反向没有在本轮实现；现有后向仍使用 tile 遍历和 warp 梯度规约。

## 固定输入 A/B

RTX 5090 D v2，驱动 596.36，Windows WDDM，CUDA 12.8，nvcc -O2 -use_fast_math -arch=native。基线为本轮修改前保存的 reference_compare_before.exe；参考端和新端链接于同一个测试程序。

既有捕获 build/capture_step100：22,229 Gaussians，640×360，60,591 instances。先检查全部输出/模型梯度，再 warm-up 10 次；每组 300 次，串行重复 3 轮，取墙钟均时的中位数。最终复测没有并行训练或编译。pair 包含 forward + backward 的主机提交、必要的计数回读及 CUDA 工作，不是逐 kernel 时间之和，也不是包含 Adam、数据加载和致密化的训练步时间。

| 场景 | 本轮修改前 ms | 修改后 ms | 耗时降低 | 同轮 gggs_reference ms |
|---|---:|---:|---:|---:|
| RGB + alpha | 0.893309 | 0.752347 | 15.78% | 0.752481 |
| RGB + alpha + median depth/normal | 2.357274 | 1.750887 | 25.72% | 1.727486 |

实际 ori_img 捕获（第 100 次 backward，48,197 Gaussians，1000×1000）也通过三轮校验，使用大列表双排序：新端 pair **3.760311 ms**，参考端 **3.719321 ms**。这里只对比新端与参考端，没有采集该场景的本轮修改前版本数据。

几何 A/B 使用真实 Gaussian/相机输入及随机深度/法线上游梯度，覆盖非零几何反传。RGB/alpha 上游来自捕获文件。RGB 所有主输出/模型梯度的最大 relative L2 约 9e-7；含几何约 1.25e-5，与修改前误差量级一致。脚本以 relative L2 <= 1e-4 或 max absolute <= 1e-5 验收，拒绝非有限结果、缺失通道和 instance count 不一致。

## 实现及原因

- 删除前向及深度导数遍历中逐 Gaussian 的 warp ballot，恢复每像素结束条件。深度细化仅遍历 max contributor 对应的批次，避免无效批次屏障。
- 删除 blenders 的逐项 screen bounds 裁剪和共享内存搬运。这套 16×2 warp 范围检查在实测场景中增加了成本；不能将 FasterGS 的 8×4 subtile 组织方式直接等同于当前实现。AccuTile 的 tile 级精确枚举继续保留。
- <=131,072 instances 使用稳定的 64 位 tile/depth 单排序，省去可见项 compact、depth pre-sort、gather 及第二次 scan 的提交。更大列表保留 FasterGS 32 位双排序。阈值是当前硬件上的保守启发式，并非跨 GPU 最优值。
- 实例 workspace 按相同 instance count 选择键宽；backward 与 point-query backward 重建完全相同的布局。两种排序都保持 Gaussian-index 的同深度 tie order。
- 可见数和实例数合并成一次 8 字节回读；SM80+ 用 warp reduction 聚合计数，旧架构保留原子加回退。
- 像素深度和点采样的首次/后续探针分别编译期特化。该项单独的实测收益很小，不归因于主要提速。
- A/B harness 增加分段及 pair 计时，捕获 Gaussian 数从文件推导，图像尺寸可传参；增加 Python 重复测量/校验/JSON 汇总工具。

## 实际训练与网格验证

- WeChat_20250712175936：3000 步，1280×720，全分辨率，显式关闭 depth-normal、multi-view geo/NCC、normal-field。保留 RGB/SSIM 和透明度训练。训练完成，training_s=6.99931，3 个训练视图 average_psnr=20.1374，average_masked_psnr=27.7131，79,655 Gaussians。
- ori_img：1000 步，1000×1000，--splat --mesh，几何从第 10 步开始；日志确认法线、深度、多视角几何及 NCC 损失非零。training_s=15.6305，59,088 Gaussians，TSDF 导出 1,074,435 顶点、2,108,330 三角面。
- 上述短训用于功能验证，不是收敛质量结论。training_s 包含启动开销，模型致密化数量也会因原子梯度差异变化；不据此声称端到端训练固定百分比提速。网格没有 ground truth 误差评估。
- Release aetherscan 和 aetherscan_splat_test 构建成功；完整 splat tests passed，含深度、点采样及相机相关既有回归；随机 RGB/geometry A/B 也执行成功。

## 复现

```powershell
./scripts/build_ab_compare.ps1
python experiments/benchmark_splat_rasterizer.py --exe build/reference_compare.exe --capture build/capture_step100 --width 640 --height 360 --output build/splat_perf_20260911/rgb_recheck
python experiments/benchmark_splat_rasterizer.py --exe build/reference_compare.exe --capture build/capture_step100 --width 640 --height 360 --geometry --output build/splat_perf_20260911/geometry_recheck
python experiments/benchmark_splat_rasterizer.py --exe build/reference_compare.exe --capture build/splat_perf_20260911/ori_step100 --width 1000 --height 1000 --geometry --output build/splat_perf_20260911/ori_recheck

build/aetherscan/Release/aetherscan.exe --images D:/ScanVideo/WeChat_20250712175936/images --colmap D:/ScanVideo/WeChat_20250712175936 --output build/splat_perf_20260911/rgb_verified.ply --splat-iterations 3000 --splat-depth-normal-weight 0 --splat-mv-geo-weight 0 --splat-mv-ncc-weight 0 --splat-normal-field=false --splat-progressive-resolution=false --splat-log-interval 500
build/aetherscan/Release/aetherscan.exe --images D:/ScanVideo/ori_img/images --colmap D:/ScanVideo/ori_img --output build/splat_perf_20260911/geo_verified.ply --splat --mesh --splat-iterations 1000 --splat-geometry-from-iter 10 --splat-progressive-resolution=false --splat-log-interval 200 --mesh-method tsdf
```

注意：这个 CLI 中几何损失由 --splat --mesh 路径启用；单独设置权重/geometry-from-iter 并不会开启几何监督。捕获文件是本地调试产物，不随源代码分发；新捕获可用 AETHERSCAN_SPLAT_GRAD_DUMP 指定前缀。

原始数据位于 build/splat_perf_20260911：rgb_baseline、geo_baseline、rgb_optimized、geo_optimized、ori_optimized 各目录包含 run_N.log 和 summary.json；训练日志为 rgb_verified.log、geo_verified.log，完整测试为 tests.log。其余带 before/after/warp/nocull 等名称的文件是探索过程记录，不用作最终结论。
