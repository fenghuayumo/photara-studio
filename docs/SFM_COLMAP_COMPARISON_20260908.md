# 当前 SfM 与 COLMAP 实测比较（2026-09-08）

## 结论

在当前 Alameda 64 张子集上，Photara 的默认 Auto 结果，以及默认焦距的鱼眼结果，与 COLMAP 鱼眼重建的相机几何仍有明显差距。此前的“64/64 注册、低于 1 px 重投影误差”是运行与自洽性检查，不能视作对齐质量已经接近 COLMAP。

仅把 COLMAP 求出的鱼眼焦距作为 Photara 初值，轨迹就明显接近 COLMAP。这支持优先修正模型自动选择、鱼眼焦距初始化和内参优化策略；目前没有证据把所有差距归因于位姿求解器本身。

## 输入与配置

- COLMAP：本地 `D:/ProgramCode/colmap-x64-windows-cuda/bin/colmap.exe`，4.1.0，commit fa8e3b3，CUDA。
- 输入：`D:/ScanVideo/alameda/images_2` 按文件名排序的前 64 张图片，1752×1168；使用此前 `artifacts/fisheye_alameda_64/images` 的硬链接。不是 1,742 张全量测试。
- COLMAP：共享相机，分别使用 OPENCV 与 OPENCV_FISHEYE；SIFT 6000 特征、peak threshold 0.005、ratio 0.8、交叉检查；GPU 特征/匹配，exhaustive 全配对，incremental mapper，CPU BA，8 线程。针孔初始 fx=fy=2102.4，鱼眼初始 fx=fy=876，主点=(876,584)，畸变初值为零。
- Photara：此前 Global 结果，SiftGPU、6000 特征、window=6、渐进匹配扩展；Auto 或显式鱼眼；CPU BA。
- 两者输入、分辨率和特征预算相同，但特征实现、匹配图和过滤策略不同。此次是产品管线对比，不是完全相同观测上的 BA 求解器对比，也不是严格的等工作量速度测试。
- COLMAP 两次都自然完成，退出码为 0，均产生一个包含 64 张图片的模型，没有命中配置的 300 秒映射上限。

## 注册与重投影

| 方案 | 注册图片 | 稀疏点 | 有效观测 | 平均误差 / px | RMS / px | 时间 / s |
|---|---:|---:|---:|---:|---:|---:|
| Photara Auto（回退针孔） | 63/64 | 22,884 | 52,834 | 0.7783 | 0.9346 | 11.15 |
| Photara 鱼眼，默认焦距 | 64/64 | 26,204 | 69,003 | 0.7798 | 0.9313 | 12.31 |
| COLMAP OPENCV | 64/64 | 17,766 | 78,205 | 1.2075 | 1.4540 | 171.33 |
| COLMAP OPENCV_FISHEYE | 64/64 | 27,453 | 170,776 | 0.5666 | 0.8059 | 62.34 |
| Photara 鱼眼，**COLMAP 焦距辅助初值** | 64/64 | 29,680 | 90,977 | 0.4729 | 0.6241 | 12.38 |

COLMAP 的误差从每个注册图像的 2D–3D 观测及最终内参、姿态重新投影计算，使用逐观测口径；不同于 `model_analyzer` 输出的按点平均误差。点数及观测数与其内置 analyzer 一致，两个 COLMAP 模型均无非正深度的有效观测。

Photara 最终过滤约为 2 px，COLMAP 默认过滤为 4 px，且保留的点/观测集合不同。因此不能把 Photara 辅助实验的更小重投影误差解读为更高的真实精度。COLMAP 鱼眼有效观测约为 Photara 默认鱼眼的 2.47 倍，平均轨迹长度约 6.22 对 2.63；注册率相同并不代表约束强度相同。

## 相机位置与朝向差异

以 COLMAP OPENCV_FISHEYE 为对照。对所有同名注册相机拟合**一次正尺度、无反射 Sim(3)**（尺度+旋转+平移），不删除离群相机。位置差异除以对照相机中心到质心距离的 RMS，报告百分比；由于没有测量尺度，不能换算成厘米。

| Photara 方案 | 公共相机 | 位置中位数 / 参考半径 % | 位置 P95 / 参考半径 % | 朝向中位数 / ° | 朝向 P95 / ° |
|---|---:|---:|---:|---:|---:|
| Auto（回退针孔） | 63 | 56.47 | 145.60 | 167.80 | 176.03 |
| 鱼眼，默认焦距 | 64 | 41.63 | 88.59 | 23.82 | 26.72 |
| 鱼眼，COLMAP 焦距辅助初值 | 64 | 1.45 | 5.87 | 0.923 | 2.36 |

这些数值是与对照的差异，**不是真值误差**。朝向差异使用同一次基于相机位置的 Sim(3)；没有额外单独旋转姿态以降低角度差。坐标约定、二进制读取与 Sim(3) 的五项既有测试全部通过；对照相机布局三个奇异值约为 28.39、8.11、5.59，不是共线或共面的退化配准。

COLMAP 自身的针孔和鱼眼模型也有明显差异：位置 P95=89.13% 参考半径，朝向 P95=31.94°。因此不能随便拿任一相机模型的 COLMAP 结果当真值。本次鱼眼模型具有更低的观测误差和更多几何支持，作为主要对照更有依据，但仍需外部真值才能断言物理精度。

![相机轨迹与姿态差异](D:/ProgramCode/C++/3dgs/Photara/artifacts/colmap_comparison_64_20260908/trajectory_comparison.png)

## 焦距诊断

- COLMAP 鱼眼最终 fx≈603.99、fy≈604.69 px。
- Photara 默认鱼眼从 876 px 开始，最后 fx≈805.58、fy≈823.12 px；平均焦距比 COLMAP 高约 34.7%。
- 辅助实验只传 `--focal 604.34`，不导入 COLMAP 位姿、点云或匹配，也不锁定内参。最终 fx≈606.11、fy≈608.01 px；相机差异大幅下降。
- 该实验从 COLMAP 获取了信息，必须单列，不能算当前默认流程的独立成绩。

Auto 这次在针孔/鱼眼几何分数接近时回退针孔，这个保守回退在此数据上并未得到接近主对照的几何。后续优先事项是让自动选择同时比较局部重建与内参收敛结果，并改善鱼眼焦距的初始化及优化，而不是只依据注册数和最终过滤后的重投影误差判断成功。

## 证据与复现

目录：`D:/ProgramCode/C++/3dgs/Photara/artifacts/colmap_comparison_64_20260908`。

- `run.json`：COLMAP 每一阶段的完整命令、退出码、耗时。
- `run_benchmark.py`：两种相机模型的运行脚本；已有数据库时拒绝覆盖。
- `analyze.py`、`comparison.json`：逐观测重投影与公共相机差异。
- `plot_comparison.py`、`trajectory_comparison.png`：轨迹与朝向图。
- `image_manifest.json`：64 张输入图片的文件大小与 SHA-256。
- `opencv/`、`opencv_fisheye/`：数据库、稀疏模型、各阶段日志。
- `aether_focal604.asfm`、`aether_focal604_sfm_diagnostics.csv`、`aether_focal604.log`：辅助实验产物。

辅助实验命令：

```powershell
.\build\photara\Release\photara.exe `
  --images artifacts\fisheye_alameda_64\images `
  --output artifacts\colmap_comparison_64_20260908\aether_focal604.asfm `
  --camera-model opencv_fisheye --focal 604.34 `
  --mode global --max-features 6000 --window 6
```
