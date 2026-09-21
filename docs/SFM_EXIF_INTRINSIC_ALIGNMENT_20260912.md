# EXIF 分组内参对齐验证（2026-09-12）

参考 `D:/ProgramCode/C++/spirula-studio/src/sfm` 的做法后，本轮修正一个混合焦距/变焦数据的
质量瓶颈：前端仍然先建立连通的全局图，但在全局定位之后读取 EXIF 的机身、镜头和焦距标识，把
不同物理焦距状态拆成独立 BA 内参组。这样避免早期硬分组把稀疏的变焦过渡边切断，又能在位姿骨架
稳定后分别优化 focal/aspect/distortion。无 EXIF 的视频抽帧仍保持原来的共享内参路径。

## 实现

- `io::load_camera_identity()` 读取 `Make/Model/BodySerialNumber/LensMake/LensModel/FocalLength/FocalLengthIn35mmFormat`，同时返回 EXIF 焦距；另读取 focal-plane tags 计算 COLMAP/Spirula 风格的像素焦距先验字段（先验只在 EXIF 覆盖率足够时使用）。
- `Scene::Image` 增加运行时 camera identity 与焦距字段；前端 track checkpoint 命中后从源图补齐元数据，冷启动与续跑一致。
- `run_global_mapping()` 在结构/位置 BA 前按 identity 分组；每组以共享解的焦距乘以“本组 EXIF 焦距 / 全体中位 EXIF 焦距”初始化，然后进入后续联合 BA。
- 前端缓存 fingerprint 追加 `camera-late-exif-split-v1`，防止旧缓存复用缺少元数据或旧相对位姿的结果。

室内 Canon 数据按 EXIF/镜头状态分成多组；日志会输出：

```text
global: split EXIF intrinsic groups=5 images_per_group=185,43,505,5,1 median_focal_length_mm=20
```

## 结果

所有运行使用 `--camera-model auto --mode global --max-features 6000 --window 6`，全新缓存。硬件为
RTX 5090 D v2。参考模型仍只是既有 COLMAP 输出的一致性对照，不是测量真值。

| 数据集 | 注册 | RMS (px) | 中心 P95 / 最大（参考半径 %） | 旋转 P95 / 最大（°） | 耗时 | 门槛 |
|---|---:|---:|---:|---:|---:|---|
| `WeChat_20250712175936` | 198/198 | 0.8225 | 2.169 / 2.809 | 0.667 / 0.969 | 27.68 s | 通过 |
| `WeChat_20250715211138` | 153/153 | 0.7739 | 0.168 / 0.184 | 0.110 / 0.118 | 33.41 s | 通过 |
| 室内 Canon 数据 | 737/739 | 0.8197 | 8.266 / 185.052 | 1.782 / 46.283 | 155.95 s | 未通过 |

室内 Canon 数据与上一版最终结果对照：

| 版本 | 注册 | RMS (px) | 中心 P95 / 最大（参考半径 %） | 旋转 P95 / 最大（°） | 耗时 |
|---|---:|---:|---:|---:|---:|
| 修改前 | 738/739 | 0.883123 | 16.243 / 188.233 | 2.727 / 47.164 | 154.378 s |
| 本轮 | 737/739 | **0.819706** | **8.266** / 185.052 | **1.782** / 46.283 | 155.948 s |

因此中心 P95 相对参考半径下降约 49%，旋转 P95 下降约 35%，重投影 RMS 下降约 7%；端到端耗时
增加约 1.0%。但新增一个相机掉出注册集合（`DCIM5012-HDR.jpg` 与 `DCIM5732.jpg`）。`DCIM5783.jpg`
的参考中心在相邻帧中先跳出约 1.0 个参考半径再返回，本结果在该相机仍有约 46°旋转分歧；不能在未
解释该参考异常前宣称 iPhone/Canon 数据的几何验收已经通过，也不能用 RMS 掩盖中心 P95 和最大值
门槛失败。

中间否决项：前端一开始按 EXIF 硬分组会只剩 548/739 张相机，registration/reference coverage
失败；按每张图独立内参过拟合，RMS 劣化到 1.0550 px、中心 P95 到 19.575%，均已撤回。

## 复现

```powershell
build/photara/Release/photara.exe --images 'D:\BaiduNetdiskDownload\室内iphone\images' --output artifacts/sfm_align_probe_20260912/iphone_exif_prior.asfm --cache-dir artifacts/sfm_align_probe_20260912/cache_iphone_exif_prior --camera-model auto --mode global --max-features 6000 --window 6
python experiments/sfm_acceptance.py --diagnostics artifacts/sfm_align_probe_20260912/iphone_exif_prior_sfm_diagnostics.csv --reference 'D:\BaiduNetdiskDownload\室内iphone\sparse\0' --output artifacts/sfm_align_probe_20260912/iphone_exif_prior_vs_reference.json
python experiments/compare_sfm_runs.py --before artifacts/sfm_gpu_iphone_20260912/final.log --after artifacts/sfm_align_probe_20260912/iphone_exif_prior.log --output artifacts/sfm_align_probe_20260912/iphone_exif_prior_stability.json
```

视频数据与对应 `_vs_reference.json` 证据在 `artifacts/sfm_align_probe_20260912/`。修改后的 BA、
two_view、mapping、checkpoint CTest 均通过。单次耗时受桌面负载影响，不做统计性提速结论。
