# AetherScan

面向大规模摄影测量的 GPU-first SfM/MVS 引擎，使用 C++20 与 CUDA。

当前里程碑聚焦 SfM 的性能瓶颈：Bundle Adjustment 后端。现已具备：

- 解析针孔 + 径向/切向畸变 Jacobian；
- OpenMP CPU 与 CUDA 并行线性化，GPU 参数和观测持久驻留；
- Huber IRLS 鲁棒加权；
- CPU 块 Schur 消元、matrix-free Schur 乘法、块预条件 PCG；
- Levenberg-Marquardt 接受/拒绝、阻尼调整、规范自由度固定；
- FreeImage 图像 IO + VLFeat SIFT/RootSIFT（默认不依赖 OpenCV）；
- 可插拔特征后端（`FeatureExtractor` / `FeatureMatcher` / registry，便于接 SuperPoint 等）；
- AVX2 L2 mutual-ratio 描述子匹配与外层任务池并行前端；
- Eigen 多视几何（E/F/H RANSAC、PnP、三角化）；
- DISK/SuperPoint + LightGlue（可选 ONNX Runtime）；
- 可选 CUDA SiftGPU 适配（因上游许可限制默认关闭）；
- 增量式 PnP/局部 BA 与全局 rotation/translation averaging；
- OpenMVS Interface（`.mvs` / MVSI）导出，可直接用 OpenMVS Viewer 查看；
- CPU/CUDA 数值一致性 benchmark 和端到端优化测试。

完整的 GPU BA 尚未完成：目前 Schur 组装、PCG 和状态更新仍在 CPU，CUDA
只负责线性化。下一里程碑会让正规方程、Schur-PCG 和参数更新全程驻留 GPU，
避免每轮 PCIe 往返。

## 目录

```text
AetherScan/
├── CMakeLists.txt                 总工程入口
├── aetherscan/
│   ├── CMakeLists.txt             核心库子项目
│   ├── include/                   稳定的公开 C++ API（ba/features/sfm/...）
│   ├── third_party/vlfeat/        精简 VLFeat SIFT（BSD）
│   ├── src/                       BA / features / geometry / sfm
│   └── tests/                     正确性测试
└── docs/                          架构与性能路线
```

## 依赖

默认 SfM 路径只需要：

- **FreeImage**（图像解码）
- **Eigen3**（线性代数与多视几何）
- OpenMP（可选，加速匹配/BA）
- CUDA（可选，BA）
- ONNX Runtime（可选，LightGlue）

**不再默认链接 OpenCV。**

## 构建

```powershell
cmake -S . -B build `
  -DAETHERSCAN_ENABLE_CUDA=ON `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

CPU-only：

```powershell
cmake -S . -B build-cpu -DAETHERSCAN_ENABLE_CUDA=OFF `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
cmake --build build-cpu --config Release --parallel
```

启用 LightGlue / ONNX（默认关闭，与「可选依赖」一致；COLMAP 风格开关）：

```powershell
# A) 本地已有 SDK（推荐，跳过下载）
cmake -S . -B build `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" `
  -DAETHERSCAN_ENABLE_ONNX=ON `
  -DAETHERSCAN_FETCH_ONNX=OFF `
  -DAETHERSCAN_ONNXRUNTIME_ROOT="D:/sdk/onnxruntime-win-x64-gpu-1.20.1"

# B) 自动 FetchContent 拉取官方包（需能访问 GitHub）
cmake -S . -B build `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" `
  -DAETHERSCAN_ENABLE_ONNX=ON `
  -DAETHERSCAN_FETCH_ONNX=ON `
  -DAETHERSCAN_ONNX_VERSION=1.20.1
```

| CMake 选项 | 默认 | 含义 |
|------------|------|------|
| `AETHERSCAN_ENABLE_ONNX` | OFF | 是否编译 ONNX/LightGlue |
| `AETHERSCAN_FETCH_ONNX` | ON | 开启 ONNX 时是否自动下载 SDK |
| `AETHERSCAN_ONNX_VERSION` | 1.20.1 | Fetch 的 ORT 版本 |
| `AETHERSCAN_ONNXRUNTIME_ROOT` | 空 | 本地 SDK；有效时优先于 Fetch |

特征后端可自由组合（提取 × 匹配），例如：

```powershell
# 默认
--extractor siftgpu --matcher gpu_mutual_ratio

# SuperPoint 提取 + LightGlue 匹配（两者独立配置）
--extractor superpoint --extractor-model ...\superpoint.onnx `
--matcher lightglue --lightglue-model ...\superpoint_lightglue_fused.onnx `
--max-features 2048
```

同一条可组合路径也支持 ALIKED 和 SIFT：

```powershell
# ALIKED 提取 + ALIKED-LightGlue
--extractor aliked --extractor-model ...\aliked-n16rot.onnx `
--matcher lightglue --lightglue-model ...\aliked-lightglue.onnx `
--extractor-min-score 0.2 --max-features 2048

# SiftGPU 提取 + SIFT-LightGlue
--extractor siftgpu `
--matcher lightglue --lightglue-model ...\sift-lightglue.onnx `
--max-features 4096
```

质量与速度兼顾时，建议使用选择性 LightGlue 救援：先做 SiftGPU
互相一致性匹配和几何验证，只将失败的时序近邻或低连接度图像对交给
LightGlue，并用更严格的几何阈值接纳救援边。

```powershell
--extractor siftgpu --matcher hybrid_lightglue `
--lightglue-model ...\sift-lightglue.onnx --lightglue-min-score 0.1 `
--hybrid-lightglue-max-features 2048
```

融合端到端仍可用 `--pipeline lightglue_end2end`（可选旁路，不是默认组合方式）。
## 运行

```powershell
.\build\aetherscan\Release\aetherscan.exe `
  --images D:\ScanVideo\chuan\images `
  --output scene.mvs `
  --window 3
```

默认使用 `global` SfM；需要实验其它后端时可显式传入 `--mode incremental` 或
`--mode hierarchical`。必填参数为 `--images`、`--output`。`--focal` 可选：省略或
`0` 时用 `1.2 * max(宽,高)` 作初始值，再由 view-graph 共识与 BA 精化；已知标定可显式传入。其余选项见 `aetherscan --help`。

稠密重建使用整条流水线质量预设，而不只是调整图像分辨率：

```powershell
.\build\aetherscan\Release\aetherscan.exe `
  --images D:\ScanVideo\ori_img\images `
  --mode global `
  --output scene.ply `
  --cache-dir cache `
  --dense --mesh `
  --dense-quality default  # preview | default | high
```

对于 360° 环拍、转台或首尾视角重叠的数据，优先使用 `--mode global`。
这类闭环序列若使用 incremental SfM，局部重投影误差即使看起来不高，累计位姿漂移仍可能在
MVS 中表现为轮廓双层、底座重叠或缺失。程序会在输出旁生成
`*_sfm_diagnostics.csv`，其中包含逐图 RMS/P95 重投影误差、相机中心、旋转和相邻位姿步长，
应先通过该报告确认 SfM，再调整 MVS 阈值。

`default` 面向常规交付；`high` 使用全分辨率、更多邻居和更严格的多视图几何/融合约束。`--dense-resolution-level` 可在预设之后单独覆盖工作分辨率。

- `scene.mvs`：OpenMVS Interface（MVSI），可用 OpenMVS Viewer 打开验证相机与稀疏点
- `scene.ply`：稀疏 XYZ；写出 PLY 时会额外生成同名 `scene.mvs`
- `scene_dense.ply`：多轮几何一致性和深度过滤后的稠密点云
- `scene_mesh.ply`：启用 `--mesh` 时生成的网格

详细设计见 [docs/SFM_ARCHITECTURE.md](docs/SFM_ARCHITECTURE.md)。
