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

启用 LightGlue 时传入 ONNX Runtime SDK：

```powershell
cmake -S . -B build `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" `
  -DAETHERSCAN_ONNXRUNTIME_ROOT="C:/sdk/onnxruntime-win-x64-gpu-1.20.1"
```

## 运行

```powershell
.\build\aetherscan\Release\aetherscan.exe `
  --images D:\ScanVideo\chuan\images `
  --focal 900 `
  --mode incremental `
  --output scene.mvs `
  --window 3
```

必填参数为 `--images`、`--focal`、`--mode`、`--output`；其余选项见 `aetherscan --help`。

- `scene.mvs`：OpenMVS Interface（MVSI），可用 OpenMVS Viewer 打开验证相机与稀疏点
- `scene.ply`：稀疏 XYZ；写出 PLY 时会额外生成同名 `scene.mvs`

详细设计见 [docs/SFM_ARCHITECTURE.md](docs/SFM_ARCHITECTURE.md)。
