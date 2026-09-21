# Photara

面向大规模场景的摄影测量与 3D 高斯溅射引擎：SfM、MVS 与 3DGS 训练一体化，
使用 C++20 编写，核心计算在 CUDA 上加速。

典型链路是 **SfM → 3DGS（Splat）→ 网格 → 贴图**，MVS 是显式的可选稠密路径：

```text
images → SfM → 稀疏点/位姿 → 3DGS 训练 →（--mesh）TSDF / PAM →（--texture）烘焙
images → SfM →（--dense）MVS PatchMatch → 融合点云 → Delaunay/TSDF mesh
```

现已具备：

- 解析针孔 + 径向/切向/鱼眼 Jacobian，原生等距柱状（360°）相机模型；
- Huber IRLS + Levenberg-Marquardt（接受/拒绝、阻尼调整、规范自由度固定）；
- CPU 与 CUDA 两套 BA：CUDA 侧线性化、Schur 组装、块预条件 PCG、
  位姿/点/内参更新与代价评估全部驻留 GPU，并用 CUDA Graph 复用 PCG 序列；
- FreeImage 图像 IO + VLFeat SIFT/RootSIFT（默认不依赖 OpenCV）；
- 可插拔特征后端（`FeatureExtractor` / `FeatureMatcher` / registry）；
- AVX2/CUDA mutual-ratio 匹配、学习型词袋检索与渐进式弱视图增强；
- Eigen 多视几何（E/F/H RANSAC、PnP、三角化）；
- DISK/SuperPoint/ALIKED + LightGlue（可选 ONNX Runtime）；
- SiftGPU 适配（`PHOTARA_ENABLE_SIFTGPU`，因上游许可边界需自行确认）；
- 增量式 PnP/局部 BA、层次化 Sim(3) 合并与全局 rotation/positioning averaging；
- CUDA Gaussian Splatting 训练（ADC+/ADC-IGS/EMC 致密化、多视图几何与 NCC 监督）；
- MVS PatchMatch（CUDA 优先，CPU 回退）+ CGAL 全局 Delaunay graph-cut / 稀疏 TSDF；
- OpenMVS Interface（`.mvs` / MVSI）导出，可直接用 OpenMVS Viewer 查看；
- CPU/CUDA 数值一致性 benchmark、端到端优化测试与 MVS/TSDF 拓扑回归。

架构按阶段拆分：[流水线总览](docs/PIPELINE.md)、[SfM](docs/SFM_ARCHITECTURE.md)、
[MVS](docs/MVS_ARCHITECTURE.md)、[稠密重建与贴图](docs/DENSE_RECONSTRUCTION.md)、
[Splat C++ 后端](docs/SPLAT_CPP.md)、[特征后端](docs/FEATURES_BACKENDS.md)。

## 目录

```text
Photara/
├── CMakeLists.txt                 总工程入口
├── apps/editor/                   Photara Studio（Vulkan + Dear ImGui 编辑器）
├── docs/                          架构、特性与质量文档
├── photara/
│   ├── CMakeLists.txt             核心库（Photara::BA/Features/SfM/MVS/Splat/Texture）
│   ├── include/                   稳定的公开 C++ API（ba/features/sfm/mvs/splat/texture/...）
│   ├── third_party/vlfeat/        精简 VLFeat SIFT（BSD）
│   ├── src/                       ba / features / io / sfm / mvs / splat / texture / project / tools
│   └── tests/                     CTest 正确性测试
└── third_party/
    └── aether_drender/            纹理烘焙 / 网格预处理（git submodule）
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

克隆后先初始化 submodule（贴图模块依赖 `third_party/aether_drender`）：

```powershell
git submodule update --init --recursive
```

```powershell
cmake -S . -B build `
  -DPHOTARA_ENABLE_CUDA=ON `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
cmake --build build --config Release --parallel
```

默认只编译 `photara.exe` 和 `photara_studio.exe`。需要正确性测试或额外工具时再打开：

```powershell
cmake -S . -B build -DPHOTARA_BUILD_TESTS=ON -DPHOTARA_BUILD_BENCHMARKS=ON
cmake --build build --config Release --target photara_tests --parallel
ctest --test-dir build -C Release --output-on-failure
```

CPU-only：

```powershell
cmake -S . -B build-cpu -DPHOTARA_ENABLE_CUDA=OFF `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
cmake --build build-cpu --config Release --parallel
```

启用 LightGlue / ONNX（默认关闭，与「可选依赖」一致；COLMAP 风格开关）：

```powershell
# A) 本地已有 SDK（推荐，跳过下载）
cmake -S . -B build `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" `
  -DPHOTARA_ENABLE_ONNX=ON `
  -DPHOTARA_FETCH_ONNX=OFF `
  -DPHOTARA_ONNXRUNTIME_ROOT="D:/sdk/onnxruntime-win-x64-gpu-1.20.1"

# B) 自动 FetchContent 拉取官方包（需能访问 GitHub）
cmake -S . -B build `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" `
  -DPHOTARA_ENABLE_ONNX=ON `
  -DPHOTARA_FETCH_ONNX=ON `
  -DPHOTARA_ONNX_VERSION=1.20.1
```

| CMake 选项 | 默认 | 含义 |
|------------|------|------|
| `PHOTARA_BUILD_TESTS` | OFF | 是否生成正确性测试可执行文件 |
| `PHOTARA_BUILD_BENCHMARKS` | OFF | 是否生成额外 CLI / benchmark 工具 |
| `PHOTARA_BUILD_STUDIO` | ON | 是否生成 Vulkan + Dear ImGui 编辑器 `photara_studio` |
| `PHOTARA_BUILD_PYTHON` | OFF | 是否生成 nanobind Python 实验模块 |
| `PHOTARA_ENABLE_CUDA` | ON | CUDA 加速（BA / splat / MVS PatchMatch） |
| `PHOTARA_ENABLE_SPLAT` | ON | CUDA Gaussian splat 训练（TinyTensor + splat_drender） |
| `PHOTARA_ENABLE_VULKAN_COMPUTE` | ON | TinyTensor 的 Vulkan 计算后端（HLSL/DXC） |
| `PHOTARA_ENABLE_ACCUTILE` | ON | opacity-aware SnugBox/AccuTile 枚举 |
| `PHOTARA_ENABLE_FEATURES` | ON | 特征提取与匹配模块 |
| `PHOTARA_ENABLE_SIFTGPU` | ON | 可选的 SiftGPU 适配（上游为非商用许可） |
| `PHOTARA_ENABLE_NATIVE_ARCH` | ON | 针对构建机 CPU 优化 |
| `PHOTARA_ENABLE_ONNX` | OFF | 是否编译 ONNX/LightGlue |
| `PHOTARA_FETCH_ONNX` | ON | 开启 ONNX 时是否自动下载 SDK |
| `PHOTARA_ONNX_VERSION` | 1.20.1 | Fetch 的 ORT 版本 |
| `PHOTARA_ONNXRUNTIME_ROOT` | 空 | 本地 SDK；有效时优先于 Fetch |
| `PHOTARA_ENABLE_TEXTURE` | ON | UVAtlas + Vulkan 贴图烘焙 |
| `PHOTARA_ENABLE_AETHER_MESH` | ON | CGAL aether_drender 网格修复/减面 |
| `PHOTARA_ENABLE_INSTANT_REMESH` | ON | CGAL 减面前先做 Instant Meshes 重拓扑 |
| `PHOTARA_AETHER_DRENDER_ROOT` | 自动 | aether_drender 路径（默认 `third_party/aether_drender`） |
| `PHOTARA_INTRINSIC_MODELS_DIR` | `${BUILD}/Models/Intrinsic` | Delight 的 `stage_*.onnx` 目录（许可见 [LICENSE-Intrinsic.md](docs/LICENSE-Intrinsic.md)） |

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
.\build\photara\Release\photara.exe `
  --images D:\ScanVideo\chuan\images `
  --output scene.mvs `
  --window 3
```

默认使用 `global` SfM；需要实验其它后端时可显式传入 `--mode incremental` 或
`--mode hierarchical`。必填参数为 `--images`、`--output`。`--focal` 可选：省略或
`0` 时用 `1.2 * max(宽,高)` 作初始值，再由 view-graph 共识与 BA 精化；已知标定可显式传入。其余选项见 `photara --help`。

`--focal` 默认只是初值；若输入是已标定的零畸变图像，可同时传入
`--trust-focal --focal <像素焦距>` 锁定内参。自动自标定退化时，全部图片注册和低重投影
误差仍可能对应错误轨迹；请先用输出旁的 `*_sfm_diagnostics.csv`（逐图 RMS/P95 误差、
相机中心、相邻位姿步长与 `alignment_reliable`）确认对齐质量。结构与质量门禁见
[SfM 架构](docs/SFM_ARCHITECTURE.md)，端到端阶段映射见[流水线总览](docs/PIPELINE.md)。

稠密重建使用整条流水线质量预设，而不只是调整图像分辨率：

```powershell
.\build\photara\Release\photara.exe `
  --images D:\ScanVideo\ori_img\images `
  --mode global `
  --output scene.ply `
  --cache-dir cache `
  --dense --mesh `
  --mesh-method auto `
  --mesh-dist-insert-px 1.25 `
  --dense-quality default  # preview | default | high
```

对于 360° 环拍、转台或首尾视角重叠的数据，优先使用 `--mode global`。
这类闭环序列若使用 incremental SfM，局部重投影误差即使看起来不高，累计位姿漂移仍可能在
MVS 中表现为轮廓双层、底座重叠或缺失。程序会在输出旁生成
`*_sfm_diagnostics.csv`，其中包含逐图 RMS/P95 重投影误差、相机中心、旋转和相邻位姿步长，
应先通过该报告确认 SfM，再调整 MVS 阈值。

`default` 面向常规交付；`high` 使用全分辨率、更多邻居和更严格的多视图几何/融合约束。
`--dense-resolution-level` 可在预设之后单独覆盖工作分辨率；不传则跟随 `--dense-quality`。

CPU PatchMatch 会一次缓存所有图像金字塔，并默认同时处理 8 个参考视图；每个视图内部再按
8 行 tile 做 red/black 并行传播。可用 `--patchmatch-concurrent-views` 和
`--patchmatch-tile-rows` 调整。`--mesh-method auto` 在 splat 路径使用 median-depth TSDF；
MVS-only 的 preview/default/high 均使用 CGAL 全局 Delaunay visibility
graph-cut，不再提供 projective mesh。全局图的输入上限由
`--mesh-max-points` 控制（默认 2,000,000，0 表示不限）。Delaunay 与 TSDF 两个 mesh 后端
都会经过统一 Clean；构建了 `aether::mesh` 且 splat mesh 超过 `--mesh-target-faces` 时，随后执行 Instant
Meshes field-aligned remesh 和 CGAL repair/decimate。已经低于目标面数的网格结果会直接保留，
避免无意义的重采样和修复引入新边界。目标面数默认为 1,000,000；Instant 的 quad 目标自动
换算为约一半，重拓扑前后会清理微小连通碎片。

Splat + TSDF 默认按 `max_depth / 2048` 取体素（与 gs2mesh.py 一致，即
`--mesh-tsdf-voxel-scale 1`），需要更粗但更规则的网格时再用
`--mesh-tsdf-voxel-scale` 放大，从而避免高密 TSDF 经通用减面产生跨孔长三角。
`--mesh-target-faces 0` 保留原始 TSDF 分辨率用于诊断。
UVAtlas 默认用 `--uv-parallel-partitions 8` 做空间分区并发展开，最后统一打包到单张 atlas；
设为 `1` 可回到串行展开。
`--mesh-remesh=false` 只关闭 remesh，目标面数设 0 可关闭整个 aether mesh 后处理。Splat mesh 模式
默认从 `--splat-geometry-from-iter`（默认 3,000 步）开启权重 0.05 的
median-depth/rendered-normal 几何一致性优化。

稠密 MVS 输入使用全部融合点初始化 splat，不再抽样。稀疏输入的 Gaussian 增长由
`--splat-densification-cap` 限制（默认 1,000,000）。
Splat mesh 导出只使用 alpha 0.5 与有效深度掩码，不再默认执行额外的 60° depth-normal 硬过滤；

当前 splat 训练借鉴 GGGS 的几何监督，但已融合 GaussianWrapping normal field 与
独立 mesh 后端，命令行统一使用 `--splat` / `--splat-*`。GaussianWrapping 几何路径可用
`--mesh-method pam`：训练默认从第 8,001 步学习四通道
normal field。PAM 不经过 TSDF：先从 Gaussian center 与 learned-normal pivot 运行
`tetra_triangulation` + Marching Tetrahedra，再将自适应采样点投影到多视图 Gaussian
occupancy 等值面，并通过第二次 CGAL Delaunay 四面体分类提取表面。可用
`--pam-pivot-max-points`、`--pam-max-points`、`--pam-occupancy-iso-value`、
`--pam-refinement-steps` 和 `--pam-neighbors` 控制质量与耗时。PAM 对完整场景重建，不使用
相机 focus、自动 ROI、场景 SubjectBounds 或外部凸包裁剪 mesh；细结构保留由致密化、
learned normal field、occupancy 和自适应采样负责。该后端要求构建时找到 CGAL。

- `scene.mvs`：OpenMVS Interface（MVSI），可用 OpenMVS Viewer 打开验证相机与稀疏点
- `scene.ply`：稀疏 XYZ；写出 PLY 时会额外生成同名 `scene.mvs`
- `scene_dense.ply`：多轮几何一致性和深度过滤后的稠密点云
- `scene_mesh.ply`：启用 `--mesh` 时生成的网格
- `scene_textured.obj` / `.mtl` / `_albedo.png`：启用 `--texture` 时由
  `aether_drender` UVAtlas + Vulkan 投影烘焙（可选 `--delight` 去光照）

贴图模块默认开启（`PHOTARA_ENABLE_TEXTURE=ON`），依赖 Vulkan SDK（含 `dxc`）与
git submodule `third_party/aether_drender`（https://github.com/fenghuayumo/aether_drender）。

`--delight` 纯 C++ **ONNX Runtime** 推理（与 LightGlue 相同开关
`-DPHOTARA_ENABLE_ONNX=ON`），无 Python/PyTorch。将 `stage_0..3.onnx` 放到
`PHOTARA_INTRINSIC_MODELS_DIR`（默认 `build-*/Models/Intrinsic`）。
Intrinsic 权重为学术/非商用许可，产品发布前必须完成许可证审查。

```powershell
.\build-cgal\photara\Release\photara.exe `
  --images D:\ScanVideo\ori_img\images `
  --output scene.mvs `
  --dense --mesh --texture --atlas-resolution 2048

# albedo（需 ONNX Runtime + stage_*.onnx）
.\build-cgal\photara\Release\photara.exe `
  --images D:\ScanVideo\ori_img\images `
  --output scene.mvs `
  --dense --mesh --texture --delight
```

## 文档

| 文档 | 内容 |
|---|---|
| [PIPELINE.md](docs/PIPELINE.md) | 端到端流水线：SfM → MVS → 3DGS → mesh → 贴图的阶段、触发条件、产物 |
| [SFM_ARCHITECTURE.md](docs/SFM_ARCHITECTURE.md) | 前端、三种 mapping 模式、BA 后端、相机模型、导出与诊断 |
| [MVS_ARCHITECTURE.md](docs/MVS_ARCHITECTURE.md) | PatchMatch 深度、深度融合、Delaunay/TSDF 网格、Clean 与质量预设 |
| [DENSE_RECONSTRUCTION.md](docs/DENSE_RECONSTRUCTION.md) | Splat/TSDF/贴图主链路设计与当前实现状态 |
| [SPLAT_CPP.md](docs/SPLAT_CPP.md) | CUDA splat 后端、致密化策略、训练数据流水线与性能分析 |
| [FEATURES_BACKENDS.md](docs/FEATURES_BACKENDS.md) | 特征提取 × 匹配后端的组合与兼容矩阵 |
| [LICENSE-Intrinsic.md](docs/LICENSE-Intrinsic.md) | Delight（Intrinsic）权重的许可边界与放置位置 |
