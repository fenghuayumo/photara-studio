# Photara 流水线总览（SfM → MVS → 3DGS）

本文按**当前代码**描述 Photara 从运动恢复结构（SfM）、多视图立体（MVS）到
3D Gaussian Splatting（3DGS / splat）的完整链路：每条链路由什么触发、经过哪些阶段、
落到哪些文件。算法细节见
[SfM 架构](SFM_ARCHITECTURE.md)、[MVS 架构](MVS_ARCHITECTURE.md)、
[稠密重建与贴图](DENSE_RECONSTRUCTION.md)、[Splat C++ 后端](SPLAT_CPP.md)。

---

## 1. 三条主链路

`photara` CLI（`photara/src/tools/reconstruct.cpp`）根据输入和开关选择链路，
不做“隐式跑 MVS 再隐式训练 splat”的组合。

```text
① 产品默认（未指定 --dense）
   images -> SfM -> 稀疏点/位姿 -> splat 稀疏初始化 -> 3DGS 训练
          ->（--mesh）TSDF 或 PAM ->（--texture）UV/投影烘焙

② 显式 MVS（--dense，且未指定 --splat）
   images -> SfM -> MVS PatchMatch 深度 -> 深度融合 dense.ply
          ->（--mesh）CGAL 全局 Delaunay graph-cut 或 TSDF ->（--texture）

③ 外部数据集（--splat-dataset）
   COLMAP / RealityCapture / OpenMVS -> 相机 + 稀疏点 -> 3DGS
   加 --dense 时改为：固定导入位姿 -> MVS -> dense/mesh（不训练 splat）
```

| 链路 | 触发条件 | 稀疏初始化 | 是否训练 splat | 是否跑 PatchMatch |
|---|---|---|---|---|
| 产品默认 | `--splat`、`--capture-mode`、内部 SfM + `--splat` | SfM 稀疏点（日志 `splat_input=sfm_sparse`） | 是 | 否 |
| 稠密初始化 | `--dense --splat` | MVS 融合点云（全部点） | 是（关闭动态致密化） | 是 |
| 外部点云初始化 | `--dense --dense-ply`（无 `--splat-dataset`） | 给定 PLY（内部 SfM 位姿） | 是 | 否 |
| 纯 MVS | `--dense`（无 `--splat`） | — | 否 | 是 |
| 外部数据集 | `--splat-dataset` | 数据集自带稀疏点或 `--dense-ply` | 是 | 否 |
| 外部数据集 + MVS | `--splat-dataset --dense` | 数据集相机位姿（点云由 PatchMatch 重建，融合会覆盖输入点云） | 否 | 是 |

`--capture-mode object|scene` 是产品级预设：显式给出时自动启用 splat 与 mesh
（`object` 还会从 SfM 稀疏点估计 `SubjectBounds`）。省略该参数则保留
“SfM-only / 低层开发”语义。

两个容易混淆的组合：

- `--splat-dataset --dense`：`parse_cli` 只在“未显式 `--dense`”时把
  `--splat-dataset` 当作 splat 训练，因此该组合走**固定导入位姿的 MVS**；
  若再显式加 `--splat`，则回到 splat 训练（使用数据集点云，不再跑 PatchMatch）。
- `--mvs-mesh-only`（配合 `--mask-mesh` 或 `--dense-ply`）：直接由已有密集点云
  建 CGAL 网格、或直接读入已有 PLY 网格，然后用 `aether_drender` 渲染
  `<stem>_masks/` 前景掩码与 `<stem>_mesh_previews/` 预览，用于外部网格的质量门禁。

---

## 2. 代码地图

```text
photara/
  include/{sfm,mvs,splat,texture,ba,features,io,project,core}/   公开 API
  src/sfm/          前端（特征/匹配/几何验证/tracks）+ 三种 mapping + BA 编排 + 导出
  src/mvs/          PatchMatch 深度、深度融合、Delaunay/TSDF mesh、Clean、导出
  src/splat/        dataset 读取、CUDA 训练器、致密化策略、mesh 提取、格式读写
  src/ba/           LM + Schur + PCG 的 CPU/CUDA 后端（SfM 与其它阶段共享）
  src/texture/      UVAtlas + aether_drender 投影烘焙 + Intrinsic delight
  src/project/      .ascan 工程容器（settings / sfm / gaussians / mesh）
  src/tools/        CLI：reconstruct.cpp（photara.exe）与 benchmark 工具
apps/editor/        Photara Studio（Vulkan + ImGui 编辑器，驱动 photara.exe）
photara/tests/      CTest 正确性/回归测试
```

对应的 CMake 目标是 `Photara::SfM`、`Photara::MVS`、`Photara::Splat`
（见 `photara/CMakeLists.txt`）。`Photara::Splat` 需要 `PHOTARA_ENABLE_SPLAT=ON`
与 CUDA；MVS 的 CUDA PatchMatch 与 CGAL mesh 分别是 `PHOTARA_ENABLE_CUDA`
和 CGAL 的可选依赖。

---

## 3. 阶段与实现对应表

日志中的 `StageScope` 名字可直接用于定位阶段耗时。

| 阶段 | 日志 stage | 主要实现 | 关键产物 |
|---|---|---|---|
| 视频抽帧 | 进度组 `extract video frames` | `src/io/video_frames.cpp` | `<video_stem>/images/*.jpg` |
| 特征/匹配/验证/tracks | `sfm.frontend`（含 `sfm.retrieve_image_pairs`、`sfm.build_tracks`） | `src/sfm/frontend.cpp`、`src/features/*` | `--cache-dir` 下的分阶段 checkpoint |
| Mapping | `sfm.incremental_mapping` / `sfm.global_mapping` / `sfm.hierarchical_mapping` | `src/sfm/{reconstruct,resection,global_rotation,global_positioning,hierarchical}.cpp` | `sfm::Scene` |
| Bundle Adjustment | `ba.cpu` / `ba.cuda` | `src/ba/optimizer_{cpu,cuda}.cpp`、`src/sfm/bundle.cpp` | 优化后的位姿/内参/点 |
| SfM 导出 | `sfm.export_openmvs` | `src/sfm/export_{mvs,colmap,nerfstudio}.cpp`、`sfm/asfm.hpp` | `.mvs` / `.ply` / `.asfm` / `.ascan` / COLMAP 文本 / `*_sfm_diagnostics.csv` |
| SubjectBounds | `sfm.subject_bounds` | `src/mvs/subject_bounds.cpp` | `<stem>_subject_bounds.txt` |
| MVS 场景构建 | `mvs.load_images` → `mvs.build_scene` | `src/mvs/scene_build.cpp` | 工作分辨率视图 + 稀疏点 |
| 邻居选择 | `mvs.select_neighbors` | `src/mvs/neighbors.cpp` | 每视图 source 列表 |
| 深度估计 | `mvs.build_pyramids` → `mvs.estimate_depth`（含 `mvs.geometric_consistency`、`mvs.filter_depth`） | `src/mvs/patchmatch.cpp`、`patchmatch_cuda.cu` | 逐视图深度/法线/置信度 |
| 深度融合 | `mvs.fuse` | `src/mvs/fuse.cpp` | `DenseCloud` → `<stem>_dense.ply` |
| Mesh 提取 | `mvs.mesh` → `mvs.mesh_global_cgal` 或 `mvs.mesh.tsdf` | `src/mvs/mesh_cgal.cpp`、`mesh_tsdf.cpp` | `Mesh` |
| Mesh 清理 | `mvs.mesh_clean` / `mvs.mesh_clean.tsdf` | `src/mvs/mesh_clean.cpp` | 清理后的单一网格 |
| aether 后处理 | `repair_and_decimate_mesh` | `reconstruct.cpp`（需要 CGAL / Instant Meshes） | 目标面数网格 |
| 3DGS 训练 | 无独立 StageScope；训练日志 `splat iteration=…`、profiler 行 | `src/splat/trainer.cpp`、`rasterizer.cu`、`densification_*.cpp` | `<stem>_splat.ply` / `.sog` / `.spz` / `.glb` |
| Splat mesh | `mvs.mesh` → `mvs.mesh.tsdf`（TSDF）或 `splat.pam`（PAM） | `src/splat/mesh.cpp`、`pam_mesh.cpp` | `<stem>_splat_mesh.ply`、`*_pam_*` |
| 贴图 | `texture.load_views` → `texture.uv_unwrap` → `texture.project` → `texture.optimize`（可选 `texture.delight`） | `src/texture/*`、`third_party/aether_drender` | `<stem>_textured.obj/.mtl/_albedo.png` |

---

## 4. CLI 蕴含规则

以下规则由 `reconstruct.cpp::parse_cli` 实现，决定“一个开关会带来哪些阶段”：

```text
--capture-mode <object|scene>   -> --splat，且未显式 --mesh 时打开 --mesh
--delight                       -> --texture
--texture（非 texture-only）     -> --mesh，进而在未指定 --splat 时 -> --dense
--mesh 且未指定 --splat          -> --dense（走 MVS，而不是 splat TSDF）
--mesh-obj                      -> --mesh
--splat-dataset                 -> splat 训练；再加 --dense 时改为固定位姿 MVS
--dense-ply（无外部数据集）      -> 打开 splat；配合 --dense 时保留内部 SfM 位姿、
                                   用该点云替换稠密初始化（日志 splat cameras=internal_sfm）
--mvs-mesh-only / --mask-mesh   -> 需要 --splat-dataset，且需 --dense-ply 或 --mask-mesh
--masks auto                    -> <images 同级>/masks；`-` 表示禁用
```

约束与当前限制：

- `--mesh-method pam` 必须与 splat 训练同用（否则报
  `--mesh-method pam requires splat training`）；
- 外部 splat 数据集 + `--texture` 目前直接报错
  （`--texture is not yet available in the direct external splat path`）；
- `--ba-backend auto|cpu|cuda` 只切换 BA 求解后端，不改变重建语义；
- `--splat` 需要 CUDA 与 `PHOTARA_ENABLE_SPLAT=ON`，否则 CLI 立即拒绝。

---

## 5. 产物清单

以 `--output <path>` 的 stem 为前缀（产物目录取该路径的父目录）：

| 文件 | 生成条件 | 内容 |
|---|---|---|
| `<stem>_sfm_diagnostics.csv` | 内部 SfM 成功且非 `--gui` | 逐图 RMS/P95 重投影误差、相机中心、旋转、相邻位姿步长 |
| `<stem>_subject_bounds.txt` | `object` 模式且 bounds 有效 | SubjectBounds（世界坐标范围） |
| `<stem>.mvs` | `--output *.mvs` 或 `--export-mvs` | OpenMVS Interface（相机 + 稀疏点，含颜色） |
| `<stem>.ply` | `--output *.ply` | 稀疏 XYZRGB 点云；同时写出同名 `.mvs` |
| `<stem>.asfm` | `--output *.asfm` | 原生 SfM 场景（相机/位姿/关键点/tracks；无描述子与 pair） |
| `<stem>.ascan` | `--output *.ascan` | 工程容器：settings + SfM ± gaussians ± mesh |
| `<stem>_dense.ply` | `--dense` | 融合点云（含权重与逐相机权重，可复现 graph-cut） |
| `<stem>_mvs_mesh.ply` | `--dense --mesh`（非 splat） | MVS 网格（Delaunay 或 TSDF） |
| `<stem>_splat.ply` / `.sog` / `.spz` / `.glb` | `--splat` | 训练后的 Gaussian 模型（后缀决定格式） |
| `<stem>_splat_mesh.ply` | `--splat --mesh`（非 PAM） | TSDF 提取并 Clean 的网格 |
| `<stem>_splat_surface.ply` | `--splat --mesh` 且存在未融合表面点 | TSDF 融合前的表面点 |
| `<stem>_pam_pivot_mesh.ply`、`<stem>_pam_candidates.ply` | `--splat --mesh --mesh-method pam` | PAM 一级 pivot 网格与二级候选点 |
| `<stem>_textured.obj` / `.mtl` / `_albedo.png` | `--texture` | 展开并烘焙的贴图模型（`--delight` 输出 albedo） |
| `<stem>_splat_iter_*_view_*.png`、`<stem>_splat_view_*.png` | `--splat` 且非 `--gui` | 训练中与最终的评估渲染 |
| 深度/法线/alpha 诊断 PNG | 设置了 TSDF 或 splat 提取诊断目录 | 用于核对几何的中间可视化 |

`--gui` 模式（编辑器子进程）跳过 ascan/asfm/PLY 旁路产物与评估 PNG，只保留
工作副本（`--working-sfm/splat/mesh/dense/texture`）与日志。

---

## 6. 不变量与边界

1. **MVS 不是 splat 的前置。** 默认链路只消费 SfM 相机、稀疏点、原图与观测，
   日志显式打印 `splat_input=sfm_sparse ... patchmatch=false`。
2. **MVS 是可选的诊断/兼容后端。** 它服务 `--dense` 校准路径、OpenMVS 互操作与
   算法 A/B；其点云/深度不得自动成为主体 Mask 或 splat 初始化输入。
3. **Mask 只约束 splat 训练。** 自动投影 Mask 仅作为训练输入，TSDF 只读取 splat
   渲染的 alpha/depth/normal；`object` 模式要求每个训练视图都有匹配 Mask，否则报错。
4. **Mesh 后端取决于来源。** splat 来源 = TSDF（默认）或 PAM；MVS 来源 =
   CGAL 全局 Delaunay graph-cut（默认）或 TSDF。`--mesh-remesh` /
   `--mesh-target-faces` 仅在构建了 aether mesh 时生效。
5. **Texture/Delight 只消费 mesh。** 贴图不反馈到 splat 训练，也不重新参与对齐。
6. **无 CUDA 不训练 splat。** `PHOTARA_ENABLE_SPLAT=OFF` 时 `--splat` 直接报错；
   MVS PatchMatch 会自动回退到 CPU 实现。

---

## 7. 快速命令

```powershell
# SfM -> 3DGS（默认稀疏初始化）
build\photara\Release\photara.exe --images images --output scene.ply --splat

# SfM -> MVS -> mesh（显式稠密路径）
build\photara\Release\photara.exe --images images --output scene.ply --dense --mesh

# 产品预设：object 模式 + 贴图（自动打开 splat 与 mesh）
build\photara\Release\photara.exe --images images --output object.ply --capture-mode object --texture

# 外部 COLMAP 数据集直接训练
build\photara\Release\photara.exe --images images --splat-dataset D:\data\colmap `
  --output scene.ply
```

回归测试入口：

```powershell
cmake -S . -B build -DPHOTARA_ENABLE_CUDA=ON -DPHOTARA_BUILD_TESTS=ON
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

当前 CTest 覆盖 `photara.ba.optimizer`、`photara.bearing_cuda`、
`photara.parallel.thread_pool`、`photara.core.logging`、`photara.features`、
`photara.io.video_frames`、`photara.sfm.{two_view,equirect,mapping,vocabulary,checkpoint,submap_recovery,hierarchical,export_mvs}`、
`photara.splat.rasterizer`、`photara.tinytensor.vulkan`、`photara.project.io`、
`photara.mvs.pipeline`。
