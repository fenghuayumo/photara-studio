# Photara SfM 架构

本文描述 `Photara::SfM`（`photara/include/sfm`、`photara/src/sfm`）的当前实现：
前端流水线、三种 mapping 模式、Bundle Adjustment 后端、相机模型、导出与质量门禁。
MVS 与 3DGS 侧见 [MVS 架构](MVS_ARCHITECTURE.md)、[Splat C++ 后端](SPLAT_CPP.md)；
端到端调用关系见[流水线总览](PIPELINE.md)。

---

## 1. 目标与边界

- 输入是图像目录（或视频抽出的静帧）；输出是注册相机位姿、稀疏 landmark 与观测。
- 默认**不依赖 OpenCV**：图像 IO 走 FreeImage / libjpeg / libpng，几何用 Eigen。
- 默认前端是 SiftGPU × GPU mutual-ratio，可用 ONNX（LightGlue / SuperPoint / DISK /
  ALIKED）替换；后端可插拔，见 `features/registry.hpp` 与 `features/compat.hpp`。
- SfM 只输出几何与外观（track 颜色）；不产生稠密点云，也不生成前景 Mask。

---

## 2. 数据模型

`sfm::Scene`（`photara/include/sfm/scene.hpp`）是唯一的重建状态容器：

```text
Scene
  cameras[]      PinholeCamera：fx,fy,cx,cy,k1,k2,p1,p2 + model + focal_prior/trust_intrinsics
  images[]       Image：path、features、camera_identity/focal_length_mm/exif_focal_px、
                 pose(Pose3D)、registered
  pairs[]        ImagePair：几何内点 matches、relative_pose/E/F/H、degenerate_planar、
                 zero_baseline、五类权重（spatial/geometry/connectivity/triplet/cycle）
  tracks[]       Track：position、observations（内点占据 [0,num_inliers)）、颜色、num_inliers
  image_tracks[] image_id -> 该图观测到的 track（resection/covisibility 热路径）
  resection_progress / registration_generation / thread_count
```

约定（与 OpenMVS / OpenMVG 一致）：`P = K R [I | -C]`，`R` 为 world→camera，
`C` 为世界坐标相机中心。`Pose3D` 提供 `transform_world_to_camera` /
`transform_camera_to_world` / 组合与求逆（`sfm/types.hpp`）。

`PinholeCamera::unproject/project/project_checked/local_reprojection/angular_error_px`
是所有几何模块共用的投影入口；等距柱状（360°）相机在切平面内计算残差与 Jacobian。

---

## 3. 前端流水线

入口：`sfm::run_frontend`（`src/sfm/frontend.cpp`），日志 stage `sfm.frontend`。

```text
① 图像指纹（SHA-256 + size + mtime）
② 特征提取（extractor）
③ 配对提案（sequential window + BoW retrieval，内容重复对标记为 zero-baseline）
④ 描述子匹配（matcher，CUDA/AVX2/ONNX）
⑤ 几何验证（E/F/H RANSAC，平面退化与零基线识别）
⑥ 视图图收尾：pair 权重 → 用标定后的焦距重算相对位姿 → 构建 tracks
```

### 3.1 特征提取

| 后端 | 说明 |
|---|---|
| `siftgpu`（默认） | CUDA SiftGPU；`--sift-contrast` 控制峰值阈值 |
| `sift` | 内置 VLFeat SIFT/RootSIFT（无 CUDA 时的默认可用路径） |
| `superpoint` / `disk` / `aliked` | ONNX（`--extractor-model`），需要 `PHOTARA_ENABLE_ONNX` |

- `--max-features`（默认 27,000）按 3×3 网格保留强特征，避免角点集中。
- 描述子默认压缩为 uint8（`compress_descriptors_u8`），匹配与检索直接消费紧凑形式。
- SiftGPU 走专有线程协调器（线程仿射），其它提取器按线程克隆并行。

### 3.2 配对提案

- 顺序窗口：`--window`（默认 3）→ `(i, i+k)`。
- BoW 检索：图像数 ≥ `retrieval_min_images`（50）时启用学习型层次词袋 +
  TF-IDF 倒排索引（`sfm::retrieve_image_pairs`，stage `sfm.retrieve_image_pairs`），
  默认 `top_k=50`、`max_descriptors_per_image=2000`、`sample_grid=3`、
  `stop_word_ratio=0.5`、`max_posting_images=64`；CLI 默认把检索结果**追加**到顺序窗口。
- 词汇表可持久化：`<cache-dir>/vocabulary.bin`。
- 字节相同的图像对被标记 `zero_baseline`，其相对位姿按“恒等约束”处理，
  不参与两视图几何估计。

### 3.3 匹配与几何验证

| matcher | 说明 |
|---|---|
| `gpu_mutual_ratio`（默认，别名 `siftgpu`） | CUDA 互一致 ratio 匹配（`--match-ratio` 默认 0.8） |
| `mutual_ratio` | CPU 版本（AVX2） |
| `lightglue` | ONNX 描述子匹配，可配 SIFT/SiftGPU/SuperPoint/DISK/ALIKED 描述子 |
| `hybrid_lightglue` | 先用 SiftGPU 快速匹配，再仅对“失败且时序近邻或弱连接”的对调用 LightGlue（救援 pass 使用更严格阈值） |
| `--pipeline lightglue_end2end` | 融合端到端配方（按对重新检测，较慢），与组合路径互斥 |

几何验证（`sfm/geometry.*`，`verify_pair_geometry`）：本质矩阵/基础矩阵/单应
RANSAC，输出 `relative_pose`、`E/F/H`、`mean_ray_angle`、`homography_ratio`、
`degenerate_planar`。平面退化或低视差对降低 `weight_geometry`，但不直接删除，
保证图仍有桥接边。

弱视图增强（默认开启，仅 `gpu_mutual_ratio`）：对“已验证度数不足 + 非平面邻居 < 3”
的图像用更高特征预算重新提取（`progressive_rescue_max_features`，
API 默认 54,000），并只对弱相关对重新匹配（`progressive_rescue_match_ratio=0.85`、
`progressive_rescue_min_inliers=20`）。`structural_pair_expansion` 仍是实验开关，
默认关闭。

### 3.4 Pair 权重与 tracks

`sfm::compute_pair_weights`（stage 内联于前端收尾）在两视图内点数之上叠加：

```text
weight_spatial       顺序/空间邻近先验
weight_geometry      平面/低视差折扣
weight_connectivity  该边在两个端点的局部重要度
weight_triplet       三视图旋转环一致性支持 [0,1]
weight_cycle         三视图强不一致惩罚（降权而非删除）
composite_weight()   min(inliers,1000) × 上述因子
```

`build_tracks`（`src/sfm/tracks.cpp`）按 `min_pair_weight` 合并观测；
`filter_tracks` 用重投影误差、最小三角化角与深度倍数剪枝。track 颜色在
`colour_triangulated_tracks`（`sfm/appearance.*`）中从原图采样，
供 splat 初始化与 MVS 稀疏点上色复用。

### 3.5 内参与相机模型

| `--camera-model` | 模型 | 备注 |
|---|---|---|
| `auto`（默认） | `automatic` | 仅前端请求；按场景选择具体模型 |
| `pinhole` | `pinhole` | fx,fy,cx,cy + k1,k2,p1,p2 |
| `fisheye` / `opencv_fisheye` | `opencv_fisheye` | k1..k4 存角向系数（写在 p1/p2 槽） |
| `equirectangular`（别名 `panorama`/`spherical`/`equirect`） | `equirectangular` | fx=W/2π、fy=H/π，主点居中，**无自由内参** |

- `--focal` 缺省或 0：初值取 `1.2 * max(W,H)`（鱼眼为 `0.5 * max(W,H)`），
  再由视图图共识（Fetzer 同相机焦距代价 + 稳健目标）与 BA 精化；
- `--trust-focal` 表示输入已标定，锁定内参不再优化；
- EXIF 焦距用于对齐初值（`load_camera_identity`、`calibrate_exif_view_graph_focals`）；
- 等距柱状相机的像素阈值会换算成等效角度（`k_equirect_threshold_scale`），
  保证“N 像素”在全景与针孔上有可比的角度含义。

---

## 4. Mapping 模式

入口 `sfm::reconstruct`（`src/sfm/reconstruct.cpp`），按 `--mode` 分派。

### 4.1 `global`（默认）

```text
rotation averaging（MST 初值 → LAD/ADMM L1 → 稳健 IRLS）
  -> 过滤不一致相对旋转后重算 pair 权重 -> build_tracks(3.0)
  -> global positioning（固定旋转，联合解相机中心/点/逐观测深度尺度）
  -> 位置离群修复 + 三角化 + filter_tracks
  -> BA 阶段 1：只解结构与平移（旋转/内参固定，12 次迭代）
  -> BA 阶段 2：放开旋转与焦距（含 aspect，25 次迭代，畸变仍固定）
  -> 完整转台/环绕序列的轨道正则化（严格检测器，非环形数据不触发）
  -> BA 阶段 3：再放开畸变（8 次迭代 + 尾部收益继续，最多 3 次续跑）
  -> 2 px 精细 filter_tracks
  -> 未注册视图用稳健 incremental resection 兜底
  -> 零基线位姿组强制一致并重新三角化
```

- rotation averaging：最大生成树初始化 + L1/ADMM + IRLS，默认权重
  `geman_mcclure`，可选 `half_norm`；可选拒绝平面主导对。
- global positioning：`GlobalPositioningConstraint` 默认 `only_points`；
  轨迹选择按视图覆盖自适应（`min_tracks_for_positioning=2500`、
  `tracks_per_registered_image=20`、上限 20,000）；相机中心可用 CUDA 稠密 Schur 热启动
  （`prefer_cuda`）；IRLS 迭代含 Huber + 权重收敛 + 约束隔离。
- 若定位后仍有视图无有效支撑，`repair_position_outliers` 会把它们重播种，
  并在末尾用“自由位姿 = 重播种视图 / 固定 = 其余视图”的锚定 BA 恢复。

### 4.2 `incremental`

`star_initialize`（openMVS StarInitializer 风格：参考视图 + 最多 36 个多视点
构星形初值）→ `register_images`（stage `sfm.resection`）：PnP 波次并行
（`max_pose_wave=8`）、局部 BA（`local_ba_every=10`，内参固定）、
周期性全量 BA（阶梯 `full_ba_every={25,50,100}`）、最终 BA
（`full_ba` 默认 40 次；尾部仍在改善时再追加一轮预算 60 次的 final polish）。
注册门禁包含内点网格覆盖、
旋转/平移一致性旁路、`min_inlier_ratio` 等。

当增量在大场景停滞时（缺失视图 ≥ 阈值且图像数 > 单簇上限），
`reconstruct` 会自动回滚到前端位姿并切换到**层次化 submap 救援**
（`incremental_hierarchical_rescue`，默认开启）。

### 4.3 `hierarchical`

`split_hierarchical_scene`（加权共视聚类，`max_views_per_cluster=2048`、
`min_views_per_cluster=10`、`min_common_tracks=25`）→ 每簇增量重建 →
`estimate_similarity_transform`（3 点 Sim(3) RANSAC + 全内点 Umeyama）→
`align_and_merge_hierarchical`（位姿/点变换 + 受保护 track 合并）→ 可选最终 BA。

三模式都以 `ReconstructionSummary` 收尾：注册数、landmark 数、观测数、
平均/RMS 重投影误差、以及 `analyze_alignment_observability` 给出的
可靠/不可靠视图计数。最终还会执行
`prune_unsupported_registrations`（默认 ≥30 个有效观测、≤2 px）剔除
“名义注册但无支撑”的相机。

---

## 5. Bundle Adjustment

`Photara::BA`（`photara/src/ba`）提供共享的 LM 求解器：

- 解析式针孔/畸变 Jacobian；Huber IRLS 鲁棒加权；
- Schur 消元 + 矩阵无关 Schur 乘法 + 块预条件 PCG；
- LM 接受/拒绝、阻尼调整、规范自由度固定（首相机/首点）；
- 内参按“内参组”独立优化：焦距、aspect、主点、畸变均为独立开关，
  带向初值的软先验与硬比例边界；
- `CudaOptimizer` 是把线性化、Schur 组装、块预条件、PCG、位姿/点/内参更新
  与代价评估全部驻留 GPU 的实现，并用 CUDA Graph 复用 PCG 的 kernel 序列。

`sfm::run_bundle_adjustment`（`sfm/bundle.hpp`）负责装配问题：

```text
BundleOptions.optimize_points / write_intrinsics
free_image_ids + fixed_image_ids     局部 BA：边界视图作为常量锚点
gate_intrinsics_by_observability     支撑不足（<3 视图或中位视差 <1°）的内参组冻结
prefer_cuda / cuda_min_observations  默认 CUDA（≥5 万观测），失败自动回退 CPU
```

`--ba-backend automatic|cpu|cuda` 只做进程级后端偏好的 A/B（`automatic` 为默认：
支持则 CUDA，静默回退 CPU）。带位姿锁的局部 BA 保持在 CPU，
以维持与全量求解一致的语义。

---

## 6. 输出与导出

| API / CLI | 产物 |
|---|---|
| `save_asfm` / `.asfm` | 原生场景（v3；只含相机、图像、位姿、关键点、track，不含描述子与 pair） |
| `project::Archive` / `.ascan` | 工程容器：settings + SfM，后续可写入 gaussians / mesh chunk |
| `export_openmvs_interface` / `.mvs`、`--export-mvs` | OpenMVS Interface：注册视图 + ≥2 观测的 landmark，含颜色 |
| `save_sparse_ply` | 稀疏 XYZRGB PLY |
| `save_colmap_text` / `--export-colmap-dir` | COLMAP 文本模型（cameras/images/points3D；全景写成 model 17），供 COLMAP 家族工具核对位姿 |
| `save_nerfstudio_transforms` | Nerfstudio / Blender `transforms.json` |

`.asfm` 版本兼容：`k_asfm_min_reader=1`，鱼眼从 v2、等距柱状从 v3 起可读。

---

## 7. 检查点、缓存与诊断

- `CheckpointStore`（`sfm/checkpoint.hpp`）支持 5 个阶段：
  `features` / `matches` / `geometry` / `tracks` / `reconstruction`。
  键由 FNV-1a 指纹（前端参数、构建标识、图像集指纹）决定；
  `--cache-dir` 打开缓存。命中 `tracks` 可直接跳过整条前端。
- 图像快照分两级校验：`identity`（路径/大小/mtime）与 `content`（SHA-256 重算）。
- `--working-sfm` 打开编辑器实时通道：`*.live`（当前特征/匹配/内点）与
  `*.preview.asfm`（每 ≥3 秒的注册快照）。
- `*_sfm_diagnostics.csv` 每图输出：内参、相机中心、四元数、观测数、
  重投影 mean/RMS/P95/max、相邻位姿步长（中心距离与旋转角）、
  `alignment_reliable`、活跃 pair 数、环支持/环不一致/平面退化的 pair 数、
  pair 权重和/最大值、加权平均射线角与相机模型。

---

## 8. 参数速查（默认值）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--mode` | `global` | `global` / `incremental` / `hierarchical` |
| `--window` | 3 | 顺序窗口（API 字段 `FrontEndOptions::neighbor_window`；取 0 表示穷举，CLI 要求 > 0） |
| `--neighbor-window` | 3 | API 中 0 表示穷举（CLI 要求 > 0） |
| `--max-features` | 27,000 | 3×3 网格筛选后的上限 |
| `--extractor` × `--matcher` | `siftgpu` × `gpu_mutual_ratio` |  |
| `--match-ratio` | 0.8 | ratio test |
| `--camera-model` | `auto` | 见 3.5 |
| `--focal` | 0 | 0 = 自动初值 |
| `--trust-focal` | false | 锁定内参 |
| `--ba-backend` | `automatic` | `cpu` / `cuda` 可强制 |
| `--cache-dir` | 空 | 打开前端/重建 checkpoint |
| `--export-colmap-dir` | 空 | COLMAP 文本导出 |

---

## 9. 测试

`photara/sfm.*` CTest 覆盖：`two_view`（E/F/H 与位姿分解）、`equirect`
（球面残差/Jacobian）、`mapping`（增量/全局 mapping 端到端）、`vocabulary`
（词袋检索）、`checkpoint`（阶段缓存与重建等价）、`submap_recovery`、
`hierarchical`（Sim(3) 合并）、`export_mvs`（MVSI 往返与朝向），
以及 `photara.project.io`（.ascan 容器）。

---

## 10. 已知限制

- 自动自标定在退化视角下可能给出“全部注册且低重投影误差”但轨迹错误的结果；
  真实数据必须先看 `*_sfm_diagnostics.csv` 与 `alignment_reliable`。
- `structural_pair_expansion` 仍是实验功能，默认关闭。
- 增量模式的层次化救援属于兜底路径，其分簇结果不保证与全局解等价。
- SfM 不产出稠密深度；前景 Mask 由用户提供或由 splat 侧主体约束产生。
