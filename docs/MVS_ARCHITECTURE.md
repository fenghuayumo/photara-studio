# Photara MVS 架构

本文描述 `Photara::MVS`（`photara/include/mvs`、`photara/src/mvs`）的当前实现：
PatchMatch 深度估计、深度融合、Delaunay / TSDF 网格后端与 Clean，
以及它在 CLI 中的定位。SfM 侧见 [SfM 架构](SFM_ARCHITECTURE.md)，
splat 侧的 TSDF/PAM 见 [Splat C++ 后端](SPLAT_CPP.md)。

---

## 1. 定位

MVS 在当前产品中**不是默认阶段**：

| 用途 | 触发方式 |
|---|---|
| 显式稠密重建（点云 / 网格 / 贴图前置） | `--dense`（可由 `--mesh`、`--texture` 隐含） |
| 外部数据集固定位姿稠密重建 | `--splat-dataset --dense` |
| OpenMVS 互操作与算法 A/B | `.mvs` 导出、benchmark 工具、诊断导出 |

它**不参与**默认 splat 链路：默认 `--splat` 只从 SfM 稀疏点初始化 Gaussian
（日志 `patchmatch=false`）。MVS 的深度与网格也不作为主体 Mask 的来源。

MVS 内部同样提供一个 **TSDF 后端**（`mvs.mesh.tsdf`），splat 侧的 median-depth
网格提取复用的就是它，因此两种来源共享同一套体素融合与 Marching Cubes 代码。

---

## 2. 阶段总览

```text
mvs::densify(scene, options)                       # src/mvs/densify.cpp
├── select_neighbors                                # stage: mvs.select_neighbors
├── estimate_depth_maps                             # stage: mvs.estimate_depth
│   ├── load_view_images / build_image_pyramids      #        mvs.load_images / mvs.build_pyramids
│   ├── PatchMatch（CUDA 或 CPU 回退）                #        每视图深度/法线/置信度
│   ├── geometric consistency（多轮）                 #        mvs.geometric_consistency
│   └── depth filter（一致视图数/冲突剔除）           #        mvs.filter_depth
├── fuse_depth_maps                                  # stage: mvs.fuse
└── reconstruct_mesh（可选）                          # stage: mvs.mesh
    ├── delaunay_cut → mvs.mesh_global_cgal（CGAL）
    ├── tsdf        → mvs.mesh.tsdf（+ mvs.mesh.tsdf.marching_cubes）
    └── clean_mesh   → mvs.mesh_clean / mvs.mesh_clean.tsdf
```

CGAL 未就绪时 `delaunay_cut` 会在 `densify` 入口直接报错
（`Global Delaunay meshing requires a CGAL-enabled build`），不会静默换后端。

---

## 3. 场景构建

`mvs::build_mvs_scene`（`src/mvs/scene_build.cpp`）把 `sfm::Scene` 转成 `MvsScene`：

1. 只保留已注册视图（少于 2 个直接报错）；
2. 按 `resolution_level`/`min_resolution`（默认 640）计算工作分辨率，
   同步缩放 fx/fy/cx/cy，同时保留 `src_*` 原始标定（供去畸变与重采样）；
3. 三角化 track → `SparsePoint`（含观测视图 id），并从原图采样稀疏点颜色；
4. 读取 `--masks` 目录（按文件名匹配）并二值化（≥128 为前景），
   再按 `mask_border_px` 腐蚀，抑制轮廓处的掠射深度片；
5. 工作分辨率图像按相机模型去畸变（鱼眼/畸变针孔走 `project_distorted`）。

`mvs::prepare_imported_scene` 用于导入数据集：只把已导入相机换算到工作分辨率，
不改位姿、不改源标定，并清空旧的深度/掩码/邻居。

`SubjectBounds`（`src/mvs/subject_bounds.cpp`，stage `sfm.subject_bounds`）从 SfM
稀疏点估计保守 3D 范围（半径过滤孤立点、半轴 ×1.15 padding），
`object` 模式下用于 TSDF 空间约束与 Clean 的裁剪边界识别，不删除稀疏点或 Gaussian。

---

## 4. 邻居选择与深度估计

`select_neighbors`：按共视点数量与基线夹角给每个参考视图排序并附加 source 列表
（`max_neighbors`、`min_shared_points`、`optim_angle_deg`）。

`estimate_depth_maps`：

- **后端**：构建时启用 CUDA 且探测到 GPU 时打印
  `mvs PatchMatch backend=cuda device=...`；否则回退 CPU
  （`mvs PatchMatch backend=cpu ...`）。CUDA 路径一次处理一个参考视图
  （单视图像素已足够占满 GPU，同时约束显存与上传压力）；CPU 路径按
  `patchmatch_concurrent_views`（默认 8）并发参考视图，视图内再按
  `patchmatch_tile_rows`（默认 8）行分块，红黑相位内并行传播。
- **多尺度**：`sub_resolution_levels + 1` 层金字塔，粗到细上采样传播。
- **光度项**：NCC 成本 + 随机 refine；`ncc_keep_threshold`（默认 0.45）保留像素。
- **几何一致性**：`geometric_iters` 轮（默认 2），每轮先快照全部邻视图深度图，
  保证读写无竞争且信息跨视图传播。
- **深度过滤**：`filter_depth_maps` + `min_views_filter`，用重投影邻居深度
  剔除自由空间冲突；`adjust_filtered_depth` 进一步用一致重投影的平均值替换。
  日志输出 `mvs depth filter: considered/kept/rejected_support/rejected_conflict`。

---

## 5. 深度融合

`mvs::fuse_depth_maps`（`src/mvs/fuse.cpp`，stage `mvs.fuse`）：

1. 每个 worker 用私有网格，再按最终 key 哈希分片归并（避免全局锁串行）；
2. 沿参考视线做**加权中位数共识**选层，再用紧内点平均抑制薄面厚度；
3. 一致性判据：`depth_diff_threshold`、`reprojection_error_px`、
   `normal_diff_threshold_deg`、`min_views_fuse`；
4. 入射角作为软权重（`grazing_weight_floor`），硬拒绝会丢失圆形轮廓；
5. 连通性 + 局部深度一致的斑块小于 `speckle_size` 时剔除。

输出 `DenseCloud` 每个点带位置、法线、权重、`views` 与 `view_weights`；
`save_dense_ply` 会把这些写进 PLY，使 graph-cut 回放可复现
（要求同一相机顺序与世界坐标，第三方 XYZ/RGB PLY 不等价）。

---

## 6. Mesh 后端

### 6.1 `delaunay_cut`（CGAL 全局可见性 graph-cut）

`src/mvs/mesh_cgal.cpp`，stage `mvs.mesh_global_cgal`：

1. 点云与相机统一变换到局部规范坐标（避免大世界坐标破坏浮点距离）；
2. 按投影间距过滤样本（`mesh_dist_insert_px`，0 = 全部插入），
   上限 `mesh_max_points`（CLI 默认 2,000,000，0 = 不限）；
3. Delaunay 四面体化，单元按空间插入顺序编号以改善 ray walk / max-flow 局部性；
4. 可见性权重（Jancosek–Pajdla 风格）：`mesh_k_sigma`、`mesh_k_qual`、
   `mesh_k_behind`；`mesh_adaptive_sigma` 按顶点局部 Delaunay 边中位数自适应
   不确定度并夹在全局尺度 `[0.25, 4]` 倍；
5. 弱表面增强：beta（前）/ gamma（后）自由空间支撑差乘以端点 sink 边，
   支撑按 `mesh_k_free_space_calibration_quantile`（默认 0.95）标定到
   `mesh_k_free_space_abs`；
6. 图割后删除最长边超过 cut facet 中位数 `mesh_max_edge_scale`（默认 4）倍的
   无支撑 webbing（0 关闭，仅用于诊断）。

### 6.2 `tsdf`（稀疏体素块 + Marching Cubes）

`src/mvs/mesh_tsdf.cpp`，stage `mvs.mesh.tsdf`（+ `mvs.mesh.tsdf.marching_cubes`）：

| 参数 | 默认 | 含义 |
|---|---|---|
| `mesh_tsdf_voxel_size` | 0（推断） | 世界单位体素；按 gs2mesh 约定取 `max_depth / 2048` |
| `mesh_tsdf_voxel_scale` | 1（CLI `-1` = 1x） | 只放大推断值，直接提取更粗但规则的面 |
| `mesh_tsdf_truncation_voxels` | 4 | 截断半宽（体素） |
| `mesh_tsdf_pixel_step` | 4 | 稀疏块分配步长（1 保留亚像素细线） |
| `mesh_tsdf_min_weight` | 0.25 | 低于该累积权重不参与提取 |
| `mesh_tsdf_support_closing_axes` | 2 | 双侧支撑轴数达到该值时填补零权重体素（0 关闭） |
| `mesh_tsdf_smooth_iters` / `lambda` / `mu` | 2 / 0.5 / -0.53 | 边界锁定的 Taubin 平滑 |
| `mesh_tsdf_bounds_padding` | 2 | 点云包围盒外扩倍数 |

### 6.3 Clean

`clean_mesh`（`src/mvs/mesh_clean.cpp`）分两条路径：

- **TSDF 路径**（`mvs.mesh_clean.tsdf`）：只做退化三角形/未引用顶点清理、
  小连通分量剔除（`mesh_tsdf_min_component_fraction`）与小洞闭合
  （`mesh_close_hole_edges`），因为 Marching Cubes 已给出一致的绕向与共享边顶点；
- **Delaunay 路径**（`mvs.mesh_clean`）：去重/退化面剔除、非流形边与蝶形点拆分、
  连通分量定向、微小岛剔除、尺度感知的 spurious 几何剔除
  （`mesh_spurious_factor`）、尖刺剔除（`mesh_remove_spikes`）、
  小边界环闭合，最后 compact + 重算法线，可选 `mesh_smooth_iters`。

`SubjectBounds` 有效时，被裁剪面围成的边界环会被识别为“有意的裁剪轮廓”，
不会被当作重建孔洞填补。

---

## 7. 质量预设

`mvs::apply_quality_preset`（`include/mvs/options.hpp`）会重置整条流水线参数，
不只是图像分辨率：

| 参数 | preview | default | high |
|---:|---|---|---|
| `resolution_level` | 2 | 1 | 0 |
| `sub_resolution_levels` | 1 | 1 | 1 |
| `estimation_iters` | 3 | 4 | 5 |
| `geometric_iters` | 1 | 2 | 3 |
| `random_iters` | 4 | 6 | 8 |
| `max_neighbors` | 8 | 12 | 16 |
| `min_patch_views` | 2 | 2 | 3 |
| `ncc_keep_threshold` | 0.50 | 0.45 | 0.40 |
| `min_views_fuse` | 2 | 3 | 3 |
| `min_views_filter` | 1 | 1 | 2 |
| `speckle_size` | 24 | 40 | 80 |
| `mesh_method` | delaunay_cut | delaunay_cut | delaunay_cut |

`--dense-resolution-level` 可在预设之后单独覆盖工作分辨率；
CLI 参数随后逐项覆盖预设值。

---

## 8. CLI 参数速查

| 参数 | 默认 | 说明 |
|---|---|---|
| `--dense` | off | 运行 MVS（`--mesh`/`--texture` 在无 `--splat` 时会隐含打开） |
| `--dense-quality` | `default` | `preview` / `default` / `high` |
| `--dense-resolution-level` | 跟随预设 | 0 = 原分辨率，1 ≈ 半分辨率 |
| `--mesh` | off | 生成网格 |
| `--mesh-method` | `auto` | MVS 路径中 `tsdf` 走体素融合，其余（含 `auto`）走 CGAL Delaunay |
| `--mesh-max-points` | 2,000,000 | 进入 Delaunay 的样本上限（0 = 不限） |
| `--mesh-dist-insert-px` | 预设（default/high 0.75） | 投影间距过滤 |
| `--mesh-free-space-support` / `--mesh-adaptive-sigma` | true | 弱表面增强 / 局部密度自适应 |
| `--mesh-max-edge-scale` | 4 | webbing 剔除（0 关闭） |
| `--mesh-free-space-quantile` | 0.95 | 支撑标定分位数 |
| `--patchmatch-tile-rows` | 8 | CPU PatchMatch 行分块 |
| `--patchmatch-concurrent-views` | 8 | CPU 并发参考视图 |
| `--masks` | `auto` | 前景掩码目录（`auto` = 同级 `masks/`，`-` = 禁用） |
| `--mesh-obj` | off | 额外输出 ASCII OBJ |

---

## 9. 导出与诊断

- `save_dense_ply` / `load_dense_ply`：稠密点云（含权重与逐相机权重）；
- `save_mesh_ply` / `load_mesh_ply` / `save_mesh_obj`：网格 IO（多边形按扇形三角化）；
- `encode_mesh` / `decode_mesh`：`.ascan` 的 mesh chunk（`k_mesh_chunk_version=1`）；
- `save_subject_bounds` / `load_subject_bounds`：`SubjectBounds` 文本；
- `save_depth_map` / `load_depth_map`：简单的 `.admap` 深度缓存；
- `--mesh-tsdf-frame-export-dir`：无损导出 TSDF 实际消费的 uint16 毫米深度帧、
  内参与 world-to-camera 位姿，用于后端 A/B；
- `mesh_tsdf_diagnostics_dir`：多视图深度一致性与 TSDF 观测权重热图。

---

## 10. 测试

`photara.mvs.pipeline`（`photara/tests/mvs_test.cpp`）覆盖：

- 场景构建（稀疏点颜色采样、鱼眼源模型保留、导入场景分辨率换算）；
- 融合（并行融合、掩码约束融合、稠密 PLY 往返）；
- TSDF（稀疏体素网格、支撑闭合修复单层间隙、孔洞边界三角化）；
- CGAL 路径（max-flow 切分、OpenMVS 能量约定、全局 Delaunay 网格、
  拓扑 Clean、蝶形孔拆分）；
- 质量预设、`SubjectBounds` 估计与 bounds 感知的 Clean、缺少 CGAL 时的早期报错。

`photara.sfm.export_mvs` 另外验证 MVSI 导出（含朝向与可见性 PLY 往返）。

---

## 11. 已知限制

- Delaunay 后端需要 CGAL；没有 CGAL 时只能使用 TSDF；
- TSDF 输出适合预览与后续修复，不承诺封闭流形或 watertight
  （薄结构、遮挡边界的孔洞属于模型层面而非参数问题）；
- MVS 质量强依赖相机位姿：自标定位姿错误会在稠密阶段放大为双层轮廓、
  连接片或缺失；`*_sfm_diagnostics.csv` 应先于 MVS 阈值调整被检查；
- 三角形数量与优化损失不是验收标准，细结构与遮挡区域必须目视检查。
