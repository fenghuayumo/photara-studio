# AetherScan 稠密重建与贴图架构

## 目标边界

在已完成的 SfM 之上，构建 RealityScan 类产品的后半段：稠密点云、网格、纹理
网格（albedo）。几何质量采用 **双路径**：

1. **Fast MVS 路径（默认可交付）**：快速稠密点云 + 初步 mesh；用户可选立刻做
   UV / Project Texture / Delight，效果满足即可结束。
2. **GGGS 精修路径（可选）**：以 MVS 稠密点云初始化 3DGS，用
   **Geometry-Grounded Gaussian Splatting (GGGS)** 优化几何并抽取最终 mesh，
   再做 UV / Project / Delight。

3DGS/GGGS **不是强制阶段**。产品应支持「只跑 MVS+贴图」与「MVS → GGGS → 贴图」
两种完整交付，由用户/配置显式选择。

参考：

- OpenMVS：`libs/MVS` 的 densify / mesh 流水线思想（不复制对象模型）；
- AIHoloImager：`TextureReconstruction`（Flatten → ShadowMap → Project → Resolve → Dilate）
  与图像域 Delighter；
- GGGS：锁定论文 *Geometry-Grounded Gaussian Splatting*（将 Gaussian 作为随机实体、
  直接渲染高质量深度并抽取形状）；
  若未来扩展，须另开插件接口，默认实现仍为 GGGS。

与 SfM 文档一致：紧凑索引与 SoA、公开 API 与执行布局分离；默认构建不依赖
OpenCV；图像 IO 继续使用 FreeImage。MVS 深度估计以 **CPU 吃满** 为第一版目标；
GGGS 优化以 **CUDA** 为主。

---

## 总览流水线

```text
SfM (已有) ──► RebuildScene
                 │
                 ▼
        ┌── Stage A: Fast MVS ──────────────────────────┐
        │  PatchMatch depth → Fuse dense cloud            │
        │  Coarse / MVS mesh（Delaunay+cut 或 Poisson）   │
        └──────────────────────┬─────────────────────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │ 用户选择几何终点            │
                 ▼                           ▼
        Path MVS-only                 Path GGGS (可选)
        (mesh = MVS mesh)             Stage C: 3DGS init
                 │                      + GGGS optimize
                 │                      + extract final mesh
                 │                           │
                 └─────────────┬─────────────┘
                               ▼
                 active_mesh = 选定几何
                               │
        ┌──────────────────────┴──────────────────────────┐
        │  Stage B/D 共用（均可选，由配置开关）              │
        │  Rasterize active_mesh → per-view masks           │
        │  [optional] Image Delight (mask 护边)             │
        │  [optional] UV + ProjectTextures + Dilate         │
        │  Export: cloud / mesh / textured albedo mesh      │
        └─────────────────────────────────────────────────┘
```

要点：

- **Mask / Texture / Delight 挂在「当前活跃 mesh」上**，不绑定「必须先 GGGS」。
- MVS-only 路径下，活跃 mesh 即 MVS mesh；开 GGGS 后，活跃 mesh 切换为
  GGGS 抽取结果（可保留 MVS mesh 作对比/fallback）。
- Delight 与 Texture 各自独立开关；可「只要 mesh」「只要带光贴图」「要 albedo」。

---

## 用户可见能力矩阵

| 开关 | 含义 | 依赖 |
|------|------|------|
| `enable_mvs` | 稠密深度 + 融合点云 + MVS mesh | SfM |
| `enable_texture` | UV 展开 + ProjectTextures + Dilate | 活跃 mesh |
| `enable_delight` | 图像域去光照，再投影得 albedo | `enable_texture`（建议）；可单独预计算 delighted 图 |
| `enable_gggs` | MVS 点云初始化 → GGGS → 最终 mesh | `enable_mvs`；纹理若开启则对 **GGGS mesh** 执行 |

典型组合：

| 场景 | 配置 |
|------|------|
| 只要稠密点云 | `mvs` |
| 快速看网格 | `mvs`（导出 mesh） |
| MVS 已够用，出贴图 | `mvs + texture` |
| MVS 出 albedo | `mvs + texture + delight` |
| 几何再精修后贴图 | `mvs + gggs + texture` |
| 全质量 | `mvs + gggs + texture + delight` |

CLI 示意（设计级，非最终参数名）：

```text
aetherscan reconstruct <images> --output out/
  --dense
  --mesh
  [--texture] [--delight]
  [--gggs]                  # 可选；开启后 texture 默认作用于 GGGS mesh
  [--texture-on mvs|gggs]   # 显式指定贴图几何源；默认随是否 --gggs
```

---

## 模块划分

```text
aetherscan/
  sfm/           # 已有
  mvs/           # Fast densify + MVS mesh
  mask/          # mesh 光栅化 → per-view mask
  gaussians/     # init from dense cloud + GGGS（可选库/目标）
  texture/       # delight, uv, flatten, project, resolve, dilate
  rebuild/       # 编排 RebuildScene、配置、checkpoint
```

CMake 建议：

- `aetherscan_mvs`：始终可构建（CPU densify/mesh）；
- `aetherscan_texture`：依赖 mvs 的 mesh/相机类型；Delight 可选 ONNX；
- `aetherscan_gggs`：`AETHERSCAN_ENABLE_GGGS`（CUDA），默认 OFF 或独立选项，
  不阻碍「仅 MVS+贴图」产品路径。

---

## 数据模型：`RebuildScene`

与 `sfm::Scene` 分离，由 `build_rebuild_scene(sfm::Scene)` 或 MVSI 导入填充。

```text
RebuildScene
  cameras[] / views[]          # 位姿、K、路径；建议支持 per-camera / undistort
  dense_cloud                  # Stage A
  mvs_mesh                     # Stage A；可带可选 UV/albedo（若在 MVS 路径贴过图）
  view_masks[]                 # 由「贴图/Delight 当时选用的 mesh」光栅化
  gaussians                    # 仅 enable_gggs
  gggs_mesh                    # 仅 enable_gggs
  active_mesh_id               # mvs | gggs
  albedo_atlas + uvs           # 仅 enable_texture
  delighted_images[]           # 可选缓存；enable_delight
```

Checkpoint 按阶段落盘（对齐 SfM `AETHCKPT` 思路）：

```text
dense-* / mesh-mvs-* / masks-* / gggs-* / mesh-gggs-* / texture-*
```

支持从任意阶段续跑，例如：已有 `mesh-mvs` 时只开 `--texture --delight`；
或已有 dense cloud 时只开 `--gggs`。

---

## Stage A — Fast MVS

### 目标

快速、高覆盖的稠密结构与 **可交付的初步 mesh**。CPU 混合视图/tile 并行 + NCC；
质量旋钮偏「快而全」，几何精修留给可选 GGGS。

### 步骤

1. **邻域选择**：共视稀疏点、基线角、尺度、重叠；
2. **PatchMatch 深度**：斜面 (depth+normal)、粗到细、几何一致性可选；
3. **融合**：多视图一致性 → `DenseCloud`（xyz、rgb、normal、view 列表）；
4. **MVS mesh**：Delaunay + 可见性图割（主推，对齐 OpenMVS 质量族）或深度图
   projective triangulation（更快预览）；随后统一 Clean。

### 输出

- 稠密点云 PLY；
- MVS mesh（OBJ/PLY）；
- `.dmap` 缓存（可选保留）。

此时若用户关闭 texture/gggs，流水线即可结束。

### 当前实现状态（2026-07-19）

PatchMatch 已实现以下 CPU 路径：

- 所有视图、所有 coarse-to-fine 层的灰度图和 mask 在进入 PatchMatch 时一次构建并缓存；
  reference/source 只持有只读引用，不再为每个参考视图重复缩放邻图；
- 深度传播采用 red/black 两阶段，阶段内以 `patchmatch_tile_rows` 行为一个 tile，避免
  相邻像素同时读写造成的数据竞争；随机种子由 view/level/iteration/tile 唯一确定；
- 默认同时调度 8 个参考视图，并在其 row tiles 之间分配总 CPU 预算。最后不足 8 个视图时，
  每个剩余视图自动获得更多线程，兼顾内存带宽吞吐和尾部利用率；
- 几何一致性每轮读取不可变的全局深度快照，因此多视图/tile 写入不会与邻图读取竞争。

全局表面重建已实现为可选 CGAL 后端：

1. 以融合点的中位 pixel footprint 建立尺度无关的体素采样，并用
   `mesh_max_points` 提供显式内存上限；
2. 构建 3D Delaunay，有限和无限 cell 都进入图；无限 cell 与相机所在 cell 连接 source；
3. 对 camera→sample 与 sample 后方的线段累计有向 facet visibility weight，质量项使用
   facet plane / circumsphere angle（与 OpenMVS 同类能量）；
4. 用无递归 FIFO push-relabel 求 s-t cut，避免百万 cell 图上的递归栈风险；
5. 提取 inside/outside 分界面，并拒绝相对中位 Delaunay 边过长的跨空洞三角形；
6. backend-independent Clean 删除非法、重复、退化和非流形面，统一连通分量朝向，删除小岛，
   封闭小边界环，可选 boundary-preserving smoothing，最后压缩顶点并重算法线。

构建时使用标准 `find_package(CGAL QUIET)`。有 CGAL 时，CLI 的 `--mesh-method auto` 在
`default/high` 选择全局 Delaunay，在 `preview` 选择 projective；无 CGAL 或图割未抽出有效面时
明确告警并回退 projective。可用 `--mesh-method projective|delaunay` 强制选择。

关键调优参数：

```text
--patchmatch-tile-rows 8
--patchmatch-concurrent-views 8
--mesh-method auto|projective|delaunay
--mesh-max-points 2000000       # 0 表示不设上限
```

---

## Stage B — Mask、Texture、Delight（共用后处理）

本阶段 **不假设** 一定经过 GGGS。输入是 `active_mesh`（MVS 或 GGGS）。

### B1. Mask（光栅化）

对每个注册视图，用 `active_mesh` Z-buffer 光栅化得到前景 mask（可羽化）。

用途：

- Delight 前景约束与 alpha 保留；
- ProjectTextures 置信度加权；
- GGGS 训练时的可选 mask 光度（若开启 GGGS）。

若用户只要点云/裸 mesh、不开 texture/delight/gggs，可跳过 mask。

### B2. Image Delight（可选）

**时机：图像域、投影之前**（对齐 AIHoloImager）。

- 默认实现：Intrinsic 分解（可移植 AIHoloImager 四段模型为 ONNX）；
- 回退：多视图统计 delight（无模型时）；
- `MergeMask`：delighted RGB + 原 alpha/mask。

关闭时 Project 使用原图 → 带光照贴图；开启 → 更接近 albedo。

### B3. UV + ProjectTextures（可选）

对齐 AIHoloImager `TextureReconstruction`：

1. **UvUnwrap**：Microsoft UVAtlas（P0 默认；chart 打包、stretch、gutter 更稳，
   与 Open3D/`pygsplat` 的 `compute_uvatlas` 同系）；xatlas 作无 UVAtlas 时的回退
   （跨平台/轻依赖）；
2. **Flatten**：UV 空间栅格化 → 世界坐标图 + 法向图；
3. **Per-view**：mesh 深度/ShadowMap → Project（遮挡 + `cos` 置信度）；
4. **融合**：P0 默认 top-1（max confidence）；high 档 top-K；
5. **Resolve + Dilate**：置信度阈值 + gutter 外扩（UVAtlas `gutter` 与 atlas 外扩配合）。

前置：mesh 须流形（或 unwrap 前自动 FixNonManifold）；失败则明确报错，不静默出坏 UV。
推荐参数起点（对齐 `pygsplat`）：`gutter=1`、`max_stretch=0.33`、按 atlas 分辨率
并行 partition。

### B4. 导出

- 无 texture：mesh ± 点云；
- 有 texture：OBJ+MTL / glTF + albedo；
- 有 delight：标注为 albedo；无 delight：标注为 projective/lit texture。

---

## Stage C — GGGS（可选几何精修）

**仅当 `enable_gggs=true`。**

### 算法锁定

后端固定为 **Geometry-Grounded Gaussian Splatting (GGGS)**：

- 将 Gaussian 原语按随机实体（stochastic solids）处理；
- 直接渲染高质量深度 / 几何场并抽取表面；
- 不与 SuGaR / 2DGS 等混为默认实现。

### 流程

1. **初始化**：MVS `DenseCloud` → Gaussian  
   （mean=xyz，尺度∝局部间距，短轴沿法向，颜色=点色；体素/曲率下采样控 N）；
2. **优化**：光度（建议 mask 内）+ GGGS 几何目标；可选与 MVS depth 一致性；
3. **抽 mesh**：GGGS 方法产出表面 → Clean / manifold / 简化 → `gggs_mesh`；
4. **切换**：`active_mesh_id = gggs`；若仍 `enable_texture`，对 **gggs_mesh**
   重跑 Stage B（mask 建议重算）。

### 失败与回退

- GGGS 失败或用户中止：保留 MVS mesh/点云；若已请求 texture，可回退到
  `texture-on=mvs` 并告警；
- 不允许在未产生合法 `gggs_mesh` 时静默宣称「GGGS 质量」。

### 资源

- CUDA 强依赖；无 GPU 构建时应编译期/运行期禁用 `enable_gggs`；
- 与 Fast MVS 的 CPU 路径解耦，避免拖慢默认「MVS+贴图」交付。

---

## 编排状态机

```text
run_rebuild(scene, cfg):
  require SfM registered

  if cfg.enable_mvs:
      dense + mvs_mesh
      active = mvs

  if cfg.enable_gggs:
      require dense_cloud
      init + optimize GGGS + extract gggs_mesh
      active = gggs

  need_mask = cfg.enable_texture or cfg.enable_delight or cfg.enable_gggs
  if need_mask and active mesh:
      rasterize masks from active mesh
      # 若先 GGGS 再 texture：mask 用 gggs_mesh
      # 若 MVS-only texture：mask 用 mvs_mesh

  if cfg.enable_delight:
      delight images (with masks)

  if cfg.enable_texture:
      uv + project + dilate on active mesh
      export textured mesh

  always export artifacts requested (cloud / meshes / masks / atlas)
```

顺序约束：

- `enable_gggs` ⇒ 需要先有 MVS dense（或从 checkpoint 加载）；
- `enable_delight` 强烈建议配合 mask；无 mesh 时不可 delight（或仅全图无 mask，不推荐）；
- `enable_texture` 需要活跃 mesh；与是否 GGGS 无关。

---

## 与现有 SfM / OpenMVS 的衔接

- 进程内：`sfm::Scene` → `RebuildScene`（保留 per-image 内参与畸变策略：
  深度/投影前 undistort 缓存为 P0 推荐）；
- 文件：继续支持 MVSI v7，便于与外部 OpenMVS densify 做 A/B；
- 已知导出缺口（单相机、无畸变、confidence=0）在自研路径中通过进程内手递规避；
  MVSI 导出可后续加固，但不阻塞本架构。

---

## 性能原则

1. **端到端墙钟时间** 为判据；分阶段记录 densify / mesh / gggs / delight / project；
2. Fast MVS：图像金字塔只构建一次；默认 8 个参考视图并行、每视图内部 row tile 并行；
   red/black phase 与几何快照保证并行确定性，并在最后一批动态重分配 CPU；
3. Texture：按 atlas 行块 / chart 并行；图像 LRU；
4. GGGS：GPU 时间单独计量；初始化与抽 mesh 后处理可 CPU；
5. 预设档：`preview` / `default` / `high`（分辨率、PM 迭代、是否 GGGS、
   atlas 尺寸、是否 Delight）。

---

## 实施顺序

1. **MVS P0（已完成）**：缓存金字塔、tile PatchMatch、depth + fuse、projective/global
   Delaunay mesh、Clean、PLY/OBJ；
2. **Mask + Texture P0**：MVS mesh 上 UVAtlas unwrap + Flatten/Project/Dilate；
3. **Delight P1**：ONNX Intrinsic + mask；开关接入；
4. **编排 P0**：配置矩阵、checkpoint、CLI（MVS-only 完整交付）；
5. **GGGS P1**：dense→init→optimize→extract；`--gggs` 后再跑 texture；
6. **产品化**：档位、回退、与 OpenMVS densify 质量对比报告。

验收优先级：先保证 **「MVS + 可选 texture/delight」** 闭环可交付，再接入 GGGS。

---

## 明确非目标（本阶段）

- 默认路径绑定 OpenMVS 式「源图矩形 atlas + LBP 视图选择」（可作实验模式，非默认）；
- 将 GGGS 并列作为官方几何后端；
- 无 CUDA 时强行启用 GGGS；
- 在粗 MVS mesh 未 Clean/流形时静默 UV（应失败或自动修复并打日志）。

---

## 小结

AetherScan 稠密段以 **Fast MVS 为可交付主干**（点云 + mesh，并可按需 texture /
delight）；**GGGS（Geometry-Grounded Gaussian Splatting）为可选几何精修**。
Mask / UV / Project / Delight 共用一套后处理，作用于用户选定的活跃 mesh。
这样既满足「MVS 已经够用就停」的产品需求，又保留「需要更高几何质量时再开 GGGS」
的升级路径。
