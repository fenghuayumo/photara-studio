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
4. 按 OpenMVS `DELAUNAY_WEAKSURF` 计算 cell 的 incoming free-space support；沿表面前后
   `3σ/4σ` 区间得到 beta/gamma，并对通过相对差、绝对差和 outlier 检测的 endpoint cell
   强化 sink t-edge。融合权重先用确定性射线样本归一化到 OpenMVS 的绝对能量尺度；
5. 用无递归 FIFO push-relabel 求 s-t cut，避免百万 cell 图上的递归栈风险；
6. 提取 inside/outside 分界面，并拒绝相对中位 Delaunay 边过长的跨空洞三角形；
7. backend-independent Clean 删除非法、重复、退化和非流形面，统一连通分量朝向，删除小岛，
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
--mesh-free-space-support true  # OpenMVS weak-surface beta/gamma 强化
--mesh-free-space-quantile 0.95 # 融合权重到 OpenMVS 能量尺度的校准分位数
```

后续几何处理顺序固定为：`Delaunay cut → Clean/manifold → photometric mesh refinement
→ 可选交付级简化/重拓扑 → UV/贴图`。photometric refinement 前不默认简化，否则会先丢失
其需要优化的小尺度自由度。当前工程只链接 `asdiff::render`，并显式设置
`ASDIFF_BUILD_MESH_TOOLS=OFF`；因此 asdiff_render 中 `asdiff::mesh` 提供的
`remesh_field_aligned` 和 `repair_and_decimate` 尚未进入 AetherScan 调用链。未来接入时，
默认仅在 refinement 之后调用保边界的 `repair_and_decimate`；Instant Meshes 重拓扑作为
显式 retopo 模式，不作为高质量扫描默认步骤。

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

- 默认实现：Intrinsic 四段网络，**ONNX Runtime 进程内推理**（无 PyTorch 依赖）；
- 模型文件：`stage_0.onnx` … `stage_3.onnx`（由上游 `.pt` 离线导出一次即可）；
- 回退：多视图统计 delight（无模型时，尚未实现）；
- 前景约束：复用 `--masks`。

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
2. **Mask + Texture P0（已完成骨架）**：`aetherscan_texture` 通过外部
   `asdiff_render`（默认同级 `../asdiffrender`，`asdiff::render`）做 UVAtlas
   unwrap + Vulkan Flatten/ShadowMap/Project；CLI `--texture` /
   `--atlas-resolution`；导出 `*_textured.obj` + MTL + albedo PNG。前景 mask
   目录复用 `--masks`；
3. **Delight P1（已完成骨架）**：Intrinsic 四段网络以 **C++ ONNX Runtime**
   进程内推理（与 LightGlue 共用 ORT，无 Python/PyTorch）；CLI `--delight`。
   模型为 `stage_0..3.onnx`（见 `AETHERSCAN_INTRINSIC_MODELS_DIR`）。权重源自
   [compphoto/Intrinsic](https://github.com/compphoto/Intrinsic)
   （**学术/非商用许可**，`docs/LICENSE-Intrinsic.md`）；
4. **编排 P0**：配置矩阵、checkpoint、CLI（MVS-only 完整交付）——部分完成
   （CLI 开关已接入；checkpoint 续跑未做）；
5. **GGGS P1**：dense→init→optimize→extract；`--gggs` 后再跑 texture；
6. **产品化**：档位、回退、与 OpenMVS densify 质量对比报告。

验收优先级：先保证 **「MVS + 可选 texture/delight」** 闭环可交付，再接入 GGGS。

### Texture / Delight 构建与用法

```powershell
# 贴图：Vulkan SDK 1.2+（含 dxc）+ UVAtlas
# Delight：再开 ONNX（与 LightGlue 相同开关），并准备 stage_*.onnx
cmake -S . -B build-cgal `
  -DAETHERSCAN_ENABLE_TEXTURE=ON `
  -DAETHERSCAN_ENABLE_ONNX=ON `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
cmake --build build-cgal --config Release --parallel

aetherscan --images ... --output out/scene.mvs --dense --mesh --texture
aetherscan --images ... --output out/scene.mvs --dense --mesh --texture --delight
```

- asdiff_render：默认使用同级本地仓库 `../asdiffrender`（单一源码，不 vendoring）。
  可用 `-DAETHERSCAN_ASDIFF_RENDER_ROOT=` 覆盖；有远端后可改为
  `third_party/asdiffrender` submodule。
- Delight：C++ ONNX only；将 `stage_*.onnx` 放到
  `AETHERSCAN_INTRINSIC_MODELS_DIR`。
- 关闭贴图：`-DAETHERSCAN_ENABLE_TEXTURE=OFF`。

---

## 明确非目标（本阶段）

- 默认路径绑定 OpenMVS 式「源图矩形 atlas + LBP 视图选择」（可作实验模式，非默认）；
- 将 GGGS 并列作为官方几何后端；
- 无 CUDA 时强行启用 GGGS；
- 在粗 MVS mesh 未 Clean/流形时静默 UV（应失败或自动修复并打日志）。

---

## ROI / Mask 闭环（已实现，2026-07-19）

ROI 与 mask 是两层独立约束：ROI 是世界坐标中的 3D OBB，决定哪些几何允许进入
深度、融合和网格；mask 是每个工作分辨率视图的 2D 前景，决定哪些像素可以参与匹配。
两者同时存在时取交集，而不是互相替代。

自动模式采用两阶段 MVS：

```text
低迭代 PatchMatch → 粗融合点云
  → RANSAC 桌面/地面 → 删除平面及背面点
  → 相机视线交汇点引导的 26 邻域体素主体分量
  → PCA OBB（带 margin）
  → ROI-aware 粗 projective mesh + Clean
  → z-buffer 回投影、膨胀并与输入 mask 求交
  → 最终 PatchMatch → filter → fusion → Delaunay/projective → Clean
```

完整约束位置：

- PatchMatch：reference patch、source patch 和候选世界点都必须位于有效区域；
- depth filter：reference/source 像素先过 mask，重投影世界点再过 OBB；
- fusion：reference/source 样本、稳健融合后的最终点均检查 mask/OBB；
- global Delaunay：ROI 外点不插入，中心在 OBB 外的 cell 强制为 source/free-space；
- Clean：删除跨出 OBB 的三角形；只保护 ROI 裁剪产生的开边界不执行 hole cap，主体内部
  的小边界环仍会补洞，避免因启用 ROI 而保留大量内部孔洞。

CLI：

```powershell
# 自动桌面/地面、主体分量、OBB、粗网格 mask，再进行最终重建
aetherscan --images images --output scene.mvs --dense --mesh --roi auto

# 手动 OBB；文件为 15 个空白分隔浮点数
# center xyz，axes 的 3x3 row-major，half_extent xyz
aetherscan --images images --output scene.mvs --dense --mesh --roi roi.txt

# 自动 OBB 每个半轴增加 10%，回投影轮廓在工作图上膨胀 7 px
aetherscan ... --roi auto --roi-margin 0.10 --roi-mask-dilate 7
```

手动 OBB 的轴矩阵读入后会以 SVD 投影到最近的正交旋转矩阵；半轴必须全部大于零。
只在地面检测失败时会退化为「视线目标 + 主体分量」OBB；若连可靠主体 OBB 也无法得到，
才保留无 ROI 的粗重建并给出 warning，不会输出一个错误裁剪的空模型。
有效 ROI 会同时写到 `<output_stem>_roi.txt`，可直接作为下一次 `--roi` 的输入，便于
自动检测后人工微调并复现最终重建。

---

## 小结

AetherScan 稠密段以 **Fast MVS 为可交付主干**（点云 + mesh，并可按需 texture /
delight）；**GGGS（Geometry-Grounded Gaussian Splatting）为可选几何精修**。
Mask / UV / Project / Delight 共用一套后处理，作用于用户选定的活跃 mesh。
这样既满足「MVS 已经够用就停」的产品需求，又保留「需要更高几何质量时再开 GGGS」
的升级路径。

### GGGS C++ 后端落地状态（2026-07-21）

已新增 `splat` 模块并完成 GGGS 原生 CUDA rasterizer 的 forward/backward、TinyTensor
参数激活与显式梯度、L1+SSIM/depth/normal loss、融合 Adam、MVS dense-cloud 初始化和
Gaussian PLY 导出。CLI 使用 `--gggs --gggs-iterations N`，会在 dense fusion 后直接训练。
`--gggs-use-mask` 已支持与 pygsplat 一致的 `transparent`（前景 RGB + alpha BCE）和
`masked`（前景 RGB + 背景 alpha 泄漏惩罚）模式，复用 `--masks` 或源图 alpha channel。

长训练收敛修复已加入 scene-scaled mean LR、`1e-15` Adam epsilon、绝对 scale 边界和默认
10:1 三轴比例约束。`D:\ScanVideo\ori_img` 上相同的 500k Gaussian / 4000 步 mask A/B 中，
三个诊断视角相对旧实现提升 `1.45 / 2.39 / 2.67 dB`，极端轴比例 p99.9 从约 170k 降到 10。

当前交付边界是“固定数量 Gaussian 的可训练后端”；动态 densify/prune、out-of-core view
cache、GGGS mesh extraction 与 `active_mesh` 切换尚未完成。因此目前 `--gggs` 输出
`*_gggs.ply`，纹理阶段仍使用 MVS mesh。实现、构建方法、性能边界和许可证风险见
[GGGS_CPP.md](GGGS_CPP.md)。

---

## 可扩展全局表面重建与 Clean（2026-07-20）

默认/高质量档的最终 mesh 现在必须使用 CGAL 全局 Delaunay visibility cut；CGAL
不可用或全局切割失败时会明确报错，不再静默退回容易产生局部碎片的 projective
mesh。projective 仅保留为 preview 的显式快速路径。

全局后端的关键实现如下：

- 先按「所有观测视图中的投影距离 + 相对深度差」过滤 Delaunay 插入点，并合并观测；
- 相机 cell 只定位一次，visibility ray 按 Delaunay segment traversal 计算；每个工作线程
  使用稀疏局部累加器，达到阈值后批量合并，避免逐 ray 原子写热点；
- s-t cut 使用 Boost Boykov-Kolmogorov。没有采用 OpenMVS 默认的 IBFS 源码，因为其
  上游许可证限定研究用途，不适合产品分发；
- 非 CGAL 构建在 densify 开始前即拒绝 default/high 全局 meshing，避免完成昂贵深度估计
  后才发现后端不可用。

Clean 与 OpenMVS 的处理尺度对齐：使用 P95 边长识别异常长三角形、使用 P55 边长与
分量 AABB 对角线剔除尺度异常小的碎片、迭代删除 spike、拆分 bow-tie 顶点、统一绕向并
补小孔。ROI 不再禁用全部补洞：只有由 OBB 裁切面产生的边界顶点被保护，主体内部的小孔
仍会关闭。`mesh_spurious_factor=0` 和 `mesh_remove_spikes=false` 可用于关闭相应 API 级步骤。
