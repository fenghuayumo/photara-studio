# Photara 稠密重建与贴图架构

## 目标边界

在已完成的 SfM 之上，默认产品主链路直接使用相机位姿与稀疏点云初始化 ADCPlus，
由 Splat 优化多视图几何，再从 Gaussian 渲染的深度、法线和 alpha 进行 TSDF 网格重建：

```text
SfM → sparse points → ADCPlus / Splat → TSDF → Clean → Texture / Delight
```

默认路径明确跳过以下阶段：

- PatchMatch 稠密深度估计；
- MVS 稠密点云融合；
- Delaunay/Poisson MVS mesh；
- 从 MVS mesh 或 MVS depth 派生前景 Mask。

原因是 MVS mesh 会在辐条、细线、薄片、遮挡边界等结构上提前丢失几何，而后续 Mask 无法恢复
这些细节。Splat 应直接利用原图光度、轮廓与多视图几何监督完成致密化。

MVS 实现保留为可选诊断、算法对照和兼容导出后端，不是 Splat、TSDF 或贴图的前置依赖，
也不参与默认物体重建。

参考：

- Photara Splat：当前产品实现；其几何监督参考 *Geometry-Grounded Gaussian
  Splatting (GGGS)*，并融合 ADCPlus、GaussianWrapping normal field 与独立 mesh 后端；
- pygsplat/GS-2M：Gaussian 深度渲染与 TSDF 提取约定；
- Open3D：稀疏体素块 TSDF 的公开接口与数值回归参考；
- OpenMVS：仅作为可选 densify/mesh 对照，不复制对象模型；
- AIHoloImager：TextureReconstruction 与图像域 Delight。

与 SfM 文档一致：紧凑索引与 SoA、公开 API 与执行布局分离；默认构建不依赖 OpenCV；
图像 IO 使用 FreeImage。ADCPlus/Splat 以 CUDA 为主，TSDF/Clean 使用多核 CPU。

---

## 总览流水线

```text
Images
  → SfM
  → registered cameras + sparse landmarks
  → sparse-point filtering / subject bounds
  → ADCPlus warmup and adaptive densification
  → Splat photometric + multi-view geometry optimization
  → median depth / normal / alpha per registered view
  → sparse-block TSDF
  → Marching Cubes → topology audit → Clean
  → optional UV / Texture / Delight
```

物体模式和场景模式共用几何主链路：

- **物体模式**：限制主体空间范围；存在外部软 Mask 时直接使用。没有外部 Mask 时，由内部
  Gaussian 主体自举产生前景约束。该自动路径属于产品 P0，失败时必须提示用户确认主体范围，
  不得静默把全图当成物体。
- **场景模式**：不做前背景分离，以空间 bounds、相机可见性和 opacity 约束控制重建范围。

产品入口不暴露 `external|bootstrap|none` 之类的 Mask 来源枚举，只保留：

```text
--capture-mode object|scene
--masks <directory>          # 可选
```

内部规则固定为：

```text
object + masks   → 使用外部软 Mask
object - masks   → Gaussian 主体自举
scene            → 不使用前景 Mask
```

### 用户可见能力矩阵

| 能力 | 含义 | 依赖 |
|---|---|---|
| `capture_mode=object` | 单物体重建，启用主体 bounds 与前景约束 | SfM |
| `capture_mode=scene` | 场景重建，不做前背景分离 | SfM |
| `enable_splat` | 稀疏点初始化 ADCPlus/Splat | SfM；默认开启 |
| `enable_mesh` | Splat depth/normal/alpha → TSDF → Clean | Splat |
| `enable_texture` | UV 展开 + ProjectTextures + Dilate | 最终 mesh |
| `enable_delight` | 图像域去光照并生成 albedo | Texture，建议开启 |
CLI 示意：

```powershell
photara --images images --output object.ply --capture-mode object
photara --images images --output object.ply --capture-mode object --masks masks
photara --images images --output scene.ply --capture-mode scene
```

显式指定 `--capture-mode` 会作为产品级完整重建预设，自动开启 Splat 与 TSDF Mesh；
省略该参数仍保留低层 SfM-only 开发流程。不再需要也不再接受 ROI 参数。
`object` 模式从 SfM 稀疏点自动计算 `SubjectBounds`；`scene` 模式禁用该范围约束。

### 模块划分

```text
photara/
  sfm/           # 相机位姿、稀疏点、观测
  splat/         # sparse init、ADCPlus、Splat、Gaussian 渲染
  subject/       # SubjectBounds、主体 Gaussian 选择、软 Mask 自举
  tsdf/          # 独立 sparse-block TSDF、MC、查询/导出
  mesh/          # topology audit、Clean、简化
  texture/       # delight、UV、project、resolve、dilate
  rebuild/       # 编排、配置、checkpoint
  mvs/           # 可选诊断/兼容后端
```

建议将 TSDF 拆为独立 `Photara::TSDF` 目标，只依赖 Eigen/OpenMP；MVS、Splat 通过 adapter
提供深度帧，不让 TSDF 公开 API 依赖 `MvsScene`。

### 数据模型：`RebuildScene`

```text
RebuildScene
  cameras[] / views[]
  sparse_landmarks + observations
  subject_bounds
  input_masks[]
  bootstrap_masks[]
  gaussians
  rendered_depth/normal/alpha[]
  final_mesh
  albedo_atlas + uvs
  delighted_images[]
```

Checkpoint 按阶段落盘：

```text
sfm-* / subject-* / splat-warmup-* / masks-* / splat-* / tsdf-* / mesh-* / texture-*
```

支持从任意阶段续跑。MVS 产物使用独立的 `diagnostics-mvs-*` 命名，不进入默认依赖图。

---

## Stage A — 稀疏初始化与主体约束

### 输入门禁

Splat 只消费 SfM 注册相机、原图、稀疏点及其观测。进入训练前必须检查：

- 注册相机比例、重投影 RMS 和轨迹连续性；
- 稀疏点的有限值、观测数和视角分布；
- 删除低观测、超大重投影误差和孤立离群点；
- object 模式下估计主体 OBB，但 OBB 只表示 3D bounds，不直接投影成前景 Mask。

稀疏点通过位置、颜色和局部 KNN 尺度初始化 Gaussian。必须显式标记
`initial_points_dense=false`，保证 ADCPlus densification 真正启用；若策略与输入类型不兼容，
立即报错，禁止静默切换策略。

### 物体主体自举

object 模式且未提供外部 Mask 时，内部执行两阶段训练：

```text
sparse subject seeds
  → short ADCPlus warmup
  → 以稀疏主体点为锚选择 Gaussian 3D 连通分量
  → 投影 full-resolution alpha/depth
  → 多视图一致性与边缘保留
  → source-resolution soft masks
  → 正式 Splat
```

自举 Mask 必须保留软 alpha、细线和孔隙，不做会删除辐条的固定半径腐蚀。若主体选择置信度
不足，输出预览并要求用户提供外部 Mask 或重新拍摄；不回退到 MVS mesh/depth Mask。

### ADCPlus / Splat

1. 以稀疏 SfM 点初始化 SH、opacity、scale 和 rotation；
2. ADCPlus 动态 grow/split/clone/prune；
3. `filter_3D` 仅作为渲染时 Mip-Splatting floor，不烘焙进 canonical scale/opacity；
4. warmup 后加入 Splat depth-normal、多视图几何往返与 plane-warp NCC；
5. object 模式使用软 Mask 监督 RGB/alpha，scene 模式使用完整图像；
6. 导出 Gaussian PLY、训练诊断与逐视图 median depth/normal/alpha。

三轴比例约束、最大 Gaussian 数和 densification 周期必须由质量档控制。几何训练当前基线显式
使用 10:1 三轴比例上限；CLI、帮助文字与策略默认值必须保持一致。

---

## Stage B — TSDF 与 Mesh

TSDF 只融合 Splat 输出的 median depth、normal 和 alpha，不依赖 MVS 数据结构。默认参数与
pygsplat/GS-2M 对齐：

```text
max_depth = 2 * scene_extent
voxel     = max_depth / 2048
sdf_trunc = 4 * voxel
```

逐相机融合后使用标准 Marching Cubes，再依次执行：

```text
raw MC
  → topology audit
  → component filtering
  → conservative small-hole handling
  → orientation repair
  → optional smoothing / remesh / decimation
  → topology audit
```

细线结构是质量门禁：

- voxel size 应小于目标最细结构直径的约 1/2；
- 使用原分辨率 depth/alpha，避免低分辨率轮廓上采样；
- 支持 2–4 voxel truncation 档位；
- alpha/normal 阈值不得在轮廓和薄片处做硬删除；
- Clean 不得仅按全局尺度删除细长小分量；
- 对小于可恢复 voxel 尺度的结构，允许同时交付 Gaussian/Surfel，而不虚假承诺封闭 mesh。

`support closing` 只允许作为可关闭的保守修复，不用于填补上游深度缺失。输出必须检查
non-manifold edge/vertex、orientability、boundary components 和 watertight 状态。

---

## Stage C — Texture 与 Delight（可选）

Texture 和 Delight 只作用于 TSDF/Clean 后的最终 mesh：

1. 从最终 mesh 光栅化每视图可见性和投影置信度；
2. 可选图像域 Delight；
3. UVAtlas 展开；
4. Flatten → ShadowMap → Project → Resolve → Dilate；
5. 导出 PLY/OBJ/glTF 与 albedo。

前景约束复用 object 模式的外部或自举软 Mask。用于贴图的最终 mesh mask 只负责可见性，
不反馈到 Splat 训练，也不作为主体分割来源。

---

## 编排状态机

```text
run_rebuild(scene, cfg):
  require and validate SfM cameras + sparse landmarks
  bounds = estimate_subject_bounds(scene.sparse_landmarks)

  if cfg.capture_mode == object:
      if cfg.input_masks:
          masks = load_soft_masks()
      else:
          warmup = train_sparse_adcplus(scene, bounds, short_schedule)
          masks = bootstrap_subject_masks(warmup, sparse_landmark_seeds)
          require mask confidence gate
  else:
      masks = none

  gaussians = train_sparse_adcplus_splat(scene, bounds, masks)
  frames = render_median_depth_normal_alpha(gaussians)
  raw_mesh = tsdf.integrate_and_extract(frames)
  final_mesh = audit_clean_and_reaudit(raw_mesh)

  if cfg.enable_texture:
      texture(final_mesh, scene.images, masks)
  if cfg.enable_delight:
      delight_and_reproject(final_mesh, scene.images, masks)

  if cfg.enable_mvs_diagnostics:
      run_mvs_out_of_band(scene)   # 不向主链路提供输入
```

顺序约束：

- Splat 依赖 SfM，不依赖 dense cloud 或 MVS mesh；
- object 模式必须通过主体 bounds 与前景置信度门禁；
- scene 模式允许无 Mask；
- TSDF 只接受 Splat 渲染帧；
- Texture/Delight 只接受通过拓扑门禁的最终 mesh；
- MVS 诊断失败不得影响默认产品结果。

---

## 与现有 SfM / OpenMVS 的衔接

- 进程内默认路径：`sfm::Scene → RebuildScene → sparse ADCPlus/Splat`；
- SfM 相机、稀疏点颜色和 observation track 必须完整传给 Splat 初始化；
- MVSI/`.mvs` 可以继续作为相机与稀疏点交换容器，但读取该容器不代表执行 MVS；
- OpenMVS densify/mesh 仅用于离线 A/B、回归和兼容导出；
- 默认路径不得因为 `--splat` 隐式触发 PatchMatch/fusion，也不得因输入被误标为 dense 而
  静默关闭 ADCPlus。

---

## 性能原则

1. 以端到端墙钟时间和最终质量为判据，不以所有阶段 CPU 100% 为目标；
2. SfM 前端、track、BA 和输入预处理充分使用多核 CPU；
3. ADCPlus/Splat 是 GPU-bound；CPU 并行负责图像解码、Mask/SubjectBounds、预取、评估和异步导出；
4. TSDF block integration、Marching Cubes、Clean 和 topology audit 使用多核 CPU；
5. Splat 与 CPU 后处理之间通过 checkpoint/队列解耦，避免 CUDA 等待串行文件 IO；
6. 质量档 `preview/default/high` 控制训练步数、Gaussian 上限、TSDF voxel/truncation、
   texture 分辨率和 Delight，不再控制 PatchMatch。

### Vulkan 编辑器与训练预览

`photara_studio` 使用 GLFW + Dear ImGui + Vulkan。编辑器创建可导出的 Vulkan image 与
timeline semaphore，以可继承 Win32 HANDLE 启动独立 `photara` 重建进程。训练进程按 Vulkan
physical-device LUID 选择同一块 CUDA GPU，导入 image/semaphore，并把该步已有的 planar-float
raster color 直接写入 Vulkan image；不克隆 GaussianModel、不额外渲染、不经过 CPU/PNG：

```powershell
cmake -S . -B build -DPHOTARA_BUILD_STUDIO=ON
cmake --build build --config Release --target photara_studio --parallel
build\photara\Release\photara_studio.exe

# 同一预览接口也可脱离 GUI 使用
photara --images images --output object.ply --splat `
  --splat-strategy adc_plus --splat-preview-interval 50 `
  --splat-preview-dir live_previews
```

同步协议为单 image、双向 timeline：CUDA 等待 `2*(N-1)`、写入并 signal `2*N-1`；Vulkan
等待奇数值，将 shared image 在 GPU 内复制到常驻显示纹理，release queue-family ownership，
再 signal `2*N`。常驻纹理允许 UI 持续显示上一帧，CUDA 只会在下一预览点等待 Vulkan 完成一次
GPU copy。CLI 未收到 external handles 时仍保留 RGB8/PNG 回退，便于无互操作环境诊断。
窗口最小化时编辑器继续在无 present 的 command buffer 中消费/释放共享帧，避免 CUDA 在下一
预览点因 WSI 暂停而阻塞。

Windows/NVIDIA 回归使用两帧验证完整往返：iteration 50 signal timeline 1，Vulkan copy/release
signal 2，iteration 100 wait 2 后 signal 3。第二帧无死锁且训练完成；`step_ms` 不再包含
GPU→CPU readback 与 PNG 编码。

---

## 实施顺序

1. **直接稀疏入口 P0（已完成）**：`sfm::Scene` 直接构造 Splat 数据集，跳过 densify；
2. **策略语义 P0（已完成）**：明确 sparse 初始化状态，ADCPlus 保持动态致密化；
3. **物体主体 P0**：支持外部软 Mask，并实现 warmup → Gaussian 主体选择 → soft-mask 自举；
4. **TSDF P0**：拆出独立 `Photara::TSDF` API，增加 Open3D 同帧回归与拓扑门禁；
5. **细结构 P0**：增加细线质量档，联合控制 alpha、voxel、truncation 与 Clean；
6. **编排/checkpoint P1**：支持从 SfM、warmup、正式 Splat、TSDF 和 Texture 任意阶段续跑；
7. **Texture/Delight P1**：只消费最终 TSDF/Clean mesh；
8. **MVS diagnostics P2**：保留现有 CUDA PatchMatch、fusion 和 Delaunay 作为独立对照工具。

当前内部 SfM 分支已经直接执行
`SfM sparse → ADCPlus/Splat → TSDF`，日志以 `splat_input=sfm_sparse` 和
`patchmatch=false` 标识。旧自动 ROI、手动 ROI、depth-ROI Mask 入口已经删除。
无外部 Mask 的 Gaussian 主体自举仍未完成，不能标记为已交付。

### Texture / Delight 构建与用法

```powershell
cmake -S . -B build-cgal -DPHOTARA_ENABLE_TEXTURE=ON -DPHOTARA_ENABLE_ONNX=ON
cmake --build build-cgal --config Release --parallel

photara --images images --output object.ply --capture-mode object --texture
photara --images images --output object.ply --capture-mode object --texture --delight

# 只导出 aether_drender 投影初值，不做光度/接缝优化
photara --images images --output object.ply --texture --texture-optimize=false
```

- aether_drender 默认使用 git submodule `third_party/aether_drender`；
- UVAtlas 展开后先由 `aether_drender::TextureBaker` 做 Vulkan 可见性投影，再默认由
  `aether_drender::TextureRefiner` 做多视图光度 Adam 优化与 seam-only polish；
- `--texture-optimize-steps`、`--texture-optimize-batch-size` 和
  `--texture-seam-samples` 控制原生优化；整个优化阶段不经过 PyTorch；
- Delight 使用 C++ ONNX Runtime；模型放在 `PHOTARA_INTRINSIC_MODELS_DIR`；
- Intrinsic 权重为学术/非商用许可，产品发布前必须完成许可证审查；
- 未通过 mesh 拓扑门禁时不进入 UV/Texture。

---

## 明确非目标（本阶段）

- 把 MVS densify/mesh 重新放回默认主链路；
- 从 MVS mesh 或 MVS depth 生成默认主体 Mask；
- 向普通用户暴露 Mask 来源枚举；
- 用固定腐蚀/闭运算牺牲细线结构换取表面看似封闭；
- 无 CUDA 时强行启用 Splat；
- 在非流形 mesh 上静默执行 UV 和贴图。

---

## SubjectBounds 与前景约束

SubjectBounds 与 Mask 是两类不同约束：

- **SubjectBounds**：世界坐标中的保守 3D AABB，只限制 Splat 几何提取与 TSDF 空间范围；
- **Mask**：原图分辨率的 2D soft alpha，只约束 object 模式中的 RGB/alpha 学习；
- SubjectBounds 不删除 SfM 点或 Gaussian，也不投影成训练 Mask。

### SubjectBounds

当前实现直接对 SfM 稀疏点执行与 `pygsplat/gs2mesh.py` 同类的保守范围估计：

1. 只接收有限坐标且至少被两个视图观测的稀疏点；
2. 以 `max(0.1, 0.05 × 原始对角线)` 为半径，删除邻居数不足 10 的孤立离群点；
3. 对保留点计算轴对齐包围盒；
4. 每个半轴乘以 1.15 的保守 padding，并设置最小退化轴厚度；
5. 输出 `<output_stem>_subject_bounds.txt` 供复现。

该范围只负责限制后续分配和提取，不做语义分割，不裁剪稀疏点、MVS 点云或 Gaussian。
旧 dense-cloud 自动 ROI、手动 ROI 文件、投影 ROI Mask 及其 CLI/Python API 均已删除。
`scene` 模式将 `SubjectBounds` 置空，TSDF 按实际可见深度分配稀疏块。

### 前景约束

产品只暴露 object/scene 模式与可选 `--masks`：

```powershell
photara --images images --capture-mode object
photara --images images --capture-mode object --masks masks
photara --images images --capture-mode scene
```

- object + 外部 Mask：按原分辨率读取 soft alpha；
- object + 无外部 Mask：内部执行 Gaussian 主体自举；
- scene：不做前背景分离；
- 不再提供 `depth|mesh|none` 这类用户可见来源参数；
- MVS depth 和 MVS mesh 不作为默认或 fallback Mask 来源。

### 细线与孔隙保护

自行车辐条、电线、栏杆和薄片必须按软覆盖处理：

- Mask 生成和训练均使用原图分辨率；
- 不使用固定半径 erosion 删除细线；
- close/tiny-hole fill 只能处理有明确尺度上限的孤立噪点；
- 轮廓 feather 应小于等于约 1 px，并保留内部真实孔隙；
- Gaussian 主体选择使用 3D 连通性与多视图支持，不能只取每帧最大 2D 连通分量；
- 自举 Mask 的 alpha、边界稳定性和跨视图一致性必须输出诊断。

最终 mesh 光栅化 Mask 只用于 Texture/Delight 的可见性和护边，不反馈给 Splat，也不改变主体
分割结果。

---

## 验证基线与已确认结论

本节只记录可复现基线、已确认结论和质量门禁。更新时覆盖旧数据，不追加每日实验流水账。

### `ori_img` 已验证部分

测试集为 `D:\ScanVideo\ori_img\images` 的 76 张 `1000×1000` 图像。当前证据如下：

| 阶段 | 实测结果 | 结论 |
|---|---:|---|
| global SfM | 23.927 s，76/76 注册，125,818 稀疏点 | 通过 |
| SubjectBounds | 半径过滤保留 125,777/125,818 | 通过；41 个孤立点被过滤 |
| sparse ADCPlus smoke | 125,818 初始 Gaussian，`densification_enabled=1`，`patchmatch=false` | 通过 |
| 稀疏 ADCPlus | 10k 步，123,557 → 402,291 Gaussians | 通过 |
| ADCPlus 30k + MV tail interval=2 | 478.244 s，125,818 → 627,470 Gaussians，PSNR 21.0766 dB | 数值通过；**mesh 质量会回退**，只在允许质量折衷时使用 |
| masked Splat | masked PSNR 28.87 dB | 通过 |
| TSDF + Clean（30k 无 Mask） | 20.392 s，1,347,617 顶点 / 2,668,110 面 | 数值通过，拓扑仍需门禁 |
| ADCPlus 30k + **MV 每步（interval=1）** + 参考点查询精度 | 968.8 s，1,000,000 Gaussians（cap），PSNR **22.2237** / SSIM 0.9343；TSDF 深度一致性 0.994–0.999 | 通过；当前 `ori_img` 的几何重建参考 |
| 同上 TSDF + Clean | 46.5 s，**5,697,315 顶点 / 11,235,411 面**（marching cubes 2286 万面 → 最大连通域 1119 万面） | 通过；密度受 1M 高斯 cap 影响，非文档 627k 档 |

当前 smoke test 已证明内部 SfM 可以完全跳过 MVS，直接进入 ADCPlus。10k 质量基线中的
Mask 来自已删除的实验性 depth-ROI 路径，只保留其数值作为历史对照，不能作为当前产品流程
的能力证明。“无 MVS、无外部 Mask 的 object 自动主体自举”仍未完成。

无 Mask 的对照会学习墙面、桌布等背景；稀疏点 OBB 投影 Mask 覆盖率曾达到 99.99%，也不能
作为物体轮廓。迁移后的下一条正式回归必须满足：

1. 命令和日志中不出现 PatchMatch、fusion 或 dense-cloud 初始化；
2. ADCPlus 日志明确 `initial_points_dense=false`、`densification_enabled=true`；
3. 先用外部软 Mask 验证完全无 MVS 的端到端结果；
4. 再用同一相机、稀疏点和训练参数验证 Gaussian 主体自举；
5. 对细线测试集单独统计轮廓召回、深度覆盖和最终 mesh 连通性。

### ADCPlus 已确认约束

- 稀疏 SfM 点是默认初始化；dense PLY 仅为兼容输入；
- 策略不兼容必须报错，不得把 `adc_plus` 静默变成无 densification；
- `filter_3D` 是独立 Mip-Splatting floor，不烘焙进 canonical scale/opacity；
- 三轴比例默认值必须与 CLI/帮助一致，当前几何基线显式使用 10:1；
- 不把 prune 数单独视为孔洞根因，同时审计 opacity、scale、多视图深度一致性和边界环。

### TSDF 对齐结论与质量门禁

相同 76 帧 uint16 毫米深度、相机、voxel、truncation 和 stride 与 Open3D
`ScalableTSDFVolume` 对照：

| 实现 | 原始顶点 | 原始三角形 |
|---|---:|---:|
| Photara，support closing 关闭 | 1,123,104 | 2,113,376 |
| Open3D | 1,127,370 | 2,122,973 |
| 相对差异 | -0.38% | -0.45% |

该结果验证了相机约定、深度尺度、TSDF 截断、稀疏体素块寻址和 Marching Cubes 主流程。
但当前 Clean 后真实网格仍有 37 条真正非流形边、约 5.4 万条边界或异常边及约 90 个非流形
顶点，并且不可定向、非 watertight。关闭 support closing 后仍存在相同非流形边。

发布前必须增加：

1. raw MC 和每个后处理步骤的独立拓扑审计；
2. non-manifold、orientability、boundary component 自动门禁；
3. 修复后重新统一绕向；
4. Open3D 同帧数值回归；
5. 细线/薄片/负坐标/边界截断的合成测试。

门禁通过前，TSDF 输出适合预览和后续修复，不能承诺 CAD、打印或物理碰撞所需的封闭流形。

### 性能结论

ADCPlus/Splat 是 GPU-bound；不要为了 CPU 100% 与 CUDA 争用内存带宽。CPU 优化重点变为
SfM、图像/Mask 预处理、TSDF integration、Marching Cubes、Clean 和 topology audit。
TSDF integration 与 support closing 已使用 OpenMP；后续优先并行化 MC block 遍历，采用连续
block allocator、扁平哈希表和两阶段计数/写出。

Splat 子阶段性能分析使用可选 CUDA event profiler：

```bash
photara ... --splat \
  --splat-profile-cuda \
  --splat-profile-interval 100
```

profiler 默认关闭；启用后在同一训练 stream 上按窗口记录并输出
`raster_forward`、`training_loss`、`raster_backward`、`densification_stats`、
`optimizer`、`adc_noise`、`refinement` 和 `filter_3d` 的每步摊销毫秒与占比。
`multi_view` 除保留总耗时外，还细分为 `unproject`、`sample_forward`、
`loss`、`sample_backward` 和 `gradient_merge`，从而区分全图反投影、邻视角
深度采样、几何/NCC loss、采样反传与主 raster backward 后的梯度合并。同时记录
Gaussian 数、tile instance、几何监督步数和拓扑更新次数。
窗口上限为 1000 步，避免意外创建无界 CUDA event 池。首个窗口包含 CUDA kernel、内存池和
图像缓存预热，只用于识别冷启动；稳定性能应比较后续多个窗口。该 profiler 不统计训练后的
最终全视图评估、PLY 导出或 TSDF 阶段。

ADCPlus 默认保持每步执行多视图；tail 调度是显式 fast mode：

```bash
# 默认值 1；只有允许质量折衷时才显式设置 2
photara ... --splat-mv-tail-interval 2
```

第 15,001–30,000 步每两步执行一次昂贵的 `sample_depth + NCC + backward`，活跃步权重乘 2，
保持多视图目标的期望不变。`ori_img` 完整 30k A/B 中，稳定 CUDA 时间由 21.2029
降到 14.8752 ms/iter（-29.84%），Splat wall time 由 575.389 降到 478.244 s
（-16.88%），三视角 PSNR 由 20.9876 提高到 21.0766 dB。**但这不是质量中性的**：
`ori_img` 的 Clean 网格面数 -1.31% 落在 ADC 跨进程非确定性范围内、不能当作无回退，实际
重建质量评估显示 mesh 质量明显下降；`antman_nomask` 同一 OpenMVS 场景 A/B 中，interval=2 虽将训练由 442.813 降至
380.596 s（-14.05%），PSNR 却由 37.8694 降到 36.7273 dB，Clean 主体网格面数由
14,736,038 降到 10,612,585（-27.98%）。ADCPlus 在 15k 后仍持续 prune、replacement、
noise 和 refine，因此按 `grow_stop_iter` 立即降频并不安全。**默认值保持 interval=1；
只要 TSDF mesh 是交付物就必须保持 interval=1**，interval=2 只能用于预览/快速模式。
该优化只降低 GPU 几何监督频率，不以提高 CPU 占用率为目标。

保持 interval=1 的 plane-warp NCC 底层优化不会减少 multi-view 监督频率。当前实现复用邻
视图 2×2 角点、在 RGBA8 上传时按需生成灰度平面，并由 `32×8` CTA 在 shared memory
复用参考视图的整数/半像素 patch。`antman_nomask` 同一 COLMAP 输入的 5k Release A/B 中，
NCC loss kernel 从 2.1737 降到 2.0694 ms（-4.80%），稳定 CUDA/iter 从 11.0843
降到 10.7495 ms（-3.02%），训练从 30.4701 降到 29.5425 秒（-3.04%）；三视角 PSNR
为 30.5035 → 30.7663 dB。最终 30k + TSDF 门禁得到 39.2708 dB 和
9,982,336 顶点 / 19,776,598 面，未观察到因减少监督频率导致的质量回退，因为频率没有变化。
详细条件和浮点非确定性说明见 `SPLAT_CPP.md`。

---

## 当前实现状态与迁移缺口

已经具备：

- SfM 相机、COLMAP/OpenMVS Interface 稀疏点加载；
- 内部 `sfm::Scene` 直接构造 Splat dataset，日志明确 `patchmatch=false`；
- `capture_mode=object|scene` 与 SfM 稀疏点自动 `SubjectBounds`；
- sparse ADCPlus 初始化保持 densification 开启；
- Splat CUDA rasterizer forward/backward、SSIM、Adam；
- sparse `default/adc_plus/adc_igs` 动态 grow/split/clone/prune；
- Mip-Splatting 3D filter、Splat 多视图几何与 NCC；
- median depth/normal/alpha → TSDF → Clean；
- 外部 `--masks` 与透明/前景训练模式；
- Texture/Delight 骨架。

仍需完成：

- object 模式 Gaussian 主体自举与置信度门禁；
- 独立 `Photara::TSDF` 公共 API；
- raw MC/后处理分阶段拓扑门禁；
- direct SfM 分支的 Texture/checkpoint 编排；
- out-of-core view cache 和 ADC-IGS edge/error ownership。

实现、构建和许可证细节见 [SPLAT_CPP.md](SPLAT_CPP.md)。

---

## 可选 MVS 诊断后端

现有 CUDA/CPU PatchMatch、depth filter/fusion、CGAL Delaunay 和 MVS Clean 代码保留，但职责为：

- 与 OpenMVS 做质量和性能 A/B；
- 输出兼容 dense cloud/MVS mesh；
- 诊断 Splat 深度覆盖；
- 独立开发和回归测试。

CGAL Delaunay mesh 核心按 OpenMVS 的全局 visibility graph-cut 原则实现。进入四面体化前，
点云与相机统一变换到局部规范坐标，避免大世界坐标破坏浮点距离；graph cell 按空间插入顺序
编号以改善 ray walk/max-flow 局部性；surface uncertainty 默认按每个顶点的局部 Delaunay
边中位数自适应，并限制在全局尺度的 `[0.25, 4]` 倍范围。可用
`--mesh-adaptive-sigma=false` 做旧全局 sigma 对照。
graph-cut 提取后还会删除最长边超过 cut-facet 中位数 4 倍的无支撑 webbing；
`--mesh-max-edge-scale=0` 可仅用于诊断时关闭该门禁。

这些产物不得自动成为 Splat 初始化、主体 Mask 或 TSDF 输入。诊断后端应使用单独命令或配置，
避免普通产品路径误触发昂贵的 densify。

---

## 小结

Photara 默认几何路线确定为：

```text
SfM cameras + sparse points
  → sparse ADCPlus / Splat
  → median depth / normal / alpha
  → TSDF
  → topology audit + Clean
  → optional Texture / Delight
```

MVS 不再是默认阶段，也不负责生成前景 Mask。物体模式使用可选外部软 Mask 或内部 Gaussian
主体自举；场景模式不做前背景分离。当前稀疏 ADCPlus 和 TSDF 核心已经实测可行，下一阶段的
实现重点是直接 SfM 编排、无 MVS 主体自举、细结构保护和 TSDF 拓扑门禁。
# Calibrated MVS regression (2026-09-06)

When calibrated COLMAP cameras already exist, use the explicit dense path:

```powershell
build/photara/Release/photara.exe --images D:/ScanVideo/ori_img/images --splat-dataset D:/ScanVideo/ori_img --output artifacts/ori_img_calibrated_high/scene.ply --dense --mesh --dense-quality high --mesh-max-points 1000000 --texture --atlas-resolution 2048 --texture-optimize-steps 100
```

This preserves the imported poses, uses the MVS quality preset for image resolution,
and writes `scene_dense.ply`, `scene_mesh.ply`, and `scene_textured.obj/.mtl/_albedo.png`.
Texture projection and Adam/seam refinement run through `aether_drender`.
Explicit `--splat` still selects Gaussian training. An external dataset alone also
keeps the existing Gaussian default; `--dense` or `--mesh` selects calibrated MVS.

Dense PLY checkpoints now retain point weights, camera indices, and per-camera
weights. Graph-cut replay requires the same camera ordering and world coordinates;
ordinary third-party XYZ/RGB PLY files are not equivalent visibility checkpoints.

The earlier `ori_img_mvs_rewrite_20260905` results used internally estimated cameras
and failed visual inspection (fragmentation and connecting sheets). With supplied
COLMAP cameras, the 640-pixel fixed-camera regression kept 6,419,346 of 6,604,910
depth samples and fused 1,863,511 points. The weak-support variant produced one
retained component with 437,899 faces. This is evidence of a major camera-dependent
quality difference, **not** an isolated measurement of a meshing-algorithm gain or
a ground-truth accuracy result. Thin details and occluded regions still need visual
inspection; triangle count and optimizer loss alone are not acceptance criteria.

Full-resolution validation through the production CLI completed successfully:

- 76 calibrated views; 15,310,182 retained depth samples; 4,033,656 fused points.
- 487,928 mesh vertices and 970,906 faces; one retained component, zero
  nonmanifold edges, 5,038 boundary edges in 36 boundary components (not watertight).
- MVS took 145.1 seconds on the local RTX 5090 D v2 system.
- Native projection took 39.7 seconds; 100 Adam steps plus 30 seam-polish steps
  took 34.8 seconds including precomputation. Reported training loss changed from
  0.03338 to 0.02813 (different sampled batches, not a held-out image-quality metric).
- 2048² atlas, 212 charts. Original-camera previews are saved in
  `artifacts/ori_img_calibrated_high/mesh_cameras.png` and `texture_cameras.png`.
  Main structure and carved appearance are much improved; side-view holes, ragged
  thin edges and some visible texture seams remain. Internal SfM pose estimation
  is not fixed by this calibrated-input route.
- Release CLI and MVS tests build successfully. All 14 CTests pass; the MVS suite
  also checks outward convex-hull facet orientation and visibility PLY round trips.
