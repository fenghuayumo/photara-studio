# 3DGS CUDA 训练：显存与性能的其他优化机会（2026-09-16）

> 2026-09-17 校订：以下以当前工作区代码（含未提交修改）及已有日志为准，未新跑训练。
> 执行顺序调整为：**缓存预算收紧＋完整显存归因 → 64/128 桶快照 → SH 梯度融合**。
> 预算收紧必须同时验证墙钟吞吐，不能以关闭预取或增加加载等待换取表面显存下降。

上一份 `VKSPLAT_OPTIMIZATION_APPLICABILITY_20260916.md` 评估的是 VkSplat 那篇论文的条目。
这一份回答「除了论文，还能从哪挤显存和性能」，全部基于本仓库当前代码与已有实测日志。

数据基准（都来自仓库里已有的 stdout/ncu 记录，未新跑）：

| 场景 | 高斯 | 实例 | 分辨率 | 步时 | 阶段占比 |
|---|---:|---:|---|---:|---|
| `antman` 23000 步附近 | 648,142 | 2,554,150 | 1728×1120 | 7.1 ms | fwd 20.6% / bwd 51.4% / loss 6.2% / optim 11.8% / 数据 7.3% |
| 早期（大 splat） | 31,536 | 529,412 | 1728×1120 | 6.8 ms | bwd **69.2%** |
| 30k 跑 | 807,567 | 3,679,918 | 1728×1120 | 7.2 ms | — |

结论先说：训练工作集的主要部分是按高斯数增长的训练状态和按实例数增长的光栅缓冲；
但**设备总占用还包含图像缓存、池保留、分配取整与其它进程，不能用工作集公式解释全部显存**。
室内 iPhone 已有日志中，图像设备缓存本身约 2.95 GiB，应优先归因并收紧其预算。

## 0. 先把账量完整（已有部分打点，分配归因未完成）

`aetherscan/third_party/tinytensor/core/vram_profiler.{hpp,cpp}` 里已经有一套完整的显存
剖析器：scope 树、live/peak bytes、按 label 归类、GPU timer、top allocations，
`SizeBucketedPool` 还会上报 `cuda_pool_reserved / fragmentation /
untracked_used`。**但 `aetherscan/src` 里一次都没调用**（只有 tinytensor 内部引用），
当前 trainer 已有设备级 `cudaMemGetInfo` 阶段打点、增长后 trim 读数，以及参数/优化器字节统计；
但尚未接入完整的分配级 profiler，`splat_memory` 不包含完整训练工作集。

建议先做这一件事，否则下面所有收益都只能靠公式估算：

1. 训练开始 `VramProfiler::setEnabled(true)`，每步 `beginIteration/setIteration`，
   在 `trainer.cpp` 已有的 `cuda_profiler.mark()` 同级加 scope（`raster_forward` /
   `raster_backward` / `optimizer` / `densify`）。
2. 每次 refinement 前后记一次 delta（增长事件就是峰值出现的地方）。
3. 收尾打印 `Tensor::print_memory_pool_stats()`。

验收统计应区分参数/Adam、模型梯度、光栅 Gaussian/Grad/Instance/Pixel/Tile 缓冲、
loss/图像展开、多视图 context、图像常驻缓存与预取槽位；分别记录请求字节、实际分配字节、
live/peak、池内空闲缓存和 CUDA reserved。嵌套池统计不能重复相加，设备级 used 仅作为
辅助读数；无法归因的差额单列，不直接认定为碎片。阶段边界采样可能遗漏阶段内部峰值，
尤其要覆盖 refine 中旧、新张量同时存活的时刻。

## 1. 显存：按收益排序

### V1. SH 组占了训练状态的六成 —— 576 B/高斯

SH（degree 3）在参数、Adam 一阶矩、梯度里各占 192 B/高斯，1M 高斯时是 576 MB，
而除 SH 外的全部训练状态（参数 44 + 一阶 44 + 二阶 48 + 梯度 44）只有 ~180 B/高斯。

| 手段 | 收益（1M 高斯） | 风险 |
|---|---:|---|
| 去掉 SH 梯度缓冲（VkSplat §4.3，见另一份文档） | −192 MB | 中（要保留 culled 行的 Adam 语义） |
| SH 一阶矩用 bf16 | −96 MB | 中（需 30k 质量对照） |
| SH 二阶矩降为每高斯 1 个标量 | 已做（`make_reduced_second_adam_state`） | — |

### V2. 二阶矩全量化：means/scales/quats/opacity 各留一份 fp32 二阶矩

`make_adam_state`（`cuda_ops.cu`）给每组参数都开 first + second 全量；SH 已经走
Brush 式 reduced-second，其余四组是 44 B/高斯。把 scales/quats 也降为「每高斯一个标量
二阶矩」可以省 ~28 B/高斯（1M 时 −28 MB）。属于改数值、需要对照实验的一类，优先级低于 V1。

### V3. 致密化增长走 `Tensor::cat`：每次 refine 整块重建，旧块与新块同时存活

`densification_strategies.cpp:54-82`（`append_model` / `append_zero_adam`）每次增长都是
`model.sh = Tensor::cat({model.sh, added.sh}, 0)` 这种写法，即「分配新块 + 拷贝 + 释放旧块」。
一次增长要重建 6 个参数数组 + 6 组 Adam 矩（12 个数组，其中 `sh`、`sh.first` 各 192 MB@800K）。
`adc_igs` 的 refine 周期是 **每 100 步一次**（`densification.cpp:strategy_schedule`），
一次 30k 训练有上百次增长事件。

- 已落地：增长后调用 `tinytensor::Tensor::trim_memory_pool()`，回收空闲缓存；
  **它不能消除已经发生的增长瞬间峰值**，也不能把历次释放量相加作为稳态收益。
- 中成本（推荐）：参数与矩张量按 `densification_cap` 预留容量，只维护 live 前缀计数，
  增长时只写新增行。建议按需分档扩容，而非无条件预分配整个 cap，避免早期常驻反而增加。
  这有望减少瞬态峰值（具体大小待测）、每 100 步一次的整块拷贝
  （host 侧看是 memcpy 带宽，GPU 侧看是 kernel 空档），以及池里的尺寸碎片。

### V4. 每步的梯度缓冲与 memset

`rasterizer.cu:290-309` 每步 `zeros_like`：`means`(12) + `sh`(192) + `refine_weight`(4) +
`densify_weight`(8) + `log_scales`(12) + `quaternions`(16) + `opacity`(4) ≈ **256 B/高斯**；
`splat_drender/src/pipeline.cu:358` 每次 backward 还会 `cudaMemset` 整个 `GradState`（68 B/高斯）。
1M 高斯时是每步 ~320 MB 的纯 memset（≈0.15~0.2 ms，约 2%~3% 步时）＋等量的池抖动。

真正需要清零的只有「这一步没被任何 CTA 写到的行」（`gaussian_backward` 对 `radius<=0`
提前返回的那些）。仓库里已经有 `zero_adam_rows_kernel(indices, ...)` 这种按行清零的设施，
把 `visibility`/`radii` 压成索引表就能复用。

### V5. 光栅侧的几何条件分配（已落地，以下为改动前分析）

| 数组 | 字节/高斯 | 用途 | geometry 关闭时 |
|---|---:|---|---|
| `GaussianState.normal` | 12 | 表面法线 | 未使用，但仍分配、仍在 preprocess 里写（`render_forward.cu:94`） |
| `GaussianState.ray_plane` | 16 | 中值深度/几何 | 未使用，同上（`:93`） |
| `GradState.d_normal` | 12 | 法线反传 | 未使用，但仍分配并在 backward 前 memset |
| `GradState.d_ray_plane` | 16 | 中值深度反传 | 未使用，同上 |

合计 **56 B/高斯**（1M 时 56 MB + 每步 56 MB 的写/读）。`PixelState::bytes(pixels, buckets,
geometry)` 已经按 `need_depth` 条件分配 `snap_normal`，`GaussianState::bytes` /
`GradState::bytes` 没有这个参数（`buffers.h:49 / 94`）。默认配置（`depth_normal_loss=0`、
`multi_view_*` 关闭）正好白占。改动很小：给两个 `bytes/from_pool` 加 geometry 形参，
preprocess 里两处赋值加 `if (need_depth)`。

### V6. tile 尺寸是实例数的乘数

实测 648K 高斯 / 2.55M 实例 = **平均 3.9 个实例/高斯**（16×16 tile）。实例数直接决定
`snap_ct`（356 MB）、`InstanceState`、排序缓冲和反向的原子上限。

把 `cfg::kTileWidth/Height` 从 16 提到 32，理论上把「大 splat」的实例数压到 ~1/3
（小 splat 反而会多算像素：5×5 的 splat 在 16px tile 下占 256 px，在 32px tile 下占 1024 px）。
本仓库平均 footprint 约 1000 px，属于偏大的一侧，**值得实测**。改动面：
`config.h:kTileWidth/kTileHeight/kTileThreads`、`render_backward.cu` 里 `<< 8`
（=256 像素/bucket）的硬编码、blend 的 launch 形状（256 线程一 tile → 1024 像素/线程块）。

### V7. 多视图/几何路径的 context 池会成倍

`RasterContextImpl` 与 `DepthSampleContextImpl` 各自持有一整套 `GaussianState`（~103 B/高斯）
+ 实例/像素/点缓冲；`multi_view` 打开时每步至少多一个 sample-depth context。
建议让 sample-depth 复用主 render 的池（两者布局不同，需要抽公共 arena），
或者限制同时存活的 context 数。

### V8. 池的桶粒度（将来才会痛）

当前 `aetherscan/third_party/tinytensor/internal/size_bucketed_pool.hpp` 已细化：
16–256 MiB 按 16 MiB、256 MiB–1 GiB 按 64 MiB、1–8 GiB 按 256 MiB、
超过 8 GiB 按 1 GiB 取整。旧版“超过 256 MB 就按 256 MB”的判断已过时。
4M 高斯的 768 MB SH 数组位于 64 MiB 粒度区间；是否继续细化应由实际取整浪费统计决定。

## 2. 性能：按收益排序

### P1. 反向 51%：每个实例 13 次（几何时 20 次）标量原子

`render_backward.cu:524-545` 每个 lane（= 一个实例）在结束时提交：

```text
dL_dcolors ×3 + d_mean2d ×3 + d_conic ×4 + refine_weight ×1 + densify_weight ×1
+ densify_weight_den ×1            → 13 个 red.add.f32
[geometry] d_normal ×3 + d_ray_plane ×4    → 再 +7
```

按 2.55M 实例算，每步是 **3300 万条标量原子指令**（几何路径 5100 万）。
这些地址在 warp 内互不冲突（同一 bucket 内实例互不相同），所以改成向量化原子是纯赚：
把 `d_mean2d`/`d_colors`/`d_normal` 改成 padded `float4`、把
`refine_weight`/`densify_weight`/`densify_weight_den` 合成一个 `float4`，
每实例的原子指令就从 13 降到 4。仓库里 `bilateral_grid.cu` 已经在用
`red.global.add.v4.f32`（sm_90+）这套手法，属于既有先例。

### P2. 前向 20.6% + 反向 51% 里的桶快照带宽

每步约 356 MB 快照写（前向）+ 356 MB 读（反向），见另一份文档的 §4.2。
与 P1 是同一块瓶颈，二者可以一起做。

### P3. 优化器 11.8%：SH 是内存带宽主导

每步要过 `参数 192 + 梯度 192 + 一阶矩 192 + 二阶矩` 的 SH 数据 ≈ 576 MB/步（1M 高斯）。
可做：SH 的 float4 向量化（12 个 128-bit）、chain 折进 Adam、矩 bf16（带宽减半）。

### P4. 数据加载与缓存预算：必须按数据集和运行配置区分

实测日志：`device_budget_bytes=0 device_hits=0 uploaded_bytes=9289728000`（1200 步），
即每步重新上传同一批 7.74 MB 的 packed RGBA8。这 0.5 ms 完全没有必要：

- 整份数据集只有 **495 MB**（`dataset_packed_bytes=495452160`，64 视图），
  在 1M 高斯（~2 GB 训练状态）的显存预算里放得下；
- 另有 `--splat-prefetch-views` + pinned host 暂存的预取路径（30k 那次跑的是 `prefetch_views=4`）。

上面的 495 MB / 设备缓存关闭只对应早期 antman 实验，不能套用到室内 iPhone。
已有 `artifacts/vram_probe_20260916/after30k.log` 中，495,452,160 B 数据已经设备常驻。
室内 `iphone_final.log:915` 则记录：

- packed 数据集 7,254,835,200 B（约 6.76 GiB），设备常驻 3,165,388,800 B（约 2.95 GiB）；
- 30k 请求中历史缓存命中 3,368 次（约 11.2%），预取命中 25,951 次（约 86.5%）；
- 另有 4 个 pending 预取、39,321,600 B（37.5 MiB）。预取命中计数包含消费时等待，不能等同于零等待。

当前已有独立 nonblocking copy stream 和 pinned staging。真正需要继续处理的是：
历史 LRU 与上传预取共用设备预算，预算为 0 或 headroom 退让禁用缓存时也会停止设备预取；
上传任务使用 `std::async`，每次上传后同步共享 copy stream，消费时还可能等待 future。
这些机制需要分段计时，不能直接认定某一项就是加载瓶颈。

优先试验明确的 0.5/1 GiB 历史缓存预算，与当前约 3 GiB 配置比较；实现上应为小规模预取
保留独立预算和槽位，避免压缩 LRU 时连预取一起关闭。1280×1920 RGBA8 的 4–8 张槽位
原始数据约 38–75 MiB，不含展开后的训练输入、几何监督和分配取整。固定工作线程、复用
pinned staging、Host 就绪后立即排上传，以及每槽位 ready/消费完成事件是后续流水线方向。

**“室内加载约 40 ms/步”尚未与具体运行对齐，不能作为该数据集的统一基线。**
`iphone_final.log` 的 30k 训练阶段约 190 秒，末段 `step_ms=6.3087`，与稳态每步加载 40 ms
不相容。需要分别统计读取/解码、Host 预取等待、staging 拷贝、H2D、消费等待及整步 CPU
墙钟，报告冷启动和稳态的均值/P95。CUDA 阶段事件不能单独解释所有 Host 等待。

### P5. 小 kernel 数量与重复的一整趟

一个训练步在光栅侧有约 10 次 launch（`preprocess_gaussians` → `emit_depth_entries` →
`gather_touched` → `emit_instances` → 2× CUB scan → 2× radix sort → `extract_ranges` →
`bucket_offsets`），另有：

- `write_camera_constants`：为了 19 个 float 起了一个 kernel（`rasterizer.cu:52`），
  可以用 `cudaMemcpyToSymbolAsync` / 常量内存 / 直接塞进 preprocess 的参数。
- `activate_kernel`：一整趟 N 大小（32 B/高斯读 + 32 B/高斯写），只为算
  `exp(log_scale)`、四元数归一化和 `sigmoid(opacity)`；而 `preprocess_gaussians` 反正也要读
  这三组参数。两者合并可以省一整趟读写（1M 高斯 32 MB 读 + 32 MB 写/步），
  顺带省掉 `RasterContextImpl.activated` 的常驻（32 B/高斯）。

合并后每步的收益在 0.05~0.2 ms 量级，属于「顺手做掉」。

### P6. 损失 6.2%：SSIM 是像素级三趟

`fused_l1_ssim_forward/backward` + `ssim_cs_error_map` 在 1728×1120 上合计约 0.44 ms
（像素级、与高斯数无关）。可选：在 `valid`/掩码外提前退出（现在掩码关闭时整图都算）、
或用 1/2 分辨率的 SSIM（质量需验证）。收益上限 ~0.2 ms（3%）。

## 3. 结构性建议

1. **给「训练状态」写一份显存预算表**，作为 `--splat-densification-cap` 的推导依据。
   以下是原始布局的粗算，当前几何条件分配已减少 56 B/高斯；具体参数/矩和梯度
   还受配置影响，不能直接当作当前完整显存账：

   ```text
   per-Gaussian  ≈ 参数 236 + Adam 一阶 236 + Adam 二阶 ~48 + 梯度 ~256
                 + 光栅 scratch ~170（GaussianState 103 + GradState 68）
                 ≈ 950 B/高斯          → 1M 高斯 ≈ 0.95 GB
   per-Instance  ≈ 桶快照 128 B + 实例数组 ~36 B + 排序 scratch
   per-Pixel     ≈ 20 B（+ 损失/图像通道）
   ```

   这只能估算训练工作集；图像缓存、池保留、取整和外部占用需另外统计。
   SH 梯度融合可减少 192 B/高斯，bf16 一阶矩另有 96 B/高斯空间；V4 的减少清零
   不等于删除缓冲，V5 已落地，不能继续作为未来收益相加。

2. **把「增长」当成一等公民**：目前增长路径（cat + index_select + 全量重建）
   既产生峰值也产生拷贝；改成容量预留后，`refinement` 会变成「只写新增行」，
   顺带让 `--splat-refine-every` 可以开得更密（现在 adc_igs 每 100 步一次）。

3. **把 VramProfiler 接上并纳入回归**：把「1M 高斯的峰值显存」和「30k 质量指标」
   一样当成 CI 指标，才不会出现「优化 A 省了 200 MB、优化 B 又还回去」。

## 4. 建议的动手顺序

| 顺序 | 事项 | 预计显存 | 预计步时 | 工作量 |
|---|---|---:|---:|---|
| 1 | 缓存预算收紧＋完整显存归因，保留独立预取能力 | 室内缓存部分约有 2–2.5 GiB 调整空间 | 必须验证墙钟不退化 | 先完成基线和预算/预取解耦 |
| 2 | 快照桶支持 64/128，保留 32 基线 | 3M 实例约 −192/−288 MB，另计 tile 余量 | 待测寄存器压力与吞吐 | 前后向及布局联动 |
| 3 | SH backward 融 Adam，删除 SH 梯度 | degree 3：−192 MB@1M | 待测 | 保留不可见行矩衰减及正则语义 |
| 4 | 致密化按需容量复用，减少 cat 重建 | 增长峰值收益待测 | 减少复制，待测 | 避免整个 cap 提前常驻 |
| 5 | chain 融 Adam、activate 融 preprocess、RGBA8 直读 loss | 数十 MB 级，按实际存活缓冲统计 | 待测 | 分项验证 |
| 6 | 去全局快照的逆序回放反向 | 剩余快照的大部分 | 待测 | 高风险，需 parity 与长程质量对照 |
| 7 | bf16 SH 一阶矩、tile 32、其它向量化 | bf16 约 −96 MB@1M，其余不保证省显存 | 待测 | 数值或调度实验 |

以上为后续计划，不代表已经实现。geometry 条件分配和增长后 trim 已落地，不再列为待做。
缓存收益是逻辑常驻数据差额，不保证设备总占用等量下降。64/128 桶与去快照针对同一块内存，
收益不能重复相加；减少 memset、float4 原子及 SH 向量化主要改善速度，不会自动减少容量。

验收用同一数据集、分辨率、视图顺序/种子、几何开关和高斯规模，分别报告冷启动、稳态、
refine 峰值、完整训练墙钟与质量。先测第 1 项，再逐项叠加 64/128 桶和 SH 融合，保留
单项 A/B，避免把不同运行的缓存占用与加载耗时拼成结论。

## 5. 已落地（2026-09-16 晚）

几何通道条件分配和增长后 trim 已实现并通过当时的测试；部分显存打点也已接入
（用 `cudaMemGetInfo`，见下方「测量口径」）。这是 2026-09-16 的历史验收记录，
不表示上方 2026-09-17 新计划的缓存收紧、64/128 桶和 SH 融合已经实现。

### 5.1 改动清单

| 文件 | 改动 |
|---|---|
| `third_party/splat_drender/include/splat_drender/buffers.h` | `GaussianState::bytes/from_pool`、`GradState::bytes/from_pool` 增加 `geometry` 形参：为 false 时不再切分 `ray_plane`/`normal`（28 B/高斯）与 `d_ray_plane`/`d_normal`（28 B/高斯），字段保持 nullptr |
| `third_party/splat_drender/src/render_forward.cu` | `preprocess_gaussians` 变为 `template <bool GEOMETRY>`，false 时跳过两处 store；host 侧按 `RenderSettings::need_depth`（或测量用的强制开关）分派 |
| `third_party/splat_drender/src/render_backward.cu` | `gaussian_backward` 对 null 的 `d_ray_plane`/`d_normal` 传零向量——与「memset 成全零的 scratch」等价，`splat_backward` 的未用分支早退照旧 |
| `third_party/splat_drender/src/pipeline.cu` | `build_draw_lists`/`rebuild_views` 分别接收 pixel/gaussian 两个 geometry 开关；backward 里的 `GradState` memset 也用同一开关（少清 28 B/高斯） |
| `third_party/splat_drender/include/splat_drender/api.h` | 新增 `RenderSettings::force_geometry_workspace`（默认 false，仅用于 A/B 复现旧布局） |
| `aetherscan/src/splat/rasterizer.cu` | `RasterizeOptions.require_depth == false` 时不再分配 `median_depth`/`normal` 两张图（各 H×W、3HW）；backward 的 depth/normal 梯度校验也随之按 `require_depth` 判定；`settings_of` 读取 `AETHERSCAN_SPLAT_FORCE_GEOMETRY_WORKSPACE` |
| `aetherscan/src/splat/cuda_ops.{hpp,cu}` | `compute_training_loss` 新增 `need_geometry_gradients`（默认 true，测试调用不受影响）：为 false 时不分配 depth/normal 梯度张量，`loss_kernel` 也不再写这两个通道（原来每步固定 4 趟全图 store） |
| `aetherscan/src/splat/trainer.cpp` | ① 一个 `need_geometry_channels` 同时决定渲染通道与损失梯度张量；② CUDA profile 窗口内加 `splat_cuda_vram` 打点；③ 每次增长事件后 `trim_memory_pool()` 并打印释放量；④ 新增 `splat_memory` 打点（参数/优化器字节与 B/高斯） |
| `scripts/measure_splat_vram.ps1`、`scripts/compare_vram_ab.ps1` | 训练 A/B 脚本：外部 `nvidia-smi` 采样 + 两条日志的对齐比较 |

### 5.2 确定性收益（与实测无关，按布局精确计算）

- 每个像素渲染：`GaussianState` 28 B/高斯 + `GradState` 28 B/高斯，且 preprocess 少 28 B/高斯的
  store、backward 少 28 B/高斯的 memset。
- 每次渲染：少分配 median depth（H·W）与 normal（3HW）两张图，少两次 `zeros` memset。
- 每个训练步：少分配 depth（H·W）与 normal（3HW）两张梯度图，`loss_kernel` 少 4 趟全图 store。

在 1280×1920、1M 高斯的室内 iPhone 数据集上是 **56 MB（池）+ 约 31 MiB（渲染输出） +
31 MiB（损失梯度）≈ 118 MB**；1728×1120/800K 高斯时约 100 MB。

### 5.3 实测（`D:/BaiduNetdiskDownload/室内iphone`，30k 步，1M 高斯封顶）

| 指标 | 值 |
|---|---|
| 高斯数 / 实例 | 1,000,000 / 约 2.9–3.0 M |
| 训练时长 | 194.3 / 194.9 / 196.6 s（三次生产配置，含数据加载） |
| 末段 step_ms | 6.35 ms（30k 时） |
| 最终质量 | PSNR 21.13–21.20、SSIM 0.9455–0.9460（三次一致） |
| 设备显存占用（cudaMemGetInfo，含桌面等其它进程） | 约 8.8–9.1 GB |
| 训练状态（`splat_memory` 打点） | 568 B/高斯（参数 250 + 优化器 318），1M 高斯 ≈ 568 MB |
| 每次增长事件 reclaim（`splat_pool_trim`） | 32–128 MiB，30k 步共 131 次 |

### 5.4 测量口径与已知问题

- **设备级 `cudaMemGetInfo` 噪声大于本次改动量级**：同机其它进程（桌面/浏览器）会让
  device-wide used 漂移 ±200 MiB，跨run 对比 100 MB 级别的差异不可靠。窗口内的
  `vram_stage_delta_mib` 仍然可信（同一时刻的相对增量），本次改动没有产生任何 > 8 MiB
  的 per-stage 增量。
- **Windows/WDDM 下按进程统计不可用**：`nvidia-smi --query-compute-apps` 的
  `used_memory` 返回 `[N/A]`；`cudaMemPoolAttrUsedMemCurrent` 在该驱动上返回的数值
  不可信（1M 高斯时给出 30 GB > 24 GB 显存），因此没有采用。
  想要精确的进程级数字，正确做法是接上 tinytensor 里现成的 `VramProfiler`
  （`aetherscan/third_party/tinytensor/core/vram_profiler.*`，按 label 统计 live/peak）。
- **测量开关偶发失败**：`AETHERSCAN_SPLAT_FORCE_GEOMETRY_WORKSPACE=1`
  （强制旧布局、但不打开深度通道）这一组合在 1M 高斯数据集上 3 次里崩了 2 次，
  报 `splat_drender: read draw counts: an illegal memory access`，紧邻的告警是
  `Failed to destroy splat prefetch CUDA stream`。生产组合（`geometry == need_depth`）
  在同一数据集上 `compute-sanitizer --tool memcheck` 跑 300 步 0 error、30k 全跑通过，
  因此该崩溃只出现在测量用的组合里；怀疑与数据加载的预取流/设备缓存的时序有关
  （告警指向预取流），**建议单独排查**——它在生产配置里也可能偶发。

### 5.5 复现方法

```powershell
# 生产配置跑一遍并记录峰值（外部采样 + 进程内打点）
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/measure_splat_vram.ps1 `
  -Tag run -Output artifacts/vram_probe_20260916/run.ply `
  -Dataset 'D:/BaiduNetdiskDownload/室内iphone' -Iterations 30000 -LogInterval 3000

# 旧布局 A/B（同时需要两个开关，见 5.4 的注意事项）
  ... -ForceGeometryChannels 1

# 比较两条日志
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/compare_vram_ab.ps1 `
  -Before artifacts/vram_probe_20260916/before.log `
  -After  artifacts/vram_probe_20260916/after.log
```

新日志的样子：

```text
splat_memory iteration=600 gaussians=294241 parameter_mb=70 optimizer_mb=88 state_bytes_per_gaussian=568
splat_pool_trim iteration=600 gaussians=302432 released_mib=128
splat_cuda_vram iterations=29901-30000 sampled_iteration=29901 vram_start_mib=9097 vram_peak_mib=9097 \
  vram_peak_stage=iteration_start vram_step_end_delta_mib=0 vram_stage_delta_mib=-
```

## 6. 64/128 桶快照与 SH 梯度融合的实测（2026-09-17）

配置：`D:/BaiduNetdiskDownload/室内iphone`，adc_igs，12,000 步（1M 高斯 / 约 2.75M 实例，
1728×1920 训练视图按 1920 上限），`--splat-prefetch-views 0`（见 6.3），同 seed，
profile 窗口 11,901–12,000：

| 配置 | 总 ms | raster_bwd | optimizer | VRAM start/peak | 训练 s | PSNR |
|---|---:|---:|---:|---:|---:|---:|
| 32 桶 / 独立 SH Adam（基线） | 6.374 | 2.886 | 1.282 | 5097 / 5289 MiB | 109.7 | 19.17 |
| **64 桶** / 独立 | 7.630 | 3.936 | 1.315 | 4905 / 5065 | 125.1 | 19.04 |
| **128 桶** / 独立 | 9.220 | 5.736 | 1.298 | 4937 / 5065 | 139.8 | 19.50 |
| 32 桶 / **融合 SH Adam** | 6.413 | 3.912 | **0.255** | **4841** / 5097 | 109.0 | 19.15 |
| 128 桶 / **融合** | 9.378 | 6.905 | 0.264 | 4585 / 4713 | 140.2 | 18.99 |

### 6.1 SH 梯度融合：显存净赢，时间打平

> 2026-09-17 更新：融合版已改为 **warp 协作**实现（`update_sh_adam_rows`）：梯度行不再放
> 寄存器（原来的 `float3 local_sh[16]` = 48 个寄存器），而是放进 49 KB 的 shared tile
> （每高斯 12 个 float4），一次 `__syncthreads()` 之后由每个 warp 用 **lane 管列**更新自己
> 的 32 行 —— 与独立优化器 kernel 相同的连续访存模式，并复刻其两棵树规约，因此与未融合
> 管线逐元素一致（`test_fused_sh_adam` 全绿）。
>
> 但 `cuobjdump --dump-resource-usage` 显示 **所有** `gaussian_backward` 实例现在是
> `REG:255 STACK:352`（同一实例在 09-16 的构建里是 72 寄存器），即这个 kernel 无论是否
> 融合都已寄存器饱和并在溢出；255 寄存器意味着占用率掉到 1 block/SM。这才是"省掉
> optimizer 0.3 ms 却没换来净收益"的原因（backward 变慢 ~0.34 ms）。**先给这个 kernel 定
> 寄存器预算（`__launch_bounds__(256, N)` / `-maxrregcount`）再谈融合收益。**

#### 寄存器压力：多轮实测与最终设置（2026-09-17）

按 `__launch_bounds__(256, N)` 的 min-blocks 逐档实测（`ori_img`，adc_plus 6k 步，
约 232K 高斯 / 1.04M 实例，其余条件相同）：

| 轮次 | launch bounds | 融合实例 regs/stack | 总 ms | raster_bwd | optimizer |
|---|---|---:|---:|---:|---:|
| R0 | 无（255 regs / 352 B spill） | 255 / 352 | 2.770 | 1.712 | 0.056 |
| R1 | `(256, 4)` | 64 / 1760 | 2.745 | 1.668 | 0.050 |
| R3 | `(256, 3)` | 80 / 1568 | 2.683 | 1.604 | 0.049 |
| R2 | `(256, 2)` | 128 / 1088 | **2.633** | **1.582** | 0.052 |

结论：一味压低寄存器（R1）把 spill 从 352 B 推到 1760 B，收益反而更小；**128 寄存器 /
2 blocks/SM 是甜点**。同时发现寄存器上限会拖慢**未融合**变体（它的 bwd 1.371 → 1.590），
因此最终按变体分别设预算：

```cpp
__global__ void __launch_bounds__(cfg::kGaussianBlock, FusedSH ? 2 : 1)
gaussian_backward(...)
```

即融合实例 128 regs、未融合实例保持编译器自选（255 regs / 352 B）。同配置复测：

| | 总 ms | raster_bwd | optimizer | VRAM |
|---|---:|---:|---:|---:|
| 融合（默认） | **2.648** | 1.577 | 0.053 | 3649 MiB |
| 未融合 | 2.693 | 1.405 | 0.285 | 3665 MiB |
| （改前 R0 融合 / 未融合） | 2.770 / 2.700 | 1.712 / 1.371 | 0.056 / 0.351 | 3697 / 3729 |

**修复后融合同时赢时间与显存**：比未融合快 1.7%、显存少 16 MiB；比改前的融合快 4.4%。
1M 高斯规模（室内 738 视角，adc_igs 12k 步）同样验证：`cuda_timeline_avg_ms`
**8.023 → 6.872 ms（−14.3%）**、`raster_bwd` 4.417 → 3.716、训练时长 109.4 → **98.5 s**。
`aetherscan_splat_test` 与 `--memory-only` 全绿。

同一数据集（`ori_img`，adc_plus 6k 步，约 232K 高斯 / 1.04M 实例）配对实测：

| | 总 ms | raster_fwd | raster_bwd | optimizer | VRAM |
|---|---:|---:|---:|---:|---:|
| 融合关闭 | 2.700 | 0.694 | 1.371 | 0.351 | 3729 MiB |
| 融合开启（协作版，默认） | 2.770 | 0.718 | 1.712 | **0.056** | **3697 MiB** |

即：optimizer −0.30 ms、backward +0.34 ms → 净 +2.6%，显存 −32 MiB（≈192 B/高斯）。

- `optimizer` 1.282 → **0.255 ms**（−80%），但 `raster_backward` 2.886 → 3.912 ms（+1.03 ms）：
  Adam 挪进 per-Gaussian backward 后，每个线程处理自己那一行 48 个系数（跨 lane 的 strided
  访存），而独立 kernel 是 warp 内连续的向量化访问。
- 净效果：整步 +0.6%（打平），**显存 −256 MiB**（SH 梯度缓冲 192 B/G + 相关 scratch，
  与 1M 高斯 × 192 B = 192 MB 一致）。
- 结论：默认开启合理。想把它也变成时间收益，需要让 Adam 在 warp 内协作完成（共享内存分段
  暂存或按列并行），否则 `float3 local_sh[16]` 的 48 个寄存器本身就是瓶颈。

### 6.2 稀疏桶快照：当前实现是负收益

> 2026-09-17：**已按此结论回退**——`snapshot_bucket_size` / `snapshot_stride` /
> `PixelState::snapshot_count` / 前缀回放代码、`--splat-snapshot-bucket-size`、
> Python 绑定与 `TrainingOptions` 字段全部删除，`PixelState` 恢复「每 32 实例一个快照」
> 的原方案（`snap_ct + (bucket_idx << 8)`），测试里的 sparsity 对照也一并移除
> （只保留 fused SH Adam 的 parity 用例）。回退后 3000 步训练正常、配置行不再打印该字段。

- 快照显存 ∝ 1/stride，但 1M 高斯/2.75M 实例下快照总量只有约 381 MB，
  stride 128 也只省 ~190 MiB（还带 `+tiles` 的 slack 项），而 backward 需要把
  checkpoint 到 bucket 起点之间的实例（最多 (stride−1)×32 个）对全部 256 个像素回放一遍：
  `raster_backward` 2.886 → 3.936（64）→ 5.736（128）ms，整步 +20% / +45%。
- 结论：stride>1 只在「实例数/tile 数比很低」或显存硬性不足时才值得；默认 32 是对的。
  若要两者兼得，正确方向是把回放变成「每像素只重放它自己的贡献」而不是整 bucket 的
  前缀，或者干脆让快照只存 (T, color) 的 8 字节压缩形式（内存减半而无需回放）。

### 6.3 阻塞性缺陷：`--splat-prefetch-views > 0` 会崩

> 2026-09-17 补充：把 `--splat-prefetch-views` 设为 0 **并不能完全避免**——在 738 视角的
> 室内数据集上，prefetch=0 时也会偶发同样的 `illegal memory access`（3 次里 1 次），
> 而且告警仍是 `Failed to destroy splat prefetch CUDA stream`。把 device cache 预算从
> 512 MB 提到 3 GB 也照样崩，说明与预算大小无关，问题在数据加载的「设备缓存上传 /
> 淘汰」路径本身（`copy_device_entry_with_pinned_staging` + `copy_stream_` 的生命周期）。
> 76 视角的 `ori_img` 数据集上多次运行均正常，因此这是随上传频率放大的竞态；测量性能时
> 建议先在小数据集上做，或暂时回退这部分 loader 改动。

> 2026-09-17 处理：**数据加载器已按此结论精简**。删除了设备侧预取整条路径
> （`device_prefetches_`、`DevicePrefetchResult`、`consume_device_prefetch`、
> `device_prefetch_*` 统计）、`copy_stream_` 与 `CUDAStreamGuard` 用法、`PinnedStagingPool`
> 及 lease 机制；H2D 改为在训练线程上 `cudaMemcpy`（pinned staging 保留为单个复用缓冲）。
> `--splat-prefetch-views` 现在只表示**主机侧**解码/打包的后台并发数（默认 4，0 关闭），
> 加载器不再从任何后台线程发起 CUDA 调用。`CacheStats`、trainer 的统计日志行、测试断言与
> CLI 文案同步更新；`aetherscan_splat_test` 与 `--memory-only` 全绿。
>
> 原崩溃配置（室内 738 视角、`--splat-prefetch-views 4`）现已在 3000 步与 12000 步两次
> 运行中干净通过（此前必然在 ~1400 步崩）。
>
> 代价：设备缓存装不下数据集时，H2D 回到关键路径——该数据集上 `data_load_ms`
> 0.043 → **0.845 ms/步**（≈10%）；ori_img（76 视角、整库可驻留）无差别。顺带实测
> `--splat-device-cache-mb 3072` 反而更差（`data_load_ms=3.36 ms`，命中率 9%），
> 当前上传路径下**不要**靠调大设备缓存来补；要恢复重叠，正确做法是在训练线程上为
> 「下一个视图」发起 H2D 并用 event 在下次 `get()` 前等待（不引入后台 CUDA 线程）。

- 现象：`--splat-prefetch-views 4` + 当前工作树，在**第 500 步附近**（adc_igs 第一次
  refinement）稳定报 `splat_drender: read draw counts: an illegal memory access`，4/4 复现；
  `--splat-prefetch-views 0` 多次干净通过。
- 已排除：本轮加入的「增长后 `trim_memory_pool()`」（去掉后仍崩；该调用目前已从
  trainer 移除，等 prefetch 竞态修好后再决定是否以更安全的形式恢复）、新增的
  「消费后的 prefetch 进 LRU」（去掉后仍崩）。
- `compute-sanitizer --tool memcheck` 跑 600 步 0 error → 是竞态，不是静态越界。
- 同一改动还把这台数据集的 device cache 预算从 3.2 GB 压到 512 MB
  （`device_budget_bytes=536870912`），eviction 频率大幅上升，很可能放大了一个潜在的
  「device entry 生命周期 / copy-stream 顺序」问题。
- 建议：修好之前保持 `--splat-prefetch-views 0`；修复方向是让 prefetch 与 eviction 的
  分配/释放都在 `copy_stream_` 上排序（`CUDAStreamGuard(copy_stream_)` 包住 eviction 与
  clear 路径），并复查 `maybe_prefetch` 中被移除的 `make_room_for_device_bytes` 预算闸门。
