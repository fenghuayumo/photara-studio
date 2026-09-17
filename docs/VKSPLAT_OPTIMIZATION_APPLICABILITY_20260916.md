# VkSplat（arXiv:2605.00219）优化点在本项目的可用性评估（2026-09-16）

> 2026-09-17 校订：当前实施顺序为 **缓存预算收紧＋完整显存归因 → 64/128 桶快照 → SH 梯度融合**。
> 本次核对当前工作区（含未提交修改）与已有日志，未重新运行性能或质量实验。
> 本文历史收益表是布局/性能估算，不是新优化已实现后的实测结果。

回答一个问题：VkSplat 里那些 3DGS 训练优化，哪些能搬到 AetherScan 当前的 CUDA 训练里，
在**显存占用**和**性能**两方面分别值多少。

论文：<https://arxiv.org/html/2605.00219v1>，代码：<https://github.com/harry7557558/vksplat>
（下文引用的 `vk_*.slang` 即该仓库 `vksplat/slang/` 下的实现，已核对。）

## 结论（先看这个）

| 论文条目 | 本仓库现状 | 可移植性 | 显存收益（648K 高斯 / 2.55M 实例） | 性能收益（当前 7.1 ms/步） |
|---|---|---|---|---|
| §4.1 精确 tile 剔除（scan-line） | **已等价实现** | 无 | — | 已吃到 |
| §4.2 光栅反向自适应调度 | 桶固定 32（FasterGS 式） | 高（分档） | 快照 356 MB → 89 MB（b=128）；彻底去快照 −356/−856 MB | 8%～15%（反向占 51%） |
| §4.3 投影反向＋优化器融合 | 完全分离（chain→Adam 多次 launch） | 高 | −190 MB（SH 梯度）～ −220 MB | 5%～8%（优化器占 12%） |
| §4.4 32-bit tile-depth 单排序 | 双排序（>131072 实例走两趟 radix） | 高 | −40 MB | 1%～2% |
| §4.5 损失全融合 + uint8 参考图 | 损失已融合；参考图驻留 RGBA8 但仍展开成 float | 中 | −38 MB（瞬态） | <1% |
| §4.6 SH/scale 的 128-bit 布局 | float3 读写、scale 与 opacity 分数组 | 中 | 0（布局中性） | 需先测（指令数↓） |
| 跨厂商 Vulkan/Slang | 不适用（本项目走 CUDA） | — | — | — |
| MCMC densification | 已有 adc_igs/adc_plus/dense_adaptive | — | 高斯数控制这一杠杆已具备 | — |

**优先级**：先收紧图像缓存预算并建立完整显存归因，再做 §4.2 的 64/128 静态桶，
随后做 §4.3 的 SH backward 融 Adam。chain 融 Adam、动态桶调度与彻底去快照放到后续。
现有独立 copy stream / pinned staging 已具备；预算收紧应保留预取能力，并以完整墙钟
不退化为验收条件。详见 [显存与性能机会文档 P4 / §4](SPLAT_TRAINING_VRAM_PERF_OPPORTUNITIES_20260916.md)。

当前已经落地：精确 tile 剔除、按配置使用的 SH reduced-second Adam、关闭几何时省略相关
scratch/输出/梯度图、增长后 trim，以及设备级和参数/优化器显存打点。尚未落地：完整
分配归因、64/128 桶、SH 梯度融合、chain 融 Adam、避免致密化全量 cat。增长后 trim 回收
的是空闲缓存，不能消除增长期间旧、新数组同时存活的峰值。

## 测量基准

显存账用真实配置算（`artifacts/igs_quality_20260915/selected/repeat`，iteration 29000）：
807,567 高斯、3,679,918 实例、1728×1120（16×16 tile → 108×70 = 7560 tile）。布局公式逐条
对齐 `third_party/splat_drender/include/splat_drender/buffers.h`。

性能账用 `--splat-profile-cuda` 的实测阶段均值（648,142 高斯、2.55M 实例、1728×1120）：

| 阶段 | 平均 | 占比 |
|---|---:|---:|
| raster_forward | 1.46 ms | 20.6% |
| raster_backward | 3.65 ms | 51.4% |
| optimizer | 0.84 ms | 11.8% |
| loss | 0.44 ms | 6.2% |
| 其余（数据/去噪/致密化统计等） | ~0.7 ms | ~10% |

即：论文 §4.2／§4.3／§4.6 直接针对的 84% 的 GPU 时间；§4.5 针对剩下的 loss。

## 逐条评估

### §4.1 精确 tile 剔除 —— 已具备，无需移植

本仓库 `third_party/splat_drender/src/device/tiles.cuh:enumerate` 已是同一件事：以
alpha=1/255 等值线定义椭圆脚印，按较短轴逐行求另一轴上的**闭式相交区间**（`sqrtf(av*threshold
- det*d0*d0)`），既无假阳也无假阴；`conic_opacity.w >= 1/255` 的阈值判断与论文一致。
投影前向只统计 `tiles_touched`，由 `gather_touched` + `emit_instances` 单独一趟填充
——与论文 §4.1 的两段式结构相同。论文提到的收益（排序/光栅化少走假阳实例）本仓库已经在吃。

### §4.2 光栅反向的自适应调度 —— 本仓库最值得做的一块

现状：`render_backward.cu:blend_bucket_backward` 是 FasterGS 式「warp 管 32 个实例、每个
lane 走全部 256 个 tile 像素、梯度留在寄存器最后一次性提交」。与论文的差异有两点：

1. **桶大小固定 32**。论文的做法是每 tile 按 `b ≈ round_up(sqrt(P·G), subgroup)` 动态取桶
   （P=256 像素、G=该 tile 的实例数），并依此调整 launch 形状。先实现静态 64/128 两档，
   保留 32 作为基线；不是只把 `bucket_offsets_kernel` 的常数替换掉，还要同步修改
   bucket 上界、前向快照边界、反向实例映射/索引和 launch。每 lane 多实例状态可能增加
   寄存器压力、导致 spill，因此显存下降不等于速度必然提高。

   - 显存：快照是 `buckets × 256 × sizeof(float4)`（`buffers.h:PixelState`），
     `buckets ≈ 实例数/b`，所以**快照显存 ∝ 1/b**。2.55M 实例、b=32 → 356 MB；b=64 → 178 MB；
   b=128 → 89 MB（开深度/法线损失时 `snap_normal` 同额，等于翻倍）。
   - 性能：波前轮数从 `G/32` 降到 `G/b`，同时前向写快照、反向读快照的流量同步下降。
     以 2.55M 实例计，当前每步约有 356 MB 快照写（前向）+ 356 MB 读（反向），
     约合 0.3～0.5 ms，即**目前就有 5% 左右的纯带宽开销**。

2. **全局桶快照**本身是论文要消掉的东西。VkSplat 的两条反向路径都不写全局快照：
   - `alphablend_shader_bwd_per_splat.slang`：按高斯并行，不要快照；
   - `alphablend_shader_bwd_tensor.slang`（论文实现二）：一个 CTA 管一个 tile，只读
     **每像素终态**（`final_pixel_state` = color+T、`n_contributors`）与每像素上游梯度态
     （`v_current_pixel_state`），按实例批次**逆序**回放，per-batch 的 alpha/状态留在
     `groupshared`（`shared_alpha[SPLAT_BATCH_SIZE][256]`）里，批次之间靠 CTA 内的
     running transmittance 传递。

   搬到本仓库就是：把 `PixelState::snap_ct[buckets][256]`（501 MB@807K 配置）换成
   `pixels` 级的终态+T（16 B/px ≈ 31 MB）与一个每像素梯度态，反向改成「一 CTA 一 tile、
   逆序批处理」。**显存 −471 MB（开几何损失时 −942 MB）**，并把前向的快照写、反向的快照读
   整体去掉。代价是 `render_backward.cu` 的 blending 反向要重写，且必须有数值对照
   （仓库已有 `tests/reference_compare.cu`，容差模型见该目录 README 的 parity 约定）。

3. 论文的 Thompson sampling 调度器（在两种反向实现间自动选）建议**先不做**：本仓库两条路径
   代码差异大，维护成本高。可以先用静态档位（按 `G` 与像素数选 b）拿到大部分收益，
   等两条路径都稳定后再考虑把 `b` 作为 bandit 的动作。

### §4.3 投影反向 + 优化器融合 —— 可移植，收益集中在 SH

现状（`rasterizer.cu:272 Rasterizer::backward` + `trainer.cpp:1498` 附近）：

```
blend_bucket_backward → gaussian_backward → chain_gradient_kernel → adam_step_structure /
adam_step_active_prefix / adam_step_reduced_second / adam_step
|grad| means12+sh192+scale12+quat16+opa4+refine4+densify8 = 256 B/高斯，全部 zeros_like 后累加
```

VkSplat 的 `fused_projection_backward_optimizer.slang` 里没有任何梯度缓冲：投影反向的
偏导在寄存器里算出后直接进 `optimizer_update()`，`g_sh_coeffs_1/2`、`g_scales_opacs`（注意
`float4` 同时装 scale.xyz 与 opacity）本身就是 Adam 的一/二阶矩。对照到本仓库，可分三层：

1. **chain 折进 Adam（低成本）**：`chain_parameter_gradients`（`cuda_ops.cu:2519` 前）
   只做逐元素/4 向量 Jacobian（`exp`、sigmoid、四元数归一化）。把它折进
   `adam_structure_kernel` / `adam_kernel`，就能删掉 `grad_scales/grad_quaternions/
   grad_opacities`（32 B/高斯 = 26 MB）与一次全量读写。
2. **SH backward 融 Adam（高收益、中风险）**：SH 是最参数量的组（48/59），
   `gradients.sh = zeros_like(model.sh)` 每步要 memset 192 B/高斯（807K 时 155 MB），
   优化器再整读一遍、写回参数。把 `sh::backward` 与 SH 的 Adam 合成一个 kernel，
   可删掉整块 `grad_sh`：**−155 MB 常驻，每步省约 155 MB memset + 155 MB 读 ≈ 0.2～0.3 ms**
   （优化器 0.84 ms 中 SH 占多数）。注意语义：`gaussian_backward` 现在对 `radius<=0` 提前
   返回，而 Adam 要求这些行的梯度为 0 且矩仍衰减，融合后要在同一 kernel 内保留该行为
   （或按 `visibility` 走第二趟），`add_sh_regularization` 也要折进去。
3. **means 一起融合（不建议）**：`gradients.means` 还要喂 `densify_blend_world_gradient`、
   多视图采样梯度（`add_sample_depth_model_gradients`）与几何正则，属于「反向之后还有
   别的写者」，论文的一步融合在这里会被打断。只融合 SH/scale/quat/opacity 组更划算。

### §4.4 32-bit tile-depth 单排序 —— 可以替换现在的双排序分支

现状（`pipeline.cu:build_draw_lists` + `buffers.h:InstanceState`）：实例数 ≤ 131072 时用
64-bit `(tile<<32|depth)` 单趟；超过则走 FasterGS 双排序（先按 32-bit 深度排可见高斯，
再按 tile 稳定排实例），后者要 `depth_key[2] / depth_value[2] / compact_offset`
（5 × 4 B/visible = 20 B/可见高斯）和 8 B/实例的 ping-pong key。**本项目在 2.5M 实例量级
一直在走双排序。**

论文 §4.4 的做法：深度先映射到 `(1,2)` 区间（`d/(1+d)` 类映射，只动尾数位），
key = `tile_id`（高位，1728×1120/16px 只需 13 bit）+ 深度尾数（低位，剩 18～19 bit），
**单趟 32-bit radix sort**，论文报告 64-bit 排序与 32-bit 排序质量指标无差异。

移植收益：删除深度预排序这一趟（3.68M 元素）与 3 个 visible 级数组，key 流量减半
（8→4 B/实例）→ 约 −40 MB、每步 −0.1 ms 级。风险是深度量化到 18～19 bit 会改变
「深度几乎相同」时的混合次序，需要跑 30k 质量基线 + `tests/reference_compare.cu` 对照。

### §4.5 损失全融合与 uint8 参考图 —— 大部分已具备

已具备：`fused_l1_ssim_loss`（`fused_ssim.cu`，11×11 高斯窗前向/反向一体）+
`loss_kernel`（`cuda_ops.cu:860`，L1 关掉后由 SSIM 反向往 `grad_color` 写），
掩码/深度/法线的梯度也在同一趟里初始化；训练视图已经以 packed RGBA8 上传并驻留
（`training_data_loader.cpp:753 / 1141`）。

差距：每步仍把 RGBA8 `unpack_training_pixels_kernel` 展开成 float 的 `rgb/gray/mask`
（3+1+1 float/px，1728×1120 约 38 MB 瞬态 + 一趟全图）。
论文的做法是参考图**只留 uint8 RGBA**，在损失 kernel 里直接读。
可移植（把 unpack 折进 loss/SSIM），收益小但改动也小：−38 MB、<1% 时间。

### §4.6 SH / scale 的 128-bit 布局 —— 布局中性，省的是指令

VkSplat 里 `sh_coeffs` 是 `float4` 数组（48 floats = **12 个 128-bit 值**），
`scales_opacs` 是**一个 float4 同时装 scale.xyz 与 opacity**。本仓库现在是
`sh: [N,K,3]` 以 `float3` 为单位读写（`device/sh.cuh:99`），scale 与 opacity 是三个独立数组。

- 显存：两者都是 48 floats/高斯、16 B/高斯，**布局不变时不省显存**（注意别为了 float4
  对齐把 48 floats 补到 64，那会 +64 B/高斯）。
- 性能：preprocess / `gaussian_backward` / Adam 里每个高斯的 SH 读写从 float3 变成
  12 次 `LDG.128` / `STG.128`，指令数下降，L1 事务更整齐。属于「先测再改」的一类：
  用 ncu 看这三个 kernel 的 LDG/STG 指令数与 L1 throughput 决定是否值得。

## 历史显存账（807,567 高斯 / 3,679,918 实例 / 1728×1120，单视图、need_depth=off）

下表保留原评估基线，不是当前进程显存总账。当前关闭几何时 GaussianState/GradState 已各
减少 28 B/高斯，渲染和 loss 的几何图也已省略；参数/Adam 随 normal_features 和配置变化。
快照分配采用 bucket 上界，而非简单的 `ceil(I/b)`：当前 b=32 时为
`(instances + 31*tiles)/32 + 1`（整数除法）。64/128 收益须按对应上界和实际池分配重算。

| 项 | 公式 | 大小 |
|---|---|---:|
| 高斯参数（3+3+4+1+48 floats） | 236 B/高斯 | 191 MB |
| Adam 一阶矩 | 236 B/高斯 | 191 MB |
| Adam 二阶矩（scale/quat/opacity 全量 + SH 单标量） | 48 B/高斯 | 39 MB |
| 每步梯度缓冲 `ModelGradients`（含 refine/densify） | 256 B/高斯 | 207 MB |
| chain 前的激活空间梯度 | 32 B/高斯 | 26 MB |
| 激活参数（每个 live render context） | 32 B/高斯 | 26 MB |
| rasterizer `GaussianState` 暂存 | ~103 B/高斯 | 83 MB |
| `GradState`（反向临时） | 68 B/高斯 | 55 MB |
| `InstanceState`（双排序 + CUB scratch） | ~36 B/实例 + 20 B/可见高斯 | ~137 MB |
| `PixelState` 非快照部分（n_contrib/total_color/dL_dmt） | 20 B/像素 | 39 MB |
| **`snap_ct` 桶快照** | 256×16 B × ⌈实例/32⌉ | **501 MB** |
| `snap_normal`（开深度/法线/多视图损失时） | 同上 | **+501 MB** |

合计：单视图约 **1.5 GB**，开几何损失约 **2.0 GB**。对照论文 Table 2（MCMC、约 1M 高斯
时 0.93 GiB）——差距主要就在桶快照：**它是当前最大单块，占 1/3**，而 VkSplat 完全没有这一块。

## 落地顺序建议

1. **缓存预算收紧＋完整显存归因。** 室内 `iphone_final.log:915` 的图像设备常驻约
   2.95 GiB，先对比 0.5/1 GiB 历史缓存预算，保留独立预取槽位；约 2–2.5 GiB 是缓存
   逻辑字节调整空间，不是保证的设备总显存降幅。现有预算为 0 时会连设备预取一起关闭，
   需处理这一耦合。分配统计覆盖 live/peak、池空闲缓存/取整、光栅、梯度和图像缓存，
   并拆分 Host 加载、预取等待和 H2D。不得把不同运行中的耗时和占用拼成一组基线。
2. **64/128 桶快照。** 先保留 32/64/128 三档静态可选，验证前后向布局和通道级 parity；
   测寄存器/spill、峰值显存及墙钟。3M 实例忽略 tile 余量时，快照从 384 MB 降到
   192/96 MB，约省 192/288 MB；几何快照另计。暂不引入动态 sqrt 或 bandit 调度。
3. **SH 梯度融合。** degree 3 时删除 `grad_sh` 可少 192 MB/1M 高斯，807K 时约
   155 MB。保留不可见行零梯度下的 Adam 矩衰减、active SH 前缀、正则和更新顺序；
   使用 parity 与 30k 质量对照验收，不提前承诺步时收益。
4. **后续候选。** 按需增长容量复用、chain 融 Adam、RGBA8 直读 loss、32-bit 排序，
   再评估完全去快照。去快照需要反向重写、通道级 parity、depth-gradient 有限差分与
   长程质量对照。float4 化主要省指令，不自动省容量。

桶增大和彻底去快照针对同一块内存，不能相加两次收益。每一阶段固定数据集、分辨率、
视图顺序/种子和几何开关，对齐高斯/实例数，报告冷启动、稳态、refine 峰值及完整墙钟。

## 附带建议：先把显存测出来

当前 trainer 已有 `splat_cuda_vram`、`splat_pool_trim`、`splat_memory`，但前两者使用设备级
读数，最后一项只统计参数与优化器。下一步是接入 tinytensor 的分配级 profiler 和标签，
区分 live/peak、缓存与分配取整，并捕获 refine 内部瞬态峰值。设备读数含其它进程，
阶段边界采样也不等于完整峰值；无法归因的差额应单列，避免重复计算嵌套池的字节数。

加载计时同样需要对齐：`iphone_final.log` 末段墙钟 `step_ms=6.3087`，30k 训练阶段约
190 秒，因此“加载约 40 ms/步”尚不能作为该运行的稳态结论。预取 hit 包含消费等待，
应增加 ready hit / waited hit 和 Host 各阶段墙钟统计。

## 不建议照搬的部分

- **Vulkan/Slang 跨厂商**：论文的主要卖点之一是脱离 CUDA/PyTorch 生态，本项目的
  splat 后端是 CUDA 原生（还用了 `red.global.add.v4` 这类 sm_90+ 指令），
  跨厂商收益为零；值钱的是它里面的 kernel 级算法。
- **论文的 "unaccounted" 时间（341 s/200 s）**：那是 PyTorch 的调度与 SH 拼接开销，
  本仓库是编译期内核，没有对应项。
- **Thompson sampling 调度器**：见 §4.2 第 3 点，先用静态档位。
- **MCMC densification**：本仓库的 adc_igs/adc_plus/dense_adaptive 已经在做「控制高斯数」
  这件事，论文里 MCMC 的价值（3M→1M 高斯、显存 3.01→0.93 GiB）本质是这一杠杆，
  不需要再引入一套策略。
