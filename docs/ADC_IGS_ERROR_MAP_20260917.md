# ADC-IGS 误差图有界重排（2026-09-17）

## 背景

`25f6522` 把 ADC-IGS 致密化回退到旧的梯度门限 + opacity/edge 采样 + gumbel
选择路径，代价是 `densify_use_error_map` 对 ADC-IGS 完全失效。本轮把 SSIM
误差图重新接入，但只作为**有界重排**，不改动任何决策结构，并补上多 seed
配对评估。验收硬指标：ori 76 视角全训练协议 PSNR ≥ 25.30。

## 实现

- 训练中 `compute_ssim_cs_error_map`（power 1）→ 渲染器按贡献加权回传到高斯
  → 每个 refine 窗口按视角累加，`refine_gaussians` 里除以观测数得到该高斯
  的时间平均误差。
- 只对**梯度合格（`score > densify_gradient_threshold`）且被 ≥2 个相机看到**
  的候选计算误差中位数 `med`，把增长权重乘以
  `1 + w * (2e/(e+med) - 1) ∈ [1-|w|, 1+|w|]`。
- 增长率（`densify_select_fraction`）、梯度门限、替换、超大强制分裂、exact
  alpha 分裂、衰减、上限全部保持原样；`w=0` 或无正误差中位数时乘子恒为 1。
- CLI：`--splat-densify-error-map`（默认 false）、
  `--splat-densify-error-weight`（[-1,1]，幅值=强度、符号=方向）、
  `--splat-seed`（默认 42，用于多 seed 配对）。
- 日志：`igs_error_refine iteration=... weight=... candidates=... reweighted=...
  error_median=... selected_error_ratio=... grown=... pruned=...`。

## 结果（同一二进制、同 seed 配对）

ori 10k / 76 视角 / cap 1M / `quality_eval_reference.exe` 每 8 张取 1（10 视角）：

| seed | 默认（误差图关） | w=0.25 | Δ |
|---|---:|---:|---:|
| 42 | 25.4481 | 25.4815 | +0.033 |
| 7 | 25.4314 | 25.4326 | +0.001 |
| 3 | 24.7465 | 25.0280 | +0.282 |
| 11 | 25.5695 | 24.8438 | -0.726 |
| 1234 | 24.7635 | 25.0089 | +0.245 |
| 2024 | 25.5658 | 25.4808 | -0.085 |
| 平均 | 25.2541 | 25.2126 | **-0.042（4 胜 2 负）** |

15k 留出：WeChat **+0.011 dB**（seed 42/7/2024：+0.034/+0.012/-0.012），
室内 iPhone **+0.054 dB**（seed 42/7/3/11：+0.086/+0.154/+0.127/-0.153）。
两个数据集都没有系统回退，但也没有超出各自 seed 抖动的收益：iPhone 前两个
seed 均为正、补到四个 seed 后出现一个同量级的负值。

被否定的用法（seed 42，或含补测）：只留高误差一半（gate）-0.48 dB；
误差取时序最大值（reduce=max）四个 seed 平均 -0.11 dB；反向 w=-0.25
-0.26 dB。即：**小幅正向重排可用，强度一大或方向反了就伤质量**。

## 关键方法学结论

ori 协议的轨迹噪声极大：同一构建同 seed 重复约 ±0.05 dB，**换 seed 达
0.82 dB**（24.75…25.57）。上一轮"w=0.25 掉 0.40 dB 因此否决"的结论来自
两次不同构建的单次运行，重测后同一配置为 25.4815（老构建那次 24.8989），
差 0.58 dB 纯属轨迹噪声。今后任何 ori 质量结论都需要多 seed 配对；本页
数据来自 `artifacts/igs_error_map_r2_20260917/`（`summary_*.csv`、
`report_stats.py`、`mesh_scan.py`）。

## 默认值

保持**关闭**：三个数据集上都没有可检测的系统性收益，打开只是重掷轨迹
骰子。想开启时用 `--splat-densify-error-map=true`（w 默认 0.25）。

## 附带发现：ori 网格连通分量选择

17 次 ori 运行中有 4 次导出的网格不是主体，而是远处背景壳（面数 3.2M–5.5M，
包围盒 x∈[0.54,1.45]，主体在原点 ±0.4）。原因不是误差图，而是 TSDF 后处理
**只保留最大连通分量**：主体分量约 2.7–2.9M 面，背景壳分量 1.8–2.9M 面，两者
接近，谁大导出谁；主体被丢掉时 landmark 遮挡指标反而降到 ~0（主体不存在，
没有东西提前遮挡 landmark），所以该指标不能单独当漂浮物判据。建议后续让
分量选择参考 subject bounds / 稀疏点云重叠，而不是只比面数。

## 2026-09-18 补充：Brush 对齐后与办公室数据集复测

统一 Brush 训练配置（渐进分辨率双策略共用、位置噪声双策略注入、背景噪声 0.1、
共享 LR/阈值、scale decay 0.002 双策略开启）后复测：

- ori（seed 42，10k，cap 1M，10 视角外部评测）：IGS 25.6755 vs IGS+EM 25.6027
  （-0.07 dB，SSIM +0.001，高斯 +9.6% 至 700,663；selected_error_ratio ~1.07，
  重排几乎不改变实际增长的父节点集合）。
- 办公室 900 视图（Photara SfM 缓存导出 COLMAP，30k，cap 3M，113 held-out）：
  IGS 25.6546 vs IGS+EM 25.6480（打平；EM 反而最早到 3M 上限 ~8.4k 步）。

结论：维持**默认关闭**（options / CLI / 策略预设三处默认均已是 false）。
根因分析暂缓，候选方向：有界乘子在高梯度合格候选内部区分度不足（ratio~1.07）、
误差的 refine 窗口时间平均稀释了瞬时误差、w=0.25 幅度太小。
数据：`artifacts/adc_scale_decay_20260918`、`artifacts/office_compare_20260918`。
