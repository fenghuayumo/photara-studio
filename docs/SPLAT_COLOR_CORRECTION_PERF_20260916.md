# 训练期颜色校正的性能修复（2026-09-16）

回答一个问题：开启 PPISP 与仿射双边网格颜色校正后，训练为什么慢，慢在哪里，能压到什么程度。
实验目录 `artifacts/color_perf_20260916/`（benchmark、ncu 输出、质量跑日志都在里面）。

## 结论（先看这个）

1. **慢的主因只有一个：双边网格的反向 kernel。** 1728×1120 视图上单个 launch 22.8 ms（Nsight
   Compute 计时），占该配置每步 GPU 时间的约 76%；它的 warp 有 84% 的周期卡在 L1 load/store
   队列（LG throttle）。两轮改写后同一 kernel 0.68 ms（**33×**），"两项全开"的逐步耗时从
   18.81 ms 降到 6.13 ms（3.07×），1500 步的墙钟从 33.7 s 降到 15.5 s（2.17×）；颜色校正本身
   的净开销从 14.5 ms/步降到 1.19 ms/步（12×）。
2. **PPISP 的每像素 kernel 改为按参数布局编译期特化**，并把 `original`（36 参数）布局的 CRF
   反传从每像素 15 次有限差分改成解析导数：该布局的反向 2.36 → 1.18–1.31 ms。第三轮进一步把它
   改成"像素组 + 颜色 pull-back"：9 参数布局 0.70 → 0.23 ms，36 参数布局 1.18 → 0.25 ms。
3. **顺带修掉一个真实缺陷。** `original` 布局的有限差分 CRF VJP 会把参数值写进调用方的局部梯度
   数组，污染曝光/暗角/色彩参数（索引 0–15）的梯度：单像素探针上解析值 21.19、有限差分 −0.24。
   三层布局现在都通过端到端有限差分校验（新增测试见下）。
4. **光度损失路径少了两趟全图往返**：掩码关闭时不再把预测/目标拷进中间缓冲；`valid_map_and_chain`
   折进 SSIM 反向。每步少 2 个 kernel、约 170 MB 流量。
5. **没有改变训练语义。** 1500 步 before/after 的损失差（≤2.4e-3）与同一二进制两次运行的噪声底
   （≤1.5e-3）同量级；15k 步留出指标与既有报告一致（见"验证"）。

## 测量口径

- 数据：`D:/ScanVideo/antman_nomask`（64 视图，1728×1120）、`D:/ScanVideo/WeChat_20250712175936`、
  `D:/BaiduNetdiskDownload/室内iphone`（738 视图，1280×1920）。
- 配置：`--splat-strategy adc_igs --splat-iterations 1500 --splat-densification-cap 1000000
  --splat-max-resolution 1920 --splat-progressive-resolution=false --splat-eval-split-every 8`
  （性能对比关掉留出评估，质量对比打开），单进程串行，RTX 5090 D v2。
- `step_ms` 取训练日志最后一行（平滑后的每步墙钟）；阶段耗时取 `--splat-profile-cuda` 的
  CUDA event 平均值；kernel 级耗时用 Nsight Compute 单次 launch（重放会抬高绝对值，只用于横向比较）。
- 优化前的数字来自同一个 benchmark 脚本、同一台机器、同一个二进制（改动前的构建），因此
  "before/after" 两列可比。

## 优化前：慢在哪里

Nsight Compute 对单个 kernel 的测量（1728×1120，约 15k–40k 高斯）：

| kernel | 优化前 | 优化后 |
|---|---:|---:|
| `bilateral_backward_kernel` | **22.82 ms** | 2.75 ms |
| `ppisp_backward_kernel<36>`（`original`） | 2.36 ms | 1.18–1.31 ms |
| `ppisp_backward_kernel<9>`（`no_crf_no_vig`） | 0.83 ms | 0.75 ms |
| `fused_l1_ssim_backward_kernel` | 0.169 ms | 0.188 ms |
| `fused_l1_ssim_forward_kernel` | 0.148 ms | 0.149 ms |
| `ssim_cs_error_map_kernel` | 0.135 ms | 0.138 ms |
| `loss_kernel` | 0.044 ms | 0.044 ms |
| `bilateral_forward_kernel` | 0.050 ms | 0.051 ms |
| `ppisp_forward_kernel` | 0.032 ms | 0.022 ms |
| `apply_mask_kernel` / `valid_map_and_chain_kernel` | 各一趟全图 | 归零（掩码关闭时） |

双边网格反向的 SASS 是 96 次标量 `LDG` + 96 次标量 `REDG.E.ADD.F32`（每像素 12 通道 × 8 角点），
warp 平均每条指令等 3151 个周期，84% 的停顿记为 LG throttle：不是算力不够，是访存指令太多。

## 三项改动

### 1. 双边网格反向（`src/splat/bilateral_grid.cu`）

- 一次遍历角点：原来第二个循环为了 z（亮度）方向的导数把 96 个系数**再读一遍**，现在同一份
  寄存器里的值同时喂给网格梯度与输入颜色梯度。
- 单元格 12 个系数是 3 个连续的 `float4`：读用 `LDG.E.128`，写用 `red.global.add.v4.f32`
  （sm_90+，旧架构回落到标量），每像素访存指令 192 → 48。
- 去掉 `%`/`/` 与分支选择，索引用 `int`（行基址仍用 64 位）。
- 每个线程走 4 个像素（stride 为 grid 跨度），让多条独立的规约流同时在飞：这一步单独贡献
  1.8 ms（4.73 → 2.90 ms 的阶段耗时）。
- 梯度全零的像素（掩码/背景）直接写出零梯度并返回。

### 2. PPISP（`src/splat/ppisp.cu`）

- `ppisp_forward_kernel` / `ppisp_backward_kernel` 以参数布局（9 / 24 / 36）为模板参数实例化：
  暗角与 CRF 的分支变成 `if constexpr`，逐参数规约循环只走该布局真正拥有的参数
  （原来 9 参数布局也要跑满 36 次带谓词的循环）。
- CRF 的 VJP 改为解析导数（`crf_channel_grad`）：原来每像素 15 次完整的色调曲线有限差分
  （每次 3 通道 × 2 次 `powf`），现在每通道一次取值 + 解析导数，顺带修掉第 3 条里的缺陷。
- 色彩单应 Jacobian 的差分步长 `k_color_eps` 从 1e-3 改到 1e-2：1e-3 时 float32 舍入主导
  估计误差（对照宿主端差分约 3%），1e-2 后残差 <1%（仍是有限差分，见"仍未做"）。

### 3. 光度损失（`src/splat/fused_ssim.cu`）

- 掩码关闭时不再把 prediction / target 拷进 `image1` / `image2`（省 48 MB 写 + 48 MB 读）。
- `valid_map_and_chain_kernel` 删除：标量项改在 SSIM 前向里累加（map 本来就在寄存器里），
  掩码乘法与最终梯度写入折进 SSIM 反向（省 72 MB 读 + 24 MB 写）。

### 4. profiler 归因（`src/splat/trainer.cpp`）

新增 `colour_forward_ms` / `colour_backward_ms` 两个阶段。此前颜色校正的前向被记在 `preview`、
反向被记在 `raster_backward` 里，日志上看不出它们的开销。

## 结果

antman，1500 步，同一脚本串行（`step_ms` 为大小时会随高斯数增长，取最后一行）：

| 配置 | step_ms 前 | step_ms 后 | 墙钟 前 | 墙钟 后 |
|---|---:|---:|---:|---:|
| 基线（都关） | 4.91 | 4.88 | 14.2 s | 13.9 s |
| PPISP `no_crf_no_vig` | 5.51 | 5.45 | 14.5 s | 14.4 s |
| PPISP `original` | 6.80 | 5.89 | 16.1 s | 14.9 s |
| 双边网格 | 18.49 | 6.79 | 33.3 s | 15.7 s |
| 双边网格 + PPISP | 18.81 | **7.38** | 33.7 s | **16.3 s** |

阶段归因（优化后，`colour_backward` 含同阶段的 densify error map，基线 0.09 ms）：

| 配置 | colour_forward_ms | colour_backward_ms | 净开销 |
|---|---:|---:|---:|
| 基线 | 0.001 | 0.091 | 0 |
| PPISP `no_crf_no_vig` | 0.017 | 0.657 | 0.57 ms |
| PPISP `original` | 0.024 | 1.059 | 0.97 ms |
| 双边网格 | 0.039 | 2.064 | 1.97 ms |
| 双边网格 + PPISP | 0.052 | 2.652 | 2.56 ms |

即：两项全开的净开销从约 14.5 ms/步降到 2.6 ms/步（5.6×），相对基线的倍数从 3.8× 降到 1.5×。

## 室内 iPhone 数据集（1280×1920、738 视图）

`D:/BaiduNetdiskDownload/室内iphone`（738 个相机、222,484 初始点、约 35 万高斯），1200 步，
其余开关与 antman 一致。这一组的结论与 antman 一致，但**墙钟收益小得多**，原因见最后一段。

| 配置 | 颜色校正 GPU 开销 前 → 后 | step_ms 前 → 后 | 墙钟 前 → 后 |
|---|---:|---:|---:|
| 基线 | 0 → 0.13（densify map） | 38.36 → 38.54 | 44.3 s → 45.1 s |
| PPISP `no_crf_no_vig` | 0.75 → 0.77 | 37.82 → 38.16 | 44.4 s → 44.9 s |
| PPISP `original` | 2.50 → 1.27 | 38.03 → 38.10 | 45.1 s → 44.7 s |
| 双边网格 | **22.43 → 2.56** | 42.03 → 38.21 | 53.4 s → 45.2 s |
| 双边网格 + PPISP | **22.84 → 3.25** | 41.71 → 38.58 | 52.9 s → **46.0 s** |

（"颜色校正 GPU 开销"取 `colour_forward_ms + colour_backward_ms` 的中位数再做基线相减；优化前
这些时间记在 `raster_backward` 里。kernel 级倍数与 antman 一致：双边网格反向 8.8×，PPISP
`original` 反向 1.9×。）

这一组最关键的一点是：**优化前 22.8 ms/步的颜色校正 GPU 工作只让每步墙钟涨了 3.3 ms**
（38.36 → 41.71），因为该数据集的训练循环本来就是**图像加载受限**：每个视图 1280×1920 的
JPEG 解码 + 上传约 40 ms/步（`data_load_ms` 中位数 38–42 ms），GPU 工作被部分盖住。于是：

- 只在 GPU 侧看，两项全开的开销降了 7.0×；
- 在墙钟上看，"两项全开相对基线"的额外耗时从 8.7%（step）／19%（1500 步以内的总时长）
  降到 0.1%（step）／2%（墙钟）；
- 进一步的墙钟收益不在这两个 kernel，而在数据加载：这个数据集可以用
  `--splat-prefetch-views`、`--splat-cache-auto`、`--splat-device-cache-mb` 调整视图缓存；
  本轮没有动这条路径。

损失轨迹的前后差（同一二进制同一配置重复运行的抖动同量级）：双边网格 ≤1.9e-3、两项全开
≤2.4e-3、基线自身 ≤3.2e-3，即改写没有把训练轨迹推到噪声之外。

## 第二轮：像素组归并（同日追加）

第一轮之后，双边网格反向仍是颜色校正里最贵的一项（antman 上 2.06 ms/步，占两项净开销的 73%），
消融显示时间几乎全在等 `red.global` 返回。第二轮把"同一个像素组共享八个格点"这件事利用起来。

### 做法

- 一个线程不再处理"跨步的 4 个像素"，而是处理**连续的一段像素**，这样组内像素大概率落在同一批
  格点上（16×16 空间格 + 1728 宽时一个 x 格覆盖约 108 像素）。
- 先算这一段的几何（颜色、输出梯度、八个邻居格点、三个分数坐标），若**整段共享八个格点**，就只
  走一遍角点循环：每个角点读一次格子（3×`float4`），把整段像素的贡献在寄存器里累加，最后只做
  3 次向量规约。每像素的规约次数从 24 次降到 1.5 次（16 像素组），每像素的格子读取也降到 1.5 次。
- 不满足条件（跨行、跨格点、像素在图像外、梯度全零）时退回原来的逐像素路径，结果一致。
- 组宽按视图大小选择：`pixels >= 1e6` 用 16，否则用 4。实测 1728×1120：组宽 4 → 1.11 ms、
  8 → 0.73 ms、16 → 0.70 ms、32 → 1.61 ms（寄存器溢出）；而 512×332 上组宽 16 反而比 4 慢
  63%（0.127 对 0.078 ms），因为 block 数不够铺满 SM，所以做了这个分档。

### A/B（antman，1200 步，其余开关一致）

| 组宽 | colour_backward_ms | step_ms | 墙钟 |
|---|---:|---:|---:|
| 逐像素（第一轮，跨步 4 像素） | 3.36 / 3.30 | 8.77 / 8.70 | 11.6 / 11.4 s |
| 连续组 + 不归并 | 3.36 / 3.30 | 8.77 / 8.70 | 11.6 / 11.4 s |
| 归并，组宽 4 | 1.11 | 6.67 | 9.3 s |
| 归并，组宽 8 | 0.73 / 0.74 | 6.31 / 6.27 | 8.9 s |
| 归并，组宽 16 | 0.71 / 0.69 | 6.24 / 6.24 | 8.9 s |
| 归并，组宽 32 | 1.61 / 1.61 | 7.01 / 7.04 | 9.9 s |

（同一行两个数字是同配置重复运行；"逐像素"那一行用的是第一轮提交的二进制。）

### 结果

antman（1500 步，`step_ms` 取最后一行，`colour_backward` 含同阶段的 densify error map，基线约 0.09 ms）：

| 配置 | 第一轮 step_ms | 本轮 step_ms | 第一轮净开销 | 本轮净开销 | 相对最初 |
|---|---:|---:|---:|---:|---:|
| 基线 | 4.88 | 4.97 | 0 | 0 | — |
| PPISP `no_crf_no_vig` | 5.45 | 5.50 | 0.57 ms | 0.57 ms | 0.65 → 0.57 |
| PPISP `original` | 5.89 | 5.88 | 0.97 ms | 0.97 ms | 2.09 → 0.97 |
| 双边网格 | 6.79 | **5.65** | 1.97 ms | **0.62 ms** | 14.4 → 0.62（23×） |
| 双边网格 + PPISP | 7.38 | **6.13** | 2.56 ms | **1.19 ms** | 14.5 → 1.19（12×） |

室内 iPhone（1200 步，1280×1920、738 视图）：

| 配置 | colour_backward 最初 → 第一轮 → 本轮 | step_ms 最初 → 本轮 | 墙钟 最初 → 本轮 |
|---|---:|---:|---:|
| 基线 | 0.13 → 0.12 | 38.36 → 38.39 | 44.3 s → 45.7 s |
| PPISP `no_crf_no_vig` | 2.14 → 0.90 | 37.82 → 38.12 | 44.4 s → 44.7 s |
| PPISP `original` | 3.89 → 1.42 | 38.03 → 38.96 | 45.1 s → 45.4 s |
| 双边网格 | **23.82 → 2.70 → 1.10** | 42.03 → 38.27 | 53.4 s → 45.0 s |
| 双边网格 + PPISP | **24.23 → 3.38 → 1.81** | 41.71 → 38.73 | 52.9 s → 45.8 s |

### 顺带做的一项（收益中性）

PPISP 反向右向的 homography Jacobian 原来是块内线程 0 串行做 17 次 `compute_homography`。
现在这 17 次求值分到 17 条 lane 上并行，八列 Jacobian 由扰动对相减得到。在 1728×1120 上
测量值与串行版无差别（0.66 对 0.66 ms，噪声内），它只是把每个 block 的串行前导去掉；保留是因为
它在小图上不再成为固定开销，而本次没有测到它的收益。

### 验证

- `aetherscan_splat_test` 全绿。双边网格的有限差分探针现在跑两组格点尺寸：细格（5×4×3，相邻像素
  基本不共享格点 → 走逐像素路径）与粗格（2×2×2，相邻像素共享格点 → 走归并路径）。为确认后者确实
  被覆盖，临时把归并结果乘 1.5 后测试立刻报 "bilateral grid coefficient gradient differs from
  finite differences"，随后已还原。
- 室内 iPhone 上以第一轮二进制为基准对比损失轨迹：双边网格 ≤3.9e-3、两项全开 ≤4.7e-3；
  归并改变的是浮点求和顺序（4 个像素先在寄存器里相加再规约），因此轨迹会离开逐位一致，
  量级与同一数据集的重跑抖动相同。

## 第三轮：PPISP 反向（同日追加）

第二轮结束时 PPISP 反向是颜色校正里最大的一项（antman 0.66 ms/步）。先看它的停顿构成
（`ppisp_backward_kernel<9>`，Nsight Compute 的 warp 停顿采样）：

| 停顿原因 | 占比 |
|---|---:|
| long scoreboard（等全局加载） | 56.4% |
| short scoreboard（等 shared / shuffle 结果） | 25.8% |
| MIO throttle | 7.5% |
| barrier | 4.9% |
| 其余 | 5.4% |

即 82% 的停顿在等内存。逐项消融（antman，900 步，看 `colour_backward_ms`）：

| 变体 | colour_backward_ms |
|---|---:|
| 完整 | 0.665 |
| 跳过块内 homography 前导（H + Jacobian） | 0.664 |
| 跳过 `apply_color_vjp`（8×9 Jacobian 收缩） | 0.193 |
| 跳过逐参数 shuffle 规约 | 0.148 |

据此做了三件事：

1. **一个线程走 4 个连续像素，颜色与梯度用 `float4` 装载**（6 次向量加载代替 24 次标量加载），
   让每个 warp 同时在飞的字节数翻两番——这一条针对的就是上面 56% 的 long scoreboard。
2. **颜色 homography 的 pull-back 上移**：8×9 的 Jacobian 对**同一视图的每个像素都一样**，所以
   原来每像素做的那 72 次融合乘加，等价于"先把 9 元素的 pull-back 向量按像素加起来，每个视图
   只做一次收缩"。现在逐像素累加 9 个 pull-back 分量，收尾由一个 `ppisp_colour_grad_kernel`
   （每视图一个线程，9 个融合乘加 × 8 行）完成；逐像素 kernel 里连 Jacobian 都不再需要，
   块内前导从 17 次 `compute_homography` 减到 1 次。
3. 4 个像素的贡献先在寄存器里相加，再做 warp 规约，规约量随之降到四分之一。

顺带**回退**了第二轮的"homography Jacobian 并行到 17 条 lane"：消融显示前导耗时是 0.664 对
0.665 ms，即完全被其他常驻 block 盖住，那项改动没有收益，保留只会增加代码。

### 结果

kernel 本体（antman，1728×1120，Nsight Compute 单次 launch）：

| kernel | 最初 | 第一轮 | 第三轮 |
|---|---:|---:|---:|
| `ppisp_backward_kernel<9>` | 0.83 ms | 0.70 ms | **0.23 ms** |
| `ppisp_backward_kernel<36>`（`original`） | 2.36 ms | 1.18 ms | **0.25 ms** |
| `ppisp_colour_grad_kernel` | — | — | 0.005 ms |

阶段与逐步耗时（antman，1500 步）：

| 配置 | colour_backward 第一轮 → 本轮 | step_ms 第一轮 → 本轮 |
|---|---:|---:|
| PPISP `no_crf_no_vig` | 0.657 → **0.295** | 5.45 → **5.24** |
| PPISP `original` | 1.062 → **0.311** | 5.88 → **5.20** |
| 双边网格 + PPISP | 1.279 → **0.905** | 6.13 → **5.77** |

（`colour_backward` 含同阶段的 densify error map，基线约 0.09 ms；扣掉它，PPISP 反向的净开销
从 0.57 ms 降到 0.21 ms，`original` 从 0.97 ms 降到 0.22 ms。）

室内 iPhone（1200 步）：

| 配置 | colour_backward 最初 → 第一轮 → 本轮 | step_ms 最初 → 本轮 |
|---|---:|---:|
| PPISP `no_crf_no_vig` | 2.14 → 0.90 → **0.40** | 37.82 → 37.77 |
| PPISP `original` | 3.89 → 1.42 → **0.41** | 38.03 → 37.74 |
| 双边网格 + PPISP | 24.23 → 1.81 → **1.38** | 41.71 → 37.88 |

三项加起来（1728×1120）：两项全开的净开销从最初的 14.5 ms/步降到 **0.69 ms/步**（21×）。

### 验证与噪声

- `aetherscan_splat_test` 全绿：三种 PPISP 布局的每一维参数与颜色都仍通过中心差分校验——颜色
  参数的梯度现在走的是完全不同的代码路径（pull-back + 每视图收缩），校验通过说明改写等价。
- 损失轨迹（室内 iPhone，与前一轮二进制对比）：PPISP ≤7.8e-3、两项全开 ≤6.5e-4，与该数据集
  重跑抖动同量级。
- **质量指标本身的抖动不可忽略**：antman 上"两项全开"用同一个二进制跑两次，15k 步留出 PSNR
  分别是 20.37 dB / 0.9358 与 21.84 dB / 0.9468（差 1.5 dB）。该配置里"高斯带的颜色"与
  "每视图曝光"本来就接近退化方向，留出相机的曝光又不在 PPISP 表里，所以这个指标本身不适合
  用来判定 1 dB 量级的差异；WeChat 同配置稳定在 20.76 / 20.81 dB。

## 第四轮：PPISP 的块级规约与"哪一项已经没空间"（同日追加）

第三轮之后 `ppisp_backward_kernel<9>` 只剩 0.23 ms，但新的停顿采样显示瓶颈换了地方：
**short scoreboard 35.6% + MIO throttle 22.4%**——两者都来自逐参数 shuffle 规约（每个值 5 步依赖
shuffle，全部走 MIO 管道）。

### 做法

块级规约换掉 shuffle 树：每个线程把自己的 `k_params + 9` 个贡献**存进共享内存**（冲突无关，
每个值一条 STS），`__syncthreads()` 之后由 `k_params + 9` 个线程各自把整块加起来，每个参数只做
一次 `atomicAdd`。共享数组按 `[值][线程]` 布局并加 1 列 padding，第二阶段的跨行读取因此不冲突；
最宽的 36 参数布局需要 45 行（46 KB），仍在 48 KB 静态共享上限之内。

顺带把"每线程像素数"提成 kernel 与 host 共用的 `k_ppisp_pixels_per_thread`：原来 host 的 block
数写死乘以 4，两者一旦不一致就会漏算一部分图像（这正是我扫描像素数时踩到的坑）。

### 结果

| kernel（antman，1728×1120） | 最初 | 第三轮 | 本轮 |
|---|---:|---:|---:|
| `ppisp_backward_kernel<9>` | 0.83 ms | 0.23 ms | **0.050 ms** |
| `ppisp_backward_kernel<36>`（`original`） | 2.36 ms | 0.25 ms | **0.06 ms** |

| 配置（antman，1500 步） | colour_backward 第三轮 → 本轮 | 净开销 第三轮 → 本轮 | step_ms 最初 → 本轮 |
|---|---:|---:|---:|
| PPISP `no_crf_no_vig` | 0.295 → **0.156** | 0.21 → **0.065 ms** | 5.51 → **5.02** |
| PPISP `original` | 0.311 → **0.195** | 0.22 → **0.105 ms** | 6.80 → **5.06** |
| 双边网格 + PPISP | 0.905 → **0.780** | 1.19 → **0.69 ms** | 18.81 → **5.68** |

室内 iPhone（1200 步）：PPISP 9p `0.90 → 0.263`、`original` `1.42 → 0.282`、两项全开
`24.23 → 1.23 ms`（该数据集逐步耗时 37–42 ms 之间波动，全部来自视图加载）。

### 已经没空间的一项：跨 warp 合并同一格点的贡献

第二轮之后双边网格反向的规约次数已经从每像素 24 次降到 1.5 次（16 像素组）。再做一次消融
（把 `red_add4` 放进永不执行的分支、其余计算保留）来量它还剩多少：

| 双边网格反向（antman，colour_backward_ms） | 值 |
|---|---:|
| 完整 | 0.6157 |
| 不做任何规约 | 0.6245 |

即**规约现在已经是免费的**（差值在噪声内、方向还是反的），跨 warp 合并没有可压缩空间。该 kernel
现在的停顿构成是 long scoreboard 66.6% + LG throttle 27.1%，寄存器 255（已溢出），每次
launch 约 120–136 MB DRAM 流量——瓶颈是逐像素 VJP 本身的装载/算子量与溢出流量，不是规约。
再往下要动的是算法级改写（例如按 z 分层做可分离累加），代价远超收益，本轮到此为止。

## 顺带修复的正确性缺陷

`original`（36 参数）布局下，CRF 的有限差分 VJP 内部有两个 `float[12]` 栈数组；编译后它们与调用
方的 `local[36]` 梯度数组重叠，等于每次都把 CRF 的参数值加进曝光/暗角/色彩参数的梯度里。
单像素探针（`parameters[1]`，暗角中心 x）：

| | 解析梯度 | 有限差分 |
|---|---:|---:|
| 修复前 | 0.0531 | 0 |
| 修复后 | 0 | 0 |

多像素探针上同一现象以 `参数值 × 像素数` 的量级出现（63 像素：21.19 对 −0.24）。改用解析 VJP
（同时去掉了那两个栈数组）后，三层布局的每一维参数、每一维颜色的解析梯度都通过中心差分校验。

## 验证

- `aetherscan_splat_test`：新增 `test_bilateral_grid_finite_differences`（网格系数 + 颜色）、
  `test_ppisp_finite_differences`（三种布局的全部参数 + 颜色）、
  `test_masked_photometric_gradient`（带掩码的光度损失对标中心差分）；原有的
  `test_ssim_loss_and_scale_constraint` 继续对齐 Python CUDA 参考值（前向损失、中心像素梯度、
  梯度场求和），说明损失路径的重构没有改变数值。
- 质量跑（15k 步、split 8、留出相机）：antman 基线 `22.23 dB / 0.9469`、两项全开
  `21.60 dB / 0.9421`；WeChat 基线 `21.29 dB / 0.9670`、两项全开 `20.74 dB / 0.9646`。
  与 [颜色校正报告](SPLAT_COLOR_CORRECTION_20260914.md) 的结论一致（两项会改变按帧对齐后的
  指标，不改善导出模型的 canonical 指标），没有出现新的退化或非有限值。
- 噪声底：同一二进制、同一配置跑两次（antman，1500 步）损失差 ≤1.5e-3、最终 PSNR 差 0.16 dB；
  before/after 的差距在 2.4e-3 以内，量级相同，因此这次改写没有把训练轨迹推到噪声之外。

## 仍未做

- **双边网格反向仍受规约延迟限制。** 消融（删掉规约、其余计算保留）显示第一轮版本
  4.73 → 1.09 ms；第二轮把 16 个像素的贡献在寄存器里合并后为 0.68 ms。再往下要动的是跨 warp
  合并同一格点上的贡献，需要 shuffle/`match_any` 或分层共享内存累加，实测代价与收益同量级，
  没有做。
- **光度损失里的 SSIM 前向/反向（各 0.15 / 0.19 ms）与 densify error map（0.14 ms）** 仍是
  固定开销。error map 用的是亮度通道的 11×11 窗口统计，和损失用的逐通道统计不是同一组量，
  融合需要在前向里多算 6 个跨通道累加器（约 +50% 前向工作量换掉 0.14 ms 的独立 kernel），
  收益接近零，没有做。
- `k_color_eps` 仍是有限差分步长。homography 的 Jacobian 现在每视图只算一次（收尾 kernel
  里），已经是块内 17 次求值的常数开销；要彻底消除百分级残差需要解析的 homography Jacobian。
- **室内 iPhone 的墙钟瓶颈是视图加载（≈40 ms/步）**，三项 kernel 优化在那上面只换到 1.15×；
  这条路径（`--splat-prefetch-views` / `--splat-cache-auto` / `--splat-device-cache-mb`）
  本轮没有动。

## 复现

```powershell
# 逐步性能（before/after 用同一脚本、同一数据、串行）
python artifacts/color_perf_20260916/run_perf.py antman
python artifacts/color_perf_20260916/summarize_perf.py

# 质量（15k 步、split 8，留出相机指标）
python artifacts/color_perf_20260916/run_quality.py antman
python artifacts/color_perf_20260916/run_quality.py wechat

# 室内 iPhone：优化前/后两个二进制各跑一遍（<suffix> 为空即"后"）
python artifacts/color_perf_20260916/run_iphone.py _before
python artifacts/color_perf_20260916/run_iphone.py ""
python artifacts/color_perf_20260916/compare_iphone.py \
  iphone_baseline_before iphone_baseline iphone_both_before iphone_both

# 正确性
build/aetherscan/Release/aetherscan_splat_test.exe
```
