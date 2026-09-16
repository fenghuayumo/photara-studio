# ADC-IGS 原生相机第 2 轮（2026-09-16）：全景反传修复与 mask 泄漏消融

沿用 `ADC_IGS_NATIVE_CAMERAS_20260916.md`（第 1 轮）的同一套数据、同一套协议，只改代码与 mask
相关开关。实验目录 `artifacts/native_camera_20260916_round2/`。

## 结论（先看这个）

1. **全景反传的两个缺陷已定位并修复**（第 1 轮列为最高优先级）。有限差分相对误差从
   `9.8% / 19.6% / 21.5%`（缝前点）和 `0.4% / 48.8% / 106.8%`（过缝点）降到 **≤0.2%**。
   `aetherscan_splat_test --fisheye-only` 现在通过。
2. **修复本身就能提高质量。** 同一命令、同一数据、同一留出视角，只换二进制（静态区 PNG）：
   全景 no-mask `24.98 → 25.91 dB`，masked(leak=1) `22.59 → 24.14 dB`。
3. **全景气泡/雾团的直接原因是 mask 的 alpha 泄漏惩罚。** leak `1 → 0.25 → 0` 单调改善
   静态区 PNG `24.14 → 25.87 → 27.02 dB`，SSIM `0.820 → 0.848 → 0.860`；目视：
   leak=1 满屏圆形半透明气泡，leak=0.25 只剩右缘余雾，leak=0 干净。
4. **全景静态区 PNG 合计提升 4.4 dB**（第 1 轮 masked `22.59` → 本轮 leak=0 `27.02`），
   其中约 1.5 dB 来自反传修复、2.9 dB 来自泄漏权重。
5. **鱼眼真正的瓶颈是容量上限，不是初始化。** `--splat-densification-cap 2000000`：
   留出 PSNR `21.13 → 21.25 dB`（同配置重跑的噪声底 0.07 dB，所以这项提升约 1.5–2× 噪声），
   SSIM `0.708 → 0.726`（噪声底 0.0005，提升 36×），增益集中在后院段
   （DSC076x–079x 逐段 +0.2…+0.6 dB）。只把初始化降到 600k（cap 仍 1M）几乎没有变化
   （`21.148 dB` / `0.709`）。
6. **仍然不能宣称"优化好了"。** 鱼眼最差段（后院 DSC076x）只有 `17.4 dB`；4K 全景跑通且干净，
   但跨分辨率的 PSNR 不能对比；反传修复只覆盖 RGB/协方差链，见"仍未验证"。
7. **指标口径已统一（键名不变）。** 训练器同时输出全图 `average_psnr` 与前景/静态区
   `average_foreground_psnr`（即逐视角行的 `foreground_psnr`）；离线 PNG 诊断一律用同一张
   静态 mask，masked 与 no-mask 才可比。

## 代码改动

| 文件 | 改动 |
|---|---|
| `third_party/splat_drender/src/device/geometry.cuh` | 全景反向：Jacobian 存储方向与 forward 对齐；补 `dL_dJ01/dL_dJ10`；equirect 走精确二阶链（pinhole 路径逐字未变） |
| `third_party/splat_drender/src/device/equirect.cuh`（新增） | 嵌套前向 dual，一次求出等距柱状投影的一阶与二阶导数 |
| `aetherscan/src/splat/cuda_ops.cu` | loss kernel 增加 `mask_alpha_leak_weight` |
| `aetherscan/include/splat/options.hpp` | 新增 `mask_alpha_leak_weight`、`initial_point_budget` |
| `aetherscan/src/splat/trainer.cpp` | 评测把 `psnr`（全图）与 `foreground_psnr`（前景）分别算对；初始化点数预算 |
| `aetherscan/src/tools/reconstruct.cpp` | 新增 `--splat-alpha-leak-weight`、`--splat-init-point-budget` |
| `aetherscan/tests/splat_test.cpp` | 全景 FD 探针打印解析值/差分值/相对误差 |

## 全景反传：到底修了什么

### 缺陷 1：反向把 Jacobian 存成了转置

forward 用 `J.m[axis][pixel] = d(pixel)/d(axis)`（即真 Jacobian 的转置），反向填的是
`J.m[pixel][axis]`。`cov2D = Tᵀ Σ T`（`T = W J`）因此被求成了另一个矩阵。pinhole 的四条
活元素都在对角线上，转置不变，所以这个错误一直看不出来；全景的 `dv/dx ≠ 0`，一转置就错。

### 缺陷 2：位置导数用了透视公式，而且少一项

反向把 `dL/dJ` 链回 `dL/d(mean)` 时用的是 3DGS 的 pinhole 恒等式
`d(f x/z)/dz = -f x/z²`、`∂J/∂x = 0` 等，且只接受 4 个非零元素。等距柱状投影
`u = (atan2(x,z)/2π+1/2)·W`、`v = (atan2(y,√(x²+z²))/π+1/2)·H` 不满足这些恒等式，
并且 `∂v/∂x` 不为零——反向完全没算这一项。

现在 `equirect::project_jacobian` 用嵌套 dual（`Dual<Dual<float>>`）对同一条源表达式求到
二阶导，六个元素与它们的三个方向导数一起得到，反向按
`dL/dt_k = Σ_{r,c} dL/dA[r][c] · ∂A[r][c]/∂t_k` 精确链式回传。

### 验证（`aetherscan_splat_test --fisheye-only`，单高斯 + 平滑探针，中心差分）

修复前后都用同一套探针，`ε=1e-3`：

| 探针位置 | 轴 | 修复前相对误差 | 修复后相对误差 |
|---|---:|---:|---:|
| (0.2, 0.1, 2) | x | 9.78% | 0.024% |
| (0.2, 0.1, 2) | y | 19.62% | 0.015% |
| (0.2, 0.1, 2) | z | 21.49% | 0.027% |
| (0.02, 0.1, -2)（过缝后向） | x | 0.42% | 0.011% |
| (0.02, 0.1, -2) | y | 48.79% | 0.20% |
| (0.02, 0.1, -2) | z | 106.79% | 0.026% |

测试现在跑四个探针，步长改成 `ε=1e-4`：

| 探针位置 | 最大相对误差（修复后） |
|---|---:|
| (0.2, 0.1, 2) 缝前中景 | 0.56% |
| (0.02, 0.1, -2) 过缝后向 | 0.69% |
| (-0.7, -0.2, 0.6) 近相机 | 0.17% |
| (0.3, 1.5, 0.8) 高仰角 | 0.14% |

近相机探针在 `ε=1e-3` 下会给出 18% 的假差异——等距柱状投影在那里曲率很大，中心差分的
三阶截断项占了主导，换成 `ε=1e-4` 后降到 0.04%，说明那是**差分步长**问题而不是反传问题。
修复后残差全部落在浮点/差分噪声里。

### 修复对训练的影响（同命令、只换二进制）

| 全景 47 留出视角 | 全图 PNG | 静态区 PNG | 球面加权静态 | foreground_psnr |
|---|---:|---:|---:|---:|
| 第 1 轮 no-mask | 21.088 | 24.979 | 24.289 | 20.934 |
| 本轮 no-mask | **21.668** | **25.912** | **25.321** | **21.525** |
| 第 1 轮 masked(leak=1) | 19.191 | 22.585 | 21.679 | 22.347 |
| 本轮 masked(leak=1) | **19.655** | **24.137** | **23.475** | **23.849** |

## mask 泄漏消融（`--splat-alpha-leak-weight`）

`--splat-alpha-mode masked` 原本对 mask 外像素无条件压 alpha：
`grad_alpha = (1-valid)/N`。持镜人挡住的那面墙在**别的视角**必须是不透明的，这个惩罚和
多视角一致性直接冲突，优化器只好在人物位置长出半透明大气泡来"两边都满足"。

新开关把这一项乘上权重（默认 1.0 = 与 pygsplat 一致；0 = 只屏蔽 RGB，遮挡背景交给其他视角）：

| 全景 masked，47 留出视角 | 全图 PNG | 静态区 PNG | 球面加权静态 | foreground_psnr | SSIM |
|---|---:|---:|---:|---:|---:|
| 第 1 轮 leak=1（旧反传） | 19.191 | 22.585 | 21.679 | 22.347 | 0.776 |
| 本轮 leak=1.00 | 19.655 | 24.137 | 23.475 | 23.849 | 0.820 |
| 本轮 leak=0.25 | 20.026 | 25.867 | 25.351 | 25.491 | 0.848 |
| 本轮 leak=0.00 | **20.285** | **27.022** | **26.405** | **26.563** | **0.860** |
| 参考：no-mask（人体进模型） | 21.668 | 25.912 | 25.321 | 21.525（=全图） | 0.825 |

要点：

- 表中"第 1 轮"行的前景值取自当轮日志的 `average_psnr`——当轮 masked 的该键**就是**仅前景
  口径（与 `foreground_psnr` 同值）。现在 `average_psnr` 是全图，前景走
  `average_foreground_psnr`，两轮的 masked 行请按前景列对齐比较。
- leak=0 的静态区比 no-mask 还高 **1.1 dB**，说明"遮住人体、不罚透明"既没有把人体学进模型，
  也没有牺牲静态几何；人体没被拟合（view 0 缝上的人不再出现），气泡消失。
- 训练器日志里 `alpha` 项：leak=1 收敛到 `5.7e-3`，leak=0.25 为 `3.9e-3`，leak=0 恒为 0。
- 全图 PSNR 仍然低（20 dB 量级）是**预期**的：被 mask 掉的像素在真值里是人，模型给的是后面的墙。
  所以拿全图 PSNR 比较 masked 与 no-mask 没有意义，必须看静态列。
- 默认仍是 1.0（pygsplat 兼容）。**固定机位、有动态遮挡物（人）的全景/鱼眼实拍建议显式用 0**。

## 训练器指标口径

第 1 轮的问题是 `psnr` 与 `foreground_psnr` 打成同一个数（都是"仅前景"），而且随 `use_mask`
换像素集合。现在两行键名都不变，只是各自回到自己的定义：

- `RenderMetrics::psnr` = **全图** PSNR；`foreground_psnr` = **前景/静态区**（mask 前景）PSNR；
  没有 mask 时两者相等。
- 逐视角行：`splat_eval_iteration=… view=… psnr=<全图> ssim=… foreground_psnr=<前景>`；
  均值行：`average_psnr=` / `average_ssim=` / `average_foreground_psnr=`。
- 与第 1 轮对照时注意：no-mask 的两者都等于全图，语义不变；第 1 轮 masked 的 `22.347`
  是"仅前景"口径，应对本轮的 `average_foreground_psnr` 比较，不要对 `average_psnr`。
  （本轮 artifacts 里的 train.log 由中间构建产生，把同一个前景数记成了
  `average_static_psnr=`，与现在的 `average_foreground_psnr=` 是同一个量。）

## 鱼眼：初始化预算 vs 容量上限

第 1 轮现象：初始化 1,072,214 点已经超过 cap 1,000,000，第 200 步被迫剪掉 72,214，
此后 `capacity=0`，IGS 只能替换不能增长（全程 `growth_selected` 合计 279）。

| 鱼眼 218 留出视角 | 初始点 | cap | 最终高斯 | 第 200 步新增 | 全程复制 | 留出 PSNR | SSIM |
|---|---:|---:|---:|---:|---:|---:|---:|
| 第 1 轮 | 1,072,214 | 1,000,000 | 1,000,000 | 0 | 279 | 21.132 | 0.708 |
| 本轮 baseline（与第 1 轮同 flags） | 1,072,214 | 1,000,000 | 1,000,000 | 0 | 0 | 21.161 | 0.708 |
| 本轮 init600k | 600,000 | 1,000,000 | 1,000,000 | 85,383 | 395,989 | 21.148 | 0.709 |
| 本轮 cap2m | 1,072,214 | 2,000,000 | 2,000,000 | 108,962 | 922,825 | **21.250** | **0.726** |

结论：**只给增长空间（init600k）没用，要同时抬高上限。** cap2m 的收益集中在后院段：

| 名称段 | 064x | 065x | 066x | 067x | 068x | 069x | 070x | 071x | 072x | 073x | 074x | 075x | 076x | 077x | 078x | 079x | 080x | 081x |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 第 1 轮 | 19.86 | 20.14 | 23.15 | 24.03 | 22.88 | 22.23 | 20.54 | 21.96 | 22.72 | 22.44 | 22.69 | 20.35 | 16.87 | 18.15 | 19.12 | 20.05 | 23.19 | 19.40 |
| cap2m | 20.03 | 19.97 | 23.12 | 23.97 | 22.84 | 22.39 | **20.75** | **22.09** | 22.86 | 22.42 | 22.80 | **20.65** | **17.39** | **18.71** | **19.35** | 20.05 | 23.00 | 19.55 |

后院（076x–079x）是全场最差段，也是这一轮唯一系统性受益的段：说明那里的瓶颈是**细节容量**
（细枝、天空、远景物），不是投影或初始化。

## 全景 4K（长边 3840）

第 1 轮只用 1920 做了结构验证，这轮把最佳配置（masked + leak=0）在长边 3840 上完整跑完
15,000 步（1609 s，1,000,000 高斯，无 CUDA 报错、无气泡）：

| 全景 47 留出视角 | 训练分辨率 | 全图 PNG | 静态区 PNG | 球面加权静态 | foreground_psnr | SSIM |
|---|---:|---:|---:|---:|---:|---:|
| masked + leak=0 | 1920×960 | 20.285 | 27.022 | 26.405 | 26.563 | 0.860 |
| masked + leak=0 | **3840×1920** | 20.056 | 26.008 | 25.378 | 25.667 | 0.971 |

**两组 PSNR 不能直接比较**：像素网格差一倍，更高分辨率要还原更多高频，PSNR 天生更低。
能说的是：4K 下反传修复与 mask 处理仍然成立（静态区依旧远高于全图、SSIM 0.97、
目视细节——百叶窗、白板字、桌面边缘——都还在，没有气泡）。真正的 4K 上限还需要在 4K 上
做 leak / cap 的对照，本轮没做。

## 复现

```powershell
# 全景：修好反传 + 静态背景优先的 mask 训练（本轮最佳）
.\build\aetherscan\Release\aetherscan.exe `
  --images D:/BaiduNetdiskDownload/VID_20260911_145523_00_007_dataset/images `
  --splat-dataset D:/BaiduNetdiskDownload/VID_20260911_145523_00_007_dataset/sparse/0 `
  --output artifacts/native_camera_20260916_round2/panorama_fix_mask_leak000/model.ply `
  --splat-strategy adc_igs --splat-iterations 15000 --splat-densification-cap 1000000 `
  --splat-max-resolution 1920 --splat-progressive-resolution=false `
  --splat-ppisp=false --splat-bilateral-grid=false `
  --splat-use-mask=true --splat-alpha-mode masked --splat-alpha-leak-weight 0 `
  --splat-undistort=false --splat-depth-normal-weight 0 `
  --splat-mv-geo-weight 0 --splat-mv-ncc-weight 0 `
  --splat-eval-split-every 8 --splat-prefetch-views=0 `
  --splat-cache-auto=false --splat-device-cache-mb=0

# 鱼眼：抬高容量上限
.\build\aetherscan\Release\aetherscan.exe `
  --images D:/ScanVideo/alameda/images_2_dataset/images `
  --splat-dataset D:/ScanVideo/alameda/images_2_dataset/sparse/0 `
  --output artifacts/native_camera_20260916_round2/fisheye_fix_cap2m/model.ply `
  --splat-strategy adc_igs --splat-iterations 15000 --splat-densification-cap 2000000 `
  --splat-max-resolution 1920 --splat-progressive-resolution=false `
  --splat-ppisp=false --splat-bilateral-grid=false --splat-use-mask=false `
  --splat-undistort=false --splat-depth-normal-weight 0 `
  --splat-mv-geo-weight 0 --splat-mv-ncc-weight 0 `
  --splat-eval-split-every 8 --splat-prefetch-views=0 `
  --splat-cache-auto=false --splat-device-cache-mb=0
```

批量脚本 `artifacts/native_camera_20260916_round2/run_round2.py`，
离线 PNG 诊断 `artifacts/native_camera_20260916_round2/analyze_round2.py`，
每个 run 目录里都有 `command.json`、`binary_sha256.txt`、`train.log`、
`image_diagnostics.json`（含逐视角结果）。

## 仍未验证 / 后续

1. **equirect 的 depth/normal 分支沿用上游的透视参数化。** `ray_plane`/footprint normal
   是按 `u=x/z、v=y/z` 定义的（forward 与 backward 一致），而 `nJ` 那一套 Jacobian 本身是
   透视的。位置链里 `dL/du·∂u/∂t` 与这个参数化自洽，但**极坐标相机下这套几何定义是否合适
   没有验证**——本轮所有 run 都把 `--splat-depth-normal-weight` 与多视角权重设为 0，
   这条支线没被触发。要用深度/法线监督的全景，需要把它换成原生表达式并重新做 FD 验证。
2. **反传修复只验证了 RGB/协方差链**（单高斯、平滑探针、四个位置）。多高斯遮挡、真实
   全景大场景还没有端到端 FD 覆盖。
3. **噪声底**：同一二进制、同一 flags 重跑不是逐位可复现（第 1 步 loss 就有 ~4e-6 的相对
   差异，来自浮点累加顺序）。鱼眼同配置两次：`21.0935` / `0.707424` 与 `21.1612` /
   `0.707968`，即 **0.07 dB / 0.0005 SSIM** 的跑动噪声。因此鱼眼 cap2m 的 +0.09…+0.16 dB
   只是 1.5–2× 噪声，SSIM(+0.018) 与后院分段(+0.2…+0.6 dB) 才算证据；全景那几个
   +1.5 / +2.9 dB 的量级远在噪声之上。
   （注：把 `--splat-iterations` 从 15000 改成 1000 会让第 1000 步的指标差 0.2 dB——那是
   训练日程随总步数变化，不是噪声，不能拿来当重复性度量。）
   另：`aetherscan_splat_test` 的 `test_training_device_cache`（异步 CUDA 预取）在**有别的
   CUDA 进程正在运行或正在退出**时会假失败（`asynchronous neighbour changed supervision`），
   GPU 空载时 6/6 通过。跑测试前先确认没有训练在跑；这条与投影/mask 改动无关，但异步预取
   这条链路本身的抗干扰性值得单独查一次。
4. **mask 的"只在从未被静态视角看过时才罚透明"没有实现。** 本轮只做了权重消融；带可见性门控
   的版本仍是待办。
5. 鱼眼最差段（后院 DSC076x，17.4 dB）还没解决：可能是细枝/天空的容量与 SH 表达问题，
   需要单独实验（更高 cap、更长训练、按区域重采样）。
