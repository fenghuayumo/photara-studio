# GGGS C++ / TinyTensor 后端

## 当前实现状态（2026-07-22）

AetherScan 已有一条可编译、可前反向传播、可由 CLI 启动的 GGGS 训练路径：

```text
images → SfM → CPU/OpenMP PatchMatch MVS → fused dense cloud ─┐
images + COLMAP sparse model ──────────────────────────────────┤
                                                              ↓
  Gaussian 初始化 → GGGS CUDA forward/backward
  → fused 11×11 L1+SSIM → TinyTensor Adam → *_gggs.ply
```

这条路径不依赖 LibTorch、PyBind 或 Python 运行时。MVS 继续负责充分利用 CPU；Gaussian
投影、排序、混合、反向传播、损失与参数更新在 CUDA 上执行。

## 目录与职责

- `aetherscan/include/splat/`：公开的模型、相机、训练配置和训练器 API；
- `aetherscan/src/splat/rasterizer.cu`：TinyTensor tensor 与 GGGS 原生 CUDA API 的桥接；
- `aetherscan/src/splat/cuda_ops.cu`：参数激活、链式梯度、融合监督损失和融合 Adam；
- `aetherscan/src/splat/fused_ssim.cu`：从 Python fused-ssim 完整移植的 11×11 CUDA
  forward/backward；
- `aetherscan/src/splat/colmap.cpp`：COLMAP 文本/二进制相机、位姿、稀疏点和 track 加载；
- `aetherscan/src/splat/trainer.cpp`：点云初始化、动态 Gaussian 管理、训练和 PLY 导出；
- `aetherscan/third_party/gggs_reference/`：从 Python 参考工程移入的原始 CUDA
  rasterizer core，不包含 Torch/PyBind wrapper；
- `aetherscan/third_party/tinytensor/`：tensor 存储、CUDA 内存和基础运算。

模型使用可训练的世界坐标均值、log-scale、四元数、opacity logit 和最高三阶 SH。
稠密点云法线用于初始化 Gaussian 朝向，像素足迹用于初始化尺度，点色用于初始化 SH0。
训练损失包含与 `pygsplat/simple_trainer.py` 对齐的 `0.8 * L1 + 0.2 * SSIM` 光度项和可选
mask/alpha loss。mesh 模式默认在第 3,000 步同时启用权重 `0.05` 的 GGGS depth-normal、
权重 `0.02` 的多视图几何往返和权重 `0.6` 的平面单应 NCC。多视图几何通过 GGGS 原生
`sampleDepth` 前后向在相邻视图查询表面点，梯度同时回传参考深度、查询点以及邻视图
Gaussian；NCC 使用半像素 7×7 patch、鲁棒 diffuse confidence 和深度/法线解析梯度。
此外默认启用 Mip-Splatting 3D filter，按所有训练相机可见距离扩大亚像素 Gaussian，并以
行列式比例补偿 opacity；filter 在拓扑变化后和训练期间周期重算，并写入 PLY 的 `filter_3D`。

depth-normal self-consistency：从 median depth 反投影中心四邻域，以
`cross(dy, dx)` 求 depth normal，
再最小化 `mean(1-dot(rendered_normal, depth_normal))`；梯度同时回传 rendered normal 和
median depth。所有参数通过显式
GGGS backward 和 TinyTensor Adam 更新。SSIM 已完整移植 Python fused-ssim 的 11×11
Gaussian separable CUDA forward/backward 和 `padding="valid"` 边界语义，不依赖 LibTorch；
mask 会在计算 L1/SSIM 前同时作用于预测图和目标图。13×13 确定性输入的 loss、中心梯度、
梯度和与绝对梯度和均有严格数值回归，误差阈值为 `2e-5`。

## 构建与运行

需要 CUDA Toolkit 和 glm：

```powershell
cmake -S . -B build -DAETHERSCAN_ENABLE_CUDA=ON -DAETHERSCAN_ENABLE_GGGS=ON
cmake --build build --config Release --target aetherscan -- /m
```

完整流程：

```powershell
build/aetherscan/Release/aetherscan.exe `
  --images data/images `
  --output output/scene.mvs `
  --gggs `
  --gggs-iterations 10000
```

mesh 质量路径会随 depth-normal loss 自动启用 Mip-Splatting 3D filter；filter 不再是
appearance-only 3DGS 的独立开关。可用 `--gggs-depth-normal-weight`、
`--gggs-mv-geo-weight`、`--gggs-mv-ncc-weight`、`--gggs-mv-neighbors`、
`--gggs-mv-pixel-noise` 和 `--gggs-geometry-from-iter` 调整；将两个 multi-view
weight 设为 0 可做关闭 A/B。

`--gggs` 隐含 `--dense`。输出包括 `scene_dense.ply` 和 `scene_gggs.ply`。当前 Gaussian
PLY 保存训练参数（opacity 和 scale 仍是 logit/log-domain），可用于检查训练结果和后续
viewer/mesh-extraction 接入。

稠密 MVS 输入默认从 fused cloud 均匀选取最多 500,000 个初始 Gaussian，可用
`--gggs-max-gaussians N` 修改，`0` 表示使用全部 dense points。稠密点云已经具有高采样密度，
因此默认关闭动态致密化；显式选择 `--gggs-strategy dense_adaptive` 时，训练器会小批量回收
低 opacity 点，并把预算重新分配到高屏幕梯度/大投影贡献区域。稀疏 COLMAP 输入则启用原有
动态 Gaussian 管理。训练结束还会保存
第一个、中间和最后相机的
`*_gggs_view_*.png`，并在日志记录 PSNR、MAE 和 alpha coverage。

可直接跳过内部 SfM/MVS，加载 COLMAP 相机位姿和稀疏点云。稀疏输入默认 30,000 步
（致密化约至 15k），稠密 MVS 仍默认 10,000 步：

```powershell
aetherscan --images D:\ScanVideo\ori_img\images `
  --colmap D:\ScanVideo\ori_img `
  --output out\scene.mvs `
  --gggs-strategy default `
  --gggs-max-gaussians 500000 `
  --gggs-densification-cap 4000000
```

调试基础优化收敛时，可用 `--gggs-densification=false` 固定 COLMAP 初始化的
Gaussian 数量；此模式禁用 split、prune 和 opacity reset，只验证 RGB、mask loss、
光栅化反向与 CUDA Adam 的参数优化。因为没有 prune，固定拓扑模式也会在整个训练中
保留 `--gggs-max-scale-fraction` 上限；启用致密化后不逐步硬夹 scale，而与 pygsplat
一样由 refine 阶段按 `0.1 * scene_scale` 清理过大的 Gaussian。不能先夹到同一个
阈值再比较，否则 `exp(log(scale))` 的浮点误差会误删边界 Gaussian。

若要验证“结构先优化、随后只收敛外观”，可加
`--gggs-structure-freeze-iter 5000`。到达该步后 means、scale、quaternion、opacity
保持不变，但 SH/颜色继续使用 CUDA Adam 更新。

### `ori_img` 收敛回归（2026-07-22）

在 `D:\ScanVideo\ori_img` 的 76 张图、83,993 个 COLMAP 初始点上，默认策略训练
5,000 步后得到 81,543 个 Gaussian。RGB+mask loss 从 1.62741 降到 0.00939；三个
固定视角的 masked PSNR 为 30.00 / 35.65 / 35.38 dB，前景像素 PSNR 为
24.84 / 30.54 / 30.24 dB。对应 Python GGGS 回归为 82,538 个 Gaussian、PSNR
34.787 dB。

本回归的关键修复是每次 GGGS backward 前清零 geometry-gradient scratch。参考
FasterGS wrapper 使用 `resizeFunctional<true>`；若 C++ 使用未初始化的 pooled memory，
`atomicAdd` 会累积旧的 conic/opacity 梯度，使 opacity 中位数错误升到约 0.96，loss
无法下降。回归测试会连续执行两次相同 backward 并比较 opacity gradient，防止复发。

同一修复也已在完整稠密路径上回归：preview MVS 融合 2,200,863 个点，均匀选择
500,000 个 Gaussian，关闭动态致密化，并在前 1,000 步优化结构参数、后续仅优化逐级开放的
SH。10,000 步训练耗时 47.13 秒，随机训练视图上的总 loss 从 0.06081 降到 0.00433；三个
固定视角在 1,000 / 5,000 / 10,000 步的 masked PSNR 分别为：

| 步数 | view 0 | view 38 | view 75 |
|---:|---:|---:|---:|
| 1,000 | 29.11 dB | 35.64 dB | 33.51 dB |
| 5,000 | 29.80 dB | 36.68 dB | 34.87 dB |
| 10,000 | 30.01 dB | 36.86 dB | 35.43 dB |

10,000 步的严格前景像素 PSNR 为 24.82 / 31.71 / 30.25 dB。渲染未出现几何破洞、针刺或
opacity 塌缩；因此稠密输入的默认 10,000 步、500,000 Gaussian 固定拓扑配置可以作为当前
质量基线，暂不需要启用 `dense_adaptive`。

稀疏 COLMAP 路径默认与 pygsplat 一致：使用原始三近邻 RMS scale 和随机 raw
quaternion；光栅化前才归一化 quaternion。稀疏云可能包含 KNN scale 很大的离群点，
正常训练由后续 prune 移除；固定拓扑稳健性实验可显式加
`--gggs-constrain-scales=true`。`--gggs-max-scale-ratio` 默认 0，不额外限制轴比。

加载器自动解析根目录、`sparse/`、`sparse/0/` 或直接 model 目录中的 `.bin` / `.txt`。
当前精确支持 `SIMPLE_PINHOLE`、`PINHOLE`、`SIMPLE_RADIAL`、`RADIAL`、`OPENCV`；无法由
现有 Brown/pinhole 相机准确表达的 fisheye、FOV、FULL_OPENCV 会明确拒绝，不做静默近似。

当前支持四种策略，前三种用于稀疏输入，`dense_adaptive` 专用于 MVS 稠密输入：

| `--gggs-strategy` | 统计与增长 | 默认调度 |
|---|---|---|
| `default` | 平均屏幕梯度；小 Gaussian clone，大 Gaussian split；opacity reset | 500–15k，每 100 步 |
| `adc_plus` | 最大 refine weight、实际 alpha 贡献可见度和屏幕半径；预算回收、ADC split/decay/noise | 全程每 200 步；15k 后停止额外增长，只回收低 opacity 点 |
| `adc_igs` | ADC+ pruning + Gumbel Top-K + 投影优先级 + 最大轴二分 | 增长至 15k，裁剪至 25k，每 200 步 |
| `dense_adaptive` | 每轮最多回收 1% 低贡献点（异常/越界点另行删除）、额外增长 0.5%，按最大 refine weight 和投影半径做表面切平面二分；不使用 ADC noise/decay | 1k–5k，每 500 步；1k 后冻结结构 Adam，仅继续 SH |

所有策略均受 `--gggs-densification-cap` 硬上限约束，新增/裁剪数量写入训练日志。
ADC+ 的“可见”要求 Gaussian 通过 alpha/transmittance 测试并实际参与至少一个像素合成；
仅投影进相机视锥但被前景遮挡的 Gaussian 不再累计支持度、参与回收采样或注入探索噪声。
稀疏 ADC+ 默认还将轴比限制为 `100`，抑制只对训练相机正面成立、在范围外视角变成漂浮片的
极薄 Gaussian；可用 `--gggs-max-scale-ratio 0` 显式关闭，或传入其他上限。

稠密点云建议配置：

```powershell
aetherscan --images D:\ScanVideo\ori_img\images --output out\scene.mvs `
  --dense --gggs --gggs-strategy dense_adaptive `
  --gggs-max-gaussians 500000 --gggs-densification-cap 600000
```

GGGS 默认启用 Mask 训练，只重建主体并抑制背景；可用 `--gggs-use-mask=false` 显式关闭。
Mask 复用 `--masks` 指定的目录（默认寻找 `images/` 的同级 `masks/`），并支持 Python
数据集相同的 stem 匹配、`.png/.jpg/.jpeg` 大小写扩展名，以及双线性软覆盖重采样，
找不到独立 mask 时回退到源图 alpha channel：

```powershell
aetherscan --images D:\ScanVideo\ori_img\images --output out\scene.mvs `
  --dense --gggs --gggs-use-mask `
  --gggs-alpha-mode transparent --gggs-match-alpha-weight 0.25 `
  --gggs-ssim-weight 0.2 `
  --gggs-min-scale-fraction 0.0001 --gggs-max-scale-fraction 0.002 `
  --gggs-max-scale-ratio 10
```

- `transparent`（默认）：前景 RGB loss + `0.25 * BCE(render_alpha, mask)`；
- `masked`：仅前景 RGB loss + 背景 alpha leakage penalty。

主体模式要求每个训练视图都有匹配 mask 或源图 alpha channel；任何视图缺失都会立即报错，
避免背景意外进入模型。日志会单独输出 `rgb/alpha/depth/normal` 四项 loss。
默认将最大 Gaussian 尺度限制为场景范围的 `0.002`，防止 splat 扩张到背景并形成不透明
雾层；可通过 `--gggs-max-scale-fraction` 显式调整。
纯光度稠密输入可在前 1,000 步 warm-up 后冻结 mean/scale/quaternion/opacity Adam；启用
depth-normal 几何目标时会自动取消该冻结，使第 7,000 步后的几何梯度能够继续更新结构参数。
SH 颜色参数始终继续训练，`dense_adaptive` 的受限回收和切平面二分也仍可执行。

`D:\ScanVideo\ori_img` 的 76/76 个 sibling masks 已用 `transparent` 模式完成 preview MVS +
500,000 Gaussian / 300 步真实回归：alpha loss 从 0.03934 降到第 200 步的 0.00223，三个
mask 内诊断视角 PSNR 为 22.87 / 22.32 / 24.55 dB，导出 PLY 的 31,000,000 个 float 标量
全部 finite。300 步仅用于验证 mask 数据链路和梯度，不能替代正式 10,000 步训练。

## 实拍端到端验证（2026-07-21）

`D:\ScanVideo\ori_img\images` 的 76 张 1000×1000 图片已完成实际验证：76/76 相机注册，
SfM reprojection RMS 为 0.591 px；default MVS 用 32 workers 在约 104 秒内融合出 2,097,765
个点。针对“训练越久反而越糊”的问题，在同一个 preview MVS、500,000 Gaussian、4000 步、
76/76 mask 配置上做了修复前后 A/B：

| 版本 | 训练时间 | view 0 PSNR | view 38 PSNR | view 75 PSNR |
|---|---:|---:|---:|---:|
| 修复前 | 33.39 s | 20.82 dB | 20.03 dB | 20.89 dB |
| 收敛修复后 | 34.40 s | 22.27 dB | 22.42 dB | 23.55 dB |

修复包括：Adam epsilon 对齐到 `1e-15`，mean 学习率按 scene scale 缩放并指数衰减，加入
L1+SSIM 光度损失，限制绝对 scale 范围与单 Gaussian 三轴最大比例。修复前 PLY 的轴比例
p90/p99.9 分别达到 `171.9 / 170011`，产生明显针刺；修复后降为 `1.77 / 10.0`，且最小
scale 从 `8.89e-8` 提升到 `8.89e-5`。两个版本 31,000,000 个 float 标量均为 finite，SSIM
在当时的近似实现中只增加约 3% 训练时间；当前完整 11×11 fused CUDA 实现需要单独做正式
性能基准，不能沿用该旧数据。

训练日志会输出当前随机采样的 `view`，因此单步 loss 不应被误读为同一张图上的单调曲线。
当前渲染已明显稳定，但仍有 floaters；修复后约 23.6% Gaussian 的 opacity 低于 `1/255`，
该 A/B 使用的是动态管理加入前的固定 Gaussian 版本，不能视为最终商业画质。

`dense_adaptive` 已在相同 76 视角数据上完成 preview MVS + 5,000 步真实验证：以 500,000
Gaussian 启动、硬上限 600,000，8 次受限 refine 后得到 520,150 Gaussian，GGGS 训练耗时
23.79 秒。第 1,000→5,000 步三个固定视角的前景内 PSNR 从 22.36 / 22.47 / 25.25 dB
变为 22.68 / 23.07 / 25.16 dB，alpha coverage 最终保持 0.323 / 0.333 / 0.325，没有出现
结构/opacity 塌缩。当前 C++ 诊断采用更严格的“仅前景像素平均”口径；按 pygsplat 将 mask
外像素置零后再对整图平均的口径，同一最终误差约为 27.6 / 27.8 / 30.0 dB。

新增 COLMAP/ADC-IGS 路径已在同一数据集实测：文本模型加载 76 个相机和 83,993 个稀疏点，
以 10,000 个 Gaussian 启动、硬上限 15,000，1000 步训练在第 800 步新增 3,987、裁剪 21，
最终为 13,966 个 Gaussian；三个诊断视角 PSNR 为 14.38 / 11.93 / 14.26 dB。该短跑用于
验证数据链路、动态 Tensor/Adam 状态和硬上限，并不代表 30k 步最终画质。

CUDA 前反向冒烟测试：

```powershell
ctest --test-dir build -C Release -R aetherscan.splat.rasterizer --output-on-failure
```

### 3D filter + multi-view 回归（2026-07-22）

`D:\ScanVideo\ori_img` 的 1,249,605 点稠密初始化、76 视角、10,000 步完整训练耗时
283.6 秒；第 3,000 步后启用 `sampleDepth` 几何、NCC 和 depth-normal，单步约 60–70 ms，
每步约 11–31 万有效 multi-view 像素。最终 TSDF mesh 为 890,029 顶点 / 1,766,231 面；
pygsplat pseudo-reference 为 887,781 / 1,762,164。

按两侧各 750,000 个均匀表面样本做精确点到三角面距离，voxel=`0.00408112`：

| 版本 | symmetric Chamfer-L1 | F@0.25 voxel | F@0.5 voxel | F@1 voxel | F@2 voxel |
|---|---:|---:|---:|---:|---:|
| depth/TSDF 修复基线 | 0.002866 | 0.2934 | 0.5280 | 0.8040 | 0.9517 |
| + 3D filter + multi-view | **0.000566** | **0.8722** | **0.9730** | **0.9936** | **0.9983** |

这里 pygsplat mesh 只是无真值条件下的实现对齐参照，不等价于真实几何精度。新 mesh 到输入
稠密点的距离 p50/p90/p99 为 `0.001889 / 0.006033 / 0.015141`，也与 pygsplat 的
`0.001873 / 0.005981 / 0.015087` 基本一致。

## 性能设计

- MVS 沿用现有并发 view/tile 调度和 OpenMP CPU 并行；
- rasterizer 沿用 GGGS 的 tile binning、CUDA 排序和前反向 kernel；
- scale/quaternion/opacity 激活和对应链式梯度使用融合 kernel；
- L1/depth/normal/alpha 主损失在一次像素 kernel 中生成 loss 与四种输出梯度；SSIM 使用完整
  的 11×11 separable forward/backward CUDA kernel，并在设备端融合 L1、valid-map 与梯度链；
- Adam 的一阶矩、二阶矩和参数更新使用单 kernel，SH0/SH-rest 在同一 launch 中使用不同学习率；
- forward context 保留 geometry/binning/image/tile buffer，backward 不重复预处理。

目前为了避免每步磁盘 IO，训练开始时会把全部训练图、mask，以及启用直接 MVS 监督时所需的
MVS depth/normal 常驻 GPU。
中小场景速度优先时合理；大场景需要改成 pinned-host 预取、有限 VRAM cache 和多 CUDA stream，
否则显存会随视图数量线性增长。

## 当前几何交付与后续工作

当前版本已打通 GGGS mesh extraction：训练后按原图分辨率渲染每个相机的 median depth、normal
和 alpha；存在输入 mask 时以 mask 为准，否则回退到 alpha 0.5，与 `gs2mesh.py` 一致。随后按
`max_depth=2*scene_extent`、`voxel=max_depth/2048`、`sdf_trunc=4*voxel` 做 Open3D-compatible
稀疏体素块投影融合，并用标准 Marching Cubes 抽取、保留最大连通分量。`--gggs --mesh` 会把
`active_mesh` 切换为 `*_gggs_mesh.ply`，不再使用 projective MVS patch mesh。

若构建时找到 CGAL，可通过非零 `--mesh-target-faces` 显式调用 asdiff_render 的 Instant Meshes
field-aligned remesh，再调用 `asdiff::mesh::repair_and_decimate`；默认保留原生 Marching Cubes 网格；
Instant 的 quad-dominant 目标会按最终三角面数的一半设置，remesh 与 CGAL 后都会剔除微小
边连通碎片。
`--mesh-remesh=false` 可跳过重拓扑做 A/B，`--mesh-target-faces 0` 可关闭整个后处理。

`D:\ScanVideo\ori_img\images` 的 76 视角稠密初始化 / 500,000 Gaussian / 10,000 步实测中，
第 7,000 步开启几何项后 normal loss 从 `0.004108` 降至 `0.000700`；三个固定视角 masked
PSNR 为 `30.44 / 36.96 / 35.98 dB`。depth-normal 提取过滤拒绝 `146,061 / 7,420,379`
个可比较样本（1.97%）；最终网格为 515,644 顶点、995,937 三角形、单边连通分量、绕序一致、
0 条非流形边。网格仍有 37,689 条开放边，因此当前交付是开放表面而不是 watertight 实体。

仍未完成的产品工作：

- 3D filter、PatchMatch sample-depth、多视图几何与 plane-warp NCC 已移植；尚未加入 Python
  glossy normal TV 分支，AetherScan MVS depth/normal 直接监督仍为可选项；
- ADC-IGS 当前以 raster refine weight、可见度和屏幕半径构造投影优先级，尚未实现 Python
  版本基于 Sobel/逐像素误差反投影的完整 edge/error ownership map；
- 未实现 checkpoint/resume、out-of-core view cache、多 GPU 和 mixed precision；
- 还需要更多数据集的质量基准和 30k 步动态致密化稳定性验证。

建议后续按“TSDF 边界/体素参数回归 → ADC-IGS pixel ownership → checkpoint/streaming →
多 GPU/图捕获”的顺序推进。

## 许可证边界

参考 GGGS CUDA 文件的源注释限定为非商业研究/评估用途，而 Python 参考目录中没有附带其
所指向的完整 `LICENSE.md`。因此 `gggs_reference` 只能视为原型验证代码，不能直接作为商业
RealityScan 类产品发布。详细记录见 `aetherscan/third_party/gggs_reference/NOTICE.md`；商业化前
必须取得明确授权，或以洁净室方式替换该 rasterizer core。

fused SSIM 移植源为 `dvsplat_utils/fused-ssim/ssim.cu`，按其 MIT 许可保留版权与许可文本，见
`aetherscan/third_party/fused_ssim/LICENSE`；这不改变上述 GGGS rasterizer core 的授权边界。
