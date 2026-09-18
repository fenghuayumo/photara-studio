# AetherScan Splat C++ / TinyTensor 后端

## 当前实现状态

AetherScan 已有一条可编译、可前反向传播、可由 CLI 启动的 Gaussian splat 训练路径。
当前实现参考了 GGGS 的几何监督与 CUDA rasterizer，但已组合 ADCPlus、GaussianWrapping
normal field、PAM/TSDF 等独立能力，因此公开接口统一称为 `splat`，不再以 GGGS 命名：

```text
images → SfM → CPU/OpenMP PatchMatch MVS → fused dense cloud ─┐
images + COLMAP sparse model ──────────────────────────────────┤
                                                              ↓
  Gaussian 初始化 → Splat CUDA forward/backward
  → fused 11×11 L1+SSIM → TinyTensor Adam → *_splat.ply
```

这条路径不依赖 LibTorch、PyBind 或 Python 运行时。MVS 继续负责充分利用 CPU；Gaussian
投影、排序、混合、反向传播、损失与参数更新在 CUDA 上执行。

外部相机对齐数据通过 `--splat-dataset` 传入，程序会从路径自动识别 COLMAP、
RealityCapture 或 OpenMVS。Gaussian 写出格式看 `--output` 后缀：`.sog` / `.spz` /
`.glb` 写到该文件，其它输出路径旁生成 `{stem}_splat.ply`。

## 目录与职责

- `aetherscan/include/splat/`：公开的模型、相机、训练配置和训练器 API；
- `aetherscan/src/splat/rasterizer.cu`：TinyTensor tensor 与 GGGS 原生 CUDA API 的桥接；
- `aetherscan/src/splat/cuda_ops.cu`：参数激活、链式梯度、融合监督损失和融合 Adam；
- `aetherscan/src/splat/bilateral_grid.cu`：仿射双边网格颜色校正（训练期、空间变化）；
- `aetherscan/src/splat/ppisp.cu`：PPISP 颜色校正，默认 `no_crf_no_vig`（曝光 +
  白平衡单应），可选暗角 / CRF 布局；
- `aetherscan/src/splat/fused_ssim.cu`：从 Python fused-ssim 完整移植的 11×11 CUDA
  forward/backward；
- `aetherscan/src/splat/colmap.cpp`：COLMAP 文本/二进制相机、位姿、稀疏点和 track 加载；
- `aetherscan/src/splat/trainer.cpp`：点云初始化、动态 Gaussian 管理、训练和 PLY 导出；
- `aetherscan/third_party/gggs_reference/`：从 Python GGGS 参考工程移入的原始 CUDA
  rasterizer core，不包含 Torch/PyBind wrapper；
- `aetherscan/third_party/tinytensor/`：tensor 存储、CUDA 内存和基础运算。

模型使用可训练的世界坐标均值、log-scale、四元数、opacity logit 和最高三阶 SH。
稠密点云法线用于初始化 Gaussian 朝向，像素足迹用于初始化尺度，点色用于初始化 SH0。
训练损失包含与 `pygsplat/simple_trainer.py` 对齐的 `0.8 * L1 + 0.2 * SSIM` 光度项和可选
mask/alpha loss。室外视频的自动曝光 / 白平衡漂移可以由两种训练期颜色校正吸收，两者都
移植自 spirula-studio，都不写入导出模型，且都默认关闭：PPISP
（`--splat-ppisp --splat-ppisp-type no_crf_no_vig|no_crf|original`）
与仿射双边网格（`--splat-bilateral-grid`）。每视图模型与场景颜色之间存在退化，因此
PPISP 把跨视图曝光/色彩均值锚定到单位变换，双边网格在每次更新后把每视图网格均值
投影回 identity 仿射——网格只表达空间变化，全局曝光归 PPISP。同时开启时默认先 PPISP、
再双边网格。实测：这些功能提升“按帧曝光对齐后”的重建指标，但不改变导出模型的留出指标，
详见 [颜色校正报告](SPLAT_COLOR_CORRECTION_20260914.md)。两者的每像素 kernel 与光度损失
路径的开销见 [颜色校正性能报告](SPLAT_COLOR_CORRECTION_PERF_20260916.md)：训练步内两项合计
约 1.5 ms（1728×1120、约 4 万高斯），profiler 的 `colour_forward_ms` / `colour_backward_ms`
就是这两项的耗时。mesh 模式默认在第 3,000 步同时启用权重 `0.05` 的 splat depth-normal、
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

### GaussianWrapping normal field 与 PAM

训练器可学习 GaussianWrapping 的四通道 `gaussian_features_0..3`：前三维经归一化得到
法线方向，第四维经 `tanh` 学习朝向符号。默认在第 8,001 步按当前 Gaussian 最短轴重置
方向和符号，再按 alignment 权重 `0.05` 与默认 depth ratio `0.6` 对齐由 median depth
微分得到的法线；特征拥有独立 Adam
状态，并随 clone、split、prune 一起维护。可用 `--splat-normal-field=false` 关闭，或用
`--splat-normal-field-weight`、`--splat-normal-field-depth-ratio`、
`--splat-normal-field-from-iter` 调整。

构建时找到 CGAL 后，`--mesh --mesh-method pam` 走 GaussianWrapping 原生的两级
`tetra_triangulation`，不依赖 TSDF。第一级为每个入选 Gaussian 生成“中心 + learned-normal
方向 3σ 偏移”的两个 pivot，计算多视图 occupancy 后做 Delaunay 与 Marching Tetrahedra，
得到 `*_pam_pivot_mesh.ply`。第二级按 `face_area / visible_camera_distance²` 从 pivot mesh
采样，用 32 邻域 normal field 将候选投影到 occupancy 等值面，再次 Delaunay，对四面体内部
occupancy 采样分类并提取 occupied/free 边界。Mask 前景视角参与最小 occupancy 融合，明确
背景投空，视锥外视角弃权。

`--pam-pivot-max-points` 控制第一级 pivot 顶点上限，`--pam-pivot-std-factor` 控制法线偏移；
`--pam-max-points`、`--pam-refinement-steps`、`--pam-neighbors` 和
`--pam-points-per-tetrahedron` 控制第二级。`--pam-occupancy-iso-value 0.3` 对应
GaussianWrapping `ours` rasterizer 的 `iso_surface_value=0.2`，适合自行车辐条等半透明细结构；
默认 0.5 更保守。`--pam-gaussian-seed-fraction` 可选地把部分候选直接从 Gaussian 生成，
默认 0 表示忠实使用 pivot mesh。PAM 始终处理完整场景，不使用相机 focus、自动 ROI、场景
SubjectBounds 或外部凸包裁剪 pivot、候选点和最终 mesh。中间候选保存为
`*_pam_candidates.ply`。Python 模块对应提供 `PamOptions` 与 `extract_pam`。

bicycle 回归可直接使用 `images_2`：

```powershell
build/aetherscan/Release/aetherscan.exe `
  --images D:\Models\360_extra_scenes_1\bicycle\images_2 `
  --splat-dataset D:\Models\360_extra_scenes_1\bicycle\sparse\0 `
  --output artifacts\bicycle\scene.mvs --splat `
  --splat-strategy adc_plus --splat-iterations 30000 `
  --splat-max-resolution 0 --splat-progressive-resolution=false `
  --mesh --mesh-method pam --pam-occupancy-iso-value 0.3
```

这里直接读取 `images_2`，`--splat-max-resolution 0` 表示保持这些文件的 2473×1643 像素，
不会先读原图再动态缩放。COLMAP 相机内参会按所选图像文件的实际宽高和像素中心规则同步标定。

## 构建与运行

需要 CUDA Toolkit 和 glm：

```powershell
cmake -S . -B build -DAETHERSCAN_ENABLE_CUDA=ON -DAETHERSCAN_ENABLE_SPLAT=ON
cmake --build build --config Release --target aetherscan -- /m
```

完整流程：

```powershell
build/aetherscan/Release/aetherscan.exe `
  --images data/images `
  --output output/scene.mvs `
  --splat `
  --splat-iterations 10000
```

mesh 质量路径会随 depth-normal loss 自动启用 Mip-Splatting 3D filter；filter 不再是
appearance-only 3DGS 的独立开关。可用 `--splat-depth-normal-weight`、
`--splat-mv-geo-weight`、`--splat-mv-ncc-weight`、`--splat-mv-neighbors`、
`--splat-mv-pixel-noise` 和 `--splat-geometry-from-iter` 调整；将两个 multi-view
weight 设为 0 可做关闭 A/B。

`--splat` 隐含 `--dense`。输出包括 `scene_dense.ply` 和 `scene_splat.ply`。当前 Gaussian
PLY 保存训练参数（opacity 和 scale 仍是 logit/log-domain），可用于检查训练结果和后续
viewer/mesh-extraction 接入。

稠密 MVS 输入使用全部 fused points 初始化 Gaussian。稠密点云已经具有高采样密度，
因此关闭动态致密化。稀疏 COLMAP 输入默认使用 ADC-IGS 动态 Gaussian 管理，增长受
`--splat-densification-cap` 约束（默认 1,000,000）。训练结束还会保存
第一个、中间和最后相机的
`*_splat_view_*.png`，并在日志记录 PSNR、MAE 和 alpha coverage。

可直接跳过内部 SfM/MVS，加载 COLMAP 相机位姿和稀疏点云。稀疏输入默认 30,000 步
（ADC 致密化持续到 95% 进度），稠密 MVS 仍默认 10,000 步：

```powershell
aetherscan --images D:\ScanVideo\ori_img\images `
  --splat-dataset D:\ScanVideo\ori_img `
  --output out\scene.mvs `
  --splat-strategy adc_igs `
  --splat-densification-cap 1000000
```

调试基础优化收敛时，可用 `--splat-densification=false` 固定 COLMAP 初始化的
Gaussian 数量；此模式禁用 split 和 prune，只验证 RGB、mask loss、
光栅化反向与 CUDA Adam 的参数优化。因为没有 prune，固定拓扑模式也会在整个训练中
保留 `--splat-max-scale-fraction` 上限；启用致密化后不逐步硬夹 scale，而与 pygsplat
一样由 refine 阶段按 `0.1 * scene_scale` 清理过大的 Gaussian。不能先夹到同一个
阈值再比较，否则 `exp(log(scale))` 的浮点误差会误删边界 Gaussian。

若要验证“结构先优化、随后只收敛外观”，可加
`--splat-structure-freeze-iter 5000`。到达该步后 means、scale、quaternion、opacity
保持不变，但 SH/颜色继续使用 CUDA Adam 更新。

稀疏 COLMAP 路径默认与 pygsplat 一致：使用原始三近邻 RMS scale 和随机 raw
quaternion；光栅化前才归一化 quaternion。稀疏云可能包含 KNN scale 很大的离群点，
正常训练由后续 prune 移除；固定拓扑稳健性实验可显式加
`--splat-constrain-scales=true`。`--splat-max-scale-ratio` 默认 0，不额外限制轴比。

加载器自动解析根目录、`sparse/`、`sparse/0/` 或直接 model 目录中的 `.bin` / `.txt`。
当前支持 `SIMPLE_PINHOLE`、`PINHOLE`、`SIMPLE_RADIAL`、`RADIAL`、`OPENCV`，以及原生
`OPENCV_FISHEYE` / `SIMPLE_RADIAL_FISHEYE` / `RADIAL_FISHEYE` 和 `EQUIRECTANGULAR`
（别名 `SPHERICAL`）。鱼眼和全景默认在原始图像上训练，不先去畸变成针孔；
`--splat-undistort` 可以把鱼眼重新采样到针孔工作相机。`FOV`、`FULL_OPENCV`、
`THIN_PRISM_FISHEYE` 仍会明确拒绝。全景无法去畸变。非针孔相机上会跳过
depth-normal 与多视图 NCC/几何项，RGB+SSIM 仍照常训练。

CLI 暴露两个稀疏输入策略，默认 `adc_igs`：

| `--splat-strategy` | 统计与增长 | 默认调度 |
|---|---|---|
| `adc_plus` | 最大 refine weight、实际 alpha 贡献可见度和屏幕半径；预算回收、ADC split/decay/noise | 全程每 200 步，95% 进度截止 |
| `adc_igs` | SSIM 对比度/结构误差图与世界梯度混合评分、屏幕梯度门槛、要求两个不同相机贡献、按分数迁移；增长候选排除已选中父点 | 与 ADC+ 相同：每 200 步、第一个窗口起，一直到 `max(14000, N−2500)` |

`adc_igs` 的差异只在"什么是证据、按什么排序增长"，几何（分裂算子、refine 间隔、
屏幕上限）与 `adc_plus` 共用。此前它使用"每次 refine 固定 +2.5%、每 100 步"的速率
增长与保持协方差的最大轴二分，并在精修时硬裁剪屏幕半径；把这套几何换成 ADC+ 的
之后，同一 7k 口径下留出 PSNR 从 21.484 提升到 21.908、SSIM 0.9425→0.9448、
近相机代理指标从 997 降到 255（0.5 阈值），并且优于 ADC+ 本身（21.791/0.9448/793）。
数值、命令与对照见 `docs/ADC_IGS_FOG_DIAGNOSIS_20260915.md`。

所有策略均受 `--splat-densification-cap` 硬上限约束，新增/裁剪数量写入训练日志。
ADC+ 的“可见”要求 Gaussian 通过 alpha/transmittance 测试并实际参与至少一个像素合成；
仅投影进相机视锥但被前景遮挡的 Gaussian 不再累计支持度、参与回收采样或注入探索噪声。
轴比默认不设上限（与 Brush 一致，三种策略相同）；需要抑制只对训练相机正面成立、在范围外
视角变成漂浮片的极薄 Gaussian 时，可显式传入 `--splat-max-scale-ratio R`（例如 `10`）。

稠密点云建议配置：

```powershell
aetherscan --images D:\ScanVideo\ori_img\images --output out\scene.mvs `
  --dense --splat
```

Splat 默认启用 Mask 训练，只重建主体并抑制背景；可用 `--splat-use-mask=false` 显式关闭。
Mask 复用 `--masks` 指定的目录（默认寻找 `images/` 的同级 `masks/`），并支持 Python
数据集相同的 stem 匹配、`.png/.jpg/.jpeg` 大小写扩展名，以及双线性软覆盖重采样，
找不到独立 mask 时回退到源图 alpha channel：

```powershell
aetherscan --images D:\ScanVideo\ori_img\images --output out\scene.mvs `
  --dense --splat --splat-use-mask `
  --splat-alpha-mode transparent --splat-match-alpha-weight 0.25 `
  --splat-ssim-weight 0.2 `
  --splat-min-scale-fraction 0.0001 --splat-max-scale-fraction 0.002 `
  --splat-max-scale-ratio 10
```

- `transparent`（默认）：前景 RGB loss + `0.25 * BCE(render_alpha, mask)`；
- `masked`：仅前景 RGB loss + 背景 alpha leakage penalty。

`masked` 的泄漏惩罚权重由 `--splat-alpha-leak-weight` 控制（默认 `1`，与 pygsplat 一致）。
动态遮挡物（例如持镜人）会挡住其他视角必须保持不透明的静态背景，惩罚与多视角一致性冲突时
会在人物位置长出半透明气泡；固定机位实拍建议 `--splat-alpha-leak-weight 0`，只屏蔽 RGB，
把被遮挡的背景交还给其他视角。详见
`docs/ADC_IGS_NATIVE_CAMERAS_R2_20260916.md`。

稀疏初始化的点数由 `--splat-init-point-budget` 限制（默认 `0` = 使用全部输入点）。稀疏 SfM
点云经常超过 `--splat-densification-cap`，此时第一次 refine 会把点云剪回上限、之后不再有
增长空间，压低初始化点数或抬高上限才能让 IGS 继续生长。

主体模式要求每个训练视图都有匹配 mask 或源图 alpha channel；任何视图缺失都会立即报错，
避免背景意外进入模型。日志会单独输出 `rgb/alpha/depth/normal` 四项 loss。
默认将最大 Gaussian 尺度限制为场景范围的 `0.002`，防止 splat 扩张到背景并形成不透明
雾层；可通过 `--splat-max-scale-fraction` 显式调整。
纯光度稠密输入可在前 1,000 步 warm-up 后冻结 mean/scale/quaternion/opacity Adam；启用
depth-normal 几何目标时会自动取消该冻结，使第 7,000 步后的几何梯度能够继续更新结构参数。
SH 颜色参数始终继续训练。

`D:\ScanVideo\ori_img` 的 76/76 个 sibling masks 已用 `transparent` 模式完成 preview MVS +
500,000 Gaussian / 300 步真实回归：alpha loss 从 0.03934 降到第 200 步的 0.00223，三个
mask 内诊断视角 PSNR 为 22.87 / 22.32 / 24.55 dB，导出 PLY 的 31,000,000 个 float 标量
全部 finite。300 步仅用于验证 mask 数据链路和梯度，不能替代正式 10,000 步训练。

CUDA 前反向冒烟测试：

```powershell
ctest --test-dir build -C Release -R aetherscan.splat.rasterizer --output-on-failure
```

## 性能设计

- MVS 沿用现有并发 view/tile 调度和 OpenMP CPU 并行；
- rasterizer 沿用 GGGS 的 tile binning、CUDA 排序和前反向 kernel；
- scale/quaternion/opacity 激活和对应链式梯度使用融合 kernel；
- L1/depth/normal/alpha 主损失在一次像素 kernel 中生成 loss 与四种输出梯度；SSIM 使用完整
  的 11×11 separable forward/backward CUDA kernel，并在设备端融合 L1、valid-map 与梯度链；
- Adam 的一阶矩、二阶矩和参数更新使用单 kernel，SH0/SH-rest 在同一 launch 中使用不同学习率；
- forward context 保留 geometry/binning/image/tile buffer，backward 不重复预处理。

数据集加载和训练缓存按流水线处理：COLMAP/OpenMVS/RealityCapture 读入时只探测图片头，
不为了记录分辨率解码像素；mask 预检只查看投影 mask、匹配路径或图片 alpha 元数据。
训练开始前先为首个 shuffled 窗口启动 host 解码，让它与相机/Adam 初始化重叠。训练中
图片解码可并行执行，预取深度按实测的 host 解码耗时与迭代耗时自适应（以
`--splat-prefetch-views` 为下限、默认 4，上限受 512MB 在飞视图约束），因此数据越大、
解码越慢，前瞻越长；已经驻留 host cache 的 view 不会重复解码，也不会占用前瞻槽位。
已完成的 host 视图会被提升为 pinned-host H2D CUDA 预取，再进入有限的 host/device
缓存。缓存淘汰不使用 LRU：一个 epoch 恰好访问每个 view 一次，LRU 会优先淘汰最近才需要
的条目；trainer 会把当前 epoch 的访问顺序发布给 loader，缓存改为淘汰"下次使用最远"的
条目，`--splat-prefetch-adaptive=false` 只关闭前瞻自适应，不改变淘汰策略。
上传流水线（默认开，`--splat-async-upload=false` 可关）：预取阶段已解码的视图会被打包进
pinned 缓冲，在独立非阻塞流上提前一迭代做 H2D，写到 **loader 自持的 device staging**（不是
显存池的块），认领时再在同一条 compute stream 上做一次 D2D 进缓存条目——因为 tinytensor 的
内存池在流/线程之间没有任何顺序保证，实测让 copy engine 直接写池内存在约 1/3 的运行里把数据
覆盖或触发 illegal memory access。实测每迭代约 -3%（alameda / iPhone，4000 迭代交错 A/B），
`get_wall_ms` 降低约 60%，10 次重复运行 0 崩溃 0 数据错（`SPLAT_VERIFY_UPLOAD=1` 校验）。
device 侧缓存预算可用 `--splat-device-cache-max-mb` 放大到 `--splat-device-cache-mb` 之上：
loader 按"这一迭代是否还需要从 host 上传"统计 device 命中率，命中率低于目标且显存有余量时逐级
放大，上限同时受数据集、空闲显存（扣除投影训练状态）与显存占比约束；下一完整统计窗口的端到端
迭代耗时至少改善 1% 才保留新增预算，否则自动回退。显存吃紧时先退还放大的部分，暂时的显存峰值
不会永久关闭后续检查。
默认 0 表示不放大——实测在 2MP / 1M Gaussian 这个量级上，上传已被计算掩盖，放大缓存反而因为占用
显存而略慢（alameda +0.8%、iPhone 生产长度 +2.4%），只有在迭代很短、loader 真在关键路径上时放大
才有意义。每次放大/收回都写入训练日志（`splat_data_cache device_budget_*`），命中率随
`--splat-profile-cuda=true` 的统计行输出。
JPEG/PNG 主路径分别直连 libjpeg/libpng，并直接写入最终 RGB/gray/alpha buffer；
FreeImage 只作为 TIFF/BMP 等其他格式和编码路径的 fallback。JPEG 训练视图根据目标
相机尺寸选择 1/2、1/4 或 1/8 DCT scale，避免 progressive-resolution 前半程先解码
完整原图再缩小。DCT 输出仍覆盖目标尺寸，剩余比例差由原有 bilinear source sampling
完成；畸变视角先将归一化射线投影到原始相机，再把像素坐标映射到 DCT-scaled bitmap。

### `Rasterizer::sample_depth` Nsight Compute 分析（2026-07-31）

在 RTX 5090 D v2、Nsight Compute 2025.1 上，使用 `ori_img` 和 `antman_nomask`
两组无 mask 数据对 `sampleDepthCUDA<2,8,5>` forward 与 `sampleDepthCUDA<2>`
backward 做了稳定窗口采集。训练配置为 ADC+、30,000 步、最大训练边长 1,000、
关闭 progressive resolution，几何项从第 3,000 步开启，每步从最多 8 个候选邻视角中
随机选择一个邻视角。`--splat-mv-neighbors 8` 只是候选池大小，并不会在一次迭代中执行
8 次 `sample_depth`；实际选择逻辑见
[`trainer.cpp`](../aetherscan/src/splat/trainer.cpp#L813)。

训练 profiler 的稳定窗口是第 15,001–30,000 步的 CUDA event 平均值：

| 数据集 | CUDA/iter | multi-view | sample forward | MV loss | sample backward | unproject | gradient merge |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ori_img` | 21.8625 ms | 12.7234 ms（58.20%） | 6.0128 ms | 3.2678 ms | 3.0834 ms | 0.3129 ms | 0.0465 ms |
| `antman_nomask` | 18.9966 ms | 10.7962 ms（56.83%） | 5.5498 ms | 2.2374 ms | 2.7406 ms | 0.2273 ms | 0.0411 ms |

对应日志为：

- `artifacts/ori_img_multiview_profile_30k_20260731/stdout.log`
- `artifacts/antman_multiview_profile_30k_20260731/stdout.log`

Nsight Compute 使用 kernel replay、19 passes 和默认 cold-cache 行为，在约第 15,000 步
捕获单次 forward/backward。这里的 kernel 时间用于判断微架构瓶颈，不能直接替代上表跨视角、
跨迭代的 CUDA event 平均值。报告保存在：

- `artifacts/nsight_sampledepth_20260731/ori_img/sample_depth_stable.ncu-rep`
- `artifacts/nsight_sampledepth_20260731/antman/sample_depth_stable.ncu-rep`

#### 全分辨率采样量与 tile batch 利用率

multi-view 在每个有效迭代中先把当前视图的整张 median-depth 图反投影到世界坐标，
再送入邻视角 `sample_depth`，见
[`trainer.cpp`](../aetherscan/src/splat/trainer.cpp#L821)。
GGGS reference 使用 16×16、256-thread CTA 和 `SAMPLE_BATCH_SIZE=2`，因此每个
duplicated-tile CTA 最多容纳 512 个点，配置见
[`config.h`](../aetherscan/third_party/gggs_reference/include/config.h#L22)。

由 forward SASS 中两组最终输出 store 的实际执行线程数可恢复进入 kernel 的点数：

| 数据集 | 图像/输入点 | 邻视角内采样点 | in-frustum | duplicated CTA | 分配槽位 | 槽位利用率 | 第 1 槽线程利用率 | 第 2 槽线程利用率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ori_img` | 1,000,000 | 801,883 | 80.2% | 2,929 | 1,499,648 | 53.5% | 79.3% | 27.7% |
| `antman_nomask` | 648,000 | 518,048 | 79.9% | 2,211 | 1,132,032 | 45.8% | 78.9% | 12.7% |

约 20% 的全分辨率点在投影到邻视角后被剔除，但它们仍参加了整图 unproject。剩余点按 tile
以 512 点为单位向上取整后，又产生 46.5%–54.2% 的空槽。第二个 per-thread sample
尤其稀疏：`ori_img` 中只有第一个 sample 数量的 34.9%，`antman_nomask` 中只有 16.0%；
但所有线程仍为双 sample 状态承担相同的寄存器分配。

tile/point binning 需要对完整 point list 做 scan、sort、range identification 和 batch
rounding，见
[`rasterizer_impl.cu`](../aetherscan/third_party/gggs_reference/src/rasterizer_impl.cu#L1378)
及
[`rasterizer_impl.cu`](../aetherscan/third_party/gggs_reference/src/rasterizer_impl.cu#L1423)。
因此减少整图输入量和减少 tile 尾块浪费是两个不同的优化层级。

#### Forward：寄存器受限的重复 tile traversal

| 指标 | `ori_img` | `antman_nomask` |
|---|---:|---:|
| 单 kernel replay 时间 | 6.477 ms | 6.754 ms |
| registers/thread | 99 | 99 |
| 实测 occupancy | 31.63% | 31.49% |
| SM throughput | 42.64% | 36.01% |
| L1 / L2 throughput | 12.27% / 5.07% | 13.13% / 4.25% |
| DRAM throughput | 0.31% | 0.30% |
| global sector 相对 ideal 冗余 | 55.6% | 55.9% |
| eligible warps/scheduler | 0.92 | 0.85 |
| branch target uniform | 89.40% | 91.58% |

forward 的主要问题不是 DRAM 带宽。99 registers/thread 使寄存器成为 occupancy 限制项：
每个 SM 只能驻留两个 256-thread CTA，理论 occupancy 上限约 33.3%。寄存器压力来自
per-sample `done/Depth/T/point_xy/last_contributor` 等状态，以及
`T_p[SAMPLE_BATCH_SIZE][SPLIT+1]`；当前配置实际为 `T_p[2][9]`，见
[`sample_forward.cu`](../aetherscan/third_party/gggs_reference/src/sample_forward.cu#L574)
和
[`sample_forward.cu`](../aetherscan/third_party/gggs_reference/src/sample_forward.cu#L694)。

每个 CTA 先遍历一次 Gaussian tile range 以确定初始深度和 `last_contributor`，然后执行
1 次完整 8-way split 和 4 次后续 refinement，总计 5 次 refinement traversal，见
[`sample_forward.cu`](../aetherscan/third_party/gggs_reference/src/sample_forward.cu#L700)
和
[`sample_forward.cu`](../aetherscan/third_party/gggs_reference/src/sample_forward.cu#L789)。
因此 forward 更准确的描述是“低 occupancy 下重复遍历同一 Gaussian range 的计算/控制流
瓶颈”，而不是显存带宽瓶颈。

`antman_nomask` 虽然输入像素少 35.2%，forward 却没有更快：它的 slot 利用率更差，
每 CTA 的平均 SASS 指令量又比 `ori_img` 高约 12.7%，说明 tile 内 Gaussian 数量和
`max_contributor` 分布比纯像素数量更能决定耗时，同时存在 CTA 间工作量不均衡。

#### Backward：local-memory/cache latency 与同步

| 指标 | `ori_img` | `antman_nomask` |
|---|---:|---:|
| 单 kernel replay 时间 | 3.201 ms | 3.105 ms |
| registers/thread | 39 | 39 |
| 实测 occupancy | 81.09% | 77.79% |
| SM throughput | 48.33% | 57.03% |
| L1 / L2 throughput | 80.95% / 60.16% | 84.28% / 50.07% |
| DRAM throughput | 1.87% | 1.87% |
| L2 theoretical local sectors | 292.42 M | 241.25 M |
| long-scoreboard stall / issue | 6.22 | 3.53 |
| barrier stall / issue | 4.24 | 4.60 |
| MIO throttle / issue | 2.75 | 3.32 |
| branch target uniform | 83.75% | 85.63% |

backward 的 occupancy 已经较高，但动态索引的 per-sample 局部数组产生了很大的 local-memory
流量。L2 hit rate 为 98%–99%、DRAM 仅使用 1.87%，说明数据大多命中 cache；真正限制
吞吐的是 local load/store 的 cache 管线和依赖延迟，而不是外部显存带宽。这与高
long-scoreboard、MIO throttle 和仅约 1.3–1.5 eligible warps/scheduler 一致。

backward 还会对 Gaussian contributor range 完整遍历两次，见
[`sample_backward.cu`](../aetherscan/third_party/gggs_reference/src/sample_backward.cu#L170)
和
[`sample_backward.cu`](../aetherscan/third_party/gggs_reference/src/sample_backward.cu#L232)；
第二次遍历对每个 Gaussian 聚合梯度，先做 warp reduction，再由每个 warp 的 lane 0
执行 10 次 `atomicAdd`，见
[`sample_backward.cu`](../aetherscan/third_party/gggs_reference/src/sample_backward.cu#L328)。
atomic/MIO 和 CTA barrier 是次要瓶颈，但在 local-memory 压力降低后会更加突出。

#### 优化顺序与预期收益

1. **补齐可观测性，不改变算法。** 在 profiler 中记录 `PN`、`num_evaluation`、
   `num_duplicated_tiles`，并增加每 tile point count、Gaussian range、`max_contributor`
   的分位数/直方图。Release CUDA 构建加入 `-lineinfo`；当前报告只有 SASS correlation，
   尚不能把 local-memory 热点自动映射回 CUDA-C 行。
2. **不要继续投入 `<1>/<2>` tail specialization。** 已分别实测“额外 tail grid”
   和“compact worklist”两种实现。前者为稀疏尾块增加空 CTA launch，后者增加 prefix scan、
   worklist 构建和 device-to-host 同步；在约 33 万 Gaussian 的真实场景中 forward 均由约
   4.2 ms 退化到约 4.9–5.0 ms。两种实现都已撤回，只保留 255/257 点边界的前后向数值测试。
3. **缩短 forward 寄存器生命周期。** 对初始深度 pass 和 split refinement 做 kernel
   拆分或状态重排，第一阶段以不改变数值语义为约束，将目标设为不超过 64
   registers/thread；理论 residency 可由 2 CTA/SM 提高到约 4 CTA/SM。需要同时测量
   中间结果写回和额外 launch 的成本，不能只看 occupancy。
4. **标量化 backward per-sample 数组。** 对 `G[2]`、`p_ids[2]`、点梯度和 contributor
   状态做显式展开，并 A/B 测试 48/64/80 registers 的编译变体，以 local sectors、
   long-scoreboard 和 kernel time 而非单独的 occupancy 作为选择标准。
5. **随后处理 atomic 与 tile 负载均衡。** 评估 block-level Gaussian 梯度聚合、
   按预计 `max_contributor` 排序 CTA，或 persistent CTA work queue；这些工作应排在
   backward local-memory 和 forward 寄存器问题之后。
6. **算法级采样调度只保留显式 fast mode。** ADC 停止增长后每 2 步执行一次、活跃步
   权重乘 2 虽能保持目标函数期望，但 `antman_nomask` 已证明它可能破坏几何覆盖。
   默认仍每步执行；后续应研究几何稳定性触发、half-resolution 或保守 focus 区域。

稳定窗口中 sample forward+backward 占总 CUDA 时间的 41.6%（`ori_img`）和 43.6%
（`antman_nomask`）。若这两个阶段整体加速 2 倍，在其他阶段不变的假设下，Amdahl 估算
每次训练迭代的 CUDA 时间可分别下降约 20.8% 和 21.8%。若 splat 训练仍占完整 pipeline 的
约 90%，且 wall time 与 CUDA 时间近似同比变化，整条 pipeline 的潜在收益约为 18%–20%；
该数字是单 kernel 优化上限估算；下面的完整 30,000 步 A/B 已验证更高层的调度收益。

#### ADC tail 多视图随机调度 A/B（2026-07-31）

新增 `--splat-mv-tail-interval N`。默认值为 1，即每步执行；`N=2` 是显式性能模式。
调度只在 `iteration > grow_stop_iter` 后生效，ADCPlus 默认即第 15,001–30,000 步。
被跳过的迭代仍消费邻视角选择 RNG，避免改变后续随机流；活跃迭代把 geometry/NCC 权重
乘以 `N`，因此是无偏的随机目标估计。它不会改变 depth-normal 的执行频率。

使用同一份 76 视角 / 125,818 稀疏点 SfM 缓存，在 RTX 5090 D v2 上按 ADCPlus、
30,000 步、`1000×1000`、geometry from 3,000、scale ratio 10 做完整 A/B：

| 指标 | interval=1 | interval=2 | 变化 |
|---|---:|---:|---:|
| Splat wall time | 575.389 s | 478.244 s | **-16.88%** |
| 15,001–30,000 CUDA/iter | 21.2029 ms | 14.8752 ms | **-29.84%** |
| 15,001–30,000 multi-view/iter | 12.3839 ms | 6.1603 ms | -50.25% |
| sample forward / loss / backward | 5.7850 / 3.2788 / 2.9687 ms | 2.8582 / 1.6546 / 1.4688 ms | 约 -50% |
| 三视角平均 PSNR | 20.9876 dB | 21.0766 dB | +0.0890 dB |
| 最终 Gaussian | 653,367 | 627,470 | -3.96% |
| TSDF Clean 顶点 / 面 | 1,365,794 / 2,703,392 | 1,347,617 / 2,668,110 | -1.33% / -1.31% |

**更正（2026-09-17）：`interval=2` 不是质量中性的加速，会降低几何重建的 TSDF mesh 质量。**
上表的网格数量差（-1.33%）落在 ADC 跨进程非确定性范围内，而数量差说明不了孔洞、薄覆盖与
噪声；实际重建质量评估显示 `interval=2` 明显拉低 mesh 质量，机制与下面 `antman_nomask` 的
A/B 一致：multi-view 几何/NCC 项是约束表面的主要信号，降到每两步一次等于单位时间的表面
约束减半，薄覆盖/低视差区域首先出现孔洞与噪声。因此**只要 mesh 是交付物就必须用
`interval=1`（默认值）；`interval=2` 只能用于预览/快速模式。** `ori_img` 上 interval=1 的
几何重建参考结果见 [DENSE_RECONSTRUCTION.md](DENSE_RECONSTRUCTION.md) 的 `ori_img` 表。
完整日志分别为
`artifacts/ori_img_mv_tail_interval1_30k_20260731/stdout.log` 和
`artifacts/ori_img_mv_tail_interval2_30k_20260731/stdout.log`。
ADC 的 GPU 原子与 refine 会造成跨进程非逐元素确定性，因此 Gaussian/mesh 数量只按区间
判断。

随后使用同一份 64 相机 / 26,127 稀疏点 OpenMVS 工程，对 `antman_nomask` 做严格同加载
路径 A/B：

| 指标 | interval=1 | interval=2 | 变化 |
|---|---:|---:|---:|
| Splat wall time | 442.813 s | 380.596 s | **-14.05%** |
| 15,001–30,000 CUDA/iter | 17.2424 ms | 12.8187 ms | **-25.66%** |
| 15,001–30,000 multi-view/iter | 9.8435 ms | 5.1150 ms | -48.04% |
| 三视角平均 PSNR | 37.8694 dB | 36.7273 dB | **-1.1421 dB** |
| 最终 Gaussian | 554,006 | 562,094 | +1.46% |
| TSDF Clean 顶点 / 面 | 7,470,759 / 14,736,038 | 5,355,665 / 10,612,585 | **-28.31% / -27.98%** |

虽然调度仍有稳定性能收益，但 `antman_nomask` 的 PSNR 和最大连通网格覆盖明显回退。
原因是 ADCPlus 的 `grow_stop_iter=15,000` 只停止新增 Gaussian，prune、replacement、
ADC noise 和每 200 步 refine 默认持续到训练结束；因此“停止增长”并不等价于拓扑与几何
已经收敛。在未实现基于几何稳定性或 refine-stop 的自适应触发前，interval=1 保持为默认值，
interval=2 只能作为允许质量折衷的显式 fast mode。对应日志为
`artifacts/antman_mv_tail_interval1_current_30k_20260731/stdout.log` 和
`artifacts/antman_mv_tail_interval2_30k_20260731/stdout.log`。

#### Plane-warp NCC 底层 kernel A/B（2026-07-31）

默认训练继续保持 `--splat-mv-tail-interval 1`，没有改变 multi-view 的迭代频率、权重、
7×7 patch、homography、NCC 目标或解析梯度。优化只作用于图像解码和
`multi_view_raw_kernel`：

1. 邻视图的双线性值与 `du/dv` 从同一组 2×2 角点计算，避免同一位置重复采样；
2. 仅在启用 NCC 时，RGBA8 上传 kernel 同步生成单通道 BT.601 灰度平面，plane warp
   不再为每个 patch sample 重读三个 RGB 平面；
3. `32×8` CTA 协作加载带 2 像素 halo 的 `36×12` 参考灰度 tile，并把固定的半像素采样
   预展开成 `69×21` shared-memory tile。相邻输出像素重叠的参考 patch 不再重复访问
   global memory 或重复做双线性插值；邻视图的非规则 warp 仍逐像素精确计算。

在同一份 `antman_nomask` COLMAP 输入（64 相机 / 14,586 稀疏点）、RTX 5090 D v2、
ADCPlus 5,000 步、`1000 px`、geometry from 3,000、无 Mask、interval=1 上，用优化前后
独立保留的 Release 二进制做同参数 A/B：

| 指标 | 基线 | Plane-warp NCC 优化 | 变化 |
|---|---:|---:|---:|
| Splat wall time | 30.4701 s | 29.5425 s | **-3.04%** |
| 3,001–5,000 CUDA/iter | 11.0843 ms | 10.7495 ms | **-3.02%** |
| 3,001–5,000 multi-view/iter | 6.6993 ms | 6.4688 ms | **-3.44%** |
| 3,001–5,000 NCC loss kernel | 2.1737 ms | 2.0694 ms | **-4.80%** |
| 三视角平均 PSNR | 30.5035 dB | 30.7663 dB | +0.2628 dB |
| 最终 Gaussian | 173,061 | 174,107 | +0.60% |

ADCPlus 的 atomic/refine 路径不是跨进程逐元素确定的，且底层优化改变了浮点加法顺序；
因此 PSNR 与 Gaussian 数只作为没有观察到质量回退的门禁，不把单次正向变化解释为质量提升。
严格性能结论优先采用同输入 A/B 的 CUDA event 分段结果。日志位于
`artifacts/antman_ncc_ab_20260731/baseline_mv/stdout.log` 和
`artifacts/antman_ncc_ab_20260731/optimized_samples_mv/stdout.log`。

最终候选又以同一 COLMAP 输入完成 30,000 步 + TSDF/Clean 检查：splat 训练 422.839 秒，
521,820 个 Gaussian，三视角平均 PSNR 39.2708 dB；第 15,001–30,000 步每窗口平均
CUDA 16.2756 ms、multi-view 9.2433 ms、NCC loss 2.3062 ms，15,000 个迭代中有
14,298 个有效 multi-view step。TSDF 三个诊断视图的跨视角深度一致率为
98.51%–99.79%，Clean 后网格为 9,982,336 顶点 / 19,776,598 面。该运行从 14,586 点
COLMAP 模型直接加载，不能与上文旧的 26,127 点 OpenMVS 30k 基线做严格速度归因。
完整日志为 `artifacts/antman_ncc_final_interval1_30k_20260731/stdout.log`。

## 当前几何交付与后续工作

当前版本已打通 splat mesh extraction：训练后按原图分辨率渲染每个相机的 median depth、normal
和 alpha；存在输入 mask 时以 mask 为准，否则回退到 alpha 0.5，与 `gs2mesh.py` 一致。随后按
`max_depth=2*scene_extent`、`voxel=max_depth/2048`、`sdf_trunc=4*voxel` 做 Open3D-compatible
稀疏体素块投影融合，并用标准 Marching Cubes 抽取、保留最大连通分量。`--splat --mesh` 会把
`active_mesh` 切换为 `*_splat_mesh.ply`，不再使用 projective MVS patch mesh。

若构建时找到 CGAL，可通过非零 `--mesh-target-faces` 显式调用 aether_drender 的 Instant Meshes
field-aligned remesh，再调用 `aether_mesh::repair_and_decimate`；默认保留原生 Marching Cubes 网格；
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
