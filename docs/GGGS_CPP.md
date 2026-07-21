# GGGS C++ / TinyTensor 后端

## 当前实现状态（2026-07-21）

AetherScan 已有一条可编译、可前反向传播、可由 CLI 启动的 GGGS 训练路径：

```text
SfM → CPU/OpenMP PatchMatch MVS → fused dense cloud
    → Gaussian 初始化
    → GGGS CUDA forward/backward
    → TinyTensor loss + Adam
    → *_gggs.ply
```

这条路径不依赖 LibTorch、PyBind 或 Python 运行时。MVS 继续负责充分利用 CPU；Gaussian
投影、排序、混合、反向传播、损失与参数更新在 CUDA 上执行。

## 目录与职责

- `aetherscan/include/splat/`：公开的模型、相机、训练配置和训练器 API；
- `aetherscan/src/splat/rasterizer.cu`：TinyTensor tensor 与 GGGS 原生 CUDA API 的桥接；
- `aetherscan/src/splat/cuda_ops.cu`：参数激活、链式梯度、融合监督损失和融合 Adam；
- `aetherscan/src/splat/trainer.cpp`：MVS 初始化、训练视图生成、随机视图训练和 PLY 导出；
- `aetherscan/third_party/gggs_reference/`：从 Python 参考工程移入的原始 CUDA
  rasterizer core，不包含 Torch/PyBind wrapper；
- `aetherscan/third_party/tinytensor/`：tensor 存储、CUDA 内存和基础运算。

模型使用可训练的世界坐标均值、log-scale、四元数、opacity logit 和最高三阶 SH。
稠密点云法线用于初始化 Gaussian 朝向，像素足迹用于初始化尺度，点色用于初始化 SH0。
训练损失包含与 `pygsplat/simple_trainer.py` 对齐的 `0.8 * L1 + 0.2 * SSIM` 光度项、相对
MVS depth、camera-space normal consistency，以及可选 mask/alpha loss；所有参数通过显式
GGGS backward 和 TinyTensor Adam 更新。SSIM 使用原生 CUDA 3×3 窗口近似，避免引入
LibTorch，并把前景 mask 同时作用于预测图和目标图。

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
  --gggs-iterations 30000
```

`--gggs` 隐含 `--dense`。输出包括 `scene_dense.ply` 和 `scene_gggs.ply`。当前 Gaussian
PLY 保存训练参数（opacity 和 scale 仍是 logit/log-domain），可用于检查训练结果和后续
viewer/mesh-extraction 接入。

当前固定模型默认从 fused cloud 均匀选取最多 500,000 个初始 Gaussian，可用
`--gggs-max-gaussians N` 修改，`0` 表示使用全部 dense points。未实现动态 pruning 前不建议
对百万级 dense cloud 使用 `0`。训练结束还会保存第一个、中间和最后相机的
`*_gggs_view_*.png`，并在日志记录 PSNR、MAE 和 alpha coverage。

Mask 训练复用 `--masks` 指定的目录（默认寻找 `images/` 的同级 `masks/`），并支持 Python
数据集相同的 stem 匹配、`.png/.jpg/.jpeg` 大小写扩展名、`>127` 二值阈值、最近邻重采样，
找不到独立 mask 时回退到源图 alpha channel：

```powershell
aetherscan --images D:\ScanVideo\ori_img\images --output out\scene.mvs `
  --dense --gggs --gggs-use-mask `
  --gggs-alpha-mode transparent --gggs-match-alpha-weight 0.25 `
  --gggs-ssim-weight 0.2 `
  --gggs-min-scale-fraction 0.0001 --gggs-max-scale-fraction 0.02 `
  --gggs-max-scale-ratio 10
```

- `transparent`（默认）：前景 RGB loss + `0.25 * BCE(render_alpha, mask)`；
- `masked`：仅前景 RGB loss + 背景 alpha leakage penalty。

若个别视图缺少 mask，该视图按无 mask 图像训练；若所有视图均缺少 mask，则立即报错，避免
用户以为 mask 已生效。日志会单独输出 `rgb/alpha/depth/normal` 四项 loss。

`D:\ScanVideo\ori_img` 的 76/76 个 sibling masks 已用 `transparent` 模式完成 preview MVS +
500,000 Gaussian / 300 步真实回归：alpha loss 从 0.03934 降到第 200 步的 0.00223，三个
mask 内诊断视角 PSNR 为 22.87 / 22.32 / 24.55 dB，导出 PLY 的 31,000,000 个 float 标量
全部 finite。300 步仅用于验证 mask 数据链路和梯度，不能替代正式 30,000 步训练。

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
只增加约 3% 训练时间。

训练日志会输出当前随机采样的 `view`，因此单步 loss 不应被误读为同一张图上的单调曲线。
当前渲染已明显稳定，但仍有 floaters；修复后约 23.6% Gaussian 的 opacity 低于 `1/255`，
说明下一阶段必须实现 ADC+ 风格的动态 split/clone/prune/recycle 与 opacity 管理，不能把本次
4000 步固定 Gaussian 结果视为最终商业画质。

CUDA 前反向冒烟测试：

```powershell
ctest --test-dir build -C Release -R aetherscan.splat.rasterizer --output-on-failure
```

## 性能设计

- MVS 沿用现有并发 view/tile 调度和 OpenMP CPU 并行；
- rasterizer 沿用 GGGS 的 tile binning、CUDA 排序和前反向 kernel；
- scale/quaternion/opacity 激活和对应链式梯度使用融合 kernel；
- L1/depth/normal/alpha 主损失在一次像素 kernel 中生成 loss 与四种输出梯度，SSIM 使用单独
  的原生 CUDA 3×3 stencil kernel；
- Adam 的一阶矩、二阶矩和参数更新使用单 kernel，SH0/SH-rest 在同一 launch 中使用不同学习率；
- forward context 保留 geometry/binning/image/tile buffer，backward 不重复预处理。

目前为了避免每步磁盘 IO，训练开始时会把全部训练图、MVS depth、normal、mask 常驻 GPU。
中小场景速度优先时合理；大场景需要改成 pinned-host 预取、有限 VRAM cache 和多 CUDA stream，
否则显存会随视图数量线性增长。

## 尚未完成的产品阶段

当前版本是“可训练 GGGS backend”，还不是文档总流程中的最终几何交付：

- 未实现 Gaussian split/clone/prune 和 opacity reset；Gaussian 数量固定为 fused cloud 点数；
- 未移植 Python wrapper 中的 PatchMatch sample-depth/refine 分支；目前直接使用 AetherScan MVS
  depth/normal 监督；
- 未实现 GGGS 最终 mesh extraction，因此 `active_mesh` 尚不会切换到 GGGS mesh，纹理仍作用于
  MVS mesh；
- 未实现 checkpoint/resume、out-of-core view cache、多 GPU 和 mixed precision；
- 还需要与 Python 11×11 fused SSIM 的数值对齐、更多数据集质量基准和 30k 步稳定性验证。

建议后续按“动态 Gaussian 管理 → checkpoint/streaming → GGGS mesh extraction → active_mesh
切换 → 多 GPU/图捕获”的顺序推进。

## 许可证边界

参考 GGGS CUDA 文件的源注释限定为非商业研究/评估用途，而 Python 参考目录中没有附带其
所指向的完整 `LICENSE.md`。因此 `gggs_reference` 只能视为原型验证代码，不能直接作为商业
RealityScan 类产品发布。详细记录见 `aetherscan/third_party/gggs_reference/NOTICE.md`；商业化前
必须取得明确授权，或以洁净室方式替换该 rasterizer core。
