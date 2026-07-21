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
训练损失包含 Charbonnier RGB、相对 MVS depth、camera-space normal consistency，以及可选
alpha BCE；所有参数通过显式 GGGS backward 和 TinyTensor Adam 更新。

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

## 实拍端到端验证（2026-07-21）

`D:\ScanVideo\ori_img\images` 的 76 张 1000×1000 图片已完成实际验证：76/76 相机注册，
SfM reprojection RMS 为 0.591 px；default MVS 用 32 workers 在约 104 秒内融合出 2,097,765
个点；从中选取 500,000 Gaussian 训练 4000 步 SH3 用时约 33 秒，TinyTensor GPU pool
约 1.72 GB。三个环绕诊断视角的 PSNR 为 24.27 / 26.81 / 27.03 dB，alpha coverage 均约
50%，正面和背面均能正确渲染。加入 Adam 非有限梯度防护后重新运行，三个视角 PSNR 为
24.01 / 26.59 / 26.70 dB；最终标准 3DGS PLY 含 500,000 个顶点、62 个 float 属性，逐值检查
确认 31,000,000 个标量全部 finite，SH-rest 非零。

这次验证同时修复了 Eigen 临时表达式导致的 W2C translation 悬空问题、Adam 极端梯度污染，
并加入非单位旋转相机转换回归测试和导出 finite gate。当前渲染仍有 floaters 和尺度毛刺，后续仍需要动态 split/prune、
opacity reset 和更长训练，不能把本次 4000 步结果视为最终商业画质。

CUDA 前反向冒烟测试：

```powershell
ctest --test-dir build -C Release -R aetherscan.splat.rasterizer --output-on-failure
```

## 性能设计

- MVS 沿用现有并发 view/tile 调度和 OpenMP CPU 并行；
- rasterizer 沿用 GGGS 的 tile binning、CUDA 排序和前反向 kernel；
- scale/quaternion/opacity 激活和对应链式梯度使用融合 kernel；
- RGB/depth/normal/alpha loss 在一次像素 kernel 中生成 loss 与四种输出梯度；
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
- 还需要真实数据集上的数值对齐、质量基准和长时间稳定性验证。

建议后续按“动态 Gaussian 管理 → checkpoint/streaming → GGGS mesh extraction → active_mesh
切换 → 多 GPU/图捕获”的顺序推进。

## 许可证边界

参考 GGGS CUDA 文件的源注释限定为非商业研究/评估用途，而 Python 参考目录中没有附带其
所指向的完整 `LICENSE.md`。因此 `gggs_reference` 只能视为原型验证代码，不能直接作为商业
RealityScan 类产品发布。详细记录见 `aetherscan/third_party/gggs_reference/NOTICE.md`；商业化前
必须取得明确授权，或以洁净室方式替换该 rasterizer core。
