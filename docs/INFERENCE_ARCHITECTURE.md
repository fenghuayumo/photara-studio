# AI 推理模块

## 阅读顺序

1. `photara/include/inference/geometry_prior.hpp`：重建模块实际消费的数据。
2. `photara/include/inference/geometry_predictor.hpp`：图像到几何先验的任务接口。
3. `photara/include/inference/options.hpp`、`runtime.hpp`：执行配置与实际设备。
4. `photara/src/inference/runtime/ggml_runtime.*`：后端初始化、图分配与执行。
5. `photara/src/sam/masks.cpp`：现有模型如何参与 Photara 业务。

## 当前实现

```text
CLI reconstruction ──→ Photara::SAM ──→ photara_sam3 ──→ ggml
                             │
Studio ──────────────→ Photara::SAMCache
                             │
                       Photara::Inference ──→ ggml Vulkan / CPU
```

`Photara::Inference` 已提供几何数据校验、模型文件定位、设备枚举，以及供
模型适配器使用的内部 ggml 会话。公共头文件不包含 ggml 或 Vulkan 类型。
`Photara::SAM` 已从 CLI 的 `tools` 中独立出来，但 SAM 的计算图与后端
生命周期仍由外部 `sam3` 库管理，不经过新会话包装。两者共享同一份 ggml。

MoGe 已提供权重转换、GGUF 检查、CPU/Vulkan 加载、ViT-B 编码器，以及
neck 和 points/normal/mask/scale head，以及公开的输出后处理与几何恢复接口。
原始 head 输出可转换为原图分辨率的 `GeometryPrior`；完整 RGB 图像预处理
和 `GeometryPredictor` 实现尚未接入。
没有可以调用但只会抛出“未实现”的模型工厂。
既有 ONNX 特征和去光照路径保持原状，当前不迁移它们的会话封装。

## 目录与职责

| 目录 | 职责 |
|---|---|
| `photara/include/inference` | 公开的数据和任务约定 |
| `photara/src/inference/runtime` | 后端、图分配和图执行；不知道模型结构 |
| `photara/src/inference` | 公共几何校验和模型缓存路径 |
| `photara/src/inference/models/moge` | MoGe 权重与模型计算图 |
| `photara/src/sam` | SAM 掩码业务、模型位置与许可记录 |
| `third_party/sam3` | 外部模型实现及当前 ggml 源码 |
| `photara/tests/inference` | 无需权重的公共契约与真实后端测试 |

MoGe 的权重加载、计算图、后续预处理和几何恢复放在模型目录中。
多个模型真正共享算子后，才抽到
公共实现；不预先创建空模型目录或通用算子框架。

后续 MoGe-2 接入以 `Ruicheng/moge-2-vitb-normal` 为首个目标模型。
模型适配器负责 token 网格、权重加载、计算图和几何恢复；调用方通过
`GeometryInput` 提供可选针孔内参。正式接入时固定权重版本，逐阶段对比
参考输出，再验证 Vulkan/CPU 精度和重建效果。

## MoGe 权重通路

`python/models/moge/convert_weights.py` 将官方 `model.pt` 转换为 GGUF，
保留原始张量名称和连续内存顺序。GGUF 的维度顺序与 PyTorch 相反，
但转换阶段不改变张量存储；计算图中的卷积、转置卷积需自行使用正确布局。
首个支持的模型为 MoGe-2 ViT-B normal。默认 F32，用于先验证数值；可选
F16 仅转换矩阵/卷积权重，位置编码及一维仿射参数保留 F32。

文件记录格式版本、原始配置、checkpoint revision 与源文件 SHA-256。
本轮验证使用 revision `ca5f0e07ff01d3e5a364c1d954ed12ee1814b368`，
源文件 SHA-256 为 `16b8110e86d5dc5a849db120ca96ef3a223fd30b0c9146d1d81db504073da5f6`。
C++ `inspect_moge_model()` 校验格式、元数据、主要编码器/输出张量形状和
数据区完整性，不分配 GPU 内存。内部 `MogeWeights` 完成后端分配与上传，
并允许设置原始张量字节上限；该上限不包含分配对齐，也不是全局显存预算。
SHA-256 记录源文件身份，不是转换后 GGUF 的完整性摘要。

```powershell
python -m pip install -r python/models/moge/requirements.txt
python python/models/moge/convert_weights.py model.pt model-f32.gguf --revision ca5f0e07ff01d3e5a364c1d954ed12ee1814b368
cmake --build build --config Release --target photara_moge_weights_test
python python/models/moge/verify_weights.py model.pt model-f32.gguf build/photara/Release/photara_moge_weights_test.exe
python python/models/moge/verify_weights.py model.pt model-f32.gguf build/photara/Release/photara_moge_weights_test.exe --vulkan
```

验证工具从源 checkpoint 的每个张量独立提取头部、内部和末尾样本，
与 C++ 从实际后端回读的值比较。F16 文件需给转换和验证脚本都指定
`--precision f16`。模型文件不进入仓库，默认 ctest 不下载或加载 checkpoint。
转换脚本同时校验固定 ViT-B neck/head 拓扑，拒绝不同归一化、激活、
残差数量或上采样配置。后续仍需完成预处理和完整预测接口；
权重回读通过不能视为模型推理正确。

## ViT 编码器与逐层对比

`models/moge/encoder.*` 实现单图 ViT-B/14：patch 投影、class token、
位置编码、12 个 Transformer block、第 5/11 层归一化和特征投影。
输出是 token-major 的 768 维融合特征与最终 class token，供后续 neck
和 scale head 使用。计算图包括多头注意力、LayerScale、残差、
`LayerNorm eps=1e-6` 和 erf GELU。

当前输入是已经准备好的 CHW 浮点 RGB 图，尺寸为 `rows*14 × columns*14`。
完整图像重采样和归一化尚未接入；patch 排布与 native DINOv2 位置插值
在编码器中实现，位置插值保留 0.1 偏移，不启用 antialias。
注意力使用显式矩阵运算，网格限制在 1..4096 patches；尚未优化 flash attention。

`compare_encoder.py` 直接实例化本地官方 MoGe 源码中的编码器，并严格加载
同一 checkpoint；通过 hooks 获取参考输出，不另写一份参考 Transformer。
校验覆盖 21 个阶段，包括全部 block、两个 tap、最终特征和 class token。
native 工具还重复执行不保留中间结果的正常路径，检查与诊断路径一致。

```powershell
cmake --build build --config Release --target photara_moge_encoder_test
python python/models/moge/compare_encoder.py model.pt model-f32.gguf build/photara/Release/photara_moge_encoder_test.exe --moge-source path/to/MoGe --rows 30 --columns 40 --report build/encoder-cpu.json
python python/models/moge/compare_encoder.py model.pt model-f32.gguf build/photara/Release/photara_moge_encoder_test.exe --moge-source path/to/MoGe --rows 30 --columns 40 --vulkan --report build/encoder-vulkan.json
```

参考源码 revision 为 `74fbce054ebed49800de42d0ad0e83495065719a`。
已验证 F32 checkpoint：CPU/Vulkan 的 8×12、30×40 网格，以及 Vulkan
的 12×8（不同随机输入）和 37×37（原生位置编码）网格。输入为确定种子的
合成 RGB 图，尚未覆盖真实图像的完整预处理。
30×40 最终特征的相对 L2 误差：CPU 约 `2.86e-6`，严格 Vulkan 约 `3.33e-6`。

默认门槛为 relative L2 ≤ `2e-4`，最大绝对误差 ≤
`2e-3 + 2e-4 * max(abs(reference))`；报告同时保存原始绝对误差与参考幅值。
后期残差层可出现超过 300 的激活，不能给所有层套同一个绝对误差门槛。

Vulkan 对比默认使用严格 F32 模式：仅对子进程设置
`GGML_VK_DISABLE_F16/COOPMAT/COOPMAT2=1`。不修改编辑器或库调用方的环境。
`GGML_PREC_F32` 只约束部分累加策略，不能保证后端不转换输入为 F16。
通过 `--vulkan-precision default` 可检查默认快路径；其 8×12 结果尚未通过
上述精度门槛（融合特征 relative L2 约 `6.1e-3`），不能与严格模式混称。
库不会自行切换这些进程级选项，后续产品接入需明确精度策略。
F16 权重的编码器前向精度尚未验收。

## Neck/head 解码器

`models/moge/decoder.*` 接收 encoder 的融合特征、class token，以及原始
输入图像的宽高比。宽高比可以与整数 token 网格不同，UV 金字塔不能直接
使用网格比例替代。实现五级 neck 和三个图像 head，通道数为
768/256/128/64/32，scale head 为独立 MLP。

3×3 卷积使用 gather 构造边缘复制填充，并显式使用 F32 im2col；
2×2/stride-2 转置卷积通过四相投影和像素重排实现，避免普通卷积与
转置卷积权重布局混淆。最后一级使用双线性上采样。

`DecoderResult` 返回网格 16 倍分辨率的 HWC 原始点图、原始法线、
mask logits 和 log metric scale，通过公共 `MogeRawGeometry` 数据约定
交给后处理，再形成可交给重建的几何先验。

```powershell
cmake --build build --config Release --target photara_moge_decoder_test
python python/models/moge/compare_decoder.py model.pt model-f32.gguf build/photara/Release/photara_moge_decoder_test.exe --moge-source path/to/MoGe --vulkan --report build/decoder.json
python python/models/moge/compare_decoder.py model.pt model-f32.gguf build/photara/Release/photara_moge_decoder_test.exe --moge-source path/to/MoGe --rows 30 --columns 40 --vulkan --native-encoder --report build/network-raw.json
```

对比工具用官方 ConvStack/MLP 和相同权重验证 104 个阶段。
默认把参考 encoder 特征直接传给 native decoder，隔离解码器误差；
`--native-encoder` 则串联 native 编码器与解码器，验证真实模块边界。
两种模式都重复执行不保留中间输出的 decoder，检查正常路径和诊断路径一致。
Vulkan 验证沿用严格 F32 子进程策略，默认快路径和 F16 权重尚未验收。

已通过 CPU 的 4×6、1×1 网格和 Vulkan 的 4×6、6×4（原图宽高比 1.7）、
30×40 网格；串联验证通过 CPU 4×6 和 Vulkan 30×40。
30×40 串联输出相对 L2：原始点图约 `5.24e-7`、法线约 `4.33e-7`、
mask logits 约 `7.27e-6`、log metric scale 约 `2.51e-6`。
该网格正常 decoder 图缓冲区约 901 MiB，不包含模型权重、诊断数据、
上传暂存或后端内部缓冲，也不代表整个进程的显存峰值。

## 输出后处理与几何恢复

公共入口为 `inference/models/moge.hpp` 中的 `recover_moge_geometry()`，
实现位于 `models/moge/postprocess.cpp`。输入包括原始 head 输出、原图尺寸、
可选针孔内参以及模型/设备元数据；输出为 `MogeGeometryResult`，包含
`GeometryPrior` 和恢复状态、z-shift、归一化焦距、重投影 RMS、采样数量。
后处理在 CPU 上执行，不要求调用方持有 ggml 会话。

处理顺序与官方 MoGe-2 一致：先双线性恢复原图尺寸（align_corners=false），
再执行 `z=exp(raw_z)`、`x/y=raw_x/y*z`、预测法线归一化、mask sigmoid
和尺度 exp。默认掩码使用严格 `probability > 0.5`，无穷、NaN、零法线和
非正深度不参与恢复。尺度将最终相机 Z 深度转换为米。

恢复使用官方的最近邻 64×64 采样网格，至少需要四个不同的有效像素。
未知内参时假设 `fx=fy`、主点位于 `(width-1)/2, (height-1)/2`，联合恢复
焦距与 z-shift。已知内参时保留完整 K，支持不同 fx/fy 和偏离中心的主点，
只恢复 z-shift。输出点图根据恢复的深度和 K 重新投影，保持针孔几何一致；
法线保留模型预测来源，confidence 留空，不把 mask 概率当成几何置信度。

原生求解器不依赖 SciPy：固定 shift 时焦距有闭式解，在正深度约束下进行
对数网格搜索和局部黄金分割。它与官方优化器实现不同，使用输出数值对比
验收。未知焦距的恒定 Z 点图判为退化；支持不足返回 `no_support`，退化或
搜索边界解返回 `degenerate`。失败结果的 valid 全为零，不补造内参。
恢复成功不等于多视图一致，调用方仍需评估重投影 RMS 和重建约束。

```powershell
cmake --build build --config Release --target photara_moge_geometry_test photara_moge_postprocess_compare
ctest --test-dir build -C Release -R '^photara\.inference\.moge_geometry$' --output-on-failure
python python/models/moge/compare_geometry.py build/photara/Release/photara_moge_postprocess_compare.exe --moge-source path/to/MoGe --report build/geometry-reference.json
```

契约测试覆盖正负 shift、已知/未知内参、非中心主点和不同 fx/fy、米制尺度、
点图投影一致性、空掩码、稀疏支持、退化平面及非法输入。参考脚本直接调用
官方 forward/infer 的后处理路径，八组合成案例通过；参考环境还需要官方
使用的 `utils3d_moge` 包和 SciPy，仅用于 Python 验证，不是 C++ 运行依赖。
`--raw-dir` 可读取 native decoder 的诊断输出，并通过 `--raw-width/--raw-height`
指定尺寸。真实 F32 权重、合成输入、严格 Vulkan 编码器/解码器的输出也通过
已知/未知焦距对比：深度 relative L2 分别约 `1.50e-7` / `2.80e-6`，法线约
`5.86e-8`，有效掩码一致。尚未验收真实图像的端到端预测，也未自动接入
SfM/MVS/Splat 或更新场景相机。

## 几何约定

- 输入为针孔 RGB 图像；鱼眼、全景重采样由调用方处理。
- 整数坐标表示像素中心，图像原点在左上角。相机轴为右、下、前。
- 深度类型显式区分相机 Z、逆 Z 和沿射线距离；尺度区分相对值与米。
- 点图是恢复后的相机空间点，不是模型未经恢复的 affine 输出。
- 法线是相机空间单位向量；必须声明预测或派生来源。
- `valid` 必须覆盖整个输出，取值为 0/1。无效像素可包含 NaN，消费方
  必须先检查掩码；置信度可以为空，不应把有效掩码当成置信度。
- 内参始终属于输入图像。输出像素通过 scale/offset 映射到输入像素，
  恢复到输入分辨率时映射为恒等。
- `GeometryPrior::validate()` 检查尺寸、有限值和基本语义，不代替多视图
  一致性、深度/点图一致性或尺度对齐。

MVS 负责先验对齐、初始化与几何筛选；Splat 负责转换成训练约束。
模型适配器不能依赖 SfM/MVS/Splat，也不修改场景相机或训练参数。
CLI/Studio 的流程层负责加载模型和传递结果。

## 构建

| 选项 | 默认 | 含义 |
|---|---|---|
| `PHOTARA_ENABLE_INFERENCE` | ON | 构建共享 ggml 与 Inference |
| `PHOTARA_INFERENCE_VULKAN` | ON | 构建 Vulkan，缺少 SDK/glslc 时警告并保留 CPU |
| `PHOTARA_INFERENCE_CUDA` | OFF | 可选的既有 SAM CUDA 后端 |
| `PHOTARA_ENABLE_SAM` | ON | 是否编译外部 SAM 模型 |
| `PHOTARA_GGML_ROOT` | `third_party/sam3/ggml` | 共享 ggml 源码位置 |

旧 `PHOTARA_ENABLE_SAM_VULKAN/CUDA` 仅用于初始化对应新选项的默认值。
新选项进入 CMake cache 后，后续修改应使用新名称。
关闭 SAM 不影响 Inference；关闭 Inference 时 SAM 模型不可用，但许可与
缓存 API、CLI 的不可用分割提示仍保留。不自动下载模型或更改许可接受状态。

新增会话默认请求 Vulkan，初始化失败可配置回退到 CPU，并公开实际设备。
不支持算子会明确失败，不在图执行中隐式混用 CPU。
会话和图分配器不支持并发调用；一份会话在一个调用线程上串行使用。
CUDA 构建开关保留兼容现有 SAM，新公共会话只支持 Vulkan/CPU。

会话自主管理 ggml 后端，不复用编辑器 VkDevice。首版以 CPU 数据为模块
边界，避免引入跨设备同步。设备空闲显存是快照，图缓冲区统计不是全局
预算。目前尚未实现跨渲染/重建的显存预留、任务队列、推理结果磁盘缓存，
也没有取消运行中 GPU dispatch 的保证；`PredictOptions` 为模型适配器
提供取消与进度约定，具体适配器需在阶段边界实现。

## 验证

```powershell
cmake --build build --config Release --target photara_inference_test photara
ctest --test-dir build -C Release -R '^photara\.inference\.' --output-on-failure
```

契约测试验证有效区域、尺度/法线元数据、文件大小检查、CPU 回退与
严格设备选择。Vulkan 测试真实执行图、上传和回读，并检查结果；没有
Vulkan 设备时报告跳过，不能视为 GPU 验证通过。没有模型权重参与这些测试。
