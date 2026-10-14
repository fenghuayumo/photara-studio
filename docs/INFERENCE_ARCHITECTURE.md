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

MoGe 已提供权重转换、GGUF 检查与 CPU/Vulkan 加载，前向预测尚未实现。
没有可以调用但只会抛出“未实现”的模型工厂。
既有 ONNX 特征和去光照路径保持原状，当前不迁移它们的会话封装。

## 目录与职责

| 目录 | 职责 |
|---|---|
| `photara/include/inference` | 公开的数据和任务约定 |
| `photara/src/inference/runtime` | 后端、图分配和图执行；不知道模型结构 |
| `photara/src/inference` | 公共几何校验和模型缓存路径 |
| `photara/src/sam` | SAM 掩码业务、模型位置与许可记录 |
| `third_party/sam3` | 外部模型实现及当前 ggml 源码 |
| `photara/tests/inference` | 无需权重的公共契约与真实后端测试 |

接入 MoGe 时创建 `photara/src/inference/models/moge`，把权重加载、
计算图、预处理和几何恢复放在那里。多个模型真正共享算子后，才抽到
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
后续仍需完成完整张量清单校验、计算图与预处理、几何恢复和前向参考对比；
权重回读通过不能视为模型推理正确。

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
