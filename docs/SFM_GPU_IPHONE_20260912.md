# iPhone 室内 SfM GPU 与性能验证（2026-09-12）

本轮在工作区已有 CUDA 联合 BA 实现上继续优化，参考 `D:/ProgramCode/C++/spirula-studio/src/sfm/ba` 和 `src/sfm/shaders/ba/cg.slang` 的分块归约、GPU 常驻求解及预条件分解复用方式。没有修改相机模型选择阈值、注册门槛、IRLS 参数或最终 BA 收敛条件。没有把已有参考模型输入重建。

## 实现

- 修复联合内参 PCG 的乘积缓冲区越界：分配长度从 `6 × poses` 改为全部位姿及内参自由度。
- 修复共享内参 Schur 乘积中普通 `+=` 与其他线程 `atomicAdd` 同时写入同一元素的竞争。
- GPU 相机和内参预条件块每个 LM 步骤只做一次 Cholesky 分解，所有 PCG 迭代复用分解；保留原有非正定块的对角回退。
- 联合内参优化按相机 CSR 分成最多 256 个观测的块，融合相机/内参乘积并做块内归约。固定内参仍使用原有按观测并行的路径：对照实验中强行改用分块反而变慢。
- 合并一次重复的代价标量回读；重新上传问题时销毁持有旧地址/拓扑的 CUDA Graph；非十倍数的 PCG 上限严格执行剩余迭代。
- 稀疏点着色最多 8 路并行解码，只保留采样后的像素。颜色按原图像顺序累加，无跨线程写轨迹、无颜色原子操作，保留原有采样坐标、缩放、均值和缺失图像处理。

GPU 实际用于稀疏 BA 的线性化、Schur 乘积、PCG 和参数更新。小问题、固定边界的局部 BA 仍保留 CPU 路径；全局位置初始化目前仍由 CPU Ceres 求解，本机 Ceres 编译为 `CERES_NO_CUDA`。这不是“稀疏流程所有步骤均已迁移 GPU”。

## 数据和口径

输入：`D:/BaiduNetdiskDownload/室内iphone/images`，739 张照片。硬件：RTX 5090 D v2，24 GB，Windows WDDM，CUDA 12.8 Release 构建。

所有全量运行使用相同参数：`--camera-model auto --mode global --max-features 6000 --window 6`。每次使用新 SfM 缓存目录，不改写数据集；不清空操作系统文件缓存。数字包含特征提取、匹配、稀疏重建、恢复和点着色，不包含后续 3DGS/MVS。

证据目录：`artifacts/sfm_gpu_iphone_20260912/`。`provenance.json` 保存源码和可执行文件 SHA256，`input_manifest.json` 保存输入文件名、尺寸和修改时间。修改前指本轮开始时工作区的版本，不是 Git HEAD。

## 全量结果

| 运行 | 注册 | 稀疏点 | 重投影 RMS / px | 重建耗时 / s |
|---|---:|---:|---:|---:|
| 修改前 `before` | 738/739 | 316685 | 0.883597 | 173.080 |
| 中间实验 `after` | 738/739 | 316740 | 0.882471 | 178.175 |
| 最终 `final` | 738/739 | 316715 | 0.883123 | 154.378 |
| 最终独立复测 `repeat` | 738/739 | 316745 | 0.881536 | 156.879 |

最终首次运行耗时减少 **10.8%**。中间实验未包含并行着色，且期间编译过独立对照程序，不作为性能成功结果。最终运行没有与本任务其他 GPU 基准或编译重叠。桌面环境和系统文件缓存仍可能引入波动，不能把全部时间差归因于 GPU 内核。

最终独立复测使用另一个全新 SfM 缓存目录，耗时减少 **9.4%**。两次最终运行平均 155.629 s，相对修改前约减少 **10.1%**。注册集合均与修改前相同。复测着色为 **2.098 s**。这些是两次运行记录，不是足以推断所有场景性能的统计样本。

`final` 的点着色阶段为 **2.107 s**。所有版本缺失的相机均为 `DCIM5732.jpg`，稳定恢复仅有 9 个独立深度对应，按原有规则拒绝注册。

相对修改前，在全部 738 个公共相机上拟合一次正尺度、无反射 Sim(3)，不删除离群点：中心变化 P95 为修改前相机中心 RMS 半径的 **0.0563%**，最大 **1.0021%**；旋转变化 P95 **0.0231°**，最大 **0.2257°**。这是版本稳定性指标，不是相对于物理真值的误差。

独立复测相对修改前：中心变化 P95 / 最大 **0.1825% / 1.1855%**，旋转变化 P95 / 最大 **0.0623° / 0.8880°**。两次最终运行互相比较：中心变化 P95 / 最大 **0.1641% / 2.0681%**，旋转变化 P95 / 最大 **0.0481° / 0.7696°**。因此只确认注册集合重复一致，不能宣称姿态逐位确定性。对应记录为 `final_comparison.json`、`repeat_comparison.json` 和 `repeat_stability.json`。

## CPU/GPU 等价性对照（`--ba-backend`）

新增 `--ba-backend automatic|cpu|cuda` 进程级开关：`cpu` 强制全部 BA 走 CPU；`cuda` 偏好 CUDA 并在任何符合条件的求解回退 CPU 时输出告警，用于审计 GPU 覆盖。该开关不参与重建缓存指纹，对照运行必须使用独立缓存目录。

64 张子集三次完整冷启动（各自独立缓存，参数一致）：

| 运行 | 后端 | 注册 | 地标 | RMS (px) | 观测 | 耗时 |
|---|---|---:|---:|---:|---:|---:|
| `ab_cpu1` | cpu | 64/64 | 19609 | 0.552505 | 58183 | 18.3 s |
| `ab_cuda` | cuda | 64/64 | 19636 | 0.551543 | 58247 | 17.2 s |
| `ab_cpu2` | cpu | 64/64 | 19609 | 0.552487 | 58183 | 19.2 s |

`ab_cuda` 中 6 个全局 BA 全部 `backend=cuda`、审计告警 0 条；32 个 `ba.cpu` 均为标定阶段的两视图小 BA（局部位姿锁定，按设计走 CPU）。CPU 运行复现了修改前基线的地标/观测计数（19609/58183），说明 CPU 路径行为保留。

全部公共相机一次正尺度无反射 Sim(3) 后的姿态差异：

| 对照 | 中心 P95 / 最大（参考半径 %） | 旋转 P95 / 最大（°） |
|---|---:|---:|
| CPU↔CPU 重复 | 0.00055 / 0.00063 | 0.00026 / 0.00028 |
| CPU↔CUDA | 0.153 / 0.194 | 0.033 / 0.041 |

CPU↔CPU 接近逐位一致；CPU↔CUDA 的差异来自 GPU 浮点归约顺序（atomicAdd）与 PCG 收敛检查节奏（图每 10 次迭代、CPU 每次迭代），经 5 遍 BA 累积后仍在本仓库历史版本间差异（约 0.05%–0.2%）的量级内。GPU 运行多保留 27 条轨道且 RMS 略低，两个都是同一优化问题的有效收敛点，不能宣称 GPU 结果更精确。单元级证据：CPU/CUDA 线性化最大差 3.783e-10；同一合成问题最终代价 CPU 5264.65 对 GPU 5264.67。

## 视频抽帧数据集（2026-09-12 追加）

两个微信视频抽帧序列，同一参数冷启动（`--camera-model auto --mode global --max-features 6000 --window 6`），全局 BA 全部 `backend=cuda`、审计告警 0 条。`0715` 的视见图焦距共识把无 EXIF 初值从 2304 px 修正到 1167.61 px（0.51×），`0712` 从 1536 px 修正到 1045.54 px（0.68×）；两者可观性审计 `reliable=全部`、`bridges=0`。

| 数据集 | 分辨率 | 注册 | 地标 | RMS (px) | 耗时 | 复跑姿态 P95（中心 % / 旋转 °） |
|---|---|---:|---:|---:|---:|---|
| `WeChat_20250712175936` | 1280×720 ×198 | 198/198 | 52060 | 0.8177 | 27.1 s | 0.169 / 0.088 |
| `WeChat_20250715211138` | 1920×1080 ×153 | 153/153 | 86972 | 0.7823 | 30.4 s | 0.077 / 0.040 |

两次运行相机集合完全相同（added/removed 为空），地标数波动 2.5%–3.3%（视频序列高重叠下边缘轨道的进出），与 iPhone 数据集记录的运行间差异同量级。`0712` 的 `global_positioning` 为 8.4 s，是 BA 之外最大的 CPU 阶段。

## GPU 独立基准

739 个相机、200000 个点、1200000 个观测，20 个 LM 步骤。开启共享焦距/横纵焦距/畸变优化；三个对照/最终运行顺序交替执行，排除上传和下载。

原始 GPU 实现有越界和读写竞争，不能直接作为可信数值基准。因此单独构建 `control/`：只修正这两项正确性错误，保留原有算法及归约路径。

| 版本 | 三次耗时 / ms | 中位数 / ms | PCG 总迭代 |
|---|---|---:|---:|
| 修正正确性的对照 | 2885.99 / 2881.53 / 2860.15 | 2881.53 | 1950 |
| 最终优化 | 2824.03 / 2808.37 / 2815.54 | 2815.54 | 1950 |

中位耗时减少 **2.29%**，各次都接受 20 步、拒绝 0 步，输出代价均为 `5.89156e6 → 110029`（日志显示精度）。初步单次测得 2.69 s，但重复测量未维持该数值，因此不宣称 6% 的稳定 GPU 收益。

同规模固定内参 CPU/GPU 单次对照：1850.30 / 1125.53 ms，GPU 为约 **1.64×**；最终代价均显示 110051。CPU/GPU PCG 总迭代为 1948/1950，停止检查节奏不同。这不是全流程加速倍数，也不是本轮修改相对于旧 GPU 版本的倍数。

## 几何限制

输入目录另有 `sparse/0`，其中 738 个参考相机与本轮有 737 个同名注册相机。仅把它作为已有模型的一致性对照，没有确认其生成流程或测量真值身份。

全部公共相机一次 Sim(3) 对照，修改前/最终的中心差异 P95 为参考半径的 **16.253% / 16.243%**，最大 **188.240% / 188.233%**；旋转 P95 为 **2.733° / 2.727°**，最大 **47.168° / 47.164°**。例如 `DCIM5783.jpg` 仍存在很大分歧。两版都未通过既有几何验收门槛；低重投影误差和高注册率不能替代独立几何精度验证。

## 验证与复现

CLI 和 Editor Release 构建成功。BA、two_view、mapping、checkpoint 四项 CTest 通过；CUDA Compute Sanitizer memcheck 为 **0 errors**。新增回归覆盖共享/多组内参、独立代价核验、不同拓扑重复上传、PCG 上限以及串/并行着色一致性（包含坐标缩放、缺失照片、无效特征和越界采样）。

```powershell
build/aetherscan/Release/aetherscan.exe --images 'D:\BaiduNetdiskDownload\室内iphone\images' --output artifacts/sfm_gpu_iphone_20260912/new.asfm --cache-dir artifacts/sfm_gpu_iphone_20260912/cache_new --camera-model auto --mode global --max-features 6000 --window 6
python experiments/compare_sfm_runs.py --before artifacts/sfm_gpu_iphone_20260912/before.log --after artifacts/sfm_gpu_iphone_20260912/final.log --output artifacts/sfm_gpu_iphone_20260912/comparison_new.json
build/aetherscan/Release/aetherscan_ba_benchmark.exe --cameras 739 --points 200000 --observations-per-point 6 --iterations 2 --gpu-solver-iterations 20 --gpu-intrinsics 1
ctest --test-dir build -C Release -R 'aetherscan\.(ba\.optimizer|sfm\.(mapping|two_view|checkpoint))$' --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode 99 build/aetherscan/Release/aetherscan_ba_test.exe
```

`compare_sfm_runs.py` 要求对应 CLI 日志和同前缀的 `_sfm_diagnostics.csv`。它报告全部公共相机，不按误差剔除相机，不将较小注册子集伪装成整体精度提升。
