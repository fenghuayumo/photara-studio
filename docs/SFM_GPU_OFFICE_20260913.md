# 办公室 SfM GPU 优化验证

数据：`D:/BaiduNetdiskDownload/标准数据集-办公室/images`，900 张照片。
硬件：NVIDIA GeForce RTX 5090 D v2，24455 MiB 显存。

## 实现

参考本地 `spirula-studio/src/sfm/feature/Matcher.h` 和
`spirula-studio/src/sfm/shaders/match/bruteforce.slang` 的描述子驻留、批量提交、
整数点积及双向共享计算思路；CUDA 内核为本项目独立实现。

- 默认 uint8 描述子路径使用原生 CUDA，支持整数 Tensor Core 的编译架构使用
  WMMA；较旧架构使用 DP4A 或等价整数计算。
- 一个 32×32 点积块同时生成两个方向的最优、次优候选，完整距离矩阵不写入显存。
- 候选按连续线程访问布局存储，避免后续归约的跨步显存读取。
- 最多 1.5 GiB 描述子缓存，受可用显存的四分之一限制；按 identity、generation、
  数据长度识别缓存失效。仅在完成的提交边界回收缓存。
- 每批最多 8 对，显存不足时按字节预算缩小批次；批内共用暂存区，一次回读结果。
- 保留 SiftGPU 的双向角距离 ratio test 和 0.7 绝对角距离阈值。
  特别保留 double acos 后转 float 的顺序：办公室第 29009 对的一个对应点
  位于 0.8 ratio 边界，直接 acosf 会造成一个最低有效位的差异。
- 去掉旧 SiftGPU 构造时按最大特征数预分配的巨大距离矩阵，旧路径按实际输入增长。
- 自动相机标定之前不再运行随后会被丢弃的几何验证；依赖初步几何筛选的
  hybrid LightGlue 路径仍保留初步验证。
- 完成匹配后释放 GPU 缓存；新增 CUDA 源文件进入特征缓存指纹，并在相关源码
  改动时重新生成指纹。

## 固定输入的匹配验证

从原版完整运行的 features 和 matches 检查点读取完全相同的输入。
没有重新提取特征，没有改变特征数、候选对或筛选阈值。

| 项目 | 结果 |
|---|---:|
| 全部候选对 | 29726 |
| 原始匹配数 | 6568970 |
| 新旧结果不一致的图像对 | 0 |
| 新实现累计匹配调用耗时 | 67.2068 秒 |
| 包含读取检查点的进程耗时 | 74.0815 秒 |
| 整卡 GPU 峰值利用率 | 99% |
| 整卡 GPU 平均利用率，包含检查点读取 | 87.85% |
| 整卡显存峰值，包含桌面及其他应用 | 3936 MiB |

最终版本均匀抽取 1024 对的独立对照：旧 SiftGPU 8.63835 秒，融合 WMMA
内核 2.63455 秒，约 **3.28 倍**；两次顺序运行均得到 241410 个相同匹配。
这是单次匹配阶段测量，不是完整 SfM 加速倍数。

原版初次完整运行的主匹配/几何流水阶段为 256.41 秒。该阶段同时包含 CPU
几何工作，且本轮开发过程中有编译和短暂 GPU 测试重叠，不能将它与纯匹配
67.21 秒直接作为严格的端到端速度对照。

## 完整重建结果

新运行使用独立空缓存，从全部原照片重新提取、匹配、验证并求解。
features 和 matches 两个检查点的载荷长度、FNV 校验值均与原版一致。
前端得到相同的 23627156 个特征、11108 个图像对、909173 条轨迹。

| 项目 | 原版初次运行 | 新版完整运行 |
|---|---:|---:|
| 注册相机 | 897/900 | 897/900 |
| 重投影 RMS / px | 0.691188 | 0.691164 |
| 三维点 | 819147 | 819119 |
| 完整重建内部计时 / 秒 | 602.907 | 298.111 |
| 前端 / 秒 | 470.820 | 182.206 |
| CUDA BA 调用数 | 4 | 4 |
| CUDA BA 累计 / 秒 | 18.005 | 17.391 |

原版总耗时受上文开发活动重叠影响，表中的约 2 倍整体差异是观察值，
不作为隔离测量的加速承诺。新版进程墙钟耗时 301.75 秒，包含最终输出。
新版主匹配阶段 67.63 秒，去掉阶段两端各一秒后，120 个整卡采样点的 GPU
利用率均值 97.76%，范围 95–99%。整个进程的 GPU 平均值为 38.72%，
因为还包含图像读取、CPU 几何验证、相机定位和轨迹处理，不能称全流程 GPU 满载。

两次未注册的照片相同：`video06_00545.jpeg`、`video08_00245.jpeg`、
`video08_00247.jpeg`。此次优化没有解决这三张的几何约束问题。

所有 897 个共同相机采用一次正尺度 Sim(3) 对齐、不剔除异常值：相机中心变化
中位数为原版相机布局 RMS 半径的 0.00140%，p95 为 0.0100%；旋转变化中位数
0.000875°，p95 为 0.00457°。这表示与原结果接近，不是真值精度测试。

## 回归与 GPU 检查

- 特征测试通过：真实 SIFT 新旧匹配、CPU 整数 oracle、双向/单向筛选、空输入、
  不完整 tile、重复描述子、跨批次、缓存 generation 失效及 0.8 边界回归。
- Compute Sanitizer memcheck：0 errors。
- Compute Sanitizer synccheck：0 errors。
- Sanitizer 使用 `--kernel-name kns=features`，聚焦本次原生匹配内核，
  不表示对所有第三方 SiftGPU 内核做了完整审计。
- `git diff --check` 通过。

证据目录：`artifacts/sfm_gpu_office_20260913/`。完整重建为 `after.asfm`，
诊断为 `after_sfm_diagnostics.csv`；`verified_all.log`、`native_final.log`、
`legacy_final.log` 保存匹配对照；`summary.json` 和 `comparison.json` 保存数值汇总。

## 复现

```powershell
cmake --build build --config Release --target aetherscan aetherscan_features_test aetherscan_sfm_match_benchmark --parallel 8
build/aetherscan/Release/aetherscan_features_test.exe
python experiments/profile_sfm_gpu.py --output artifacts/sfm_gpu_office_20260913/recheck -- build/aetherscan/Release/aetherscan_sfm_match_benchmark.exe artifacts/sfm_gpu_office_20260913/cache_before/features-782dcc4bbb107403.bin artifacts/sfm_gpu_office_20260913/cache_before/matches-38f6b939463a9235.bin native 0
```

`legacy` 可替换 `native`；最后的 `0` 表示全部图像对，非零表示均匀抽样。
另可传入起始索引以检查一个连续区间。验证工具对任何不同的对应关系返回非零。
采样 CSV 中的利用率和显存属于整张显卡，不是进程独占指标。
