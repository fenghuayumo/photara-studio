# 相机热启动与 CUDA bearing 扩展验证（2026-10-13）

## 结论

全景 375 张及 Playroom 225 张适合继续试验相机中心热启动 + CUDA bearing。全景重复结果稳定；Playroom mapping 减少约 72%，但相对原初始化路径的参考位姿误差有升有降。739 张室内 iPhone 和 1511 张 Berlin 已达到默认热启动门槛，在相同结构下 CPU/GPU 对照也确认 GPU 的性能收益。

Berlin 中默认时间预算下 CPU/GPU 的位姿误差差异，主要与分轮短预算使 CPU 提前停在不同优化终点相关。关闭共享总预算后，CPU/GPU 的最终代价和位姿误差非常接近。性能结果支持使用 CUDA bearing，但不能据此认定降低热启动门槛后所有质量指标都会改善。

## 方法与范围

- 基线提交 525a780，定位算法、生产默认参数不变。初始十次运行使用同一已验证的 benchmark 二进制；额外两次 Berlin 对照仅增加 benchmark 的可选时间预算参数。
- 每个数据集使用同一 tracks 检查点，固定 tracks_per_image=20、固定 CUDA BA，顺序运行并导出 ASFM，与各自 COLMAP 参考模型对齐。
- 全景补两次 CUDA 热启动复测；Playroom 测默认 CPU、热启动 CPU、两次热启动 CUDA；室内 iPhone/Berlin 测 CPU/CUDA 默认门槛对照；Berlin 再补预算对照。共新增 12 次 mapping 和参考评估。
- 小场景 CUDA 热启动门槛设为 1；大场景保持 512。默认 CPU 对照同样保持门槛 512。739/1511 张的默认 automatic 路径本来就可使用 CUDA，本表的 CPU 强制对照不代表当前 automatic 产品耗时。
- mapping 指标包含旋转、轨迹构建、定位、三角化和 BA，排除图片前端、恢复、最终管线审计、着色和导出。因此注册数也仅表示 mapping 阶段，不与以前完整 CLI 的最终注册数直接比较。
- 全景 CPU 指标引用上一轮同输入、同定位代码的两次诊断运行。其他 CPU/GPU 成对测试在本轮执行。每卡 GPU 采样包含其他进程，未用于宣称 SM 利用率。
- 原始命令、输入/源码/程序哈希、日志、ASFM、GPU 采样、参考评估及 summary.json 位于 artifacts/sfm_bearing_validation_20261013/；上一轮全景 CPU 位于 artifacts/sfm_positioning_analysis_20261013/。

## 性能

| 数据集 | CPU 定位 | CUDA 热启动定位 | CPU mapping | CUDA mapping | mapping 注册数 |
|---|---:|---:|---:|---:|---:|
| 全景 375 张 | 46.529–47.513 s | 3.005 / 3.131 s | 54.583–55.499 s | 11.279 / 11.599 s | 375 |
| Playroom 225 张 | 9.333 s | 1.000 / 1.018 s | 12.387 s | 3.413 / 3.449 s | 225 |
| 室内 iPhone 739 张 | 50.450 s | 11.273 s | 83.280 s | 44.122 s | 739 |
| Berlin 1511 张 | 49.869 s | 13.238 s | 76.897 s | 40.508 s | 1511 |

全景约减少 79%–80%，Playroom 约减少 72%，室内 iPhone/Berlin 的强制 CPU 对照分别减少约 47%。大场景各一次 CPU/GPU，不能将数值视为稳定置信区间。

Playroom 的 CPU bearing 热启动定位为 2.357 s、mapping 为 4.840 s；与 CUDA 的 1.000 s、3.413 s 相比，说明初始化结构和 GPU 执行均有收益。Playroom 默认 CPU 运行存在相机图 reseed 和 anchored component BA 失败警告，热启动两种后端均没有这些警告；对照不是只有 kernel 替换。

## 质量

| 数据集 | CPU RMS | CUDA RMS | 参考平均旋转误差 CPU → CUDA | 参考中心 P95 CPU → CUDA（参考步长） |
|---|---:|---:|---:|---:|
| 全景 | 0.368983–0.369131 | 0.369777 | 0.1951–0.1954° → 0.2054° | 0.1728–0.1731 → 0.1663 |
| Playroom | 0.647591 | 0.643963 | 0.05045° → 0.06124° | 0.01006 → 0.01186 |
| 室内 iPhone | 0.776696 | 0.776682 | 0.93313° → 0.93321° | 2.02966 → 2.02985 |
| Berlin，默认时间预算 | 0.638628 | 0.638626 | 0.21993° → 0.23158° | 0.09280 → 0.10998 |

全景两次复测的轨迹数、观测数、RMS 在输出精度下一致；Playroom 两次同样如此，参考误差有微小浮点变化。Playroom 参考模型有 222 张，CPU 默认对齐 inlier 为 220，热启动为 222，因此参考配准也发生了变化。参考模型是比较依据，不被假定为绝对真值。

## Berlin 的预算控制

默认分轮时间预算下，CPU bearing 首轮达到 30 次迭代，但后续多轮因时间耗尽只完成 1–4 次有效更新；GPU 则能够执行更多更新。因此参考误差差异不能直接视为 CUDA 数值精度问题。

新增可选 solver_time_seconds 参数，设为 0 后取消共享尝试时间预算。现有 bearing 实现仍保留每次 Solve 的 45 s 默认上限；本次所有 bearing pass 均因迭代上限或 function tolerance 退出，未因超时退出。两种后端的首轮/IRLS 迭代上限与容差保持相同。

| 对照 | CPU | CUDA |
|---|---:|---:|
| 全局定位 | 67.555 s | 12.338 s |
| mapping | 95.269 s | 40.645 s |
| RMS | 0.638628 | 0.638626 |
| 平均中心误差（参考步长） | 0.0604809 | 0.0604504 |
| 中心 P95（参考步长） | 0.1100216 | 0.1099604 |
| 平均旋转误差 | 0.2315999° | 0.2315900° |

CUDA 定位约快 5.48 倍。两者参考对齐 inlier 都为 1510，末轮定位代价均约为 3.33405，轨迹/观测数相同；完整输出未验证逐位一致。更充分优化的 CPU 解也没有恢复到短预算 CPU 的较低中心参考误差，说明还需关注优化目标与参考位姿质量的关系。

## 使用实验工具

可执行文件：build/photara/Release-fixed-deps/photara_sfm_positioning_benchmark.exe。位置参数如下，输出文件必须不存在：

```text
photara_sfm_positioning_benchmark tracks-HEX.bin cpu|cuda|vulkan NEW_OUTPUT.asfm
    [tracks_per_image] [warm_start_min_images] [automatic|cpu|cuda|vulkan BA]
    [solver_time_seconds]
```

全景候选方案尾部参数为 `cuda NEW_OUTPUT.asfm 20 1 cuda`；Berlin 预算对照尾部为 `cpu|cuda NEW_OUTPUT.asfm 20 512 cuda 0`。省略 solver_time_seconds 保持 GlobalPositioningOptions 的默认预算，参数须有限且非负。GPU bearing 的观测数/相机数限制及 CPU 回退逻辑保持原值。

本轮未改变生产默认门槛，下一步宜加入可选实验配置，并以注册数、参考位姿、RMS、尾部相机质量共同评估。完整 CLI 的构建目前受现有贴图子模块接口不匹配阻止（photara_texture 中找不到 fill_texture_on_surface）；独立定位实验工具编译和本轮全部实验成功。本轮未修改该子模块或贴图算法。
