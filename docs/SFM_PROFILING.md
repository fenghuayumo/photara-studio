# SfM 阶段计时

当前新增计时覆盖 native CUDA SIFT、native CUDA 描述子匹配、GPU 匹配协调器、CPU 三角化与轨迹拆分、CUDA BA。CPU BA 保留原有计时。Vulkan 后端没有新增设备端计时。

## 开启日志

CLI 使用现有环境变量控制控制台日志。在 PowerShell 中，先设置环境变量，再运行原有重建命令并保存输出：

```powershell
$env:PHOTARA_LOG_LEVEL = 'debug'
# 运行原有 photara.exe 命令，将 stdout/stderr 重定向到本次实验日志。
```

`debug` 包含图片级 SIFT 汇总、匹配批次、三角化/split 和 BA 明细。需要 SIFT 每个 octave/level 的明细时使用 `trace`。通过 C++ API 调用时，使用 `core::Logger::configure` 或 `set_levels` 设置日志等级。CLI 没有默认开启文件日志。

缓存命中会跳过对应计算和计时日志。分析完整前端时应使用独立的冷缓存目录；分析映射阶段时可复用相同前端缓存。不要把两者的总耗时直接比较。

## 字段与口径

| 日志前缀 | 字段 | 含义 |
|---|---|---|
| `CUDA SIFT timing` | `wall_ms` | 单张图片 CUDA extractor 调用内的墙钟时间，不含外部解码与 finalize/grid/mask 后处理 |
| 同上 | `sync_count`, `sync_wait_ms` | 原有 stream 同步调用次数及主机阻塞时间总和 |
| 同上 | `count_wait_ms`, `keys_download_wait_ms`, `sorted_upload_wait_ms` | 检测数量、关键点下载、排序后上传对应同步点的等待 |
| 同上 | `pyramid_wait_ms`, `orientation_download_wait_ms`, `descriptor_download_wait_ms` | 金字塔尾部、方向下载、描述子下载对应同步点的等待 |
| 同上 | `sort_cpu_ms`, `expand_cpu_ms` | CPU 关键点排序和多方向展开 |
| `CUDA SIFT layer timing` | `phase`, `octave`, `level`, `wall_ms`, `sync_count`, `sync_wait_ms` | 每层 detect/descriptor 的墙钟时间、同步次数和等待，trace 等级 |
| `CUDA match timing` | `pairs`, `batches` | 本次 match_batch 的图像对数与实际子批次数 |
| 同上 | `cache_hits`, `cache_misses`, `cache_resets`, `uploaded_bytes` | 实际描述子查找命中/未命中、容量触发的缓存清空、实际上传字节数；命中统计是查找次数，不是独立图片数 |
| 同上 | `upload_ms`, `compute_ms`, `download_ms` | CUDA event 测得的 stream 区间；上传含缓存查找/填充，计算含 tile 与 finish，下载为输出回传 |
| 同上 | `sync_wait_ms`, `result_cpu_ms`, `wall_ms` | 批次同步的主机等待、CPU 匹配结果组装、整个调用耗时 |
| `match/verify timing` | `match_wall_s` | 协调器调用 matcher 的累计墙钟时间 |
| 同上 | `throttle_wait_s`, `throttle_waits`, `peak_in_flight` | 几何验证队列满导致的主线程阻塞时间、次数、并发队列峰值 |
| 同上 | `geometry_work_sum_s`, `geometry_queue_sum_s` | 所有 CPU 验证任务的工作时间、从提交到启动的等待时间之和 |
| 同上 | `drain_wait_s`, `wall_s` | 匹配提交完后的尾部排空等待、整个协调器耗时；不含最终 live flush |
| `stage finished: sfm.verify_geometry` | `elapsed_s` | 独立 CPU 几何验证循环的墙钟时间；自动相机选择或匹配缓存命中后可能使用此路径，协调器的 `verify=0` 不表示跳过了几何验证 |
| `triangulation timing` | `solve_wall_ms`, `candidates`, `selected`, `outliers_only`, `split_enabled` | 首轮三角化循环的墙钟时间和执行范围，包含循环内进度计数 |
| `track split timing` | `scan_wall_ms`, `peel_wall_ms`, `index_wall_ms` | 候选扫描/fast gate、递归子轨迹求解与提交、索引重建的墙钟时间 |
| 同上 | `merge_wall_ms`, `workers`, `roots` | 并行分支确定性合并耗时（包含在 peel_wall_ms 内）、请求的拆分线程数、初始待拆分父轨迹数 |
| `CUDA BA timing` | `linearize_ms`, `assemble_ms`, `solve_ms`, `update_cost_ms` | 重投影线性化、系统组装/预条件器、PCG/点回代/步长范数、更新/代价/接受拒绝/回滚的 stream 区间 |
| `CUDA BA transfer timing` | `upload_wall_ms`, `download_wall_ms` | optimize_cuda 包装入口的 CPU 打包/分配/上传与结果下载耗时，不包含在 optimizer.total_time_ms 中 |

CUDA BA 的四项结果同时填入 `OptimizerSummary` 原有字段，因此 `brief_report()` 中的四项子阶段不再固定为零。直接调用持久化 optimizer 的 upload/download 不经过包装入口的传输日志。

轨迹拆分在至少 64 条待处理父轨迹、CPU 线程预算大于 1 时使用共享线程池。各父轨迹及其递归子轨迹独立求解，再按原串行 LIFO 顺序提交；子轨迹 ID 和观测排序保持不变。小批次、单线程以及嵌套线程池调用保留串行路径。候选扫描与最终索引重建仍串行执行。

## 并行拆分实测（2026-10-12）

同一环境下顺序运行原串行版本（1798820）与并行版本，复用同一前端 tracks 缓存，每次实际执行完整 global mapping，未命中 reconstruction 缓存。两组数据各运行一次串行基线、两次并行版本；全部使用 CUDA BA（第二次并行通过 automatic 选择 CUDA）。日志、命令、源码与二进制哈希、参考模型评估保存在 `artifacts/sfm_parallel_split_20261012/`。

| 数据集 | 串行 sfm.reconstruct | 并行 sfm.reconstruct（两次） | 串行累计 peel | 并行累计 peel（两次） | 注册数 |
|---|---:|---:|---:|---:|---:|
| Berlin，1511 张 | 121.605 s | 97.377 / 95.046 s | 29.878 s | 1.506 / 1.499 s | 1508/1511 |
| 全景视频，375 张 | 59.850 s | 57.913 / 58.070 s | 11.009 s | 0.526 / 0.534 s | 375/375 |

这些是复用前端缓存的重建耗时，不代表冷缓存端到端提速。Berlin 重建减少约 20%–22%，恢复阶段从 72.102 s 降至 48.752 / 46.657 s；RMS 为 0.624601 / 0.624600 px，与串行 0.624601 px 接近。峰值工作集从 1571 MB 到 1575 / 1583 MB。

全景数据集拆分明显加速，但完整流程仅减少约 3%。拆分之前的全局定位已从串行 34.169 s 变为并行 43.606 / 43.708 s，IRLS 轮数也不同，抵消了多数收益。RMS 从 0.367801 px 变为 0.368939 / 0.368938 px（约 +0.31%）；参考模型旋转/中心误差 P95 下降约 5.5% / 9.0%，平均旋转误差则从 0.1680° 升至 0.1936°。因此不能宣称完整重建逐位一致或所有质量指标都改善，也不能将这些定位和质量变化直接归因于拆分。本次未修改定位算法。

mapping 测试用相同输入逐项验证串行与并行拆分的精确一致性，覆盖递归深度、fast gate 开关、选定轨迹去重、未注册观测、子轨迹 ID、观测顺序及嵌套线程池回退。BA、mapping、hierarchical、submap recovery 四项 Release 测试通过。GPU 利用率采样是整卡指标；本次优化减少 CPU 拆分等待，未修改 GPU kernel，也未证明 GPU SM 利用率提升。

## 解释限制

- 本次没有增加 `cudaStreamSynchronize`、`cudaDeviceSynchronize` 或 event synchronize。SIFT 测量现有同步点；匹配和 BA 的 event 在原有同步完成后读取。
- CUDA event 区间可能包含主机提交间隙、同步后的 CPU 决策和其他 GPU 工作影响，不能作为纯 kernel 时间或 SM 利用率。字段明确标注 `stream_event_intervals`。
- 主机同步等待与 GPU 区间重叠，不能相加。并行任务的 `*_sum_s` 是累计任务时间，可能远大于流水线墙钟时间，同样不能直接相加。
- SIFT 同步等待可能包含同步点之前的金字塔计算等未完成工作，并非对应 PCIe 传输的单独耗时。
- pageable host memory 的 `cudaMemcpyAsync` 调用本身也可能阻塞。很小的 `sync_wait_ms` 不能证明整个传输路径没有 CPU 等待，需结合层级 `wall_ms` 或 CUDA API 时间线进一步定位。
- 计时本身有 event 创建/记录/查询、时钟读取和日志开销。比较性能时使用同一计时版本和日志等级，trace 日志尤其应独立用于诊断。
- 空输入、缓存命中、禁用 split 或零迭代可能自然产生零值；不应视为计时缺失。
