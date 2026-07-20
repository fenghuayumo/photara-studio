# MVS 质量回归：statue_alex（2026-07-14）

## 数据与基线

- 输入：`D:/ScanVideo/ori_img/images`，76 张图，配套前景 mask。
- SfM：76/76 注册，133,915 个稀疏点，平均重投影误差 0.565 px。
- AetherScan 旧基线：`scene_v6_dense.ply`。
- OpenMVS 对照：`D:/ScanVideo/ori_img/scene_dense.ply`。

## 本轮改动

1. 几何一致性改为全局多轮同步：每轮重新快照全部邻居深度图。
2. 几何重投影和融合使用投影点周围的深度候选搜索，降低取整误差。
3. 融合前增加多视图深度/法线过滤与一致深度加权校正。
4. 融合使用沿参考视线的加权中位深度选层，再对层内样本求均值。
5. 增加 `preview/default/high` 全流水线质量预设。

## 端到端结果

| 配置 | MVS 时间 | 有效深度 | 融合点 | Mesh 面数 | 峰值内存 |
|---|---:|---:|---:|---:|---:|
| v6 旧默认 | 115.4 s | 6.499 M | 2.642 M | 2.870 M | 3.54 GB |
| v7 新 default | 133.3 s | 6.231 M | 2.128 M | 2.590 M | 3.33 GB |
| v8 新 high（测试时 ultra mesh step=1） | 702.6 s | 13.537 M | 4.363 M | 18.592 M | 9.03 GB |

最终 `high` 预设的 mesh step 已改为 2，以降低默认网格体积和峰值内存；
深度图与点云配置和上述 high 测试相同。API 调用者仍可设为 1 输出 ultra mesh。

PatchMatch 运行期间 Ryzen 9 7950X 的实测平均有效占用约 27–29 个逻辑核；
剩余损失主要来自视图任务尾部不均衡、网格合并和文件导出，而非线程池空闲。

## 局部表面指标

从每个点云等量随机采样，构建 k=8 邻域。`thickness ratio` 是邻域位移在点法线
方向上的中位绝对分量除以邻域间距，越低越好；`normal agreement` 是邻域法线
绝对点积，越高越好。这些指标用于回归，不替代有真值数据集上的 accuracy / completeness。

### default 对旧实现（各采样 250k）

| 指标 | v6 | v7 default | 变化 |
|---|---:|---:|---:|
| thickness ratio median | 0.2630 | 0.2403 | -8.6% |
| thickness ratio P90 | 0.5331 | 0.5131 | -3.8% |
| normal agreement median | 0.9925 | 0.9974 | 提升 |
| normal agreement P10 | 0.9106 | 0.9624 | 明显提升 |

### high 对 OpenMVS（各采样 300k）

| 指标 | OpenMVS | AetherScan high |
|---|---:|---:|
| 点数 | 1.250 M | 4.363 M |
| thickness ratio median | 0.2278 | 0.1943 |
| thickness ratio P90 | 0.4832 | 0.4652 |
| normal agreement median | 0.9911 | 0.9960 |
| normal agreement P10 | 0.9142 | 0.9560 |

## 仍未解决的产品级差距

- CGAL Delaunay/visibility graph-cut 与通用 Clean 已接通，但仍需 mask/ROI、DTU/Tanks and
  Temples 真值评测，以及 decimation/remesh/refine，才能宣称 RealityScan 级最终表面质量。
- CPU PatchMatch 的 high 档约 10.7 分钟只用于质量上限；产品交互速度需要 CUDA
  PatchMatch、GPU 代价聚合和设备端几何一致性。
- 仍需 DTU/Tanks and Temples 等有真值数据集验证 accuracy、completeness 和 F-score。
- 纹理阶段需要全局视图选择、曝光/白平衡补偿、接缝优化和 delight，点云指标不能代表
  最终 RealityScan 式纹理模型观感。

## 轮廓掠射伪影回归（v9）

针对圆形底座切线位置的外翻薄片，high 预设新增工作分辨率 mask 2 px 保护带，并在
深度过滤和融合时要求 `-dot(normal, viewing_ray) >= 0.20`。低入射角参考样本被拒绝，
但同一表面仍可由更正面的邻居视图保留。

| 指标 | v8 high | v9 silhouette guard |
|---|---:|---:|
| 融合点 | 4.363 M | 4.105 M |
| thickness ratio median | 0.1943 | 0.1906 |
| thickness ratio P90 | 0.4652 | 0.4610 |
| normal agreement median | 0.9960 | 0.9962 |
| normal agreement P10 | 0.9560 | 0.9609 |

v9 稠密重建耗时 694.0 s，峰值内存 6.67 GB（未生成 mesh）。同坐标、同尺度对比图为
`D:/ScanVideo/ori_img/aetherscan_out/scene_v8_v9_silhouette_comparison.png`。

## Tile PatchMatch、金字塔缓存与全局表面回归（2026-07-19）

### 实现

- 全场景图像金字塔一次缓存，reference/source 层只读复用；
- red/black row-tile PatchMatch；默认 8 个参考视图共享 CPU，每视图内部 4 个工作线程
  （32 逻辑核机器），最后一批自动重分配线程；
- 几何轮读取不可变深度快照，消除邻图读写竞态；
- CGAL 3D Delaunay + OpenMVS 同类双向 visibility weight + facet quality + s-t cut；
- max-flow 改为无递归 FIFO push-relabel；
- 统一 Clean：非法/重复/退化/非流形面、小分量、朝向、小孔、压缩和法线。

### Preview 性能

机器为 Ryzen 9 7950X，32 逻辑核。旧版是 view-level PatchMatch；新版结果使用
`tile_rows=8, concurrent_views=8`。金字塔构建在 antman 上仅 0.003 s。

| 数据 | 注册视图 | 旧深度阶段 | 单视图 tile 实验 | 混合 view+tile | 结论 |
|---|---:|---:|---:|---:|---|
| antman_nomask | 64 | 57.797 s | 61.660 s | 56.483 s | 比旧版快 2.3% |
| chuan | 109 | 106.514 s | 121.484 s | 104.019 s | 比旧版快 2.3% |
| ori_img | 76 | 44.124 s | 未测 | 37.726 s | 比旧版快 14.5% |

单视图 tile 是必要的正确性/尾部构件，但不应独占整机。混合调度恢复了多视图吞吐，
同时保留 tile 的确定性传播和最后一批扩容。

三组混合调度的金字塔缓存构建分别为 0.003 s、0.004 s、0.007 s；错误日志均为空。
由于 red/black 传播改变了更新顺序，结果不要求与旧 Gauss-Seidel 路径逐点一致；当前 preview
分别得到 4.797 M、5.641 M、6.035 M 个有效深度像素，融合点为 2.222 M、1.160 M、
2.825 M。后续仍应在有真值数据集上以 F-score 而不是点数选择默认并发参数。

### antman 全局网格

命令使用 `--dense-quality preview --mesh-method delaunay --mesh-max-points 250000`。
像素尺度体素去重后实际插入 65,787 点，生成 418,009 个 Delaunay cell。

| 指标 | 结果 |
|---|---:|
| 深度阶段 | 61.660 s |
| 全局 Delaunay + graph cut + Clean | 约 17.4 s |
| MVS 总时间 | 82.794 s |
| 峰值工作集 | 2.47 GiB |
| 输出面数 | 60,359 |
| 非流形边 | 0 |
| 边界边 | 1,313 |
| 连通分量 | 9 |
| 最大分量占面数 | 99.3% |

第一版只对有限 cell 建图且使用常数平滑项，虽只有 325 条边界边，却出现跨空洞的大三角封片，
视觉检查判定失败。修正版把无限 cell、相机 free-space、sample 后方 visibility、
plane/circumsphere quality 和长边拒绝全部纳入后，封片消失。`antman_nomask` 仍重建出与人物
相连的桌面，这是输入没有前景 mask/ROI 的预期结果，不应由 Clean 猜测删除。

回归产物位于 `_mvs_next/antman_delaunay_v2/`，包括 PLY、日志和最大连通分量预览图。

## ori_img 全局 mesh 与 OpenMVS 对照（2026-07-20）

使用 76 张 `ori_img`、自动 ROI、default 稠密质量和 `mesh_max_points=500000`。本次
AetherScan 与 OpenMVS 使用同一套 SfM 相机，但采样数和裁切范围不同，因此拓扑统计用于
判断碎片/破洞趋势，不作为 accuracy/completeness 真值评测。

### 性能

| 阶段 | 旧 AetherScan | 当前 AetherScan |
|---|---:|---:|
| visibility（约 160 万 ray） | 串行实测约 2036 s | 0.91 s，32 线程 |
| max-flow（约 63.5 万 cell） | 51155 s（约 14.2 h） | 2.59 s |
| global Delaunay + cut | 不可交互 | 6.99 s |
| Clean | 0.07 s，ROI 时不补洞 | 0.10 s，15 spikes / 19 holes |

max-flow 从自研通用 push-relabel 切换为 Boost Boykov-Kolmogorov；这是本轮从「能跑」到
「产品可用」的主要性能变化。完整两阶段 ROI MVS 本次为 156.8 s，峰值工作集 3.45 GiB。

### 拓扑

| 输出 | 顶点 | 面 | 连通分量 | 最大分量 | 边界边 | 边界环 | 绕向一致 |
|---|---:|---:|---:|---:|---:|---:|---|
| 旧 projective preview | 646456 | 1118653 | 258 | 94.28% | 202815 | 13544 | 否 |
| 当前 global + Clean | 42918 | 85568 | 1 | 100% | 304 | 11 | 是 |
| OpenMVS full | 401111 | 802140 | 1 | 100% | 136 | 1 | 是 |

当前全局结果已经消除 258 个局部 sheet/碎片和绕向冲突，非流形边为 0。实测剩余 11 个
边界环中有 4 个小环暴露了 bow-tie 顶点问题，因此 Clean 随后增加了 one-ring fan 拆分并
加入「双四面体共享顶点、两个孔独立关闭」的回归测试。尚未解决的主要差距转为几何分辨率：
500k 候选经投影过滤后仅插入约 99k 顶点，输出 85k 面，明显低于 OpenMVS 的 802k 面。
下一轮应优先评估 `mesh_dist_insert_px`、采样上限与局部 refine/remesh，而不是回到
projective 拼片或无约束增大补洞阈值。

## DELAUNAY_WEAKSURF 对齐回归（2026-07-20）

在 `ori_img` 的同一 SfM/ROI 配置上，对 OpenMVS weak-surface free-space support、
beta/gamma 检测和 endpoint sink t-edge 强化做固定输入 A/B。直接使用 OpenMVS 的
`kAbs=1000` 会在 AetherScan 的融合权重尺度上过度触发：约 58% 的有效射线成为候选，
并产生明显大三角和孔洞。确定性抽样将 ratio-qualified `beta-gamma` 的 P95 映射到
OpenMVS 绝对尺度后，候选降到 450,881 / 10,081,895（4.47%）。

| 指标 | weak-surface 关闭 | P95 尺度校准后 |
|---|---:|---:|
| 清理后顶点 / 面 | 415144 / 830303 | 412809 / 825643 |
| 边界边 / 非流形边 | 37 / 0 | 37 / 0 |
| dense→mesh P50 / P95 | 0.002176 / 0.007384 | 0.002185 / 0.007630 |
| 法线夹角 P50 / P95 | 19.56° / 53.48° | 19.60° / 54.51° |
| weak-surface 阶段 | 关闭 | 1.79 s |

该项当前结论是“不再破坏基准拓扑和几何，但尚未证明整体误差下降”。它保留为默认的
弱表面能量项，并提供 `--mesh-free-space-support=false` 做逐场景 A/B；进入 photometric
mesh refinement 前仍需在 `antman`、`chuan` 以及有真值的数据集上验证薄片区域。
