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

- 当前默认 mesh 是多深度图局部三角化再焊接；RealityScan 级输出仍需要可扩展的
  Delaunay/visibility graph-cut 或 TSDF/Poisson 全局表面后端。
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
