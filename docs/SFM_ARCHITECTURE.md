# AetherScan SfM 架构与性能路线

## 目标边界

RealityScan 类产品不是单个重建算法，而是图像导入、特征、检索、匹配、几何验证、
初始化、增量/全局 SfM、BA、稠密 MVS、网格与纹理的一整条流水线。AetherScan
先把 BA 做成可独立压测的基础设施，再向前接 SfM，能避免前端完成后被后端吞吐量
卡住。

参考 openMVS `libs/SFM` 的职责划分，但不复制其对象模型：AetherScan 的核心数据
采用紧凑索引、SoA 热数据和可持久驻留设备内存，公开 API 与执行期布局分离。

## 目标模块

```text
image/io -> features -> retrieval/pairs -> matching -> geometry
                                                   |
                                                   v
scene/track <- triangulation <- incremental mapper/resection
     |                              |
     +-------------> BA <-----------+
                       |
                       v
                 MVS interface
```

- `features`：FreeImage IO + VLFeat SIFT/RootSIFT；可选 LightGlue ONNX；
- `retrieval`：词汇树或全局描述子召回，避免全图像对匹配；
- `matching`：AVX2 L2 / 后续批量 GPU KNN、ratio/cross-check；
- `geometry`：Eigen Essential/Fundamental/Homography RANSAC 与 PnP；
- `scene`：相机、视图、特征、track、3D point 的稳定 ID 数据库；
- `mapper`：高内点种子初始化、PnP/resection、局部/全局 BA 调度；
- `ba`：当前正在实现的高性能非线性最小二乘后端；
- `mvs`：后续稠密深度、融合、网格与纹理入口。

默认构建**不依赖 OpenCV**；图像 IO 使用 FreeImage。

## BA 当前实现

每个观测产生 `2x6` 位姿 Jacobian 与 `2x3` 点 Jacobian。CPU 后端按点建立邻接，
形成相机块 `U`、点块 `V` 与交叉块 `W`，消去点变量：

```text
S = U - W V^-1 W^T
b_s = b_c - W V^-1 b_p
```

`S` 不显式展开成巨型稀疏矩阵，而由 camera/point 双邻接执行 matrix-free 乘法；
PCG 使用 `6x6 U` 块作预条件。第一相机和第一点默认固定以消除单目系统的规范自由度。

当前 CUDA kernel 只生成每观测 Jacobian。它非常快，但若把结果下载给 CPU 再做 Schur，
大规模问题会受 PCIe 和同步限制，因此这不是最终架构。

## GPU BA 下一阶段

1. 将观测按 point 主序排序，并建立 camera 反向索引；
2. 一个 warp/CTA 负责一个 track，寄存器内累计 `V`、`b_p` 和 `W`；
3. 分层归约相机 `U/b_c`，避免全局双精度 atomic 成为瓶颈；
4. GPU matrix-free Schur-PCG，向量、预条件块和临时量全程驻留；
5. GPU 参数更新与 cost reduction，使用 CUDA Graph 固化 LM 一轮；
6. 可选 mixed precision：Jacobian/`W` 用 FP32，归约与小块求逆用 FP64；
7. 超大场景采用局部 BA、共视子图和分层全局 BA，而不是每加入一张图就全局求解。

性能判断必须报告端到端时间，并分别记录线性化、组装、PCG、更新、代价评估和传输；
kernel-only 数字只用于定位算子上限。

## SfM 实现顺序

1. Scene/View/Camera/Track 稳定数据模型和二进制缓存；
2. 特征后端接口与 CUDA SIFT 基线；
3. 图像对召回、GPU 匹配和两视几何；
4. track 合并、优质种子选择、三角化和 PnP；
5. 增量 mapper、局部 BA 调度、异常点过滤；
6. 全局 rotation/translation averaging 作为大数据集可选路径；
7. COLMAP/OpenMVS 互操作和 MVS 数据出口。

当前已完成第 1 步的内存数据模型、union-find track 合并、E/F/H RANSAC 与
N-view DLT 三角化；增量/全局 mapper 首版；以及 OpenMVS Interface（`.mvs` / MVSI）
导出（`sfm::export_openmvs_interface`），可用 OpenMVS Viewer 直接打开。

当前 mapper 首版也已落地：

- 增量式：种子相对位姿、2D-3D 候选评分、EPNP RANSAC、LM 精修、补三角化、
  共视局部 BA、最终统一重三角化和全场 BA；
- 全局式：BFS 旋转初始化、Lie IRLS rotation averaging、联合相机中心/边尺度的
  robust position averaging、全局三角化和两阶段 BA；
- 平移系统执行秩检测，共线或不具平行刚性的 view graph 会明确失败，不输出伪解。

在 `D:/ScanVideo/chuan/images` 的 109 张序列上（FreeImage + VLFeat，无 OpenCV），
邻接窗口 3 约产生 318 条验证边和 3.3 万 tracks。增量式注册 109/109 相机并输出
约 3 万点（特征密度低于此前 OpenCV SIFT 配置，可通过降低 `contrast_threshold`
提高密度）。7950X 上端到端约 8.6s：提取 ~1.8s、匹配+验证 ~2.0s、mapping ~4.7s。

全局路径在该近顺序、局部低视差数据上的下一改进重点是 track-based global
positioning 和 triplet scale averaging。

前端为外层任务池并行；单对匹配使用 AVX2 L2，并对 ratio 候选做 reverse NN。

## 真实序列回归

`D:/ScanVideo/chuan/images` 共 109 张 1280x720 图片。以 900 px 近似焦距运行全部
108 个相邻对的 CPU SIFT + mutual-ratio + E/H RANSAC：108/108 有效，Essential
内点最小 317、中位数 1934、平均 1883.5、最大 2908。低视差/近似平面段应根据
Homography 占比从初始化候选中排除，但可以保留为 view-graph 连接边。

每一阶段都应有合成正确性测试、公开数据集回归和单独 benchmark；不要仅以 CPU/GPU
利用率判断性能，利用率高但访存、原子冲突或无效图像对过多，端到端仍可能更慢。
